"""
custom_trainer.py — Custom Ultralytics Trainer for YOLOv8nPANOnly with Brevitas QAT.
"""

import copy

import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

from models.yolov8PanOnly import YOLOv8nPANOnly
import quantizers as q
from quantizers.manager import QuantizerManager
from training_harness.engine_utils import LossPlateauDetector


class CustomYOLOv8nTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass that builds our clean YOLOv8nPANOnly nn.Module
    instead of parsing yolov8n.yaml.

    Only get_model() is overridden. Everything else — loss, optimizer,
    scheduler, augmentation, logging, checkpointing — is Ultralytics stock.

    Compatibility notes
    -------------------
    set_model_attributes() sets model.nc, model.names, model.args.
    Our YOLOv8nPANOnly doesn't define those, but Python allows setting arbitrary
    attributes on nn.Module instances, so this works without any change.

    The DFL freeze ("always_freeze_names = ['.dfl']") matches our
    detect.dfl submodule name, so DFL weights are correctly frozen
    during training_harness (they are fixed by construction anyway).

    The loss function (v8DetectionLoss) reads model.model[-1] to get the
    Detect head's stride, nc, and reg_max. We attach a .model attribute
    that exposes this so the loss can find it.
    """
    def __init__(self, *args, checkpoint: str = None, qat_patience: int = 5, **kwargs):
        # Store checkpoint path before super().__init__ validates overrides
        self._checkpoint = checkpoint
        self.qat_patience = qat_patience
        super().__init__(*args, **kwargs)
        # Disable EMA to prevent it from averaging/corrupting quantizer scales.
        # `ema` is not a valid override argument, so we nullify it after init.
        self.ema = None
        
        # Initialize plateau detector and QAT state
        self.loss_plateau_detector = LossPlateauDetector(patience=qat_patience)
        self.qat_activated = False
        
        # Deactivate quantizers initially for normal float training
        QuantizerManager().disable_quantization()
        
        # Register callback to check plateau at end of each epoch
        self.add_callback("on_train_epoch_end", self._check_qat_plateau)

    def _check_qat_plateau(self, trainer):
        """Callback triggered at the end of each training epoch."""
        if self.qat_activated:
            return

        # Safety check: ensure we are still in the "not quantizing at all" phase
        if not QuantizerManager().is_not_quantizing_at_all:
            return

        # Extract current training loss from Ultralytics trainer state
        loss_val = 0.0
        if hasattr(trainer, 'tloss') and trainer.tloss:
            # Ultralytics stores running averages per loss component in tloss
            loss_val = sum(m.avg for m in trainer.tloss if hasattr(m, 'avg')) / len(trainer.tloss)
        elif hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            if isinstance(trainer.loss_items, dict):
                loss_val = sum(v.item() for v in trainer.loss_items.values()) / len(trainer.loss_items)
            elif hasattr(trainer.loss_items, 'mean'):
                loss_val = trainer.loss_items.mean().item()

        # Check for plateau
        is_plateau = self.loss_plateau_detector.step(loss_val)
        if is_plateau:
            print(f"[QAT Harness] Training loss plateaued after {self.qat_patience} epochs. Activating QAT...")
            self.qat_activated = True
            
            mgr = QuantizerManager()
            mgr.set_annealing_for_n_inferences(6)
            mgr.quantization_start_gap = 20
            print(f"[QAT Harness] QAT activated. Annealing alpha step set, start_gap=20.")

    def save_model(self):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
        }
        torch.save(ckpt, self.last)

        export_model = copy.deepcopy(self.model).float().cpu().eval()

        dummy = torch.zeros(1, 3, 640, 640)
        torch.onnx.export(
            export_model, dummy, str(self.last) + ".onnx",
            dynamo=False,
            opset_version=13,
            custom_opsets={"Quantify": 1},
            do_constant_folding=False,  # keep the custom node visible
            input_names=["input"],
            output_names=["output"],
        )
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
            torch.onnx.export(
                export_model, dummy, str(self.best) + ".onnx",
                dynamo=False,
                opset_version=13,
                custom_opsets={"Quantify": 1},
                do_constant_folding=False,  # keep the custom node visible
                input_names=["input"],
                output_names=["output"],
            )
        del export_model
        return True

    def final_eval(self):
        export_model = copy.deepcopy(self.model).float().cpu().eval()
        self.metrics = self.validator(model=export_model)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build our custom YOLOv8nPANOnly, optionally loading a saved state dict."""
        nc = self.data["nc"]
        
        # Explicitly instantiate a local QuantizerManager to avoid global state leakage.
        # This manager coordinates inference gating, annealing, and recalibration
        # specifically for this training run.
        quantizer_mgr = QuantizerManager()
        # Note: Initial deactivation is handled in __init__ via QuantizerManager().disable_quantization()
        # The plateau callback will override these settings when QAT is triggered.
        
        # Note: Brevitas DI instantiates quantizer classes internally.
        # If your quantizer classes inherit from BaseQuantizer, you can pass the manager
        # via a subclass or wrapper. For now, the manager is configured and ready
        # to be attached to quantizer proxies post-instantiation if needed.
        model = YOLOv8nPANOnly(nc=nc, weight_quant=q.FixedPointPerTensorWeightQuant, act_quant=q.FixedPointPerTensorActivationQuant)
        model = model.to(self.device)

        # Load a previously saved state dict if provided via --checkpoint.
        # self.args.checkpoint is set from overrides in main().
        checkpoint = self._checkpoint
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,  # may also appear
            np.int64,
            np.float64,
            np.ndarray,
        ])
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Support both raw state dicts and Ultralytics-style ckpt dicts
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt  # already a state dict
            else:
                state_dict = ckpt.state_dict()
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if verbose and RANK in {-1, 0}:
                print(f"  Loaded checkpoint: {checkpoint}")
                if missing:
                    print(f"  ⚠️  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

        model.detect.nc = nc
        model.detect.stride = model.stride
        model.end2end = False

        if verbose and RANK in {-1, 0}:
            n_params = sum(p.numel() for p in model.parameters())
            mode = "fine-tuning" if checkpoint else "scratch"
            print(f"Custom YOLOv8nPANOnly ({mode}): nc={nc}, strides={model.stride.tolist()}, {n_params:,} parameters")

        return model
