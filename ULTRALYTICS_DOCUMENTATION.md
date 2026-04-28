# Ultralytics Framework Documentation

## 1. Core Concepts & Architecture
- **YOLO High-Level API**: The `YOLO` class acts as a unified wrapper for model loading, training, validation, prediction, and export. It internally manages `DetectionModel`, `DetectionTrainer`, `DetectionValidator`, and `Predictor` instances.
- **DetectionModel**: The base PyTorch `nn.Module` for object detection. It parses YAML configuration files into a `torch.nn.Sequential` backbone and head. Key methods include `forward()` (handles both training loss and inference), `init_criterion()`, and `predict()`.
- **DetectionTrainer**: Inherits from `BaseTrainer`. Orchestrates the training loop, validation, checkpointing, logging, and optimizer/scheduler setup. It is the primary extension point for custom training logic.
- **Callbacks**: Ultralytics uses a callback system (`on_train_start`, `on_fit_epoch_end`, `on_model_save`, etc.) to inject logic at specific lifecycle stages. Callbacks accept a `Trainer`, `Validator`, or `Predictor` instance. They are preferred over full trainer subclassing for simple hooks.
- **Loss & Criterion**: Detection models use `v8DetectionLoss` or `E2ELoss`. The loss is computed in `model.loss(batch, preds)` and returns a tuple `(loss, loss_items)`. Custom losses must match this signature.

## 2. Training Pipeline & Customization
- **Custom Trainer Pattern**: To customize training without breaking the core loop, subclass `DetectionTrainer` and override specific methods:
  ```python
  from ultralytics.models.yolo.detect import DetectionTrainer

  class CustomTrainer(DetectionTrainer):
      def get_model(self, cfg, weights, verbose):
          # Return custom Brevitas-quantized model
          return MyQuantizedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
      
      def build_optimizer(self, model, name, lr, momentum, decay, iterations):
          # Custom optimizer setup if needed
          return super().build_optimizer(model, name, lr, momentum, decay, iterations)
  ```
  Pass it via `model.train(trainer=CustomTrainer, ...)`.
- **Key Overridable Methods**:
  - `get_model(cfg, weights)`: Instantiate the model.
  - `build_optimizer(model, ...)`: Configure optimizer parameter groups.
  - `validate()`: Run validation and return `(metrics, fitness)`. Override to inject custom metrics or early stopping logic.
  - `save_model()`: Handle checkpoint saving. Override to export quantized states or custom metadata.
  - `get_validator()`: Return a custom validator instance.
- **Callback Injection**: For non-invasive hooks, use `model.add_callback("on_train_epoch_end", my_callback)`. Useful for logging Brevitas scales, saving intermediate ONNX exports, or triggering calibration.

## 3. Quantization Integration (QAT & PTQ)
- **PTQ (Post-Training Quantization)**: Ultralytics supports built-in PTQ via `model.export(format="onnx", int8=True, data="calibration.yaml")`. It uses a calibration dataloader to collect activation statistics and applies per-channel/per-tensor quantization during export. No training loop modification is needed.
- **QAT (Quantization-Aware Training) with Brevitas**: 
  - Replace standard layers (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`) with Brevitas `qnn.QuantConv2d`, `qnn.QuantLinear`, etc., in your custom `DetectionModel`.
  - **Custom Quantizers**: Use Brevitas `ExtendedInjector` to define quantizers. Example structure:
    ```python
    from brevitas.inject import ExtendedInjector
    from brevitas.core import RescalingIntQuant, ScalingImpl

    class MyFixedPointQuant(ExtendedInjector):
        proxy_class = WeightQuantProxyFromInjector
        tensor_quant = RescalingIntQuant
        scaling_impl = ScalingImpl(...)
        bit_width = 8
        signed = True
    ```
  - Inject quantizers via model kwargs: `qnn.QuantConv2d(..., weight_quant=MyFixedPointQuant, input_quant=MyActQuant)`.
- **AMP Compatibility**: Ultralytics enables Automatic Mixed Precision (`amp=True`) by default. **Warning**: Brevitas fake quantization can conflict with AMP due to gradient scaling and precision mismatches in `autocast`. 
  - **Workaround**: Disable AMP during QAT by passing `amp=False` to `model.train(amp=False)`. If AMP is required, ensure Brevitas quantizers use `float32` accumulation or wrap quantization math outside `autocast` scope. Monitor gradient norms closely.

## 4. Calibration & Data Flow
- **QAT Statistics Collection**: During QAT, Brevitas collects activation scales/zero-points per batch. Ensure your Ultralytics dataloaders (`DetectionTrainer.get_dataloader()`) provide a representative, shuffled dataset. Small or biased batches will skew quantization parameters.
- **PTQ Calibration Flow**: When exporting with `int8=True`, Ultralytics runs a calibration pass over the dataset specified in `data=`. It iterates through batches, runs forward passes, and collects histograms/statistics. Ensure calibration data matches training distribution.
- **Brevitas Calibration Mode**: Before export, you can run Brevitas' `calibration_mode` context manager to freeze scales based on a calibration set, then switch to `eval()` for inference/export:
  ```python
  from brevitas.graph.calibrate import calibration_mode
  with calibration_mode(model):
      for batch in loader: model(batch)
  model.eval()
  ```
- **Data Pipeline Integration**: Ultralytics handles preprocessing, augmentation, and batching. Brevitas quantization happens inside the model's `forward()`. No special dataloader modifications are needed, but ensure `batch["img"]` is passed correctly to the quantized model.

## 5. Export & Deployment
- **Standard Export**: `model.export(format="onnx", half=True, dynamic=True, simplify=True)`. Ultralytics handles graph construction, opset selection, and optimization.
- **QCDQ Export**: For Brevitas models, use `export_onnx_qcdq(model, input, path)` to insert `QuantizeLinear`/`DeQuantizeLinear` nodes. This is ORT-compatible and supports arbitrary bit-widths via clipping.
- **Custom ONNX Nodes**: If using Brevitas custom ops (`torch.autograd.Function.symbolic`), you **must** use the legacy exporter: `torch.onnx.export(..., dynamo=False)`. Modern `torch.export` does not support `symbolic`.
- **TensorRT Export**: `model.export(format="engine", int8=True, data="cal.yaml")`. Requires NVIDIA GPU and TensorRT. Handles PTQ calibration internally.
- **ONNX Runtime Compatibility**: QCDQ models run natively in ORT. Custom `mydomain::` nodes will cause fallback warnings. For ORT deployment, prefer QCDQ or implement custom ORT kernels.

## 6. Common Pitfalls & Debugging
- **AMP + Brevitas Conflicts**: Mixed precision can cause `RuntimeError` or silent accuracy drops during QAT. Disable `amp=True` in trainer args. If gradients explode, check Brevitas scale initialization and learning rate.
- **Trainer Override Scope**: Overriding `get_model()` is usually sufficient. Avoid rewriting `_do_train()` or `train()` unless absolutely necessary, as it breaks Ultralytics' DDP, EMA, and callback integration.
- **Loss Return Format**: Custom loss functions must return `(loss, loss_items)`. `loss_items` should be a tensor or dict for logging. Mismatched formats break `save_metrics()` and progress bars.
- **Callback Timing**: `on_fit_epoch_end` runs *after* validation. `on_train_epoch_end` runs *before*. Use `on_fit_epoch_end` for checkpointing/metrics that depend on validation results.
- **Calibration Data Mismatch**: PTQ accuracy drops significantly if calibration data doesn't represent inference distribution. Use a subset of training data or a dedicated calibration set.
- **Dynamic Axes & Dummy Inputs**: When exporting, ensure dummy inputs match expected runtime dimensions. Pass `dynamic_axes` if batch/height/width vary. Brevitas quantization is shape-agnostic but scales are per-tensor/per-channel.

## 7. Missing Information / Future Expansion
- **Brevitas DI Wiring in Ultralytics**: The exact integration pattern for Brevitas `ExtendedInjector` within Ultralytics' `parse_model()` YAML parser is not fully documented. Needs a dedicated example showing how to register custom quantizers globally or per-layer in Ultralytics configs.
- **EMA & Quantization**: Ultralytics uses Exponential Moving Average (`ModelEMA`) for checkpointing. Brevitas quantization scales are updated per-batch. Document how EMA interacts with quantized models (usually EMA should track FP weights, not quantized scales).
- **DDP & Quantization**: Distributed Data Parallel training with Brevitas fake quantization may require gradient synchronization of scales. Ultralytics' DDP setup (`_setup_ddp`) doesn't explicitly handle quantization metadata. Needs verification.
- **Custom Validator Metrics**: How to inject Brevitas-specific metrics (e.g., scale drift, bit-width utilization) into Ultralytics' validation loop without breaking `DetMetrics` structure.
- **ONNX Opset & Brevitas**: Specific opset version requirements for Brevitas QCDQ vs custom nodes. Ultralytics defaults to latest, which may break legacy custom ops.
