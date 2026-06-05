"""
train_custom_cifar10.py
---
Train a custom quantized CIFAR-10 model from scratch using the
training_harness infrastructure (optimizer, scheduler, augmentation,
logging, checkpointing — all unchanged).

The key mechanism: We instantiate a quantized MobileNet-style model,
wrap it in the training_harness Trainer, and run QAT with calibration
and EMA handling built into the harness.

Run
---
    # Full CIFAR-10 training (50 epochs, default batch):
    python train_custom_cifar10.py \\
        --workdir ./runs/cifar10_qat

    # Quick smoke-test (1 epoch, batch=16):
    python train_custom_cifar10.py \\
        --smoke-test

    # With explicit device and learning rate:
    python train_custom_cifar10.py \\
        --device cuda --lr 0.01 --epochs 50
"""

import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from training_harness.trainer import Trainer
from training_harness.config import TrainerConfig
from models.cifar10_quant import MobileNetCIFAR


# Example usage: python -m examples.train_custom_cifar10 --device cuda --batch 64 --epochs 50

MAX_BATCHES = 5
counter = {"n": 0}

def stop_early(trainer):
    counter["n"] += 1
    if counter["n"] >= MAX_BATCHES:
        trainer.stop = True


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train quantized CIFAR-10 from scratch using training_harness")
    p.add_argument("--workdir", default="./runs/cifar10_qat",
                   help="Output directory")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--imgsz", type=int, default=32, help="Input image size")
    p.add_argument("--device", default="cuda", help="Device (cuda / cpu)")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a previously saved state dict (.pt) to resume from")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--smoke-test", action="store_true",
                   help="Override to 1 epoch, batch=16, and 1000 samples for a quick sanity check")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.batch = 16
        print("Smoke-test mode: 1 epoch, batch=16")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Quantized CIFAR-10 — {'Fine-tuning' if args.checkpoint else 'Training from Scratch'}")
    print("=" * 60)
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Imgsz   : {args.imgsz}")
    print(f"  Device  : {args.device}")
    print(f"  Workdir : {workdir}")
    if args.checkpoint:
        print(f"  Checkpoint : {args.checkpoint}")

    # 1. Data Loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    if args.smoke_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
        val_dataset = torch.utils.data.Subset(val_dataset, range(500))

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 2. Model
    model = MobileNetCIFAR(num_classes=10)
    model = model.to(args.device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Trainer Config
    config = TrainerConfig(
        workdir=str(workdir),
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        amp=False,  # QAT requires amp=False to avoid autocast conflicts with fake-quant arithmetic
        checkpoint_path=args.checkpoint,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        scheduler=scheduler,
    )

    # trainer.add_callback("on_train_batch_end", stop_early)
    trainer.train()

    # 6. Save clean state dict
    best_ckpt = Path(trainer.save_dir) / "weights" / "best.pt"
    out_path = Path(trainer.save_dir) / "weights" / "best_custom_statedict.pt"

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model") or ckpt
        if hasattr(model_state, "state_dict"):
            torch.save(model_state.state_dict(), out_path)
            print(f"\nClean state dict saved to: {out_path}")
    else:
        print(f"\n⚠️  best.pt not found at {best_ckpt}")

    print(f"\nTraining complete. Results in: {trainer.save_dir}")


if __name__ == "__main__":
    main()
