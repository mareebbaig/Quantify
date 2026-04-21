"""
ImageNet Validation for pretrained MobileNetV2.

This script loads a pretrained MobileNetV2 model and validates it on the 
ImageNet validation dataset.

Run
---
    python examples/validate_imagenet_mobilenetv2.py --data-dir /path/to/imagenet/val --workdir ./runs/imagenet_val
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from utils import add_workspace_args, workspace_from_args
from utils.logging import CSVLogger


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
        
    return 100.0 * correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Validate pretrained MobileNetV2 on ImageNet")
    add_workspace_args(p, name="imagenet_mobilenetv2_val")
    p.add_argument("--data-dir", type=str, required=True, 
                   help="Path to the ImageNet validation directory (containing subfolders per class)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main(args):
    ws = workspace_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Workspace: {ws.root}")
    print(f"Device:    {device}")
    print(f"Data Dir:  {args.data_dir}")

    # ---------------- data ----------------
    # MobileNetV2 pretrained weights expect specific normalization and resizing
    weights = MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    # Using ImageFolder as it is the standard way to load ImageNet validation 
    # when organized in class-named subdirectories.
    val_set = torchvision.datasets.ImageFolder(
        root=args.data_dir, 
        transform=preprocess
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # ---------------- model ----------------
    print("Loading pretrained MobileNetV2...")
    model = mobilenet_v2(weights=weights).to(device)
    model.eval()

    # ---------------- evaluation ----------------
    log_path = ws.logs / "validation_log.csv"
    
    with CSVLogger(log_path, fieldnames=["top1_acc"]) as log:
        acc = evaluate(model, val_loader, device)
        log.log(top1_acc=f"{acc:.2f}")
        print(f"\nValidation Top-1 Accuracy: {acc:.2f}%")

    print(f"Results logged to: {log_path}")


if __name__ == "__main__":
    main(parse_args())
