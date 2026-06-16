"""
benchmark_dataloader.py — isolates where time is lost in the data pipeline.

Run from the repo root:
    python scripts/benchmark_dataloader.py

What it measures (in order):
  1. Raw HF dataset item access (Arrow deserialization only, no image decode)
  2. PIL JPEG decode only (no transforms)
  3. Val transforms  (Resize + CenterCrop + ToTensor + Normalize)
  4. Train transforms (RandAugment + RandomErasing on top of val)
  5. Full DataLoader at various worker counts

Each stage builds on the previous — the delta between stages shows which
operation is eating time.
"""

import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

N_SAMPLES   = 200   # samples timed per per-item benchmark
N_BATCHES   = 30    # batches timed per DataLoader benchmark
BATCH_SIZE  = 64    # small batch so worker count is the variable, not GPU
HF_DATASET  = "ILSVRC/imagenet-1k"
WORKER_COUNTS = [0, 4, 8, 12, 16, 20, 24]

# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------

print(f"Loading {HF_DATASET} validation split …")
t0 = time.perf_counter()
hf_val = load_dataset(HF_DATASET, split="validation")
print(f"  dataset loaded in {time.perf_counter()-t0:.1f}s  ({len(hf_val)} samples)\n")


# ---------------------------------------------------------------------------
# Stage 1: Raw Arrow access (no PIL, no transforms)
# ---------------------------------------------------------------------------

print("─" * 60)
print("Stage 1: Raw HF Arrow item access (no image decode)")
t0 = time.perf_counter()
for i in range(N_SAMPLES):
    _ = hf_val[i]   # returns dict with raw bytes + label
elapsed = time.perf_counter() - t0
print(f"  {N_SAMPLES} items in {elapsed*1000:.1f} ms  →  {elapsed/N_SAMPLES*1000:.2f} ms/item\n")


# ---------------------------------------------------------------------------
# Stage 2: PIL JPEG decode (Arrow access + image.convert("RGB"))
# ---------------------------------------------------------------------------

print("─" * 60)
print("Stage 2: Arrow access + PIL JPEG decode (.convert('RGB'))")
t0 = time.perf_counter()
for i in range(N_SAMPLES):
    item = hf_val[i]
    _ = item["image"].convert("RGB")
elapsed = time.perf_counter() - t0
print(f"  {N_SAMPLES} items in {elapsed*1000:.1f} ms  →  {elapsed/N_SAMPLES*1000:.2f} ms/item\n")


# ---------------------------------------------------------------------------
# Stage 3: Val transforms (decode + Resize + CenterCrop + ToTensor + Normalize)
# ---------------------------------------------------------------------------

bicubic = T.InterpolationMode.BICUBIC
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_tf = T.Compose([
    T.Resize(236, interpolation=bicubic),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize,
])

print("─" * 60)
print("Stage 3: Decode + val transforms (Resize/CenterCrop/ToTensor/Normalize)")
t0 = time.perf_counter()
for i in range(N_SAMPLES):
    item = hf_val[i]
    _ = val_tf(item["image"].convert("RGB"))
elapsed = time.perf_counter() - t0
print(f"  {N_SAMPLES} items in {elapsed*1000:.1f} ms  →  {elapsed/N_SAMPLES*1000:.2f} ms/item\n")


# ---------------------------------------------------------------------------
# Stage 4: Train transforms (adds RandAugment + RandomErasing)
# ---------------------------------------------------------------------------

train_tf = T.Compose([
    T.RandomResizedCrop(224, interpolation=bicubic),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9, interpolation=bicubic),
    T.ToTensor(),
    normalize,
    T.RandomErasing(p=0.25),
])

print("─" * 60)
print("Stage 4: Decode + train transforms (adds RandAugment + RandomErasing)")
t0 = time.perf_counter()
for i in range(N_SAMPLES):
    item = hf_val[i]
    _ = train_tf(item["image"].convert("RGB"))
elapsed = time.perf_counter() - t0
ms_per_item = elapsed / N_SAMPLES * 1000
print(f"  {N_SAMPLES} items in {elapsed*1000:.1f} ms  →  {ms_per_item:.2f} ms/item")
# Project: how many workers are needed to saturate batch_size=1024 in real training?
# One batch of 1024 needs to arrive in ~<GPU_compute_time>.  With 1 worker:
batch1024_time = ms_per_item * 1024 / 1000
print(f"  → single-worker time for batch=1024: {batch1024_time:.1f} s")
print(f"  → workers needed to keep up at 0.5s/batch: {batch1024_time/0.5:.0f}\n")


# ---------------------------------------------------------------------------
# Stage 5: DataLoader throughput at various worker counts
# ---------------------------------------------------------------------------

class _ValWrapper(Dataset):
    def __init__(self, ds, tf):
        self.ds, self.tf = ds, tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return self.tf(item["image"].convert("RGB")), item["label"]

print("─" * 60)
print(f"Stage 5: DataLoader throughput  (batch={BATCH_SIZE}, {N_BATCHES} batches each)")
print(f"  {'Workers':>8}  {'batches/s':>10}  {'samples/s':>11}  {'ms/batch':>10}")
print(f"  {'-------':>8}  {'---------':>10}  {'-----------':>11}  {'--------':>10}")

for nw in WORKER_COUNTS:
    persistent = nw > 0
    prefetch   = 3 if nw > 0 else None
    loader = DataLoader(
        _ValWrapper(hf_val, val_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    it = iter(loader)
    # warm-up: 2 batches (worker startup + cache warm)
    for _ in range(min(2, N_BATCHES)):
        next(it)
    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        next(it)
    elapsed = time.perf_counter() - t0
    bps = N_BATCHES / elapsed
    sps = bps * BATCH_SIZE
    print(f"  {nw:>8}  {bps:>10.2f}  {sps:>11.0f}  {1000/bps:>10.1f}")

print()
print("Done. The largest delta between stages = the bottleneck.")
print("The worker count where samples/s stops growing = the CPU saturation point.")
