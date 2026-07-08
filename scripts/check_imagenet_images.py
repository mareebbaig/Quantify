"""
Scan all images in an ImageNet-layout dataset, find broken files,
attempt recovery via PIL's truncated-image mode, and save fixable
images to an output directory.

Usage:
    python scripts/check_imagenet_images.py \
        --data-dir /home/th/tmp/datasets/imagenet \
        --output-dir /home/th/maybe_fixed/train_dataset

Output:
  - Fixed images written to <output-dir>/{split}/{class}/{file}.jpg
  - broken_train.txt / broken_val.txt listing unrecoverable paths
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _worker_init():
    # Start each worker in strict mode; step 2 enables truncation tolerance.
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def _check_file(args: Tuple[str, str, str]) -> Tuple[str, str]:
    """
    Returns (status, rel_path) where status is:
      'ok'     — image decoded normally; nothing written
      'fixed'  — image was corrupt/truncated but recoverable; saved to output_dir
      'broken' — unrecoverable; path will be logged
    """
    abs_path, rel_path, output_dir = args

    # Step 1: strict check — LOAD_TRUNCATED_IMAGES is False, so truncated
    # files raise OSError and are caught here rather than silently passing.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(abs_path) as img:
            img.load()
        return ('ok', rel_path)
    except Exception:
        pass

    # Step 2: recovery — enable truncation tolerance, re-decode, save clean JPEG.
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(abs_path) as img:
            img_rgb = img.convert('RGB')
        out_path = Path(output_dir) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_rgb.save(str(out_path), 'JPEG', quality=95)
        return ('fixed', rel_path)
    except Exception:
        return ('broken', rel_path)


# ---------------------------------------------------------------------------
# Per-split scan
# ---------------------------------------------------------------------------

def _collect_files(split_dir: Path, data_dir: Path) -> List[Tuple[str, str]]:
    """Collect (abs_path, rel_path) tuples for all image files in a split."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    files = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in exts:
                files.append((str(f), str(f.relative_to(data_dir))))
    return files


def scan_split(
    split: str,
    data_dir: Path,
    output_dir: Path,
    num_workers: int,
) -> dict:
    split_dir = data_dir / split
    if not split_dir.exists():
        print(f"  [skip] {split_dir} not found")
        return {}

    print(f"\nCollecting {split} file list …")
    files = _collect_files(split_dir, data_dir)
    print(f"  {len(files):,} files found")

    tasks = [(abs_p, rel_p, str(output_dir)) for abs_p, rel_p in files]

    counts = {'ok': 0, 'fixed': 0, 'broken': 0}
    broken_paths: List[str] = []

    with multiprocessing.Pool(num_workers, initializer=_worker_init) as pool:
        for status, rel_path in tqdm(
            pool.imap_unordered(_check_file, tasks, chunksize=64),
            total=len(tasks),
            desc=f"  {split}",
            unit="img",
        ):
            counts[status] += 1
            if status == 'broken':
                broken_paths.append(rel_path)

    broken_file = Path(f"broken_{split}.txt")
    if broken_paths:
        broken_paths.sort()
        broken_file.write_text('\n'.join(broken_paths) + '\n')
        print(f"  Broken paths written to {broken_file}")

    print(f"\n=== {split} ===")
    for k, v in counts.items():
        print(f"  {k:<8}: {v:>10,}")

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check and fix ImageNet images")
    p.add_argument(
        "--data-dir",
        default="/home/th/tmp/datasets/imagenet",
        help="Root with train/ and val/ subdirectories",
    )
    p.add_argument(
        "--output-dir",
        default="/home/th/maybe_fixed/train_dataset",
        help="Where to save fixed images (only broken-but-recoverable files)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to check (default: train val)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(multiprocessing.cpu_count(), 32),
        help="Number of parallel worker processes",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Dataset : {data_dir}")
    print(f"Output  : {output_dir}")
    print(f"Workers : {args.workers}")
    print(f"Splits  : {args.splits}")

    total = {'ok': 0, 'fixed': 0, 'broken': 0}
    for split in args.splits:
        counts = scan_split(split, data_dir, output_dir, args.workers)
        for k in total:
            total[k] += counts.get(k, 0)

    print(f"\n{'='*40}")
    print("TOTAL")
    for k, v in total.items():
        print(f"  {k:<8}: {v:>10,}")
    if total['broken'] > 0:
        print(f"\n  {total['broken']} unrecoverable files listed in broken_{{split}}.txt")
    if total['fixed'] > 0:
        print(f"  {total['fixed']} fixed files saved to {output_dir}")


if __name__ == "__main__":
    main()
