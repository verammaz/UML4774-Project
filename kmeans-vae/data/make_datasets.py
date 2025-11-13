#!/usr/bin/env python3
"""
Driver to generate BOTH gaussian_raw/ and bernoulli_raw/ under data_set/,
sized so their combined on-disk size is ~target_mb (default 256 MB).
Stores unsplit per-component arrays; use data_io.load_and_split(...) later.

Usage ex:
  python make_datasets.py --k 5 --dims 64 --target-mb 256 --seed 42
"""

import os
import argparse
import numpy as np
from typing import Tuple, List
from gaussian_gen import generate_gaussian_raw
from bernoulli_gen import generate_bernoulli_raw

def compute_counts_for_target_bytes(
    target_bytes_total: int,
    dims: int,
    k: int,
    datasets: int = 2,
    overhead_bytes_per_dir: int = 1_000_000,
) -> List[int]:
    """
    Approximate per-component counts (equal across components) for each dataset,
    assuming float32 features (4 bytes * dims) + int32 labels (4 bytes) per example.
    We split the total budget evenly across 'datasets'.
    """
    bytes_per_example = 4 * dims + 4  # X + y
    budget_each = max(1, (target_bytes_total // datasets) - overhead_bytes_per_dir)
    n_total_each = max(1, budget_each // bytes_per_example)
    per_comp = max(1, n_total_each // k)
    return [int(per_comp)] * k

def dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total / (1024 * 1024)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--target-mb", type=float, default=256.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outroot", type=str, default="data_set")
    args = ap.parse_args()

    np.random.seed(args.seed)

    target_bytes_total = int(args.target_mb * 1024 * 1024)
    n_per_each = compute_counts_for_target_bytes(
        target_bytes_total=target_bytes_total, dims=args.dims, k=args.k, datasets=2
    )

    os.makedirs(args.outroot, exist_ok=True)
    gauss_dir = os.path.join(args.outroot, "gaussian_raw")
    bern_dir  = os.path.join(args.outroot, "bernoulli_raw")

    print(f"Per-dataset per-component counts: {n_per_each} (dims={args.dims})")

    generate_gaussian_raw(
        outdir=gauss_dir,
        k=args.k,
        dims=args.dims,
        n_per=n_per_each,
        seed=args.seed + 1,
    )
    generate_bernoulli_raw(
        outdir=bern_dir,
        k=args.k,
        dims=args.dims,
        n_per=n_per_each,
        seed=args.seed + 2,
    )

    gsz = dir_size_mb(gauss_dir)
    bsz = dir_size_mb(bern_dir)
    tot = gsz + bsz

    print(f"Wrote {gauss_dir}  (~{gsz:.2f} MB)")
    print(f"Wrote {bern_dir}   (~{bsz:.2f} MB)")
    print(f"Total on disk ≈ {tot:.2f} MB (target {args.target_mb:.2f} MB)")
    if abs(tot - args.target_mb) > 10:
        print("Tip: adjust --dims or --target-mb to dial in size more closely.")


if __name__ == "__main__":
    main()
