#!/usr/bin/env python3
"""
I/O helpers for saving component-wise raw datasets and loading stratified splits.
- save_components(outdir, components, meta)
- load_and_split(indir, splits=(0.7,0.15,0.15), seed=42, normalize=True, save_manifest=True)
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Sequence

Array = np.ndarray

def save_components(outdir: str,
                    components: Sequence[Dict[str, Array]],
                    meta: Dict):
    """
    Save per-component arrays and metadata.

    components: list of dicts like:
        {"X": float32 array (n_i, d), "y": int32 array (n_i,), "name": str}
    meta: dict with high-level info, merged into metadata.json
    """
    os.makedirs(outdir, exist_ok=True)
    index = []
    for i, comp in enumerate(components):
        name = comp.get("name", f"comp_{i}")
        x_path = os.path.join(outdir, f"{name}_X.npy")
        y_path = os.path.join(outdir, f"{name}_y.npy")
        np.save(x_path, comp["X"])
        np.save(y_path, comp["y"])
        index.append({"name": name, "X": os.path.basename(x_path), "y": os.path.basename(y_path)})
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump({"index": index, **meta}, f, indent=2)

def _stratified_split_indices(y: Array,
                              splits: Tuple[float, float, float],
                              rng: np.random.Generator):
    assert np.isclose(sum(splits), 1.0), "splits must sum to 1"
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(splits[0] * n))
        n_val   = int(round(splits[1] * n))
        # keep at least 1 for test where possible
        n_train = min(n_train, n - 2) if n >= 3 else max(0, n - 2)
        n_val   = min(n_val,   n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train - 1)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)

def _standardize_fit(X: Array):
    mu = X.mean(axis=0).astype(np.float32, copy=False)
    sd = X.std(axis=0, ddof=0).astype(np.float32, copy=False)
    sd = np.where(sd == 0, 1.0, sd).astype(np.float32)
    return ((X - mu) / sd).astype(np.float32), mu, sd

def _standardize_apply(X: Array, mu: Array, sd: Array):
    return ((X - mu) / sd).astype(np.float32)

def load_and_split(indir: str,
                   splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                   seed: int = 42,
                   normalize: bool = True,
                   save_manifest: bool = True):
    """
    Load unsplit components from 'indir', concatenate, make stratified splits,
    and (optionally) standardize using train-only statistics.

    Returns a dict with X_train/X_val/X_test, y_*, train_mean/train_std, meta.
    Also writes splits.json if save_manifest=True.
    """
    with open(os.path.join(indir, "metadata.json")) as f:
        meta = json.load(f)

    parts_X, parts_y = [], []
    for item in meta["index"]:
        X = np.load(os.path.join(indir, item["X"]))
        y = np.load(os.path.join(indir, item["y"]))
        parts_X.append(X)
        parts_y.append(y)

    X = np.vstack(parts_X).astype(np.float32)
    y = np.concatenate(parts_y).astype(np.int32)

    rng = np.random.default_rng(seed)
    tr, va, te = _stratified_split_indices(y, splits, rng)

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    if normalize:
        X_tr, mu, sd = _standardize_fit(X_tr)
        X_va = _standardize_apply(X_va, mu, sd)
        X_te = _standardize_apply(X_te, mu, sd)
    else:
        mu = X_tr.mean(axis=0).astype(np.float32)
        sd = X_tr.std(axis=0, ddof=0).astype(np.float32)
        sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

    if save_manifest:
        manifest = {
            "ratios": splits,
            "seed": seed,
            "splits": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
        }
        with open(os.path.join(indir, "splits.json"), "w") as f:
            json.dump(manifest, f)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_va, "y_val": y_va,
        "X_test": X_te, "y_test": y_te,
        "train_mean": mu, "train_std": sd,
        "meta": meta
    }
