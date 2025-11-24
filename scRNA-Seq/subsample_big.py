#!/usr/bin/env python3
"""
Preprocessing for DESC (Option A: NO PCA).
- Stratified subsample, HVG selection, normalize, log1p, MinMax scale.
- Saves: desc_data/X_scaled_hvg.npy, desc_data/adata.h5ad, desc_data/metadata_subset.csv
"""

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import time

# ----------------- CONFIG -----------------
FILEPATH = "/home/ryanghosh/UML4774-Project/scRNA-Seq/data/expression_matrix.hdf5"
METADATA_PATH = "/home/ryanghosh/UML4774-Project/scRNA-Seq/data/metadata.csv"
OUT_DIR = "desc_data_no_pca"
PRESET = "large"   # "test","small","medium","large"
TARGET_N_CELLS = 100_000
RANDOM_SEED = 42
MIN_PER_REGION = None   # optional
# PRESETS define #HVG target size (n_genes)
PRESETS = {
    'test':   {'n_cells': 1000,  'n_genes': 2000},
    'small':  {'n_cells': 5000,  'n_genes': 2500},
    'medium': {'n_cells': 10000, 'n_genes': 3000},
    'large':  {'n_cells': 100000, 'n_genes': 3000},  # use 100k target cells by default
}
# ------------------------------------------

def stratified_sample_meta(meta_df, region_col="region_label", sample_col="sample_name",
                           target_total=100000, min_per_region=None, seed=42):
    np.random.seed(seed)
    total_cells = len(meta_df)
    regions = meta_df[region_col].value_counts().sort_index()
    n_regions = len(regions)
    region_target = {}
    if min_per_region is None:
        floats = (regions / total_cells) * target_total
        floored = np.floor(floats).astype(int)
        remainder = int(target_total - floored.sum())
        fracs = floats - floored
        order = np.argsort(-fracs)
        if remainder > 0:
            floored[order[:remainder]] += 1
        for r, n in zip(regions.index, floored):
            region_target[r] = max(1, int(n))
    else:
        remaining = target_total - int(min_per_region) * n_regions
        if remaining < 0:
            raise ValueError("MIN_PER_REGION * n_regions > target_total")
        prop = (regions / total_cells) * remaining
        floored = np.floor(prop).astype(int)
        remainder = int(remaining - floored.sum())
        fracs = prop - floored
        order = np.argsort(-fracs)
        if remainder > 0:
            floored[order[:remainder]] += 1
        for r, n in zip(regions.index, floored):
            region_target[r] = int(n) + int(min_per_region)

    sampled = []
    for region, targ in region_target.items():
        pool = meta_df[meta_df[region_col] == region][sample_col].values
        if len(pool) == 0:
            continue
        if len(pool) <= targ:
            chosen = pool.tolist()
        else:
            chosen = np.random.choice(pool, size=targ, replace=False).tolist()
        sampled.extend(chosen)

    sampled = list(dict.fromkeys(sampled))
    return sampled

def main():
    start_time = time.time()
    np.random.seed(RANDOM_SEED)

    if PRESET not in PRESETS:
        raise ValueError(f"Unknown preset {PRESET}")
    n_genes_target = PRESETS[PRESET]['n_genes']

    print(f"\n=== SUBSAMPLING (stratified by region) + preprocessing for DESC (NO PCA) ===")
    print(f"Target cells: {TARGET_N_CELLS:,}, preset genes: {n_genes_target:,}")
    print(f"HDF5 file: {FILEPATH}")
    print(f"Metadata: {METADATA_PATH}")

    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f"HDF5 file not found: {FILEPATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    meta = pd.read_csv(METADATA_PATH, dtype=str)
    sample_col = "sample_name"
    if sample_col not in meta.columns:
        for alt in ["samples", "Sample", "sampleID", "cell"]:
            if alt in meta.columns:
                sample_col = alt
                break
    if "region_label" not in meta.columns:
        raise ValueError("metadata CSV must contain a 'region_label' column for stratified sampling")

    sampled_names = stratified_sample_meta(meta, region_col="region_label", sample_col=sample_col,
                                          target_total=TARGET_N_CELLS, min_per_region=MIN_PER_REGION,
                                          seed=RANDOM_SEED)
    print(f"Selected {len(sampled_names):,} sample names (target was {TARGET_N_CELLS:,})")

    with h5py.File(FILEPATH, 'r') as f:
        data_group = f['data']
        samples_ds = data_group['samples'][:]
        samples = [s.decode('utf-8') if isinstance(s, (bytes, bytearray)) else str(s) for s in samples_ds]
        sample_to_idx = {s: i for i, s in enumerate(samples)}
        n_genes_total, n_cells_total = data_group['counts'].shape
        print(f"✓ HDF5 reports: {n_cells_total:,} cells × {n_genes_total:,} genes")

    present = [s for s in sampled_names if s in sample_to_idx]
    missing = set(sampled_names) - set(present)
    if missing:
        print(f"⚠ {len(missing):,} sampled names not in HDF5; skipping")

    selected_indices = np.array([sample_to_idx[s] for s in present], dtype=int)
    selected_indices.sort()
    print(f"Using {len(selected_indices):,} indices from HDF5")

    # select genes
    with h5py.File(FILEPATH, 'r') as f:
        gene_ids_raw = f['data']['gene'][:]
    gene_ids = [g.decode('utf-8') if isinstance(g, (bytes,bytearray)) else str(g) for g in gene_ids_raw]
    np.random.seed(RANDOM_SEED)
    if n_genes_target >= len(gene_ids):
        selected_gene_idx = np.arange(len(gene_ids), dtype=int)
        print(f"Requested genes >= total genes; keeping all {len(gene_ids)} genes")
    else:
        selected_gene_idx = np.sort(np.random.choice(len(gene_ids), size=n_genes_target, replace=False))
        print(f"Randomly selected {len(selected_gene_idx):,} genes")

    selected_gene_ids = [gene_ids[i] for i in selected_gene_idx]

    # extract matrix in blocks (cells x genes)
    n_sel_cells = len(selected_indices)
    n_sel_genes = len(selected_gene_idx)
    X_subset = np.zeros((n_sel_cells, n_sel_genes), dtype=np.float32)
    write_pos = 0
    max_block_cells = 5000
    sel_gene_idx_array = np.array(selected_gene_idx, dtype=int)

    with h5py.File(FILEPATH, 'r') as f:
        counts = f['data']['counts']
        for start in range(0, n_sel_cells, max_block_cells):
            end = min(n_sel_cells, start + max_block_cells)
            cols_block = selected_indices[start:end]
            cols_block_sorted = np.sort(cols_block)
            block_all = counts[:, cols_block_sorted]
            block_sel = np.array(block_all[sel_gene_idx_array, :], dtype=np.float32)
            block_np = block_sel.T
            X_subset[write_pos:write_pos + block_np.shape[0], :] = block_np
            write_pos += block_np.shape[0]
            print(f"  extracted cells {start:,}..{end:,} (wrote {write_pos:,}/{n_sel_cells:,})")

    # create AnnData and preprocess
    selected_sample_ids = []
    with h5py.File(FILEPATH, 'r') as f:
        samples_ds2 = f['data']['samples'][:]
        samples_list = [s.decode('utf-8') if isinstance(s, (bytes,bytearray)) else str(s) for s in samples_ds2]
        selected_sample_ids = [samples_list[i] for i in selected_indices]

    adata = sc.AnnData(X=X_subset)
    adata.obs_names = selected_sample_ids
    adata.var_names = selected_gene_ids

    sc.pp.filter_genes(adata, min_cells=10)
    print(f"After filtering: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes selection (keeps up to n_genes_target)
    n_top_genes = min(n_genes_target, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    print(f"Selected top {n_top_genes} HVGs → {adata.shape[1]} genes")

       # -------------------------------
    # STEP 5 (modified): Scale for DESC using Scanpy standardization + clipping
    # (zero-center & unit variance, then clip values to [-10, 10])
    # -------------------------------
    print("\n[STEP 5] Scaling for DESC (zero-mean, unit-variance, clipped)...")
    # sc.pp.scale modifies `adata.X` in-place
    # This matches the original DESC pipeline which uses sc.pp.scale(max_value=10)
    sc.pp.scale(adata, max_value=10)

    # Extract dense numpy array (float32) for DESC
    X = adata.X
    if sp.issparse(X):
        X = X.toarray().astype(np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    # Name it X_scaled to match downstream code
    X_scaled = X
    print(f"✓ Scaled matrix: {X_scaled.shape}, approx range [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

    # Save outputs (NO PCA)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_scaled_hvg.npy"), X_scaled.astype(np.float32))
    adata.write(os.path.join(OUT_DIR, "adata.h5ad"))

    meta_subset = meta[meta[sample_col].isin(selected_sample_ids)].copy()
    meta_subset['__order__'] = meta_subset[sample_col].apply(lambda s: selected_sample_ids.index(s))
    meta_subset = meta_subset.sort_values('__order__').drop(columns='__order__')
    meta_subset.to_csv(os.path.join(OUT_DIR, "metadata_subset.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"DESC preprocessing (NO PCA)\n")
        f.write(f"Subset: {X_scaled.shape[0]:,} cells × {X_scaled.shape[1]:,} HVGs\n")
        f.write("Normalization: total counts → log1p → sc.pp.scale (zero-mean, unit-variance) clipped to [-10, 10]\n")

    elapsed = time.time() - start_time
    print(f"\n✅ Done! Data ready for DESC in '{OUT_DIR}/' ({elapsed/60:.1f} min)")
    print(f"  • X_scaled_hvg.npy ({X_scaled.shape})")
    print(f"  • adata.h5ad ({adata.shape})")
    print(f"  • metadata_subset.csv")
    print(f"  • summary.txt")


if __name__ == "__main__":
    main()
