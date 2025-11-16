import os
import pandas as pd
import numpy as np
import h5py
import anndata as ad
import scanpy as sc
from tqdm import tqdm


def download_data(datadir):
    os.makedirs(datadir, exist_ok=True)

    os.system(
        f'wget -c "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/expression_matrix.hdf5" -O {os.path.join(datadir, "expression_matrix.hdf5")}'
    )

    os.system(
        f'wget -c "https://idk-etl-prod-download-bucket.s3.amazonaws.com/aibs_mouse_ctx-hpf_10x/metadata.csv" -O {os.path.join(datadir, "metadata.csv")}'
    )



def load_hd5a(datapath):
    adata = sc.read_h5ad(datapath)
    print(f"Loaded expression data: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata


def load_hdf5_expr_matrix(datapath):
    f = h5py.File(datapath, 'r')

    for key in f['data']:
        print(key, f['data'][key].shape)
    
    counts = f['data']['counts'][:]
    genes = np.array(f['data']['gene']).astype(str)
    samples = np.array(f['data']['samples']).astype(str)

    adata = ad.AnnData(X=counts.T)
    adata.var_names = genes
    adata.obs_names = samples

    return adata


def load_metadata(metapath, label_col='subclass_label'):
    # assumes all samples in expr_matrix == samples in meta_matrix
    meta = pd.read_csv(metapath)
    print(f"No. cell types: {meta[label_col].nunique()}")
    return meta



def create_proportional_subset_h5ad(
    expr_path,
    metadata_path,
    output_h5ad='subset_expression.h5ad',
    output_meta_csv='subset_metadata.csv',
    max_cells=5000,
    celltype_col='subclass_label',
    min_prop=0.001,
    random_state=42,
    chunk_size=1000,
):

    rng = np.random.default_rng(random_state)

    # Load metadata
    meta = pd.read_csv(metadata_path)
    if celltype_col not in meta.columns:
        raise ValueError(f"{celltype_col} not found in metadata columns")
    meta['sample_name'] = meta['sample_name'].astype(str).str.strip()
    meta = meta.set_index('sample_name')

    # Inspect HDF5
    with h5py.File(expr_path, 'r') as f:
        n_genes, n_samples = f['data']['counts'].shape
        print(f"Expression matrix: {n_genes:,} genes × {n_samples:,} cells")

    # Build mapping sample → index
    print("Building sample name → index map...")
    sample_to_idx = {}
    with h5py.File(expr_path, 'r') as f:
        sample_ds = f['data']['samples']
        total = len(sample_ds)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = [s.decode('utf-8').strip() for s in sample_ds[start:end]]
            for i, name in enumerate(chunk, start):
                if name in meta.index:
                    sample_to_idx[name] = i

    print(f"Found {len(sample_to_idx)} overlapping samples")
    assert len(sample_to_idx) > 0, "No overlapping cells found."

    meta = meta.loc[meta.index.intersection(sample_to_idx.keys())]

    # Proportions
    type_counts = meta[celltype_col].value_counts()
    proportions = type_counts / type_counts.sum()

    print("Original proportions:")
    print(proportions)

    # Skip small labels
    valid_labels = [
        ct for ct in proportions.index
        if proportions[ct] >= min_prop
    ]

    print(f"Skipping labels: {set(proportions.index) - set(valid_labels)}")
    print(f"Using labels: {valid_labels} ({len(valid_labels)})")

    proportions = proportions.loc[valid_labels]
    type_counts = type_counts.loc[valid_labels]

    # Determine sample sizes
    n_cells_per_type = (proportions * max_cells).round().astype(int)
    n_cells_per_type[n_cells_per_type == 0] = 1

    sampled_cells = []
    for ct, n in n_cells_per_type.items():
        ct_cells = meta.index[meta[celltype_col] == ct].tolist()
        n_pick = min(n, len(ct_cells))
        sampled_cells.extend(rng.choice(ct_cells, size=n_pick, replace=False))

    subset_idx = [sample_to_idx[s] for s in sampled_cells]
    subset_idx_sorted = np.sort(subset_idx)

    print(f"Sampling {len(subset_idx_sorted)} cells from {len(valid_labels)} labels.")

    # Load genes
    with h5py.File(expr_path, 'r') as f:
        genes = [g.decode('utf-8') for g in f['data']['gene']]

    # Read expression subset
    print("Reading expression matrix...")
    chunks = []

    with h5py.File(expr_path, 'r') as f:
        dset = f['data']['counts']
        for start in tqdm(range(0, len(subset_idx_sorted), chunk_size)):
            end = min(start + chunk_size, len(subset_idx_sorted))
            idx_chunk = subset_idx_sorted[start:end]
            chunk = dset[:, idx_chunk].T.astype(np.float32)
            chunks.append(chunk)

    counts_subset = np.vstack(chunks)

    # Build AnnData
    adata = ad.AnnData(X=counts_subset)
    adata.var_names = genes
    adata.obs_names = sampled_cells
    adata.obs = meta.loc[sampled_cells]

    adata.write(output_h5ad)
    adata.obs.to_csv(output_meta_csv)

    print(f"Saved: {output_h5ad} (shape {adata.shape})")
    print(f"Saved: {output_meta_csv} (shape {adata.obs.shape})")

    return adata



def get_sample_labels(meta_df, label_col='subclass_label'):
    cluster_lables = meta_df[['sample_name', label_col]]
    n_clusters = cluster_lables[label_col].nunique()
    return n_clusters, cluster_lables, 