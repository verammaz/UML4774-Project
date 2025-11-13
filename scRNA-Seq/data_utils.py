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
    celltype_col='class_label',
    random_state=42,
    chunk_size=500
):
    
    rng = np.random.default_rng(random_state)

    # Load metadata
    meta = pd.read_csv(metadata_path)
    if celltype_col not in meta.columns:
        raise ValueError(f"{celltype_col} not found in metadata columns")
    meta['sample_name'] = meta['sample_name'].astype(str).str.strip()
    meta = meta.set_index('sample_name')

    # Open HDF5 file and inspect structure
    with h5py.File(expr_path, 'r') as f:
        n_genes, n_samples = f['data']['counts'].shape
        print(f"Expression matrix shape: {n_genes:,} genes x {n_samples:,} samples")

    # Build a mapping of sample_name → index without loading all names
    print("Building sample name → index mapping (streaming in chunks)...")
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

    if len(sample_to_idx) == 0:
        raise ValueError("No overlapping samples between metadata and HDF5")

    # Restrict metadata to only those overlapping samples
    meta = meta.loc[meta.index.intersection(sample_to_idx.keys())]

    # Compute proportional sample sizes per cell type
    type_counts = meta[celltype_col].value_counts()
    proportions = type_counts / type_counts.sum()
    
    print(proportions)

    n_cells_per_type = (proportions * max_cells).round().astype(int)
    n_cells_per_type[n_cells_per_type == 0] = 1
   
    sampled_cells = []
    for ct, n in n_cells_per_type.items():
        ct_cells = meta.index[meta[celltype_col] == ct].tolist()
        n_pick = min(n, len(ct_cells))
        sampled_cells.extend(rng.choice(ct_cells, size=n_pick, replace=False))

    
    subset_idx = [sample_to_idx[s] for s in sampled_cells]
    subset_idx_sorted = np.sort(subset_idx)
    print(f"Sampling {len(subset_idx_sorted)} total cells across {len(n_cells_per_type)} cell types")

    # Load gene names (small, safe to load fully)
    with h5py.File(expr_path, 'r') as f:
        genes = [g.decode('utf-8') for g in f['data']['gene']]

    # Load subset of expression matrix
    print("Reading subset of expression matrix (this may take a few minutes)...")

    # Load expression matrix subset in chunks
    print(f"Loading expression data in chunks of {chunk_size} cells...")
    chunks = []

    with h5py.File(expr_path, 'r') as f:
        counts_dset = f['data']['counts']
        for start in tqdm(
            range(0, len(subset_idx_sorted), chunk_size),
            desc="Reading expression chunks",
            unit="chunk"
        ):
            end = min(start + chunk_size, len(subset_idx_sorted))
            idx_chunk = subset_idx_sorted[start:end]
            # Read genes x cells chunk, then transpose
            chunk = counts_dset[:, idx_chunk].T.astype(np.float32)
            print(chunk)
            chunks.append(chunk)

    counts_subset = np.vstack(chunks)

    # Create AnnData and save
    adata = ad.AnnData(X=counts_subset)
    adata.var_names = genes
    adata.obs_names = sampled_cells
    adata.obs = meta.loc[sampled_cells]

    adata.write(output_h5ad)
    adata.obs.to_csv(output_meta_csv)

    print(f"Saved subset AnnData (shape {adata.shape}) {output_h5ad}")
    print(f"Saved subset metadata (shape {adata.obs.shape}): {output_meta_csv}")
    
    return adata


def get_sample_labels(meta_df, label_col='subclass_label'):
    cluster_lables = meta_df[['sample_name', label_col]]
    n_clusters = cluster_lables[label_col].nunique()
    return n_clusters, cluster_lables, 