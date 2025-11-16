import scanpy as sc
import sys
import yaml

import data_utils


def preprocess(adata, min_cells=3, n_top_genes=2000, subset_hgv=True, max_value=10, target_sum=1e4):
    # Filter genes 
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalize per cell (counts per 10,000)
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log-transform
    sc.pp.log1p(adata)

    # Select highly variable genes (HVGs)
    if subset_hgv:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True, flavor='seurat')

    # Scale data
    sc.pp.scale(adata, max_value=max_value)
    print(f"Finished preprocessing: {adata.shape[0]} cells x {adata.shape[1]} HVGs")

    return adata


def main():
    # Load YAML input arguments
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)  
    
    adata = data_utils.load_hd5a(params['expression'])

    processed_adata = preprocess(adata)

    processed_adata.write(params['expression_processed'])


if __name__ == "__main__":
    main()

