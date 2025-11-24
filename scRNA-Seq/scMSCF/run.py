import sys
import os
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import meta_cluster
import pca_kmeans
import transformer_cluster

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_utils
import data_preprocess
import plot


def main():
    class_col = 'subclass_label'
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)

    raw = data_utils.load_hd5a(params['expression'])
    
    # step 1: preprocess
    adata = data_preprocess.preprocess(raw)

    # step 2: pca + kmeans
    meta = data_utils.load_metadata(params['metadata'], label_col=class_col)
    n_clusters, true_labels = data_utils.get_sample_labels(meta, label_col=class_col)
    pca_dims_list = params['pca_dims']

    all_labels = pca_kmeans.multi_pca_kmeans(adata, pca_dims_list, k_range=range(2, n_clusters+1), plots_dir=params['plots'])

    cluster_df = pd.DataFrame(
    {col: labels for col, labels in all_labels.items()},
    index=adata.obs_names
    )
    cluster_df.index.name = "cell_name"
    cluster_df.to_csv(params['kmeans_pca_out'])


    # step 3: meta clustering
    # Load base cluster results
    cluster_df = pd.read_csv(params['kmeans_pca_out']).set_index("cell_name")
    all_labels = [cluster_df[col].to_numpy() for col in cluster_df.columns]
    sample_names = cluster_df.index.to_numpy()


    results = meta_cluster.meta_clustering_with_confidence(all_labels)
    # Extract outputs
    finalC = results['finalC']
    conf_df = results['confidence_df']
    top_conf = results['top_confident_cells']

    # Save full meta-clustering assignments
    meta_out = pd.DataFrame({
        'cell_name': sample_names,
        'final_cluster': finalC
    })
    meta_out.to_csv(params['meta_cluster_out'], index=False)
    print(f"Saved full meta-cluster assignments in {params['meta_cluster_out']}")

    # Save top-confidence subset for Transformer
    top_conf_cells = pd.DataFrame({
        'cell_name': sample_names[top_conf['cell_index'].astype(int)],
        'final_cluster': top_conf['meta_cluster'].astype(int)
    })
    top_conf_path = params.get('top_conf_cells_out')
    top_conf_cells.to_csv(top_conf_path, index=False)
    print(f"Saved top confident cells in {top_conf_path}")

    ari = adjusted_rand_score(true_labels[class_col], meta_out['final_cluster'])
    nmi = normalized_mutual_info_score(true_labels[class_col], meta_out['final_cluster'])
    metrics = pd.DataFrame(columns=["ARI", "NMI"])

    metrics.loc[len(metrics)] = [ari, nmi]


    # step 4: transformer
    # Load config
    gene_expr_path = params['expression_processed']
    top_conf_cells_path = params['top_conf_cells_out']
    output_predictions_path = params['output_predictions']
    meta_path = params['metadata']
    num_heads = int(params.get('num_heads', 4))
    num_layers = int(params.get('num_layers', 2))
    d_model = int(params.get('d_model', 128))
    dropout = float(params.get('dropout', 0.1))
    batch_size = int(params.get('batch_size', 32))
    num_epochs = int(params.get('num_epochs', 10))
    lr = float(params.get('learning_rate', 0.001))

    transformer_cluster.run_transformer_pipeline(gene_expr_path, top_conf_cells_path, output_predictions_path, meta_path,
                             num_heads=num_heads, num_layers=num_layers,
                             num_epochs=num_epochs, d_model=d_model, dropout=dropout, batch_size=batch_size)


    # evaluate
    final_clusters_file = params['output_predictions']
    final_clusters = pd.read_csv(final_clusters_file)

    ari = adjusted_rand_score(true_labels[class_col], final_clusters['final_cluster'])
    nmi = normalized_mutual_info_score(true_labels[class_col], final_clusters['final_cluster'])
    metrics.loc[len(metrics)] = [ari, nmi]

    metrics.to_csv(os.path.join(params['outdir'], "metrics.csv"), index=False)


    # plot
    plot.plot_embedding(adata, final_clusters['final_cluster'], outdir=params['plots'], method="umap")
    plot.plot_embedding(adata, final_clusters['final_cluster'], outdir=params['plots'], method="tsne")


if __name__ == "__main__":
    main()