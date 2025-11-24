import sys
import os
import numpy as np
import pandas as pd
import yaml

import meta_cluster
import pca_kmeans
import transformer_cluster

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_utils
import data_preprocess
import plot


# --------------------------
# ADVERSARIAL ATTACK FUNCTIONS
# --------------------------

def gaussian_noise_attack(adata, epsilon):
    X = adata.X.copy()
    noise = epsilon * np.random.normal(0, 1, X.shape)
    adata.X = X + noise
    return adata


def fgsm_style_attack(adata, labels, epsilon):
    X = adata.X.copy()

    # compute pseudo-gradients via class centroids
    unique = np.unique(labels)
    centroids = {c: X[labels == c].mean(axis=0) for c in unique}

    grad = np.zeros_like(X)
    for i in range(X.shape[0]):
        c = labels[i]
        grad[i] = X[i] - centroids[c]

    adata.X = X + epsilon * np.sign(grad)
    return adata


def random_feature_corruption(adata, epsilon, pct=0.02):
    X = adata.X.copy()
    n_genes = X.shape[1]
    k = int(pct * n_genes)

    gene_mask = np.random.choice(n_genes, size=k, replace=False)
    X[:, gene_mask] += epsilon

    adata.X = X
    return adata


def apply_attack(adata, attack_cfg, true_labels):
    attack = attack_cfg["type"]
    eps = float(attack_cfg.get("epsilon", 0.1))

    if attack == "gaussian_noise":
        return gaussian_noise_attack(adata, eps)

    elif attack == "fgsm":
        return fgsm_style_attack(adata, true_labels, eps)

    elif attack == "feature_corruption":
        pct = attack_cfg.get("percent_genes", 0.02)
        return random_feature_corruption(adata, eps, pct)

    else:
        raise ValueError(f"Unknown attack type: {attack}")


# --------------------------
# ADVERSARIAL PIPELINE
# --------------------------

def main():
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)

    raw = data_utils.load_hd5a(params['expression'])
    adata = data_preprocess.preprocess(raw)

    # Load metadata + labels
    meta = data_utils.load_metadata(params['metadata'])
    n_clusters, true_labels = data_utils.get_sample_labels(meta)
    true_labels_array = true_labels['subclass_label'].to_numpy()

    # --------------------------
    # Apply adversarial attack here
    # --------------------------
    if "attack" in params and params["attack"].get("enable", False):
        print("Applying adversarial attack:", params["attack"]["type"])
        adata = apply_attack(adata, params["attack"], true_labels_array)
        params['plots'] = f'data/out/adv/plots/{params['attack']['type']}'
    else:
        print("No adversarial attack applied.")

    # --- PCA + KMeans ---
    pca_dims_list = params['pca_dims']
    all_labels = pca_kmeans.multi_pca_kmeans(
        adata,
        pca_dims_list,
        k_range=range(2, n_clusters + 1),
        plots_dir=params['plots']
    )

    cluster_df = pd.DataFrame(
        {col: labels for col, labels in all_labels.items()},
        index=adata.obs_names
    )
    cluster_df.index.name = "cell_name"
    cluster_df.to_csv(params['kmeans_pca_out'])

    # --- Meta-clustering ---
    cluster_df = pd.read_csv(params['kmeans_pca_out']).set_index("cell_name")
    all_labels = [cluster_df[col].to_numpy() for col in cluster_df.columns]
    sample_names = cluster_df.index.to_numpy()

    results = meta_cluster.meta_clustering_with_confidence(all_labels)
    finalC = results['finalC']
    top_conf = results['top_confident_cells']

    # Save full assignments
    meta_out = pd.DataFrame({
        'cell_name': sample_names,
        'final_cluster': finalC
    })
    meta_out.to_csv(params['meta_cluster_out'], index=False)

    # Save only confident subset
    top_conf_cells = pd.DataFrame({
        'cell_name': sample_names[top_conf['cell_index'].astype(int)],
        'final_cluster': top_conf['meta_cluster'].astype(int)
    })
    top_conf_cells.to_csv(params['top_conf_cells_out'], index=False)

    # --- Evaluate ---
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels_array, finalC)
    nmi = normalized_mutual_info_score(true_labels_array, finalC)
    metrics = pd.DataFrame(columns=["ARI", "NMI"])
    metrics.loc[len(metrics)] = [ari, nmi]

    # --- Transformer ----
    transformer_cluster.run_transformer_pipeline(
        params['expression_processed'],
        params['top_conf_cells_out'],
        params['output_predictions'],
        params['metadata'],
        num_heads=params.get('num_heads', 4),
        num_layers=params.get('num_layers', 2),
        num_epochs=params.get('num_epochs', 10),
        d_model=params.get('d_model', 128),
        dropout=params.get('dropout', 0.1),
        batch_size=params.get('batch_size', 32)
    )

    final_clusters = pd.read_csv(params['output_predictions'])
    ari = adjusted_rand_score(true_labels_array, final_clusters["final_cluster"])
    nmi = normalized_mutual_info_score(true_labels_array, final_clusters["final_cluster"])
    metrics.loc[len(metrics)] = [ari, nmi]

    metrics.to_csv(os.path.join(params['outdir'], "adv", f"{params["attack"]["type"]}_metrics.csv"), index=False)

    # --- Plot UMAP ---
    plot.plot_embedding(adata, final_clusters['final_cluster'], outdir=params['plots'], method="umap")
    plot.plot_embedding(adata, final_clusters['final_cluster'], outdir=params['plots'], method="tsne")


if __name__ == "__main__":
    main()
