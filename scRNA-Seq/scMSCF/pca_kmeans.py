import scanpy as sc
import pandas as pd
import sys
import yaml
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_utils

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import scanpy as sc

def multi_pca_kmeans(adata, pca_dims, k_range=range(3, 43), plots_dir=None):

    results = {}

    for n_comp in pca_dims:
        print(f"\n====== Running with PCA n_comp={n_comp} ======")

        # --- PCA ---
        sc.tl.pca(adata, n_comps=n_comp, svd_solver='arpack', copy=False)
        Xp = adata.obsm["X_pca"]

        inertias = []
        sil_scores = []

        # --- compute inertia + silhouette for each k ---
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = km.fit_predict(Xp)

            inertias.append(km.inertia_)

            # Silhouette undefined for k=1 so safe
            sil = silhouette_score(Xp, labels)
            sil_scores.append(sil)

        inertias = np.array(inertias)
        sil_scores = np.array(sil_scores)

        # =================================================================
        #   ELBOW METHOD (use curvature / second derivative)
        # =================================================================
        d1 = np.diff(inertias)
        d2 = np.diff(d1)

        # two strongest curvature drops
        elbow_idxs = np.argsort(np.abs(d2))[:2]
        elbow_ks = sorted([k_range[i] for i in (elbow_idxs + 2)])

        # =================================================================
        #   SILHOUETTE METHOD (global max)
        # =================================================================
        best_sil_k = k_range[np.argmax(sil_scores)]

        print(f"Elbow optimal ks: {elbow_ks}")
        print(f"Silhouette optimal k: {best_sil_k}")

        # =================================================================
        #   PLOTS
        # =================================================================
        if plots_dir is not None:
            os.makedirs(plots_dir, exist_ok=True)

            # Elbow plot
            plt.figure(figsize=(6,4))
            plt.plot(k_range, inertias, marker='o')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Inertia (WCSS)")
            plt.title(f"Elbow Method (PCA {n_comp})")
            plt.xticks(k_range)
            plt.savefig(os.path.join(plots_dir, f"pca{n_comp}_elbow.png"))
            plt.close()

            # Silhouette plot
            plt.figure(figsize=(6,4))
            plt.plot(k_range, sil_scores, marker='o')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Silhouette score")
            plt.title(f"Silhouette Scores (PCA {n_comp})")
            plt.xticks(k_range)
            plt.savefig(os.path.join(plots_dir, f"pca{n_comp}_silhouette.png"))
            plt.close()

        # Print clean result
        print(f"Final recommended ks for PCA={n_comp}:")
        print(f"  - Elbow: {tuple(map(int, elbow_ks))}")
        print(f"  - Silhouette: {int(best_sil_k)}")
        
        optimal_ks = elbow_ks
        optimal_ks.append(best_sil_k)

        # --- run k-means for optimal ks ---
        for k in optimal_ks:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = km.fit_predict(Xp)

            colname = f"kmeans_pc{n_comp}_k{k}"
            adata.obs[colname] = pd.Categorical(labels.astype(str))

            results[f"pca{n_comp}_kmeans{k}"] = labels

    return results

    



def main():
    # Load YAML input arguments
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)  
    
    adata = data_utils.load_hd5a(params['expression_processed'])
    meta = data_utils.load_metadata(params['metadata'])
    n_clusters, _ = data_utils.get_sample_labels(meta)
    pca_dims_list = params['pca_dims']

    all_labels = multi_pca_kmeans(adata, pca_dims_list, k_range=range(2, n_clusters+1))

    cluster_df = pd.DataFrame(
    {col: labels for col, labels in all_labels.items()},
    index=adata.obs_names
    )   

    cluster_df.to_csv(params['kmeans_pca_out'])


if __name__ == "__main__":
    main()


