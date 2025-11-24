import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


def plot_embedding(adata, labels, outdir=None, method="umap", title=None,
                   n_neighbors=15, min_dist=0.3, random_state=42):
    """
    Plot UMAP or t-SNE embedding for an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Processed data matrix.
    labels : array-like
        Cluster labels to color cells.
    outdir : str or None
        Directory to save plot. If None, the plot is shown.
    method : {"umap", "tsne"}
        Embedding algorithm to use.
    title : str or None
        Title for the plot.
    """
    
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # --- compute embedding ---
    if method.lower() == "umap":
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            random_state=random_state
        )
        embedding = reducer.fit_transform(X)
        default_title = "UMAP Projection"
        fname = "umap_clusters.png"

    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=2, 
            perplexity=30, 
            learning_rate="auto", 
            init="pca", 
            random_state=random_state
        )
        embedding = reducer.fit_transform(X)
        default_title = "t-SNE Projection"
        fname = "tsne_clusters.png"

    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    print(f"{method.upper()} shape: {embedding.shape}")

    # --- prepare plot ---
    plt.figure(figsize=(7, 6))

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab20" if num_clusters <= 20 else "tab20b",
        s=6
    )

    plt.title(title if title else default_title)
    plt.xlabel(f"{method.upper()}-1")
    plt.ylabel(f"{method.upper()}-2")

    # Optional: add legend for < 25 clusters
    if num_clusters <= 25:
        for lab in unique_labels:
            plt.scatter([], [], c=plt.cm.tab20(lab % 20), label=str(lab), s=20)
        plt.legend(
            title="Clusters",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=6
        )

    plt.tight_layout()

    # --- save or show ---
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, fname), dpi=300)
    else:
        plt.show()

    plt.close()
