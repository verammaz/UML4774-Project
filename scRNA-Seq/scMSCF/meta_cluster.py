import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import sys
import yaml

def _onehot_matrix(labels):
    """Return a dense one-hot (n_cells x n_clusters) for a single label vector."""
    lb = LabelBinarizer(sparse_output=False)
    O = lb.fit_transform(labels)
    # If LabelBinarizer returns shape (n_samples,) for binary case, expand
    if O.ndim == 1:
        O = O.reshape(-1, 1)
    return O, lb.classes_

def compute_coassociation(all_labels):
    """
    all_labels: list of label arrays (each length N_cells)
    Returns AA: co-association matrix (N x N), averaged across runs
    """
    n_runs = len(all_labels)
    N = len(all_labels[0]) # number of cells 
    
    AA = np.zeros((N, N), dtype=np.float64)

    for labels in all_labels:
        O, _ = _onehot_matrix(labels)
        # O shape: (N, k_run)
        # accumulate O @ O.T
        AA += O @ O.T  # shape N x N

    AA /= float(n_runs)
    return AA


def meta_clustering_with_confidence(
    all_labels,
    n_meta=None,
    linkage_method='average',
    confidence_ratio=0.8, 
    delta=0.1
):
    """
    Perform meta-clustering with confidence

    Parameters
    ----------
    all_labels : list of numpy arrays
        Each array is a run's labels (length Ncells). Columns correspond to runs.
    n_meta : int or None
        Desired number of meta clusters. If None, it will default to number unique labels in first run.
    linkage_method : str
        Linkage for hierarchical clustering (e.g., 'average', 'ward').
    confidence_ratio : float
        Fraction of top confident cells to pick per meta-cluster (0..1).
    
    Returns
    -------
    result : dict
        Contains:
          - 'finalC' : final cluster label per cell (array of length N)
          - 'x0' : N x n_meta counts of how many runs assigned each cell to each meta cluster
          - 'confidence_df' : DataFrame with per-cell proportions per meta-cluster
          - 'top_confident_cells' : DataFrame of top confident cells (indices + proportions)
          - 'S_cell' : AA co-association matrix (N x N)
          - 'W_cell' : W = AA*(1-AA)
          - 'S_cluster' : cluster-to-cluster similarity matrix (n_all_clusters x n_all_clusters)
    """

    if len(all_labels) == 0:
        raise ValueError("all_labels empty")

    N = len(all_labels[0]) # number of cells
    C = len(all_labels) # number of cluster runs

    # Ensure all runs length match
    for arr in all_labels:
        if len(arr) != N:
            raise ValueError("All label arrays must have same length")

    # 1) Compute AA (co-association) efficiently
    AA = compute_coassociation(all_labels)  # shape N x N

    # 2) Compute W = AA * (1 - AA) per pairwise entry
    W = AA * (1.0 - AA)

    # 3) compute per-cell weight: w0 = 4/N * rowSums(W), w1 = (w0 + e)/(1+e)
    w0 = (4.0 / float(N)) * W.sum(axis=1)  # shape (N,)
    e = 0.01
    w1 = (w0 + e) / (1.0 + e)  # shape (N,)

    # 4) Make unique cluster ids across runs and create membership lists
    # newnC: N x C with entries like "<label>__runIdx"
    newnC = np.vstack(
        [
            np.array([f"{lab}__{ri}" for lab in all_labels[ri]])
            for ri in range(C)
        ]
    ).T  # transpose to N x C


    # R: unique cluster identifiers across all runs
    R = np.unique(newnC.flatten())
    allC = len(R)

    # Map cluster id -> member indices
    cluster_to_idx = {}
    for r in R:
        # find cells with label r in any run (but r includes run idx, so it's membership in its run)
        rows = np.where((newnC == r).any(axis=1))[0]
        cluster_to_idx[r] = rows

    
    # 5) Build cluster-to-cluster similarity S_cluster
    # S_cluster[p,q] = average pairwise W between members of cluster p and cluster q:
    #   S_pq = sum_{i in p} sum_{j in q} W[i,j] / (|p| * |q|)
    S_cluster = np.zeros((allC, allC), dtype=np.float64)

    R_list = list(R)
    # compute similarity for upper triangle
    for a in range(allC):
        idx_a = cluster_to_idx[R_list[a]]
        size_a = len(idx_a)
        if size_a == 0:
            continue
        for b in range(a, allC):
            idx_b = cluster_to_idx[R_list[b]]
            size_b = len(idx_b)
            if size_b == 0:
                continue
            # average pairwise W
            # If a == b, compute internal cohesion (average W among members)
            # if a == b:
            #     if size_a == 1:
            #         val = 1.0  # self similarity
            #     else:
            #         sub = W[np.ix_(idx_a, idx_a)]
            #         # exclude diagonal?
            #         # we'll include diagonal but that is fine; normalize by size^2
            #         val = sub.sum() / float(size_a * size_a)
            # else:
            sub = W[np.ix_(idx_a, idx_b)]
            val = sub.sum() / float(size_a * size_b)
            S_cluster[a, b] = val
            S_cluster[b, a] = val

    # make diagonal ones for clustering convenience
    np.fill_diagonal(S_cluster, 1.0)

    # 6) hierarchical clustering on cluster similarity; choose number of meta clusters
    if n_meta is None:
        # default: take number of unique labels in first run (fall-back)
        n_meta = max([len(np.unique(all_labels[i])) for i in range(len(all_labels))])

    print(f"Number of meta clusters: {n_meta}")

    # convert similarity -> distance
    dist_cluster = 1.0 - S_cluster
    # condensed vector
    condensed = squareform(dist_cluster, checks=False)
    Z = linkage(condensed, method=linkage_method)
    # assign cluster ids
    meta_assign = fcluster(Z, t=n_meta, criterion='maxclust')  # length allC, 1..n_meta

    # 7) Map original cluster ids R -> meta id
    tf = { R_list[i]: int(meta_assign[i]) for i in range(allC) }
    
    # 8) Create newnC_mapped: N x C where each entry is the mapped meta id
    newnC_mapped = np.vectorize(tf.get)(newnC)
  
    # 9) Voting per cell: choose majority meta id across runs
    # finalC per cell:
    finalC = []
    for i in range(N):
        vals, counts = np.unique(newnC_mapped[i, :], return_counts=True)
        # pick largest
        max_idx = np.argmax(counts)
        finalC.append(int(vals[max_idx]))
    finalC = np.array(finalC, dtype=int)

    # 10) Build per-cell proportion table: fraction of runs assigning to each meta cluster
    unique_meta = np.arange(1, n_meta + 1)
    proportion_matrix = np.zeros((N, n_meta), dtype=np.float64)
    for k in unique_meta:
        proportion_matrix[:, k-1] = (newnC_mapped == k).sum(axis=1) / float(C)

    confidence_df = pd.DataFrame(proportion_matrix, columns=[f"Cluster_{k}" for k in unique_meta])
    # add final cluster
    confidence_df['final_cluster'] = finalC

    # 11) For each meta cluster, pick top confidence_ratio fraction of cells by proportion
    top_conf_list = []
    for k in unique_meta:
        colname = f"Cluster_{k}"
        cluster_cells = confidence_df[confidence_df['final_cluster'] == k].copy()
        if cluster_cells.shape[0] == 0:
            continue
        # compute ranking by that column
        cluster_cells = cluster_cells.sort_values(colname, ascending=False)
        n_top = max(1, int(round(confidence_ratio * cluster_cells.shape[0])))
        top_k = cluster_cells.head(n_top).copy()
        top_k['meta_cluster'] = k
        top_k['cell_index'] = top_k.index
        top_conf_list.append(top_k)

    if len(top_conf_list) > 0:
        top_confident_cells = pd.concat(top_conf_list, axis=0)
    else:
        top_confident_cells = pd.DataFrame(columns=confidence_df.columns.tolist() + ['meta_cluster', 'cell_index'])

    # 12) x0 : counts of how many times a cell was assigned to each meta cluster
    x0 = np.zeros((N, n_meta), dtype=np.int32)
    for k in unique_meta:
        x0[:, k-1] = (newnC_mapped == k).sum(axis=1)

    result = {
        'finalC': finalC,
        'x0': x0,
        'confidence_df': confidence_df,
        'top_confident_cells': top_confident_cells,
        'S_cell': AA,
        'W_cell': W,
        'S_cluster': S_cluster,
        'newnC_mapped': newnC_mapped
    }
    return result


def main():
    params_file = sys.argv[1]
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)

    # Load base cluster results
    cluster_df = pd.read_csv(params['kmeans_pca_out'])
    all_labels = [cluster_df[col].to_numpy() for col in cluster_df.columns if col != 'sample_name']
    sample_names = cluster_df['sample_name'].to_numpy()

    # Run meta-clustering
    results = meta_clustering_with_confidence(all_labels)

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

    
if __name__ == "__main__":
    main()