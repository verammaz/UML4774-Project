from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import gzip


def save_compressed_array(filepath: str, array: np.ndarray):
    """Save numpy array using gzip compression."""
    with gzip.open(filepath, "wb") as f:
        np.save(f, array)

