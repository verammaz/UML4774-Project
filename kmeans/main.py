#!/usr/bin/env python3
import os
import json
import re
import argparse
import numpy as np
import torch

from vae.model import VAE
from data.data_io import load_and_split
from kmeans.model import KMeansClustering
from kmeans.utils import save_compressed_array

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_metadata(data_dir: str):
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"[ERROR] metadata.json not found in {data_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Normalize keys to be consistent
    meta_norm = {
        "type": meta.get("type", meta.get("distribution", "unknown")),
        "k": meta.get("k") or meta.get("n_clusters"),
        "dims": meta.get("dims") or meta.get("input_dim"),
    }
    return meta_norm


def get_vae_model_config(checkpoint: dict):
    model_config = checkpoint['config']['model']
    return {"input_dim": checkpoint['model_state_dict']['encoder.0.weight'].shape[1],
            "latent_dim": model_config['latent_dim'],
            "hidden_dims": model_config['hidden_dims'],
            "likelihood": model_config['likelihood'],
            "kl_beta": model_config['kl_beta'],
            "activation": model_config.get('activation', 'LeakyReLU')}



# ---------------------------------------------------------------------
# ---- KMeans experiment runner ---------------------------------------
# ---------------------------------------------------------------------
def run_kmeans_experiment(data_dir: str, vae_path: str, n_clusters: int = None):

    data_meta = get_data_metadata(data_dir)

    checkpoint = torch.load(vae_path, map_location=DEVICE)

    vae_config = get_vae_model_config(checkpoint)

    vae = VAE(
        input_dim=vae_config['input_dim'],
        latent_dim=vae_config['latent_dim'],
        hidden_dims=vae_config['hidden_dims'],
        likelihood=vae_config['likelihood'],
        beta=vae_config['kl_beta'],
        activation=vae_config.get('activation', 'LeakyReLU')
    )

    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(DEVICE)
    vae.eval()

    data = load_and_split(
        data_dir,
        splits=(0.85, 0.0, 0.15),
        normalize=True,
        save_manifest=True
    )
    
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    n_clusters = data_meta["k"] or len(np.unique(y_test))
   


    # === Encode data into latent space ===
    with torch.no_grad():
        z_mean, _ = vae.encode(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        Z = z_mean.cpu().numpy()

    # === KMeans in input space ===
    print("Running kmeans in input space...")
    kmeans_input = KMeansClustering(n_clusters=n_clusters)
    kmeans_input.fit(X_test)
    res_input = kmeans_input.evaluation_metrics(X_test, y_test)

    # === KMeans in latent space ===
    print("Running kmeans in latent space...")
    kmeans_latent = KMeansClustering(n_clusters=n_clusters)
    kmeans_latent.fit(Z)
    res_latent = kmeans_latent.evaluation_metrics(X_test, y_test)

    # === Compare results ===
    print("\n=== KMeans Comparison ===")
    print(f"Input-space  ARI: {res_input['ARI']:.3f}, NMI: {res_input['NMI']:.3f}, Inertia: {res_input['Inertia']:.3f}")
    print(f"Latent-space ARI: {res_latent['ARI']:.3f}, NMI: {res_latent['NMI']:.3f}, Inertia: {res_latent['Inertia']:.3f}")


    # === Save results === 
    out_dir = os.path.join(os.path.dirname(os.path.abspath(vae_path)), "kmeans_results") 
    os.makedirs(out_dir, exist_ok=True) 
    np.savez_compressed( os.path.join(out_dir, "summary_results.npz"), 
                        res_input=res_input, 
                        res_latent=res_latent, 
                        data_meta=data_meta, 
                        vae_meta=vae_config, ) 
    # Save cluster centers & embeddings separately (gzipped) 
    save_compressed_array(os.path.join(out_dir, "centers_input.npy.gz"), kmeans_input.centroids()) 
    save_compressed_array(os.path.join(out_dir, "centers_latent.npy.gz"), kmeans_latent.centroids()) 
    save_compressed_array(os.path.join(out_dir, "embeddings_latent.npy.gz"), Z)
  
    print(f"\nResults saved to {out_dir}")


# ---------------------------------------------------------------------
# ---- CLI entrypoint -------------------------------------------------
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run KMeans on input and latent spaces")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory containing metadata.json")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Override number of clusters (optional)")
    args = parser.parse_args()

    run_kmeans_experiment(
        data_dir=args.data_dir,
        vae_path=args.vae_path,
        n_clusters=args.n_clusters
    )


if __name__ == "__main__":
    main()
