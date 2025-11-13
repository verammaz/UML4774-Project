#!/usr/bin/env python3

import os, math
import numpy as np
from typing import Sequence, Tuple, Dict
from data_io import save_components

def generate_gaussian_raw(
    outdir: str = "data_set/gaussian_raw",
    k: int = 5,
    dims: int = 64,
    n_per: Sequence[int] | int = 50_000,
    mean_range: Tuple[float, float] = (-6.0, 6.0),
    var_range: Tuple[float, float] = (0.3, 2.5),
    seed: int | None = 123,
) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if isinstance(n_per, int):
        n_per = [n_per] * k
    assert len(n_per) == k and all(n > 0 for n in n_per)

    centers = rng.uniform(mean_range[0], mean_range[1], size=(k, dims)).astype(np.float32)
    variances = rng.uniform(var_range[0], var_range[1], size=k).astype(np.float32)

    components = []
    for c in range(k):
        std = math.sqrt(float(variances[c]))
        Xc = centers[c] + rng.standard_normal(size=(n_per[c], dims)).astype(np.float32) * std
        yc = np.full(n_per[c], c, dtype=np.int32)
        components.append({"name": f"gauss_c{c}", "X": Xc.astype(np.float32), "y": yc})

    meta = {
        "type": "gaussian",
        "k": k,
        "dims": dims,
        "mean_range": list(mean_range),
        "var_range": list(var_range),
        "seed": seed,
        "n_per": n_per,
    }
    save_components(outdir, components, meta)
    np.save(os.path.join(outdir, "centers.npy"), centers)
    np.save(os.path.join(outdir, "variances.npy"), variances)
    return {"centers": centers, "variances": variances, **meta}

if __name__ == "__main__":
    generate_gaussian_raw()
