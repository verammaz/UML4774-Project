#!/usr/bin/env python3

import os
import numpy as np
from typing import Sequence, Tuple, Dict
from data_io import save_components

def generate_bernoulli_raw(
    outdir: str = "data_set/bernoulli_raw",
    k: int = 5,
    dims: int = 64,
    n_per: Sequence[int] | int = 50_000,
    p_range: Tuple[float, float] = (0.1, 0.9),
    seed: int | None = 42,
) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if isinstance(n_per, int):
        n_per = [n_per] * k
    assert len(n_per) == k and all(n > 0 for n in n_per)

    probs = rng.uniform(p_range[0], p_range[1], size=(k, dims)).astype(np.float32)

    components = []
    for c in range(k):
        Xc = rng.binomial(1, probs[c], size=(n_per[c], dims)).astype(np.float32)
        yc = np.full(n_per[c], c, dtype=np.int32)
        components.append({"name": f"bern_c{c}", "X": Xc, "y": yc})

    meta = {
        "type": "bernoulli",
        "k": k,
        "dims": dims,
        "p_range": list(p_range),
        "seed": seed,
        "n_per": n_per,
    }
    save_components(outdir, components, meta)
    np.save(os.path.join(outdir, "probs.npy"), probs)
    return {"probs": probs, **meta}

if __name__ == "__main__":
    generate_bernoulli_raw()
