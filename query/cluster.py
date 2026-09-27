"""Look-alike site groups (DESIGN.md §3, M7): centred eo_vec -> UMAP (UMAP_DIM) -> HDBSCAN (unknown k).

Centring removes the shared direction that makes mean-pooled ViT tokens all look alike (cosine ~0.9 everywhere).
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN


def cluster(vecs: NDArray[np.float32], min_cluster_size: int, dim: int, seed: int = 0) -> NDArray[np.int64]:
    """Cluster label per row; -1 = noise (not in any group)."""
    x = vecs - vecs.mean(axis=0, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    if len(x) > dim + 2:
        import umap  # numba JIT on first use; imported lazily so the API starts fast

        x = umap.UMAP(
            n_components=dim, n_neighbors=min(15, len(x) - 1), metric="cosine", random_state=seed
        ).fit_transform(x)
    return (
        HDBSCAN(min_cluster_size=min(min_cluster_size, max(2, len(x) // 2))).fit_predict(x).astype(np.int64)
    )
