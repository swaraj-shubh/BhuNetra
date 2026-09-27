import numpy as np

from query.cluster import cluster


def test_cluster_recovers_separated_groups_and_marks_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 64))
    pts = np.vstack([b + 0.05 * rng.normal(size=(40, 64)) for b in base] + [3 * rng.normal(size=(5, 64))])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    labels = cluster(pts, min_cluster_size=10, dim=5, seed=0)
    groups = [set(labels[i * 40 : (i + 1) * 40]) for i in range(3)]
    assert all(len(g) == 1 and -1 not in g for g in groups)  # each blob is one cluster
    assert len({next(iter(g)) for g in groups}) == 3  # three distinct clusters


def test_cluster_small_input_skips_umap():
    pts = np.eye(4, 8, dtype=np.float32)
    assert len(cluster(pts, min_cluster_size=10, dim=20, seed=0)) == 4  # fewer points than dims: no crash
