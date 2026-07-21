"""
metrics.py
----------
Shared mathematical utilities for intrinsic dimension estimation.

    - nearest_neighbor_distances: compute d_min for each point
    - volume_ball:               V^n(r) = volume of n-dimensional ball of radius r
    - remove_outliers:           trim boundary effects from distance distribution
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma


def nearest_neighbor_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute the distance to the nearest neighbor for each point.

    Uses sklearn's efficient KD-tree / ball-tree implementation.

    Args:
        data: Array of shape (N, D).

    Returns:
        Array of shape (N,) with nearest-neighbor distances.
    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(data)
    distances, _ = nbrs.kneighbors(data)
    # distances[:, 0] is distance to self (=0), [:, 1] is nearest neighbor
    return distances[:, 1]


def volume_ball(r: np.ndarray, n: int) -> np.ndarray:
    """
    Volume of an n-dimensional ball of radius r.

    V^n(r) = (pi^(n/2) / Gamma(n/2 + 1)) * r^n

    Args:
        r: Array of radii.
        n: Dimension.

    Returns:
        Array of volumes, same shape as r.
    """
    coeff = (np.pi ** (n / 2)) / gamma(n / 2 + 1)
    return coeff * (r ** n)


def remove_outliers(distances: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """
    Remove points with abnormally large nearest-neighbor distances.

    These typically correspond to boundary points where the local density
    estimate is unreliable. Following Ivanov et al. (2021), we remove
    the top percentile.

    Args:
        distances:  Array of nearest-neighbor distances.
        percentile: Keep only distances below this percentile.

    Returns:
        Filtered array of distances.
    """
    threshold = np.percentile(distances, percentile)
    return distances[distances < threshold]


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=2000, noise=0.0, random_state=42)

    dists = nearest_neighbor_distances(X)
    print(f"Swiss Roll ({X.shape[0]} pts in R^{X.shape[1]})")
    print(f"  d_min range: [{dists.min():.4f}, {dists.max():.4f}]")
    print(f"  d_min mean:  {dists.mean():.4f}")

    dists_clean = remove_outliers(dists, percentile=95)
    print(f"  After outlier removal: {len(dists_clean)} points (was {len(dists)})")

    # Volume of 2D ball with radius 1
    print(f"\n  V^2(1.0) = {volume_ball(np.array([1.0]), 2)[0]:.4f}  (should be pi = {np.pi:.4f})")
