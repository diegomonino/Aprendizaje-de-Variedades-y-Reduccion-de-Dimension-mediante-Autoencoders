"""
flattening.py
-------------
Flattening transformation from Ivanov et al. (2021).

Transforms each coordinate by its empirical CDF so that the marginal
distributions become uniform on [0, 1]. This locally flattens the
manifold, making Euclidean distances in the transformed space a better
approximation of geodesic distances on the manifold.

The transformation is: x_i' = F(x_i), where F is the empirical CDF
of the i-th coordinate across the dataset.
"""

import numpy as np


def flatten_data(data: np.ndarray) -> np.ndarray:
    """
    Apply coordinate-wise CDF flattening.

    For each coordinate dimension j, replaces values by their
    normalized rank (empirical CDF), mapping to [0, 1].

    Args:
        data: Array of shape (N, D).

    Returns:
        Flattened array of shape (N, D) with values in [0, 1].
    """
    N = data.shape[0]
    data_flat = np.zeros_like(data)

    for j in range(data.shape[1]):
        # Double argsort gives the rank of each element
        ranks = np.argsort(np.argsort(data[:, j]))
        data_flat[:, j] = ranks / (N - 1)

    return data_flat


if __name__ == "__main__":
    # Test: uniform data should stay roughly uniform after flattening
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 3)  # Gaussian -> after flattening should be uniform
    X_flat = flatten_data(X)
    print(f"Original range:  [{X.min():.2f}, {X.max():.2f}]")
    print(f"Flattened range: [{X_flat.min():.4f}, {X_flat.max():.4f}]")
    print(f"Flattened mean per dim: {X_flat.mean(axis=0)}")  # Should be ~0.5
