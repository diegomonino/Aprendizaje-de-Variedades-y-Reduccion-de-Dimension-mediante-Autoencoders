"""
topology.py
-----------
Topological evaluation metrics for representation quality.

    - trustworthiness: measures whether neighbors in latent space are
                       also neighbors in data space (no false neighbors)
    - continuity:      measures whether neighbors in data space remain
                       neighbors in latent space (no missing neighbors)

Both metrics range from 0 to 1, where 1 means perfect preservation.
"""

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 12) -> float:
    """
    Trustworthiness: penalizes points that are neighbors in the
    low-dimensional embedding but far apart in the original space.

    Wrapper around sklearn's implementation.

    Args:
        X_high:      Original high-dimensional data (N, D).
        X_low:       Low-dimensional embedding (N, d).
        n_neighbors:  Number of neighbors to consider.

    Returns:
        Trustworthiness score in [0, 1].
    """
    return sklearn_trustworthiness(X_high, X_low, n_neighbors=n_neighbors)


def continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 12) -> float:
    """
    Continuity: penalizes points that are neighbors in the
    original space but far apart in the embedding.

    This is the dual of trustworthiness.

    C(k) = 1 - (2 / (Nk(2N-3k-1))) * sum of (r(i,j) - k)
    where the sum is over points j that are among the k nearest
    neighbors of i in the original space but NOT among the k nearest
    neighbors in the embedding, and r(i,j) is the rank of j among
    i's neighbors in the embedding.

    Args:
        X_high:      Original high-dimensional data (N, D).
        X_low:       Low-dimensional embedding (N, d).
        n_neighbors:  Number of neighbors to consider.

    Returns:
        Continuity score in [0, 1].
    """
    N = X_high.shape[0]
    k = n_neighbors

    # Find k nearest neighbors in original space
    nbrs_high = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # remove self

    # Find ALL neighbor rankings in embedding space
    nbrs_low = NearestNeighbors(n_neighbors=N, algorithm="auto").fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)

    # Build rank lookup for embedding space
    # ranks_low[i][j] = rank of point j among neighbors of point i in embedding
    ranks_low = np.zeros((N, N), dtype=int)
    for i in range(N):
        for rank, j in enumerate(indices_low[i]):
            ranks_low[i, j] = rank

    # Compute continuity
    penalty = 0.0
    for i in range(N):
        # Neighbors of i in original space
        high_neighbors = set(indices_high[i])
        # Neighbors of i in embedding space
        low_neighbors = set(indices_low[i, 1:k + 1])

        # Points that are high-space neighbors but NOT low-space neighbors
        missing = high_neighbors - low_neighbors
        for j in missing:
            r_ij = ranks_low[i, j]  # rank in embedding
            penalty += r_ij - k

    normalization = N * k * (2 * N - 3 * k - 1)
    if normalization == 0:
        return 1.0

    return 1.0 - (2.0 / normalization) * penalty


def evaluate_topology(X_high: np.ndarray, X_low: np.ndarray,
                      n_neighbors: int = 12) -> dict:
    """
    Compute both trustworthiness and continuity.

    Args:
        X_high:      Original data (N, D).
        X_low:       Embedding (N, d).
        n_neighbors:  Number of neighbors.

    Returns:
        Dict with 'trustworthiness' and 'continuity' scores.
    """
    T = trustworthiness(X_high, X_low, n_neighbors=n_neighbors)
    C = continuity(X_high, X_low, n_neighbors=n_neighbors)
    return {"trustworthiness": T, "continuity": C}


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    from sklearn.decomposition import PCA

    X, color = make_swiss_roll(n_samples=500, noise=0.0, random_state=42)
    X_pca = PCA(n_components=2).fit_transform(X)

    scores = evaluate_topology(X, X_pca, n_neighbors=12)
    print(f"PCA on Swiss Roll (500 pts):")
    print(f"  Trustworthiness: {scores['trustworthiness']:.4f}")
    print(f"  Continuity:      {scores['continuity']:.4f}")
