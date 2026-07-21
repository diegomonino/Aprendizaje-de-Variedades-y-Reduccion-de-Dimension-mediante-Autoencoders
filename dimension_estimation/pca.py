"""
pca.py
------
PCA-based (projective/linear) intrinsic dimension estimation.

This is the classical, linear-algebra counterpart to the box-counting
(minkowski.py) and geometrically-probabilistic (geometric_prob.py)
estimators. It answers a different question: "how many linear directions
are needed to explain the data's variance?" rather than "how does local
neighborhood volume scale?".

Two complementary estimators are implemented:

    1. Variance-threshold dimension
       Smallest k such that the first k principal components explain at
       least `threshold` (e.g. 95%) of the total variance. This is the
       standard "elbow"/threshold heuristic used in practice.

    2. Participation ratio (effective dimension)
       PR = (sum_i lambda_i)^2 / sum_i (lambda_i^2)
       A continuous, threshold-free measure of how "spread out" the
       variance is across components. PR = D if variance is uniform
       across all D components; PR = 1 if a single component dominates.
       Scale-invariant: using explained_variance_ratio_ or raw eigenvalues
       gives the same result, since PR is homogeneous of degree 0 in the
       eigenvalues.

Known limitation (expected, not a bug)
---------------------------------------
PCA is a LINEAR method. On a linear manifold (e.g. H_{d,D}) it recovers
the true intrinsic dimension d almost exactly. On a NONLINEAR manifold
(e.g. the Swiss Roll, intrinsic dim 2, embedded in R^3) it will typically
report a HIGHER dimension (close to the ambient dimension), because a
curved manifold cannot be spanned by a small linear subspace. This is
precisely the limitation that motivates nonlinear methods (autoencoders,
Isomap, LLE, ...) and should be reported as expected behavior, not an
implementation error.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Dict


def pca_spectrum(data: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the PCA explained-variance spectrum of a point cloud.

    Args:
        data:         Array of shape (N, D).
        n_components: Number of components to compute. Defaults to
                       min(N, D) (the maximum possible).

    Returns:
        Tuple of:
            - explained_variance_ratio: array of shape (n_components,),
              fraction of total variance explained by each component,
              sorted in decreasing order.
            - cumulative_variance: array of shape (n_components,),
              running cumulative sum of explained_variance_ratio.
    """
    N, D = data.shape
    if n_components is None:
        n_components = min(N, D)
    n_components = min(n_components, N, D)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return explained_variance_ratio, cumulative_variance


def pca_dimension(
    data: np.ndarray,
    threshold: float = 0.95,
    n_components: int = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Estimate intrinsic dimension via the variance-threshold heuristic.

    Args:
        data:         Array of shape (N, D).
        threshold:    Minimum fraction of cumulative variance to explain
                       (default 0.95, i.e. 95%).
        n_components: Number of components to compute (see pca_spectrum).

    Returns:
        Tuple of:
            - dim_estimate: smallest k (>= 1) such that the first k
              components explain >= threshold of the total variance.
            - explained_variance_ratio: full spectrum (for plotting).
            - cumulative_variance: cumulative curve (for plotting).
    """
    explained_variance_ratio, cumulative_variance = pca_spectrum(data, n_components)

    idx = np.searchsorted(cumulative_variance, threshold)
    dim_estimate = int(min(idx + 1, len(cumulative_variance)))

    return dim_estimate, explained_variance_ratio, cumulative_variance


def participation_ratio_dimension(
    data: np.ndarray,
    n_components: int = None,
) -> Tuple[float, np.ndarray]:
    """
    Estimate the "effective" intrinsic dimension via the participation ratio.

        PR = (sum_i lambda_i)^2 / sum_i (lambda_i^2)

    A continuous, threshold-free complement to pca_dimension. Scale-invariant:
    computed here from explained_variance_ratio_, which gives the same value
    as using raw eigenvalues.

    Args:
        data:         Array of shape (N, D).
        n_components: Number of components to compute (see pca_spectrum).

    Returns:
        Tuple of:
            - effective_dim: participation ratio (float, generally not an
              integer).
            - explained_variance_ratio: full spectrum (for reference/plotting).
    """
    explained_variance_ratio, _ = pca_spectrum(data, n_components)

    numerator = np.sum(explained_variance_ratio) ** 2
    denominator = np.sum(explained_variance_ratio ** 2)
    effective_dim = float(numerator / denominator) if denominator > 0 else 0.0

    return effective_dim, explained_variance_ratio


def estimate_dimension_pca(
    data: np.ndarray,
    threshold: float = 0.95,
    n_components: int = None,
) -> Dict:
    """
    Convenience wrapper: runs both PCA-based estimators and packages
    everything needed for plotting and reporting in a single dict.

    Args:
        data:         Array of shape (N, D).
        threshold:    Variance threshold for pca_dimension (default 0.95).
        n_components: Number of components to compute.

    Returns:
        Dict with:
            'variance_threshold_dim':   int, from pca_dimension
            'participation_ratio_dim':  float, from participation_ratio_dimension
            'explained_variance_ratio': np.ndarray
            'cumulative_variance':      np.ndarray
            'threshold':                float (echoed back for plotting)
    """
    dim_thresh, evr, cum_var = pca_dimension(data, threshold=threshold, n_components=n_components)
    dim_pr, _ = participation_ratio_dimension(data, n_components=n_components)

    return {
        "variance_threshold_dim": dim_thresh,
        "participation_ratio_dim": dim_pr,
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_var,
        "threshold": threshold,
    }


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    print("== Swiss Roll (nonlinear manifold, intrinsic dim = 2) ==")
    print("   Expected: PCA OVERESTIMATES (needs ~3 dims, since it's curved)")
    X, _ = make_swiss_roll(n_samples=2000, noise=0.0, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    res = estimate_dimension_pca(X, threshold=0.95)
    print(f"   Variance-threshold dim (95%): {res['variance_threshold_dim']}")
    print(f"   Participation ratio dim:      {res['participation_ratio_dim']:.2f}")
    print(f"   Cumulative variance:          {np.round(res['cumulative_variance'], 3)}")

    print("\n== Linear H_3,30 (linear manifold, intrinsic dim = 3) ==")
    print("   Expected: PCA RECOVERS ~3 (it's exactly linear)")
    rng = np.random.RandomState(42)
    Z = rng.uniform(0, 1, size=(10000, 3))
    A_raw = rng.randn(30, 3)
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :3]
    X_lin = Z @ A.T
    res_lin = estimate_dimension_pca(X_lin, threshold=0.95)
    print(f"   Variance-threshold dim (95%): {res_lin['variance_threshold_dim']}")
    print(f"   Participation ratio dim:      {res_lin['participation_ratio_dim']:.2f}")
    print(f"   Cumulative variance (first 5): {np.round(res_lin['cumulative_variance'][:5], 4)}")
