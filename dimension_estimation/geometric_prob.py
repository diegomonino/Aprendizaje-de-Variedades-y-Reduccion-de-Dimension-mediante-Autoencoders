"""
geometric_prob.py
-----------------
Geometrically-probabilistic intrinsic dimension estimation.
Based on Ivanov et al. (2021): "Manifold Hypothesis in Data Analysis:
Double Geometrically-Probabilistic Approach to Manifold Dimension Estimation."

Core idea:
    For points uniformly distributed on an n-dimensional manifold,
    the nearest-neighbor distance d_min satisfies that V^n(d_min)
    follows an exponential distribution.

    For the correct n:
        - A1^2(n) ≈ A2(n)   (mean^2 ≈ variance, exponential property)
        - D_KS(n) -> min     (best Kolmogorov-Smirnov fit)

Algorithm:
    1. Flatten data (CDF transformation per coordinate)
    2. Compute nearest-neighbor distances
    3. Remove boundary outliers
    4. For each candidate n, compute V^n(d_min) and test exponentiality
    5. Select n where KS statistic is minimized and A1^2 ≈ A2
"""

import numpy as np
from scipy.stats import kstest
from typing import Tuple, Dict

from .flattening import flatten_data
from .metrics import nearest_neighbor_distances, volume_ball, remove_outliers


def evaluate_dimension_candidate(distances: np.ndarray, n: int) -> Dict[str, float]:
    """
    Evaluate a single dimension candidate n.

    Computes V^n(d_min) for each point and checks whether the
    resulting distribution is exponential.

    Args:
        distances: Array of nearest-neighbor distances (after outlier removal).
        n:         Candidate dimension.

    Returns:
        Dict with:
            'A1_sq':   E[V^n(d_min)]^2  (squared mean)
            'A2':      Var[V^n(d_min)]   (variance)
            'ks_stat': Kolmogorov-Smirnov statistic vs exponential
            'ratio':   A1_sq / A2 (should be ~1.0 for correct dimension)
    """
    V = volume_ball(distances, n)

    mean_V = np.mean(V)
    var_V = np.var(V)

    A1_sq = mean_V ** 2
    A2 = var_V

    # KS test against exponential with estimated rate parameter
    lambda_hat = 1.0 / mean_V if mean_V > 0 else 1e10
    ks_stat, _ = kstest(V, "expon", args=(0, 1.0 / lambda_hat))

    ratio = A1_sq / A2 if A2 > 0 else float("inf")

    return {"A1_sq": A1_sq, "A2": A2, "ks_stat": ks_stat, "ratio": ratio}


def estimate_dimension(
    data: np.ndarray,
    n_min: int = 1,
    n_max: int = None,
    apply_flattening: bool = True,
    outlier_percentile: float = 95.0
) -> Tuple[int, Dict]:
    """
    Estimate the intrinsic dimension of a point cloud using the
    geometrically-probabilistic method.

    Args:
        data:                Array of shape (N, D).
        n_min:               Minimum candidate dimension to test.
        n_max:               Maximum candidate dimension. Defaults to D.
        apply_flattening:    Whether to apply CDF flattening before estimation.
        outlier_percentile:  Percentile for outlier removal on distances.

    Returns:
        Tuple of:
            - best_n:  estimated intrinsic dimension
            - results: dict mapping each candidate n to its evaluation metrics,
                       plus 'candidates' (list of n values tested)
    """
    N, D = data.shape

    if n_max is None:
        n_max = min(D, 30)  # cap to avoid numerical issues with V^n

    # Step 1: Flatten (CDF transformation)
    if apply_flattening:
        data_proc = flatten_data(data)
    else:
        data_proc = data.copy()

    # Step 2: Nearest-neighbor distances
    dists = nearest_neighbor_distances(data_proc)

    # Step 3: Remove boundary outliers
    dists = remove_outliers(dists, percentile=outlier_percentile)

    # Step 4: Evaluate each candidate dimension
    candidates = list(range(n_min, n_max + 1))
    results = {"candidates": candidates}
    A1_sq_vals = []
    A2_vals = []
    ks_vals = []

    for n in candidates:
        eval_n = evaluate_dimension_candidate(dists, n)
        results[n] = eval_n
        A1_sq_vals.append(eval_n["A1_sq"])
        A2_vals.append(eval_n["A2"])
        ks_vals.append(eval_n["ks_stat"])

    results["A1_sq_vals"] = np.array(A1_sq_vals)
    results["A2_vals"] = np.array(A2_vals)
    results["ks_vals"] = np.array(ks_vals)

    # Step 5: Select best n (minimize KS statistic)
    best_idx = int(np.argmin(ks_vals))
    best_n = candidates[best_idx]

    return best_n, results


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    print("== Swiss Roll (expected dim = 2) ==")
    X, _ = make_swiss_roll(n_samples=15000, noise=0.0, random_state=42)
    dim, results = estimate_dimension(X, n_min=1, n_max=7)
    print(f"  Estimated dimension: {dim}")
    print(f"  KS statistics: {[f'{v:.4f}' for v in results['ks_vals']]}")
    print(f"  A1^2 / A2 ratios: ", end="")
    for n in results["candidates"]:
        r = results[n]["ratio"]
        print(f"n={n}: {r:.2f}  ", end="")
    print()

    print("\n== Linear H_3,30 (expected dim = 3) ==")
    rng = np.random.RandomState(42)
    Z = rng.uniform(0, 1, size=(10000, 3))
    A_raw = rng.randn(30, 3)
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :3]
    X_lin = Z @ A.T
    dim_lin, results_lin = estimate_dimension(X_lin, n_min=1, n_max=7)
    print(f"  Estimated dimension: {dim_lin}")
    print(f"  KS statistics: {[f'{v:.4f}' for v in results_lin['ks_vals']]}")
