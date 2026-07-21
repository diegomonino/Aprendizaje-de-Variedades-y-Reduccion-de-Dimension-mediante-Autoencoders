"""
minkowski.py
------------
Minkowski (box-counting) dimension estimation.

For a point cloud C near a d-dimensional manifold M in R^D:
    log N(r) ~ log V - d * log r

where N(r) is the number of occupied cubes of side r.
The dimension d is estimated as the slope of log N vs -log r
in the linear regime.

For large D, we enumerate cubes by hashing point coordinates
(iterating over data points) rather than constructing a full grid.

Implementation note (bug fixed 2026-05)
----------------------------------------
The original version used r_min = 2 * median(NN distance) and fitted a
line over the ENTIRE scale range [r_min, r_max].  For manifolds embedded
in high-dimensional spaces (e.g. H_{3,30} in R^30), r_min landed inside
the *flat* region where each point occupies its own box (slope ≈ 0),
causing the fit to mix flat+linear regions and severely underestimate the
dimension (e.g. H_{3,30}: ~1.16 instead of ~3).

The fix has two parts:
  1. Start the curve *before* the flat region ends by using
     r_min = 0.5 * median(NN distance) so the flat-to-linear transition
     is visible.
  2. Auto-detect the linear regime with a sliding-window search that
     finds the window maximising slope subject to R² ≥ 0.95, falling back
     to max(slope × R²) when no window passes the threshold.

Known limitation
----------------
For a d-dimensional flat manifold embedded in R^D with D >> d, the
ambient box-counting overestimates the number of boxes relative to the
intrinsic case.  This "overcounting" is scale-dependent at moderate
sample sizes, compressing the apparent linear regime.  Ivanov et al.
(2021) use N = 10^6 samples for H_{3,30}; with N = 10^4 our estimate is
~2.4–2.7 instead of 3.  This is a sample-size limitation, not a code
bug.  Users should treat Minkowski estimates for high-D embeddings as
lower bounds unless N >> 10^5.
"""

import numpy as np
from typing import Tuple


def count_occupied_cubes(data: np.ndarray, r: float) -> int:
    """
    Count the number of distinct D-dimensional cubes of side r
    that contain at least one data point.

    Efficient implementation: assigns each point to a cube via
    floor(x / r) and counts unique cube indices.

    Args:
        data: Array of shape (N, D).
        r:    Cube side length.

    Returns:
        Number of occupied cubes.
    """
    cube_indices = np.floor(data / r).astype(np.int64)
    unique_cubes = set(map(tuple, cube_indices))
    return len(unique_cubes)


def _find_linear_regime(
    neg_log_r: np.ndarray,
    log_N: np.ndarray,
    window: int = 5,
    r2_thresh: float = 0.95,
) -> Tuple[int, int]:
    """
    Find the most linear window in the log N vs -log r curve.

    The box-counting curve has three regions:
        - flat  (small r) : N ≈ N_points, slope ≈ 0   (each point in own box)
        - linear (medium r): N ~ r^{-d},   slope ≈ d   (the regime we want)
        - saturation (large r): N → 1

    Strategy: among all sliding windows of length `window`, choose the one
    with the HIGHEST slope where R² ≥ r2_thresh.  If no window passes the
    R² threshold, fall back to maximising slope × R².

    High slope + high R² jointly select the linear regime over:
      - the flat region (slope ≈ 0, regardless of R²)
      - the noisy saturation region (high slope but low R²)

    Args:
        neg_log_r : array of -log(r) values.
        log_N     : array of log N(r) values.
        window    : number of consecutive points per local fit (default 5).
        r2_thresh : minimum R² for the primary criterion (default 0.95).

    Returns:
        (start, end) indices of the best window (inclusive).
    """
    n = len(neg_log_r)
    best_slope = -np.inf
    best_score = -np.inf   # fallback: slope * R²
    best_start_primary = None
    best_start_fallback = 0

    for i in range(n - window + 1):
        x = neg_log_r[i: i + window]
        y = log_N[i: i + window]
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        if slope <= 0:
            continue
        residuals = y - np.polyval(coeffs, x)
        ss_res = np.dot(residuals, residuals)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        # Primary criterion: highest slope with acceptable fit quality
        if r2 >= r2_thresh and slope > best_slope:
            best_slope = slope
            best_start_primary = i

        # Fallback criterion: slope × R²
        score = slope * r2
        if score > best_score:
            best_score = score
            best_start_fallback = i

    best_start = best_start_primary if best_start_primary is not None else best_start_fallback
    return best_start, best_start + window - 1


def minkowski_dimension(
    data: np.ndarray,
    r_min: float = None,
    r_max: float = None,
    n_scales: int = 30,
    window: int = 5,
    r2_thresh: float = 0.95,
    auto_linear: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the Minkowski (box-counting) dimension of a point cloud.

    Args:
        data       : Array of shape (N, D).
        r_min      : Minimum cube side.  Defaults to 0.5 × median NN distance
                     (starting before the flat region ends).
        r_max      : Maximum cube side.  Defaults to max per-dim range / 2.
        n_scales   : Number of log-spaced scale values (default 30).
        window     : Sliding-window size for linear-regime detection (default 5).
        r2_thresh  : R² threshold for primary window selection (default 0.95).
        auto_linear: If True (default), restrict the final fit to the
                     auto-detected linear regime.  False = legacy full-range fit.

    Returns:
        Tuple of:
            - dim_estimate : estimated dimension (float)
            - neg_log_r    : array of ALL -log(r) values (full curve, for plotting)
            - log_N        : array of ALL log(N(r)) values (full curve, for plotting)
    """
    from .metrics import nearest_neighbor_distances

    N, D = data.shape

    # ── Scale range ─────────────────────────────────────────────────────────
    if r_min is None:
        dists = nearest_neighbor_distances(data)
        r_min = 0.5 * np.median(dists)   # start before flat region ends
    if r_max is None:
        # Two competing estimates:
        #   data_range/2 works for low-D embeddings (Swiss Roll in R^3) where
        #     per-dimension ranges are comparable to the Euclidean diameter.
        #   diam/3 works for flat manifolds in high-D spaces (H_{3,30} in R^30)
        #     where each dimension carries only a small fraction of the total
        #     extent, making data_range/2 too conservative.
        # Taking the max ensures we cover the linear regime for both cases.
        data_range = data.max(axis=0) - data.min(axis=0)
        r_max_dr = np.max(data_range) / 2.0
        idx = np.random.choice(N, size=min(N, 500), replace=False)
        from scipy.spatial.distance import cdist
        diam = cdist(data[idx], data[idx]).max()
        r_max = max(r_max_dr, diam / 3.0)

    r_min = max(r_min, 1e-10)
    if r_max <= r_min:
        r_max = r_min * 20

    # ── Compute N(r) ────────────────────────────────────────────────────────
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    log_N_list: list = []
    neg_log_r_list: list = []

    for r in r_values:
        n_cubes = count_occupied_cubes(data, r)
        if n_cubes > 1:
            log_N_list.append(np.log(n_cubes))
            neg_log_r_list.append(-np.log(r))

    log_N = np.array(log_N_list)
    neg_log_r = np.array(neg_log_r_list)

    if len(neg_log_r) < 3:
        raise ValueError(
            "Not enough valid scale points for linear fit. "
            "Try adjusting r_min or r_max."
        )

    # ── Linear fit ──────────────────────────────────────────────────────────
    if auto_linear and len(neg_log_r) >= window:
        i0, i1 = _find_linear_regime(neg_log_r, log_N,
                                     window=window, r2_thresh=r2_thresh)
        x_fit = neg_log_r[i0: i1 + 1]
        y_fit = log_N[i0: i1 + 1]
    else:
        x_fit, y_fit = neg_log_r, log_N

    coeffs = np.polyfit(x_fit, y_fit, deg=1)
    dim_estimate = float(coeffs[0])

    return dim_estimate, neg_log_r, log_N


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    print("== Swiss Roll N=2000 (expected ~2, paper: 1–2) ==")
    X, _ = make_swiss_roll(n_samples=2000, noise=0.0, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    dim, _, _ = minkowski_dimension(X)
    print(f"  Estimated: {dim:.2f}")

    print("\n== Linear H_3,30 (expected ~3, paper uses 10^6 samples) ==")
    rng = np.random.RandomState(42)
    for n in [2000, 10000]:
        Z = rng.uniform(0, 1, size=(n, 3))
        A_raw = rng.randn(30, 3)
        A, _ = np.linalg.qr(A_raw)
        A = A[:, :3]
        X_lin = Z @ A.T
        dim_lin, _, _ = minkowski_dimension(X_lin)
        print(f"  N={n:6d}: {dim_lin:.2f}")

    print("\n== Linear H_4,30 (expected ~4) ==")
    rng2 = np.random.RandomState(7)
    Z4 = rng2.uniform(0, 1, size=(10000, 4))
    A4, _ = np.linalg.qr(rng2.randn(30, 4))
    A4 = A4[:, :4]
    X4 = Z4 @ A4.T
    dim4, _, _ = minkowski_dimension(X4)
    print(f"  N=10000: {dim4:.2f}")
