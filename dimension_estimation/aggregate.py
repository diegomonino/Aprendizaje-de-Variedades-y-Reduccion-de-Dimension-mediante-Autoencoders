"""
aggregate.py
------------
Convenience aggregator that runs all intrinsic-dimension estimators of
this package on a single point cloud and returns their results in one
dict. Used by Step 3 of the main experiment to fix, for each dataset,
the latent dimension d used by every model in the later steps
(`choose_latent_dim`, which applies d = ceil(max(Minkowski, geom-prob))).

The three estimators answer complementary questions:
    - Minkowski (box-counting): how does local neighbourhood *count* scale?
    - Geometric-probabilistic (Ivanov et al.): for which n is V^n(d_min)
      exponentially distributed?  -> returns an integer.
    - PCA: how many *linear* directions explain the variance?  (upper
      bound on nonlinear manifolds such as the Swiss Roll.)
"""

import numpy as np

from .minkowski import minkowski_dimension
from .geometric_prob import estimate_dimension
from .pca import estimate_dimension_pca


def estimate_all_dimensions(
    data: np.ndarray,
    n_max: int = 10,
    pca_threshold: float = 0.95,
) -> dict:
    """
    Run Minkowski, geometric-probabilistic and PCA estimators on `data`.

    Args:
        data:          Array (N, D).
        n_max:         Max candidate dimension for the geometric-prob method.
        pca_threshold: Cumulative-variance threshold for PCA (default 0.95).

    Returns:
        Dict with:
            'minkowski'          : float slope estimate
            'geometric_prob'     : int (KS-optimal candidate)
            'pca_threshold'      : int (min components for `pca_threshold`)
            'pca_participation'  : float (participation ratio)
            'geometric_prob_ks'  : list of KS statistics per candidate
    """
    mink, _, _ = minkowski_dimension(data)
    geo, geo_res = estimate_dimension(data, n_min=1, n_max=n_max)
    pca_res = estimate_dimension_pca(data, threshold=pca_threshold)

    return {
        "minkowski": float(mink),
        "geometric_prob": int(geo),
        "pca_threshold": int(pca_res["variance_threshold_dim"]),
        "pca_participation": float(pca_res["participation_ratio_dim"]),
        "geometric_prob_ks": [float(v) for v in geo_res["ks_vals"]],
    }


def choose_latent_dim(estimates: dict, expected: int = None) -> int:
    """
    Pick the single latent dimension d used by the models downstream.

    Rule (the one stated in the thesis):

        d = ceil( max( d*_Minkowski , d*_geom-prob ) )

    The two estimators are lower bounds of different kinds -- Minkowski
    measures how neighbourhood *counts* scale, the geometric-probabilistic
    one for which n the volume V^n(d_min) is exponentially distributed --
    so taking the maximum errs on the side of NOT throwing away structure:
    an under-sized latent space loses information irrecoverably, an
    over-sized one is only wasteful (and shows up in `jacobian_dim_pr`).
    When both agree the max is that common value.

    Minkowski returns NaN whenever box-counting finds no linear scaling
    regime -- the generic situation in high ambient dimension with few
    samples (`mnist`, `coil20_object`; see results/experiment_config.json).
    NaN is ignored via np.nanmax instead of being propagated to d; if BOTH
    estimates are unusable the known/expected dimension is the fallback.

    Caveat, for the record: the max rule does not repair an estimator that
    is itself wrong. In `coil20_object` (N=72 points in D=16384) Minkowski
    is NaN and the geometric-probabilistic one returns 5 on a manifold of
    true dimension 1 (a pure rotation), so the rule returns 5. That is a
    small-sample failure of the estimator in very high dimension, not of
    the selection rule, and it is deliberately NOT patched here: the
    over-sizing is visible downstream in the participation ratio of the
    decoder Jacobian (Step 11), which is the honest way to report it.

    Args:
        estimates: Output dict of estimate_all_dimensions.
        expected:  Known/reference intrinsic dimension, if any.

    Returns:
        Chosen latent dimension d (int, >= 1).
    """
    vals = np.array([estimates.get("minkowski", np.nan),
                     estimates.get("geometric_prob", np.nan)], dtype=float)
    # A degenerate estimate (<= 0, or non-finite) counts as missing, not as
    # a candidate for the maximum.
    vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)

    if np.all(np.isnan(vals)):
        d = int(expected) if expected else 1
    else:
        d = int(np.ceil(np.nanmax(vals)))
    return max(1, d)


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=3000, noise=0.0, random_state=42)
    X = (X - X.mean(0)) / X.std(0)
    est = estimate_all_dimensions(X, n_max=7)
    print("Swiss Roll estimates:", est)
    print("Chosen d:", choose_latent_dim(est, expected=2))
