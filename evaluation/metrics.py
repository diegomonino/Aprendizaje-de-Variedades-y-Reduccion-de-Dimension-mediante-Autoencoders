"""
metrics.py
----------
Common evaluation metrics shared across all four models (linear AE,
nonlinear AE, VAE, DAE) so that Step 8 can build a single comparison
table with identical columns.

Three families of metrics:

    1. Reconstruction error
       reconstruction_mse — mean squared error between x and g(f(x)).

    2. Neighbourhood preservation (topology)
       Re-exported from evaluation.topology: trustworthiness and its
       symmetric counterpart continuity (already rank-based in this repo).

    3. Local distortion / effective dimension via the DECODER Jacobian
       decoder_jacobian, effective_dimension_jacobian.

       The generative map g : R^d -> R^D parameterises the learned
       manifold. Its Jacobian J = dg/dz at a latent point z spans the
       tangent space of that manifold. The number of singular values of
       J above a relative threshold (default 1% of the largest) is the
       local dimension of the image manifold — how many latent directions
       actually move the output. Averaging over many z gives an effective
       dimension estimated *from the model itself* (used in Step 7 and
       compared against the data-driven estimates of Step 3).

       Implemented with torch.func.jacfwd + vmap for a batched, exact
       Jacobian. Forward-mode is used deliberately: decoders map a LOW
       latent dimension d to a HIGH ambient dimension D (d << D), and
       jacfwd's cost scales with the input dimension d rather than the
       output dimension D. jacrev (reverse-mode) would instead build an
       internal (D, D)-sized cotangent basis per batch element -- fine for
       small D (Swiss Roll, MNIST) but a memory blow-up for datasets with
       large pixel-space D such as COIL-20 (D=16384: a single batch of 50
       points needs 50*16384*16384*4 bytes =~ 53 GB with jacrev).
       autograd.functional.jacobian is provided as a fallback.
"""

import numpy as np
import torch

from .topology import evaluate_topology

try:
    from torch.func import jacfwd, vmap
    _HAS_TORCH_FUNC = True
except Exception:  # pragma: no cover - old torch
    _HAS_TORCH_FUNC = False


@torch.no_grad()
def reconstruction_mse(model, data: torch.Tensor, device: str = "cpu") -> float:
    """
    Mean squared reconstruction error over the full dataset.

    Args:
        model: Trained model with a forward returning dict key 'x_rec'.
        data:  Tensor of shape (N, D) (clean data).
        device: 'cpu' or 'cuda'.

    Returns:
        Scalar MSE (float).
    """
    model.eval()
    model.to(device)
    x = data.to(device)
    x_rec = model(x)["x_rec"]
    return torch.mean((x_rec - x) ** 2).item()


def decoder_jacobian(decoder, z: torch.Tensor) -> torch.Tensor:
    """
    Batched Jacobian of a decoder g : R^d -> R^D at latent points z.

    Args:
        decoder: Callable mapping (d,) -> (D,) (an nn.Module works, since
                 nn.Linear broadcasts over leading dims).
        z:       Latent points, shape (B, d).

    Returns:
        Tensor of shape (B, D, d): jac[b] = dg/dz evaluated at z[b].
    """
    if _HAS_TORCH_FUNC:
        def single(z_i):            # (d,) -> (D,)
            return decoder(z_i)
        return vmap(jacfwd(single))(z)      # (B, D, d)

    # Fallback: loop with autograd.functional.jacobian
    from torch.autograd.functional import jacobian
    jacs = []
    for z_i in z:
        J = jacobian(lambda zz: decoder(zz), z_i)   # (D, d)
        jacs.append(J)
    return torch.stack(jacs)


def effective_dimension_jacobian(
    decoder,
    z_points: torch.Tensor,
    rel_threshold: float = 0.01,
    device: str = "cpu",
) -> dict:
    """
    Estimate the effective manifold dimension from the decoder Jacobian.

    For each latent point we compute the singular values of J = dg/dz and
    count how many exceed `rel_threshold` times the largest singular value
    at that point. The mean count over points is the effective dimension.

    Args:
        decoder:       Decoder module (or model.decode), R^d -> R^D.
        z_points:      Latent points to probe, shape (B, d). Typically the
                       encoded training set (or a random subsample of it).
        rel_threshold: Singular values below this fraction of the top one
                       are treated as numerically zero (default 0.01 = 1%).
        device:        'cpu' or 'cuda'.

    Returns:
        Dict with:
            'effective_dim':   mean number of active singular values (float)
            'per_point_dim':   int array of shape (B,)
            'singular_values': array (B, d) of singular values (descending)
            'rel_threshold':   echoed threshold
    """
    z_points = z_points.to(device)
    J = decoder_jacobian(decoder, z_points)           # (B, D, d)
    # Singular values per point; shape (B, min(D, d)) sorted descending.
    sv = torch.linalg.svdvals(J).detach().cpu().numpy()

    top = sv[:, :1]                                    # (B, 1) largest sv
    active = sv > (rel_threshold * top)                # boolean (B, d)
    per_point_dim = active.sum(axis=1).astype(int)

    return {
        "effective_dim": float(per_point_dim.mean()),
        "per_point_dim": per_point_dim,
        "singular_values": sv,
        "rel_threshold": rel_threshold,
    }


def evaluate_common_metrics(
    model,
    X_high: np.ndarray,
    z_latent: np.ndarray,
    n_neighbors: int = 12,
    jacobian_points: int = 200,
    rel_threshold: float = 0.01,
    topo_max_points: int = 1500,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Compute the full set of shared metrics for one (model, dataset) pair.

    This is the single entry point used by Step 8 so every row of the
    comparison table is produced the same way.

    Args:
        model:           Trained model (AE / DAE / VAE).
        X_high:          Original data, array (N, D).
        z_latent:        Latent codes for X_high, array (N, d).
        n_neighbors:     k for trustworthiness / continuity.
        jacobian_points: Number of latent points sampled for the Jacobian
                         effective-dimension estimate (subsampled for speed).
        rel_threshold:   Relative singular-value threshold.
        topo_max_points: Cap on the number of points used for the topology
                         metrics. Continuity is O(N^2) in pure Python, so we
                         subsample (consistently across X and z) above this
                         size. Set to 0/None to disable subsampling.
        device:          'cpu' or 'cuda'.
        seed:            Seed for the Jacobian / topology subsamples.

    Returns:
        Dict with 'mse', 'trustworthiness', 'continuity', 'jacobian_dim'.
    """
    X_t = torch.tensor(X_high, dtype=torch.float32)
    mse = reconstruction_mse(model, X_t, device=device)

    # Topology metrics on a (consistent) subsample for speed.
    X_topo, z_topo = X_high, z_latent
    if topo_max_points and X_high.shape[0] > topo_max_points:
        rng_t = np.random.RandomState(seed)
        sub = rng_t.choice(X_high.shape[0], size=topo_max_points, replace=False)
        X_topo, z_topo = X_high[sub], z_latent[sub]
    topo = evaluate_topology(X_topo, z_topo, n_neighbors=n_neighbors)

    rng = np.random.RandomState(seed)
    n = min(jacobian_points, z_latent.shape[0])
    idx = rng.choice(z_latent.shape[0], size=n, replace=False)
    z_probe = torch.tensor(z_latent[idx], dtype=torch.float32)
    jac = effective_dimension_jacobian(
        model.decode, z_probe, rel_threshold=rel_threshold, device=device
    )

    return {
        "mse": mse,
        "trustworthiness": topo["trustworthiness"],
        "continuity": topo["continuity"],
        "jacobian_dim": jac["effective_dim"],
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.autoencoder import build_dae

    # A decoder R^2 -> R^3 should have local dimension ~2.
    model = build_dae(input_dim=3, latent_dim=2)
    z = torch.randn(50, 2)
    jac = effective_dimension_jacobian(model.decode, z)
    print(f"Decoder R^2 -> R^3, effective dim (expect ~2): "
          f"{jac['effective_dim']:.2f}")
    print(f"Jacobian batch shape: {decoder_jacobian(model.decode, z).shape}")

    X = torch.randn(200, 3)
    print(f"Reconstruction MSE (random model): "
          f"{reconstruction_mse(model, X):.4f}")
