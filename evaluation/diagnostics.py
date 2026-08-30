"""
diagnostics.py
--------------
Probes of a trained autoencoder's latent space, reused for the plain
nonlinear AE (Step 5) and for the VAE / DAE (Step 6) so the three models
are diagnosed identically. The first two apply to any model of the repo;
the third is VAE-specific.

Diagnostic 1 — latent holes (latent_hole_probe)
    A plain AE only constrains the decoder on the region where training
    codes land; elsewhere the decoder can produce arbitrary output. We
    sample latent points *between* / *around* the training codes, decode
    them, and measure how far the decoded points fall from the real data
    manifold (nearest-neighbour distance in data space). A large ratio of
    hole-distance to genuine-reconstruction-distance signals gaps where
    the decoder generates implausible samples.

Diagnostic 2 — input sensitivity (noise_sensitivity)
    We perturb inputs with small Gaussian noise and measure (a) how much
    the reconstruction error grows and (b) how far the latent code moves
    relative to the input perturbation (an amplification factor). A robust
    encoder keeps both small; a brittle one amplifies noise. This is
    precisely the property the DAE is trained to improve, so contrasting
    the plain AE against the DAE here is the point of the experiment.

Diagnostic 3 — posterior collapse (posterior_variance_stats, VAE only)
    A VAE dimension that carries no information settles on the prior:
    sigma^2_phi(x) ~ 1 and mu_phi(x) ~ constant, so its KL contribution
    vanishes. Counting how many of the d latent dimensions are in that
    state ("active units") is the standard way to tell a VAE that uses its
    whole latent space from one that has silently thrown part of it away,
    and is what separates the two seed groups observed on COIL-20 (some
    seeds end with a decoder participation ratio ~1, others with ~2-4).
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@torch.no_grad()
def latent_hole_probe(
    model,
    z_codes: np.ndarray,
    X: np.ndarray,
    n_probes: int = 200,
    mode: str = "interpolation",
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Decode latent points away from the training codes and measure how far
    the results land from the real data manifold.

    Args:
        model:    Trained model with a decode method.
        z_codes:  Training latent codes, array (N, d).
        X:        Original data, array (N, D).
        n_probes: Number of latent probe points to generate.
        mode:     'interpolation' (midpoints of random code pairs) or
                  'uniform' (uniform samples in the code bounding box).
        seed:     RNG seed.
        device:   'cpu' or 'cuda'.

    Returns:
        Dict with:
            'hole_dist_mean' : mean NN distance (decoded holes -> data)
            'real_dist_mean' : mean NN distance (decoded real codes -> data)
            'ratio'          : hole_dist_mean / real_dist_mean (>=1; large
                               => decoder invents off-manifold points)
    """
    model.eval()
    model.to(device)
    rng = np.random.RandomState(seed)
    N = z_codes.shape[0]

    if mode == "interpolation":
        i = rng.randint(0, N, size=n_probes)
        j = rng.randint(0, N, size=n_probes)
        z_holes = 0.5 * (z_codes[i] + z_codes[j])
    elif mode == "uniform":
        lo, hi = z_codes.min(axis=0), z_codes.max(axis=0)
        z_holes = rng.uniform(lo, hi, size=(n_probes, z_codes.shape[1]))
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    # Decode the holes and a matching sample of genuine training codes.
    z_holes_t = torch.tensor(z_holes, dtype=torch.float32, device=device)
    x_holes = model.decode(z_holes_t).cpu().numpy()

    idx_real = rng.choice(N, size=n_probes, replace=True)
    z_real_t = torch.tensor(z_codes[idx_real], dtype=torch.float32, device=device)
    x_real = model.decode(z_real_t).cpu().numpy()

    nn = NearestNeighbors(n_neighbors=1).fit(X)
    hole_dist, _ = nn.kneighbors(x_holes)
    real_dist, _ = nn.kneighbors(x_real)

    hole_mean = float(hole_dist.mean())
    real_mean = float(real_dist.mean())
    ratio = hole_mean / real_mean if real_mean > 1e-12 else float("inf")

    return {
        "hole_dist_mean": hole_mean,
        "real_dist_mean": real_mean,
        "ratio": ratio,
    }


@torch.no_grad()
def noise_sensitivity(
    model,
    X: np.ndarray,
    sigma: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Measure how a small input perturbation propagates through the model.

    Args:
        model:  Trained model with encode and forward (dict with 'x_rec').
        X:      Data array (N, D).
        sigma:  Std of the additive Gaussian perturbation.
        seed:   RNG seed.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with:
            'mse_clean'        : reconstruction MSE on clean input
            'mse_noisy'        : reconstruction MSE on perturbed input
                                 (target = clean x)
            'mse_increase'     : mse_noisy - mse_clean
            'latent_shift'     : mean ||f(x+eps) - f(x)||_2
            'input_shift'      : mean ||eps||_2
            'amplification'    : latent_shift / input_shift
    """
    model.eval()
    model.to(device)
    torch.manual_seed(seed)

    x = torch.tensor(X, dtype=torch.float32, device=device)
    eps = sigma * torch.randn_like(x)
    x_noisy = x + eps

    z = model.encode(x)
    z_noisy = model.encode(x_noisy)

    x_rec = model(x)["x_rec"]
    x_rec_noisy = model(x_noisy)["x_rec"]

    mse_clean = torch.mean((x_rec - x) ** 2).item()
    mse_noisy = torch.mean((x_rec_noisy - x) ** 2).item()

    latent_shift = torch.linalg.norm(z_noisy - z, dim=1).mean().item()
    input_shift = torch.linalg.norm(eps, dim=1).mean().item()
    amplification = latent_shift / input_shift if input_shift > 1e-12 else float("inf")

    return {
        "mse_clean": mse_clean,
        "mse_noisy": mse_noisy,
        "mse_increase": mse_noisy - mse_clean,
        "latent_shift": latent_shift,
        "input_shift": input_shift,
        "amplification": amplification,
    }


@torch.no_grad()
def posterior_variance_stats(
    model,
    X: np.ndarray,
    collapse_var_threshold: float = 0.9,
    collapse_kl_threshold: float = 0.01,
    device: str = "cpu",
) -> dict:
    """
    Per-dimension state of a VAE posterior q(z|x) = N(mu(x), sigma^2(x)).

    A latent dimension is "collapsed to the prior" when the encoder stops
    using it: sigma^2_phi(x) drifts up to 1 and mu_phi(x) stops depending
    on x, so the per-dimension KL against N(0,1) goes to 0 and the decoder
    receives pure noise there. Reporting the mean posterior variance and
    the per-dimension KL is therefore the direct measurement behind an
    effective latent dimension smaller than d.

    Two counts are returned because the two signals can disagree while a
    dimension is on its way to collapsing: `n_collapsed_var` uses only the
    variance (mean sigma^2 >= collapse_var_threshold, the textbook signal),
    `n_collapsed_kl` uses the stricter and scale-free criterion that the
    dimension carries almost no information (KL_j < collapse_kl_threshold
    nats, the usual "active units" definition).

    Args:
        model:  Trained VAE (must expose an `encoder` returning
                (mu, log_var)). Raises AttributeError for a plain AE/DAE.
        X:      Data array (N, D).
        collapse_var_threshold: mean sigma^2 above which a dimension counts
                as collapsed (default 0.9, i.e. within 10% of the prior).
        collapse_kl_threshold:  per-dimension KL (nats) below which a
                dimension counts as inactive (default 0.01).
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with:
            'var_by_dim'      : list of mean sigma^2_phi(x) per latent dim
            'var_mean'        : mean of the above over dimensions
            'var_max'         : max of the above (closest dim to the prior)
            'mu_std_by_dim'   : list of std_x(mu_phi(x)) per latent dim
            'kl_by_dim'       : list of mean per-dimension KL (nats)
            'kl_total'        : sum of kl_by_dim (the KL of the loss)
            'n_collapsed_var' : dims with mean sigma^2 >= threshold
            'n_collapsed_kl'  : dims with KL below threshold
            'n_active_kl'     : d - n_collapsed_kl
            'latent_dim'      : d
    """
    if not hasattr(model, "encoder"):
        raise AttributeError(
            f"{type(model).__name__} has no probabilistic encoder: "
            "posterior_variance_stats only applies to a VAE.")

    model.eval()
    model.to(device)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    mu, log_var = model.encoder(x)

    var = log_var.exp()                       # sigma^2_phi(x), shape (N, d)
    var_by_dim = var.mean(dim=0)              # (d,)
    mu_std_by_dim = mu.std(dim=0)             # (d,)

    # Per-dimension KL( N(mu_j, sigma_j^2) || N(0,1) ), averaged over x.
    kl_by_dim = (0.5 * (var + mu.pow(2) - 1.0 - log_var)).mean(dim=0)

    var_np = var_by_dim.cpu().numpy()
    kl_np = kl_by_dim.cpu().numpy()

    return {
        "var_by_dim": [float(v) for v in var_np],
        "var_mean": float(var_np.mean()),
        "var_max": float(var_np.max()),
        "mu_std_by_dim": [float(v) for v in mu_std_by_dim.cpu().numpy()],
        "kl_by_dim": [float(v) for v in kl_np],
        "kl_total": float(kl_np.sum()),
        "n_collapsed_var": int((var_np >= collapse_var_threshold).sum()),
        "n_collapsed_kl": int((kl_np < collapse_kl_threshold).sum()),
        "n_active_kl": int((kl_np >= collapse_kl_threshold).sum()),
        "latent_dim": int(var_np.shape[0]),
    }
