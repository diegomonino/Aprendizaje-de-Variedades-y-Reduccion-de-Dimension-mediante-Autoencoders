"""
losses.py
---------
Loss functions for DAE and VAE training.

    - dae_loss:  MSE reconstruction loss
    - vae_loss:  MSE reconstruction + beta * KL divergence (ELBO)
"""

import torch
import torch.nn.functional as F


def dae_loss(x_clean: torch.Tensor, model_output: dict) -> dict:
    """
    DAE loss: MSE between clean input and reconstruction.

    Args:
        x_clean:      Original (uncorrupted) input.
        model_output:  Output dict from DAE.forward() with key 'x_rec'.

    Returns:
        dict with:
            'loss':      total loss (scalar)
            'mse':       reconstruction MSE (scalar)
    """
    x_rec = model_output["x_rec"]
    mse = F.mse_loss(x_rec, x_clean, reduction="mean")
    return {"loss": mse, "mse": mse}


def vae_loss(x_clean: torch.Tensor, model_output: dict, beta: float = 1.0) -> dict:
    """
    VAE loss: reconstruction MSE + beta * KL divergence.

    The KL divergence is computed analytically for Gaussian encoder
    against N(0, I) prior:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Args:
        x_clean:      Original input.
        model_output:  Output dict from VAE.forward() with keys
                       'x_rec', 'mu', 'log_var'.
        beta:          Weight for KL term (beta-VAE framework).

    Returns:
        dict with:
            'loss':  total loss (scalar)
            'mse':   reconstruction MSE (scalar)
            'kl':    KL divergence (scalar)
    """
    x_rec = model_output["x_rec"]
    mu = model_output["mu"]
    log_var = model_output["log_var"]

    mse = F.mse_loss(x_rec, x_clean, reduction="mean")

    # KL divergence: analytical formula for Gaussian vs N(0,I)
    kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

    loss = mse + beta * kl
    return {"loss": loss, "mse": mse, "kl": kl}


if __name__ == "__main__":
    # Quick test
    batch_size, input_dim, latent_dim = 16, 3, 2
    x = torch.randn(batch_size, input_dim)

    # Fake DAE output
    dae_out = {"x_rec": x + 0.1 * torch.randn_like(x), "z": torch.randn(batch_size, latent_dim)}
    d_loss = dae_loss(x, dae_out)
    print(f"DAE loss: {d_loss['loss']:.4f}")

    # Fake VAE output
    vae_out = {
        "x_rec": x + 0.1 * torch.randn_like(x),
        "z": torch.randn(batch_size, latent_dim),
        "mu": torch.randn(batch_size, latent_dim),
        "log_var": torch.randn(batch_size, latent_dim),
    }
    v_loss = vae_loss(x, vae_out, beta=1.0)
    print(f"VAE loss: {v_loss['loss']:.4f} (mse={v_loss['mse']:.4f}, kl={v_loss['kl']:.4f})")
