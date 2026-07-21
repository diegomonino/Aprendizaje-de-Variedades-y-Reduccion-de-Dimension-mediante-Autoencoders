"""
vae.py
------
Variational Autoencoder (VAE) for manifold learning.

Same encoder-decoder backbone as the DAE, but with:
    - Probabilistic encoder: outputs mu and log_var
    - Reparameterization trick for differentiable sampling
    - KL divergence term in the loss (see models/losses.py)
"""

import torch
import torch.nn as nn
from typing import List


class VAEEncoder(nn.Module):
    """
    Probabilistic encoder: R^input_dim -> (mu, log_var) in R^latent_dim.
    Shares hidden layers, then splits into two parallel heads.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """MLP decoder: R^latent_dim -> R^output_dim (identical to DAE decoder)."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Args:
        input_dim:   Dimensionality of input data.
        hidden_dims: List of hidden layer sizes.
        latent_dim:  Bottleneck dimensionality.
        beta:        Weight for the KL divergence term (beta-VAE).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.
        Only samples during training; returns mu during eval.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space (returns the mean, no sampling)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to data space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass with reparameterization.

        Returns:
            dict with keys:
                'z':       sampled latent vector
                'x_rec':   reconstruction
                'mu':      encoder mean
                'log_var': encoder log-variance
        """
        mu, log_var = self.encoder(x)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu

        x_rec = self.decoder(z)
        return {"z": z, "x_rec": x_rec, "mu": mu, "log_var": log_var}


def build_vae(input_dim: int, hidden_dims: List[int] = None, latent_dim: int = 2,
              beta: float = 1.0) -> VAE:
    """Convenience builder with sensible defaults."""
    if hidden_dims is None:
        if input_dim <= 10:
            hidden_dims = [32, 16]
        elif input_dim <= 100:
            hidden_dims = [128, 64]
        else:
            hidden_dims = [512, 256, 128]
    return VAE(input_dim, hidden_dims, latent_dim, beta)


if __name__ == "__main__":
    model = build_vae(input_dim=3, latent_dim=2)
    print(model)
    x = torch.randn(16, 3)
    out = model(x)
    print(f"Input:    {x.shape}")
    print(f"Latent z: {out['z'].shape}")
    print(f"mu:       {out['mu'].shape}")
    print(f"log_var:  {out['log_var'].shape}")
    print(f"Recon:    {out['x_rec'].shape}")
