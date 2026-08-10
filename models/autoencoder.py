"""
autoencoder.py
--------------
Denoising Autoencoder (DAE) for manifold learning.

Architecture: symmetric MLP encoder-decoder with configurable hidden dims
and latent bottleneck. Noise corruption is applied externally via data/noise.py
so the model itself is a standard autoencoder.
"""

import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """MLP encoder: R^input_dim -> R^latent_dim."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """MLP decoder: R^latent_dim -> R^output_dim."""

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


class DAE(nn.Module):
    """
    Denoising Autoencoder.

    The noise is NOT applied inside this module. During training,
    the caller corrupts the input and passes both clean and corrupted
    versions. This keeps the model clean and reusable.

    Args:
        input_dim:   Dimensionality of input data.
        hidden_dims: List of hidden layer sizes (e.g. [128, 64]).
        latent_dim:  Bottleneck dimensionality.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to data space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            x: Input tensor (possibly corrupted by the caller).

        Returns:
            dict with keys:
                'z':     latent representation
                'x_rec': reconstruction
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return {"z": z, "x_rec": x_rec}


class LinearAE(nn.Module):
    """
    Linear Autoencoder (no nonlinearities).

    A single linear map encodes R^input_dim -> R^latent_dim and a single
    linear map decodes back. With an MSE objective the optimal solution
    spans the same subspace as the top `latent_dim` principal components
    (Baldi & Hornik, 1989), so this model is the natural bridge between
    autoencoders and PCA used in Step 4. Biases are included so the model
    can centre the data, matching PCA on mean-centred inputs.

    Exposes the same encode/decode/forward dict contract as DAE, so it
    plugs into the shared training loop and metrics unchanged.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_rec = self.decode(z)
        return {"z": z, "x_rec": x_rec}


def build_linear_ae(input_dim: int, latent_dim: int = 2) -> LinearAE:
    """Convenience builder for the linear autoencoder."""
    return LinearAE(input_dim, latent_dim)


def build_dae(input_dim: int, hidden_dims: List[int] = None, latent_dim: int = 2) -> DAE:
    """
    Convenience builder with sensible defaults.

    Args:
        input_dim:   Data dimensionality.
        hidden_dims: Encoder hidden layers. Defaults based on input_dim.
        latent_dim:  Bottleneck size.

    Returns:
        DAE instance.
    """
    if hidden_dims is None:
        if input_dim <= 10:
            hidden_dims = [32, 16]
        elif input_dim <= 100:
            hidden_dims = [128, 64]
        else:
            hidden_dims = [512, 256, 128]
    return DAE(input_dim, hidden_dims, latent_dim)


if __name__ == "__main__":
    # Quick test with Swiss Roll dimensions
    model = build_dae(input_dim=3, latent_dim=2)
    print(model)
    x = torch.randn(16, 3)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Latent: {out['z'].shape}")
    print(f"Recon:  {out['x_rec'].shape}")
