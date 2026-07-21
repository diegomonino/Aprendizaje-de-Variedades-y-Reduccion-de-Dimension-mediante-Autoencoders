"""
latent_analysis.py
------------------
Analysis tools for the learned latent space.

    - interpolate_latent:  linear interpolation between two latent points
    - encoder_jacobian:    compute the Jacobian of the encoder at a point
    - local_distortion:    measure how much the encoder distorts local geometry
"""

import torch
import numpy as np
from typing import List


def interpolate_latent(
    model,
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    n_steps: int = 10,
    device: str = "cpu"
) -> dict:
    """
    Linear interpolation in latent space between two data points.

    Encodes both endpoints, interpolates in latent space,
    then decodes each interpolated point.

    Args:
        model:    Trained autoencoder (DAE or VAE) with encode/decode methods.
        x_start:  Starting point in data space, shape (D,).
        x_end:    Ending point in data space, shape (D,).
        n_steps:  Number of interpolation steps (including endpoints).
        device:   'cpu' or 'cuda'.

    Returns:
        Dict with:
            'z_interp':  latent interpolations, shape (n_steps, d)
            'x_interp':  decoded interpolations, shape (n_steps, D)
            'alphas':    interpolation coefficients, shape (n_steps,)
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        x_s = x_start.unsqueeze(0).to(device)
        x_e = x_end.unsqueeze(0).to(device)

        z_s = model.encode(x_s)
        z_e = model.encode(x_e)

        alphas = torch.linspace(0, 1, n_steps)
        z_interp = []
        x_interp = []

        for alpha in alphas:
            z = (1 - alpha) * z_s + alpha * z_e
            x_dec = model.decode(z)
            z_interp.append(z.squeeze(0))
            x_interp.append(x_dec.squeeze(0))

        z_interp = torch.stack(z_interp).cpu().numpy()
        x_interp = torch.stack(x_interp).cpu().numpy()
        alphas = alphas.numpy()

    return {"z_interp": z_interp, "x_interp": x_interp, "alphas": alphas}


def encoder_jacobian(
    model,
    x: torch.Tensor,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute the Jacobian matrix of the encoder at a given point.

    J[i, j] = d(z_i) / d(x_j)

    This reveals how the encoder maps local perturbations in data
    space to the latent space — the singular values indicate
    metric distortion.

    Args:
        model: Trained autoencoder with an encode method.
        x:     Single data point, shape (D,).
        device: 'cpu' or 'cuda'.

    Returns:
        Jacobian matrix of shape (d, D).
    """
    model.eval()
    model.to(device)

    x_input = x.clone().unsqueeze(0).to(device).requires_grad_(True)
    z = model.encode(x_input)  # shape: (1, d)

    d = z.shape[1]
    D = x_input.shape[1]
    jacobian = np.zeros((d, D))

    for i in range(d):
        model.zero_grad()
        if x_input.grad is not None:
            x_input.grad.zero_()
        z[0, i].backward(retain_graph=True)
        jacobian[i] = x_input.grad[0].cpu().numpy()

    return jacobian


def local_distortion(
    model,
    data: torch.Tensor,
    n_samples: int = 100,
    device: str = "cpu"
) -> dict:
    """
    Measure local metric distortion of the encoder across multiple points.

    For each point, computes the Jacobian and its singular values.
    The ratio max_sv / min_sv (condition number) measures distortion:
    a value of 1 means isometric (no distortion).

    Args:
        model:     Trained autoencoder.
        data:      Dataset tensor, shape (N, D).
        n_samples: Number of points to evaluate.
        device:    'cpu' or 'cuda'.

    Returns:
        Dict with:
            'condition_numbers': array of shape (n_samples,)
            'singular_values':   list of arrays, each of shape (d,)
            'mean_condition':    average condition number
    """
    n_samples = min(n_samples, len(data))
    indices = np.random.choice(len(data), n_samples, replace=False)

    condition_numbers = []
    singular_values_list = []

    for idx in indices:
        x = data[idx]
        J = encoder_jacobian(model, x, device=device)
        sv = np.linalg.svd(J, compute_uv=False)

        if sv[-1] > 1e-10:
            cond = sv[0] / sv[-1]
        else:
            cond = float("inf")

        condition_numbers.append(cond)
        singular_values_list.append(sv)

    condition_numbers = np.array(condition_numbers)

    return {
        "condition_numbers": condition_numbers,
        "singular_values": singular_values_list,
        "mean_condition": np.mean(condition_numbers[np.isfinite(condition_numbers)]),
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from models.autoencoder import build_dae

    model = build_dae(input_dim=3, latent_dim=2)
    x = torch.randn(100, 3)

    # Test interpolation
    result = interpolate_latent(model, x[0], x[1], n_steps=5)
    print(f"Interpolation: z shape = {result['z_interp'].shape}, "
          f"x shape = {result['x_interp'].shape}")

    # Test Jacobian
    J = encoder_jacobian(model, x[0])
    print(f"Jacobian shape: {J.shape}")  # should be (2, 3)

    # Test distortion
    dist = local_distortion(model, x, n_samples=10)
    print(f"Mean condition number: {dist['mean_condition']:.2f}")
