"""
dae_vs_vae.py
-------------
Experiment 4.3: Comparison of DAE vs VAE latent space geometry.

Trains both models on Swiss Roll and compares:
    - Latent space visualizations (side-by-side)
    - Trustworthiness and continuity metrics
    - Intrinsic dimension estimation on each latent space

Usage:
    python -m experiments.dae_vs_vae
"""

import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets import get_dataset
from data.noise import get_noise_fn
from models.autoencoder import build_dae
from models.vae import build_vae
from models.losses import dae_loss, vae_loss
from evaluation.topology import evaluate_topology
from evaluation.visualization import plot_dae_vs_vae_latent, plot_latent_space_2d
from dimension_estimation.geometric_prob import estimate_dimension


def train_model(model, dataset, loss_fn, noise_fn=None, epochs=200, lr=1e-3,
                batch_size=256, device="cpu"):
    """Generic training loop for DAE or VAE."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            x = batch.to(device)
            x_input = noise_fn(x) if noise_fn else x
            output = model(x_input)
            loss_dict = loss_fn(x, output)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_loss += loss_dict["loss"].item()
            n += 1
        losses.append(epoch_loss / n)

        if epoch % 50 == 0:
            print(f"    Epoch {epoch}/{epochs} | Loss: {losses[-1]:.6f}")

    return losses


@torch.no_grad()
def get_latent(model, dataset, device="cpu"):
    model.eval()
    z = model.encode(dataset.data.to(device))
    return z.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "results/dae_vs_vae"
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Dataset ──
    dataset = get_dataset("swiss_roll", n_samples=2000, noise=0.0)
    X_high = dataset.data.numpy()
    color = dataset.labels
    input_dim = dataset.data.shape[1]

    # ── Train DAE ──
    print("Training DAE...")
    model_dae = build_dae(input_dim, latent_dim=2).to(device)
    noise_fn = get_noise_fn("gaussian", sigma=0.2)
    losses_dae = train_model(model_dae, dataset, dae_loss, noise_fn,
                             epochs=200, device=device)
    z_dae = get_latent(model_dae, dataset, device)

    # ── Train VAE ──
    print("\nTraining VAE...")
    model_vae = build_vae(input_dim, latent_dim=2, beta=1.0).to(device)
    vae_loss_fn = lambda x, out: vae_loss(x, out, beta=1.0)
    losses_vae = train_model(model_vae, dataset, vae_loss_fn,
                             epochs=200, device=device)
    z_vae = get_latent(model_vae, dataset, device)

    # ── Topology metrics ──
    print("\nEvaluating topology...")
    topo_dae = evaluate_topology(X_high, z_dae, n_neighbors=12)
    topo_vae = evaluate_topology(X_high, z_vae, n_neighbors=12)

    print(f"  DAE - Trust: {topo_dae['trustworthiness']:.4f}, "
          f"Cont: {topo_dae['continuity']:.4f}")
    print(f"  VAE - Trust: {topo_vae['trustworthiness']:.4f}, "
          f"Cont: {topo_vae['continuity']:.4f}")

    # ── Dimension estimation on latent spaces ──
    print("\nEstimating intrinsic dimension on latent spaces...")
    dim_dae, res_dae = estimate_dimension(z_dae, n_min=1, n_max=5)
    dim_vae, res_vae = estimate_dimension(z_vae, n_min=1, n_max=5)
    print(f"  DAE latent intrinsic dim: {dim_dae}")
    print(f"  VAE latent intrinsic dim: {dim_vae}")

    # ── Save everything ──
    results = {
        "dae": {
            "final_loss": losses_dae[-1],
            "trustworthiness": topo_dae["trustworthiness"],
            "continuity": topo_dae["continuity"],
            "estimated_dim": dim_dae,
        },
        "vae": {
            "final_loss": losses_vae[-1],
            "trustworthiness": topo_vae["trustworthiness"],
            "continuity": topo_vae["continuity"],
            "estimated_dim": dim_vae,
        },
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Visualizations
    plot_dae_vs_vae_latent(
        z_dae, z_vae, color=color,
        save_path=os.path.join(save_dir, "dae_vs_vae_latent.png")
    )
    plot_latent_space_2d(
        z_dae, color=color, title="DAE Latent Space",
        save_path=os.path.join(save_dir, "dae_latent.png")
    )
    plot_latent_space_2d(
        z_vae, color=color, title="VAE Latent Space",
        save_path=os.path.join(save_dir, "vae_latent.png")
    )

    np.save(os.path.join(save_dir, "z_dae.npy"), z_dae)
    np.save(os.path.join(save_dir, "z_vae.npy"), z_vae)

    print(f"\nAll results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
