"""
ablation_noise.py
-----------------
Experiment 4.2: Ablation study on corruption type and noise level.

Trains DAEs with different noise configurations on Swiss Roll and
evaluates representation quality via trustworthiness and continuity.

Usage:
    python -m experiments.ablation_noise
"""

import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets import get_dataset
from data.noise import get_noise_fn
from models.autoencoder import build_dae
from models.losses import dae_loss
from evaluation.topology import evaluate_topology


def train_dae(dataset, noise_fn, latent_dim=2, epochs=200, lr=1e-3, batch_size=256,
              device="cpu"):
    """Train a DAE and return the model + loss history."""
    input_dim = dataset.data.shape[1]
    model = build_dae(input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            x = batch.to(device)
            x_noisy = noise_fn(x)
            output = model(x_noisy)
            loss_dict = dae_loss(x, output)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_loss += loss_dict["loss"].item()
            n += 1
        losses.append(epoch_loss / n)

    return model, losses


@torch.no_grad()
def get_latent(model, dataset, device="cpu"):
    model.eval()
    z = model.encode(dataset.data.to(device))
    return z.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "results/ablation_noise"
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    dataset = get_dataset("swiss_roll", n_samples=2000, noise=0.0)
    X_high = dataset.data.numpy()

    # Configurations to test
    configs = [
        {"noise_type": "gaussian", "params": {"sigma": 0.05}, "label": "gauss_0.05"},
        {"noise_type": "gaussian", "params": {"sigma": 0.2},  "label": "gauss_0.2"},
        {"noise_type": "gaussian", "params": {"sigma": 0.5},  "label": "gauss_0.5"},
        {"noise_type": "masking",  "params": {"p": 0.1},      "label": "mask_0.1"},
        {"noise_type": "masking",  "params": {"p": 0.3},      "label": "mask_0.3"},
        {"noise_type": "masking",  "params": {"p": 0.5},      "label": "mask_0.5"},
        {"noise_type": "salt_pepper", "params": {"p": 0.1},   "label": "sp_0.1"},
        {"noise_type": "salt_pepper", "params": {"p": 0.3},   "label": "sp_0.3"},
    ]

    results = []
    print("=" * 70)
    print("ABLATION: Noise Type and Level on Swiss Roll")
    print("=" * 70)

    for cfg in configs:
        print(f"\n  Training: {cfg['label']}...")
        noise_fn = get_noise_fn(cfg["noise_type"], **cfg["params"])

        model, losses = train_dae(dataset, noise_fn, epochs=200, device=device)
        z = get_latent(model, dataset, device)

        # Topology metrics
        scores = evaluate_topology(X_high, z, n_neighbors=12)

        result = {
            "label": cfg["label"],
            "noise_type": cfg["noise_type"],
            "params": cfg["params"],
            "final_loss": losses[-1],
            "trustworthiness": scores["trustworthiness"],
            "continuity": scores["continuity"],
        }
        results.append(result)
        print(f"    Loss: {losses[-1]:.6f} | "
              f"Trust: {scores['trustworthiness']:.4f} | "
              f"Cont: {scores['continuity']:.4f}")

    # Save results
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot results ──
    labels = [r["label"] for r in results]
    trust = [r["trustworthiness"] for r in results]
    cont = [r["continuity"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, trust, width, label="Trustworthiness", color="steelblue")
    bars2 = ax.bar(x_pos + width/2, cont, width, label="Continuity", color="darkorange")

    ax.set_ylabel("Score")
    ax.set_title("Ablation: Effect of Noise Type and Level on Representation Quality")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_noise.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to: {save_dir}/")


if __name__ == "__main__":
    main()
