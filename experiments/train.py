"""
train.py
--------
Training script for DAE and VAE models.

Usage:
    python -m experiments.train --model dae --dataset swiss_roll --noise gaussian --sigma 0.2
    python -m experiments.train --model vae --dataset swiss_roll --beta 1.0
    python -m experiments.train --model dae --dataset linear --intrinsic_dim 3 --ambient_dim 30
    python -m experiments.train --model dae --dataset mnist --latent_dim 12 --epochs 50
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets import get_dataloader, get_dataset
from data.noise import get_noise_fn
from models.autoencoder import build_dae
from models.vae import build_vae
from models.losses import dae_loss, vae_loss
from evaluation.visualization import plot_training_loss, plot_latent_space_2d


def train_epoch(model, loader, optimizer, loss_fn, noise_fn=None, device="cpu"):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_clean = batch.to(device)

        # Apply noise for DAE
        if noise_fn is not None:
            x_input = noise_fn(x_clean)
        else:
            x_input = x_clean

        output = model(x_input)
        losses = loss_fn(x_clean, output)

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        total_loss += losses["loss"].item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def extract_latent(model, dataset, device="cpu"):
    """Extract latent representations for the full dataset."""
    model.eval()
    model.to(device)
    data = dataset.data.to(device)
    z = model.encode(data)
    return z.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train DAE or VAE")
    # Model
    parser.add_argument("--model", type=str, default="dae", choices=["dae", "vae"])
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    # Dataset
    parser.add_argument("--dataset", type=str, default="swiss_roll")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--intrinsic_dim", type=int, default=3)
    parser.add_argument("--ambient_dim", type=int, default=30)
    # Noise (DAE)
    parser.add_argument("--noise", type=str, default="gaussian",
                        choices=["gaussian", "masking", "salt_pepper"])
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--mask_p", type=float, default=0.3)
    # VAE
    parser.add_argument("--beta", type=float, default=1.0)
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--save_dir", type=str, default="results")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ──
    ds_kwargs = {"n_samples": args.n_samples}
    if args.dataset == "linear":
        ds_kwargs["intrinsic_dim"] = args.intrinsic_dim
        ds_kwargs["ambient_dim"] = args.ambient_dim

    dataset = get_dataset(args.dataset, **ds_kwargs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    input_dim = dataset.data.shape[1]

    print(f"Dataset: {args.dataset}, shape: {dataset.data.shape}")

    # ── Model ──
    if args.model == "dae":
        model = build_dae(input_dim, args.hidden_dims, args.latent_dim).to(device)
        loss_fn = dae_loss
        # Noise function
        if args.noise == "gaussian":
            noise_fn = get_noise_fn("gaussian", sigma=args.sigma)
        elif args.noise == "masking":
            noise_fn = get_noise_fn("masking", p=args.mask_p)
        else:
            noise_fn = get_noise_fn("salt_pepper", p=args.mask_p)
    else:
        model = build_vae(input_dim, args.hidden_dims, args.latent_dim, args.beta).to(device)
        loss_fn = lambda x, out: vae_loss(x, out, beta=args.beta)
        noise_fn = None

    print(f"Model: {args.model.upper()}, latent_dim={args.latent_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, loss_fn, noise_fn, device)
        losses.append(avg_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.6f}")

    # ── Save ──
    run_name = f"{args.model}_{args.dataset}_{args.noise}_s{args.sigma}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Save config
    config = vars(args)
    config["input_dim"] = input_dim
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save loss curve
    plot_training_loss(losses, title=f"{args.model.upper()} Training Loss",
                       save_path=os.path.join(save_dir, "figures", "loss_curve.png"))

    # Save latent space visualization (if 2D)
    if args.latent_dim == 2:
        z = extract_latent(model, dataset, device)
        plot_latent_space_2d(z, color=dataset.labels,
                             title=f"{args.model.upper()} Latent Space ({args.dataset})",
                             save_path=os.path.join(save_dir, "figures", "latent_space.png"))
        # Save latent coordinates
        np.save(os.path.join(save_dir, "latent_z.npy"), z)

    # Save losses
    np.save(os.path.join(save_dir, "losses.npy"), np.array(losses))

    print(f"All results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
