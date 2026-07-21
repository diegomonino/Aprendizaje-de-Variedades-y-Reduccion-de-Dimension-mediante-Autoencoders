"""
dimension_pipeline.py
---------------------
Experiment 4.4: Intrinsic dimension estimation pipeline.

Applies both Minkowski and geometrically-probabilistic methods to:
    1. Original data in R^D (validation of methods on known ground truth)
    2. DAE latent space (does it preserve intrinsic dimension?)
    3. VAE latent space (does the KL prior distort dimension?)

Generates the composite 4-panel figure for the report.

Usage:
    python -m experiments.dimension_pipeline
    python -m experiments.dimension_pipeline --dataset linear --intrinsic_dim 3 --ambient_dim 30
"""

import argparse
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
from dimension_estimation.minkowski import minkowski_dimension
from dimension_estimation.geometric_prob import estimate_dimension
from dimension_estimation.pca import estimate_dimension_pca
from evaluation.visualization import (
    plot_minkowski, plot_probabilistic_dim, plot_pca_dim, plot_composite_figure
)


def quick_train(model, dataset, loss_fn, noise_fn=None, epochs=200, lr=1e-3,
                batch_size=256, device="cpu"):
    """Train a model and return it (trained)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            x = batch.to(device)
            x_input = noise_fn(x) if noise_fn else x
            output = model(x_input)
            loss_dict = loss_fn(x, output)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                all_x = dataset.data.to(device)
                out = model(all_x)
                l = loss_fn(all_x, out)["loss"].item()
            print(f"    Epoch {epoch}/{epochs} | Loss: {l:.6f}")

    return model


@torch.no_grad()
def get_latent(model, dataset, device="cpu"):
    model.eval()
    z = model.encode(dataset.data.to(device))
    return z.cpu().numpy()


def run_dimension_analysis(data, name, n_max=None, pca_threshold=0.95, save_dir="."):
    """Run all three dimension estimation methods on a dataset."""
    print(f"\n  [{name}] Minkowski dimension...")
    try:
        mink_dim, neg_lr, log_N = minkowski_dimension(data)
        print(f"    Minkowski estimate: {mink_dim:.2f}")
        plot_minkowski(neg_lr, log_N, mink_dim,
                       title=f"Minkowski: {name}",
                       save_path=os.path.join(save_dir, f"minkowski_{name}.png"))
    except ValueError as e:
        print(f"    Minkowski failed: {e}")
        mink_dim, neg_lr, log_N = None, None, None

    if n_max is None:
        n_max = min(data.shape[1], 15)

    print(f"  [{name}] Probabilistic dimension (n_max={n_max})...")
    prob_dim, prob_results = estimate_dimension(data, n_min=1, n_max=n_max)
    print(f"    Probabilistic estimate: {prob_dim}")
    plot_probabilistic_dim(prob_results,
                           title=f"Probabilistic: {name}",
                           save_path=os.path.join(save_dir, f"probabilistic_{name}.png"))

    print(f"  [{name}] PCA dimension (threshold={pca_threshold:.0%})...")
    pca_results = estimate_dimension_pca(data, threshold=pca_threshold)
    print(f"    PCA variance-threshold estimate: {pca_results['variance_threshold_dim']}")
    print(f"    PCA participation-ratio estimate: {pca_results['participation_ratio_dim']:.2f}")
    plot_pca_dim(pca_results["explained_variance_ratio"],
                 pca_results["cumulative_variance"],
                 pca_results["variance_threshold_dim"],
                 threshold=pca_threshold,
                 title=f"PCA: {name}",
                 save_path=os.path.join(save_dir, f"pca_{name}.png"))

    return {
        "minkowski_dim": mink_dim,
        "probabilistic_dim": prob_dim,
        "pca_variance_threshold_dim": pca_results["variance_threshold_dim"],
        "pca_participation_ratio_dim": pca_results["participation_ratio_dim"],
        "mink_neg_log_r": neg_lr,
        "mink_log_N": log_N,
        "prob_results": prob_results,
        "pca_results": pca_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swiss_roll")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--intrinsic_dim", type=int, default=3)
    parser.add_argument("--ambient_dim", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_max", type=int, default=7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = f"results/dimension_{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Dataset ──
    ds_kwargs = {"n_samples": args.n_samples}
    if args.dataset == "linear":
        ds_kwargs["intrinsic_dim"] = args.intrinsic_dim
        ds_kwargs["ambient_dim"] = args.ambient_dim

    dataset = get_dataset(args.dataset, **ds_kwargs)
    X = dataset.data.numpy()
    input_dim = X.shape[1]
    print(f"Dataset: {args.dataset}, shape: {X.shape}")

    # ── 1. Dimension estimation on original data ──
    print("\n" + "=" * 60)
    print("STEP 1: Dimension estimation on ORIGINAL data")
    print("=" * 60)
    orig_results = run_dimension_analysis(X, "original", n_max=args.n_max,
                                          save_dir=save_dir)

    # ── 2. Train DAE and estimate dimension on latent space ──
    print("\n" + "=" * 60)
    print("STEP 2: Train DAE, estimate dimension on LATENT SPACE")
    print("=" * 60)
    print("  Training DAE...")
    model_dae = build_dae(input_dim, latent_dim=args.latent_dim).to(device)
    noise_fn = get_noise_fn("gaussian", sigma=0.2)
    model_dae = quick_train(model_dae, dataset, dae_loss, noise_fn,
                            epochs=args.epochs, device=device)
    z_dae = get_latent(model_dae, dataset, device)
    dae_results = run_dimension_analysis(z_dae, "dae_latent", n_max=args.n_max,
                                         save_dir=save_dir)

    # ── 3. Train VAE and estimate dimension on latent space ──
    print("\n" + "=" * 60)
    print("STEP 3: Train VAE, estimate dimension on LATENT SPACE")
    print("=" * 60)
    print("  Training VAE...")
    model_vae = build_vae(input_dim, latent_dim=args.latent_dim, beta=1.0).to(device)
    vae_loss_fn = lambda x, out: vae_loss(x, out, beta=1.0)
    model_vae = quick_train(model_vae, dataset, vae_loss_fn,
                            epochs=args.epochs, device=device)
    z_vae = get_latent(model_vae, dataset, device)
    vae_results = run_dimension_analysis(z_vae, "vae_latent", n_max=args.n_max,
                                         save_dir=save_dir)

    # ── Composite figure (for Swiss Roll with 3D data) ──
    if args.dataset == "swiss_roll" and orig_results["mink_neg_log_r"] is not None:
        print("\n  Generating composite figure...")
        plot_composite_figure(
            data_3d=X,
            z_latent=z_dae,
            color=dataset.labels,
            mink_neg_log_r=orig_results["mink_neg_log_r"],
            mink_log_N=orig_results["mink_log_N"],
            mink_dim=orig_results["minkowski_dim"],
            prob_results=orig_results["prob_results"],
            save_path=os.path.join(save_dir, "composite_figure.png")
        )

    # ── Summary ──
    summary = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "original": {
            "minkowski_dim": orig_results["minkowski_dim"],
            "probabilistic_dim": orig_results["probabilistic_dim"],
            "pca_variance_threshold_dim": orig_results["pca_variance_threshold_dim"],
            "pca_participation_ratio_dim": orig_results["pca_participation_ratio_dim"],
        },
        "dae_latent": {
            "minkowski_dim": dae_results["minkowski_dim"],
            "probabilistic_dim": dae_results["probabilistic_dim"],
            "pca_variance_threshold_dim": dae_results["pca_variance_threshold_dim"],
            "pca_participation_ratio_dim": dae_results["pca_participation_ratio_dim"],
        },
        "vae_latent": {
            "minkowski_dim": vae_results["minkowski_dim"],
            "probabilistic_dim": vae_results["probabilistic_dim"],
            "pca_variance_threshold_dim": vae_results["pca_variance_threshold_dim"],
            "pca_participation_ratio_dim": vae_results["pca_participation_ratio_dim"],
        },
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original data:  Mink={orig_results['minkowski_dim']}, "
          f"Prob={orig_results['probabilistic_dim']}, "
          f"PCA(95%)={orig_results['pca_variance_threshold_dim']}, "
          f"PCA(PR)={orig_results['pca_participation_ratio_dim']:.2f}")
    print(f"  DAE latent:     Mink={dae_results['minkowski_dim']}, "
          f"Prob={dae_results['probabilistic_dim']}, "
          f"PCA(95%)={dae_results['pca_variance_threshold_dim']}, "
          f"PCA(PR)={dae_results['pca_participation_ratio_dim']:.2f}")
    print(f"  VAE latent:     Mink={vae_results['minkowski_dim']}, "
          f"Prob={vae_results['probabilistic_dim']}, "
          f"PCA(95%)={vae_results['pca_variance_threshold_dim']}, "
          f"PCA(PR)={vae_results['pca_participation_ratio_dim']:.2f}")
    print(f"\nAll results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
