"""
visualization.py
----------------
Plotting utilities for manifold learning experiments.

All functions save figures to disk and optionally display them.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional
import os


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def plot_swiss_roll_3d(
    data: np.ndarray,
    color: np.ndarray = None,
    title: str = "Swiss Roll",
    save_path: str = None
):
    """Plot Swiss Roll in 3D with color mapping."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                         c=color, cmap="Spectral", s=5, alpha=0.7)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, shrink=0.6)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_space_2d(
    z: np.ndarray,
    color: np.ndarray = None,
    title: str = "Latent Space",
    save_path: str = None
):
    """Plot 2D latent space with color mapping."""
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=color, cmap="Spectral", s=5, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_loss(
    losses: list,
    title: str = "Training Loss",
    save_path: str = None
):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_minkowski(
    neg_log_r: np.ndarray,
    log_N: np.ndarray,
    dim_estimate: float,
    title: str = "Minkowski Dimension",
    save_path: str = None
):
    """
    Plot log N(r) vs -log(r) with fitted line.
    The slope of the fitted line is the dimension estimate.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(neg_log_r, log_N, c="green", s=30, zorder=3, label="Data")

    # Fitted line
    coeffs = np.polyfit(neg_log_r, log_N, deg=1)
    fit_line = np.polyval(coeffs, neg_log_r)
    ax.plot(neg_log_r, fit_line, "r--", linewidth=1.5,
            label=f"Fit (slope = {dim_estimate:.2f})")

    ax.set_xlabel("$-\\log r$")
    ax.set_ylabel("$\\log N(r)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_probabilistic_dim(
    results: Dict,
    title: str = "Geometrically-Probabilistic Estimation",
    save_path: str = None
):
    """
    Plot A1^2 vs A2 and KS statistic for each candidate dimension.
    Replicates the visualization style of Ivanov et al. (2021).
    """
    candidates = results["candidates"]
    A1_sq = results["A1_sq_vals"]
    A2 = results["A2_vals"]
    ks = results["ks_vals"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel a: A1^2 vs A2 in log scale
    ax1.plot(candidates, A1_sq, "o", color="red", markersize=8, label="$A_1^2$")
    ax1.plot(candidates, A2, "x", color="blue", markersize=8, label="$A_2$")
    ax1.set_yscale("log")
    ax1.set_xlabel("Candidate dimension $n$")
    ax1.set_ylabel("Value")
    ax1.set_title("$A_1^2(n)$ vs $A_2(n)$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel b: KS statistic
    ax2.plot(candidates, ks, "o-", color="darkgreen", markersize=8)
    best_idx = int(np.argmin(ks))
    ax2.axvline(x=candidates[best_idx], color="red", linestyle="--", alpha=0.5,
                label=f"Best: n={candidates[best_idx]}")
    ax2.set_xlabel("Candidate dimension $n$")
    ax2.set_ylabel("K-S statistic")
    ax2.set_title("Kolmogorov-Smirnov Statistic")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dae_vs_vae_latent(
    z_dae: np.ndarray,
    z_vae: np.ndarray,
    color: np.ndarray = None,
    save_path: str = None
):
    """Side-by-side comparison of DAE and VAE latent spaces."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sc1 = ax1.scatter(z_dae[:, 0], z_dae[:, 1], c=color, cmap="Spectral", s=5, alpha=0.7)
    ax1.set_title("DAE Latent Space")
    ax1.set_xlabel("$z_1$")
    ax1.set_ylabel("$z_2$")
    plt.colorbar(sc1, ax=ax1)

    sc2 = ax2.scatter(z_vae[:, 0], z_vae[:, 1], c=color, cmap="Spectral", s=5, alpha=0.7)
    ax2.set_title("VAE Latent Space")
    ax2.set_xlabel("$z_1$")
    ax2.set_ylabel("$z_2$")
    plt.colorbar(sc2, ax=ax2)

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_composite_figure(
    data_3d: np.ndarray,
    z_latent: np.ndarray,
    color: np.ndarray,
    mink_neg_log_r: np.ndarray,
    mink_log_N: np.ndarray,
    mink_dim: float,
    prob_results: Dict,
    save_path: str = None
):
    """
    Composite figure with 4 panels for the report:
        (a) Original data (3D)
        (b) Latent space (2D)
        (c) Minkowski dimension plot
        (d) Probabilistic dimension plot (KS statistic)
    """
    fig = plt.figure(figsize=(14, 10))

    # Panel a: 3D data
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                c=color, cmap="Spectral", s=3, alpha=0.6)
    ax1.set_title("(a) Swiss Roll in $\\mathbb{R}^3$")

    # Panel b: latent space
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(z_latent[:, 0], z_latent[:, 1], c=color, cmap="Spectral", s=3, alpha=0.6)
    ax2.set_title("(b) DAE Latent Space")
    ax2.set_xlabel("$z_1$")
    ax2.set_ylabel("$z_2$")

    # Panel c: Minkowski
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(mink_neg_log_r, mink_log_N, c="green", s=20)
    coeffs = np.polyfit(mink_neg_log_r, mink_log_N, deg=1)
    ax3.plot(mink_neg_log_r, np.polyval(coeffs, mink_neg_log_r), "r--",
             label=f"slope = {mink_dim:.2f}")
    ax3.set_xlabel("$-\\log r$")
    ax3.set_ylabel("$\\log N(r)$")
    ax3.set_title("(c) Minkowski Dimension")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel d: Probabilistic KS statistic
    ax4 = fig.add_subplot(2, 2, 4)
    candidates = prob_results["candidates"]
    ks = prob_results["ks_vals"]
    ax4.plot(candidates, ks, "o-", color="darkgreen", markersize=6)
    best_idx = int(np.argmin(ks))
    ax4.axvline(x=candidates[best_idx], color="red", linestyle="--", alpha=0.5,
                label=f"Best: n={candidates[best_idx]}")
    ax4.set_xlabel("Candidate dimension $n$")
    ax4.set_ylabel("K-S statistic")
    ax4.set_title("(d) Probabilistic Dimension Estimation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
