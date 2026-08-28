"""
training.py
-----------
Reusable training utilities shared by the main experiment notebook.

Centralises three things that were previously duplicated across the
experiment scripts:

    - set_seed:        deterministic seeding of python/numpy/torch
    - train_model:     generic training loop for AE / DAE / VAE
    - get_latent:      extract latent codes for a full dataset
    - reconstruct:     decode a full dataset back to data space

The loop is model-agnostic: it relies only on the shared contracts of
the repo (models return a dict with 'x_rec'/'z'; loss functions take
(x_clean, model_output) and return a dict with key 'loss'). The optional
`noise_fn` corrupts the *input* only, so the loss is always measured
against the clean target x (the denoising objective).
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    """
    Seed python, numpy and torch RNGs for reproducibility.

    Every stochastic step in the pipeline (weight init, mini-batch
    shuffling, noise corruption, latent sampling) draws from these
    generators, so a single call fixes the whole run. The functions in
    this project all take an explicit `seed` argument so the entire
    experiment can be re-run under several seeds later on.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model,
    dataset,
    loss_fn,
    noise_fn=None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    log_every: int = 50,
    progress_bar: bool = True,
    desc: str = None,
    return_components: bool = False,
):
    """
    Generic training loop for any model in this repo (AE, DAE, VAE).

    Args:
        model:      Module returning a dict with key 'x_rec' (and 'z', ...).
        dataset:    NumpyDataset (yields clean input tensors).
        loss_fn:    Callable (x_clean, model_output) -> dict with 'loss'.
        noise_fn:   Optional callable x -> x_corrupted (DAE). If None the
                    input equals the target (plain / variational AE).
        epochs:     Number of passes over the data.
        lr:         Adam learning rate.
        batch_size: Mini-batch size.
        device:     'cpu' or 'cuda'.
        seed:       Seed for weight-independent stochasticity (shuffling,
                    noise, reparameterisation). Set here so the loop is
                    reproducible on its own.
        verbose:    Print progress (used only when progress_bar=False, e.g.
                    non-interactive/script runs).
        log_every:  Epoch interval between progress prints (verbose mode).
        progress_bar: Show a live tqdm bar with the running loss instead of
                    the periodic prints. Default on -- this is the mode used
                    throughout the notebook.
        desc:       Label shown on the progress bar (e.g. "swiss_roll · DAE").
        return_components: Also return the per-epoch average of every other
                    scalar in the loss dict ('mse', 'recon', 'kl', ...).
                    Needed for the VAE: its total loss is
                    alpha * E[||x-g(z)||^2] + beta * KL, whose scale is not
                    comparable with the plain per-element MSE the other
                    models minimise, so the loss curves have to be drawn
                    from the reconstruction term alone (divided by D).

    Returns:
        List of per-epoch average losses, or the tuple (losses, history)
        when return_components=True, with history a dict
        {component_name: [per-epoch average]}.
    """
    set_seed(seed)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    history = defaultdict(list)
    epoch_iter = range(1, epochs + 1)
    pbar = tqdm(epoch_iter, desc=desc or "training", leave=False) if progress_bar else epoch_iter
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        comp_sums = defaultdict(float)
        n_batches = 0
        for batch in loader:
            x_clean = batch.to(device)
            x_input = noise_fn(x_clean) if noise_fn is not None else x_clean

            output = model(x_input)
            loss_dict = loss_fn(x_clean, output)

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

            epoch_loss += loss_dict["loss"].item()
            for key, value in loss_dict.items():
                if key != "loss" and torch.is_tensor(value) and value.ndim == 0:
                    comp_sums[key] += value.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        losses.append(avg)
        for key, total in comp_sums.items():
            history[key].append(total / n_batches)
        if progress_bar:
            pbar.set_postfix(loss=f"{avg:.4f}")
        elif verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"    Epoch {epoch:4d}/{epochs} | Loss: {avg:.6f}")

    if return_components:
        return losses, dict(history)
    return losses


@torch.no_grad()
def get_latent(model, dataset, device: str = "cpu") -> np.ndarray:
    """Encode the full dataset to latent codes (uses the deterministic
    encode path — the mean for a VAE, no sampling)."""
    model.eval()
    model.to(device)
    z = model.encode(dataset.data.to(device))
    return z.cpu().numpy()


@torch.no_grad()
def reconstruct(model, dataset, device: str = "cpu") -> np.ndarray:
    """Encode-then-decode the full dataset, returning reconstructions."""
    model.eval()
    model.to(device)
    out = model(dataset.data.to(device))
    return out["x_rec"].cpu().numpy()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.datasets import get_dataset
    from data.noise import get_noise_fn
    from models.autoencoder import build_dae
    from models.losses import dae_loss

    ds = get_dataset("swiss_roll", n_samples=1000)
    model = build_dae(input_dim=3, latent_dim=2)
    losses = train_model(model, ds, dae_loss,
                         noise_fn=get_noise_fn("gaussian", sigma=0.1),
                         epochs=30, log_every=10)
    print(f"First loss: {losses[0]:.4f}  Last loss: {losses[-1]:.4f}")
    z = get_latent(model, ds)
    print(f"Latent shape: {z.shape}")
