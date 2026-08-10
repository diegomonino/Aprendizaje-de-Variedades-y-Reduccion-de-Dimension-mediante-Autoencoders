"""
datasets.py
-----------
Unified dataset generation and loading interface.

Datasets:
    - SwissRoll:   2D manifold embedded in R^3 (generated locally)
    - LinearH:     d-dimensional manifold linearly embedded in R^D (generated locally)
    - MNIST:       784-dimensional image dataset (auto-downloaded via torchvision)

Usage:
    dataset = get_dataset("swiss_roll", n_samples=2000, noise=0.0)
    dataset = get_dataset("linear", n_samples=10000, intrinsic_dim=3, ambient_dim=30)
    dataset = get_dataset("mnist", n_samples=10000)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_swiss_roll
import torchvision
import torchvision.transforms as transforms
import os


class NumpyDataset(Dataset):
    """
    Wraps a numpy array into a PyTorch Dataset.
    Stores optional labels for visualization (coloring).
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels  # kept as numpy for matplotlib colormaps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_swiss_roll_dataset(
    n_samples: int = 2000,
    noise: float = 0.0,
    random_state: int = 42
) -> NumpyDataset:
    """
    2D manifold embedded in R^3.
    Intrinsic dimension: 2, Ambient dimension: 3.
    """
    X, color = make_swiss_roll(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return NumpyDataset(data=X, labels=color)


def make_linear_manifold_dataset(
    n_samples: int = 10000,
    intrinsic_dim: int = 3,
    ambient_dim: int = 30,
    random_state: int = 42
) -> NumpyDataset:
    """
    Uniform sampling from [0,1]^d, linearly embedded into R^D via
    a random orthonormal matrix. Replicates H_{d,D} from Ivanov et al. (2021).
    """
    assert ambient_dim >= intrinsic_dim, (
        f"ambient_dim ({ambient_dim}) must be >= intrinsic_dim ({intrinsic_dim})"
    )
    rng = np.random.RandomState(random_state)
    Z = rng.uniform(0, 1, size=(n_samples, intrinsic_dim))

    A_raw = rng.randn(ambient_dim, intrinsic_dim)
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :intrinsic_dim]

    X = Z @ A.T
    return NumpyDataset(data=X, labels=Z[:, 0])


def make_mnist_dataset(
    n_samples: int = 10000,
    data_dir: str = "./data/raw",
    train: bool = True,
    flatten: bool = True
) -> NumpyDataset:
    """
    MNIST handwritten digits (auto-downloaded).
    Intrinsic dimension: ~10-12 (estimated). Ambient dimension: 784.
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist = torchvision.datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )

    n_samples = min(n_samples, len(mnist))
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(mnist), generator=generator)[:n_samples]
    subset = torch.utils.data.Subset(mnist, indices)
    loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
    X, y = next(iter(loader))

    if flatten:
        X = X.view(n_samples, -1).numpy()
    else:
        X = X.numpy()

    return NumpyDataset(data=X, labels=y.numpy())


def make_mnist_digit_dataset(
    digit: int = 3,
    n_samples: int = 5000,
    data_dir: str = "./data/raw",
    train: bool = True,
) -> NumpyDataset:
    """
    Single-digit MNIST subset (auto-downloaded).

    Loads all images of a specific digit from the MNIST training (or test)
    split, normalises pixels to [0, 1] by dividing by 255, and then applies
    standard z-score normalisation (mean 0, std 1) per pixel dimension so
    that the data lives on a comparable scale to the other datasets.

    Preprocessing steps:
        1. Select images whose label == digit
        2. Flatten 28×28 → 784-dimensional vector
        3. Scale to [0, 1]:  x / 255.0
           (ToTensor already does this, so we rely on that)
        4. Per-feature z-score:  (x - mean) / (std + eps)
           Applied across samples for each of the 784 pixels.
           This removes the trivial "always-black" background pixels
           and puts the variation we care about on an equal footing.

    The labels stored are NOT digit class labels (all the same digit) but
    the pixel-intensity of the centre-of-mass column — a proxy for
    left-right writing style — useful for coloring scatter plots.

    Args:
        digit:     Which digit to load (0–9). Default: 3.
        n_samples: Maximum number of samples to return. If fewer images of
                   that digit exist, all available images are returned.
        data_dir:  Where to cache the MNIST download.
        train:     If True use the training split (60 000 images),
                   else use the test split (10 000 images).

    Returns:
        NumpyDataset with:
            data   — float32 array of shape (N, 784), z-score normalised
            labels — float32 array of shape (N,) for coloring
    """
    os.makedirs(data_dir, exist_ok=True)

    # Load raw pixels in [0, 1] via ToTensor (no per-channel normalisation
    # yet — we apply our own normalisation below).
    transform = transforms.ToTensor()   # scales uint8 → float32 in [0,1]

    mnist = torchvision.datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )

    # Collect all images for the requested digit
    all_x, all_y = [], []
    loader_full = DataLoader(mnist, batch_size=1024, shuffle=False)
    for xb, yb in loader_full:
        mask = (yb == digit)
        if mask.any():
            all_x.append(xb[mask])
            all_y.append(yb[mask])

    X = torch.cat(all_x, dim=0)   # shape: (N_digit, 1, 28, 28)
    y = torch.cat(all_y, dim=0)   # shape: (N_digit,)

    if len(X) == 0:
        raise ValueError(f"No images found for digit={digit}.")

    # Flatten: (N, 1, 28, 28) → (N, 784)
    X = X.view(len(X), -1)        # float32, values in [0, 1]

    # Subsample if needed (random, reproducible)
    n_samples = min(n_samples, len(X))
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(X), generator=rng)[:n_samples]
    X = X[indices]

    # Per-feature normalisation across samples.
    #
    # We only z-score features with sufficient variance (std >= min_std).
    # Near-constant "background" pixels (std ≈ 0) are simply mean-centred.
    # Using eps = 1e-8 on such pixels would divide by a near-zero number,
    # creating extreme outliers (values > 50) that corrupt geometry-based
    # analyses (Minkowski dimension, topology metrics) and can destabilise
    # training with large-gradient updates.
    X_np   = X.numpy()
    mean   = X_np.mean(axis=0)          # shape: (784,)
    std    = X_np.std(axis=0)           # shape: (784,)
    min_std = 0.05                       # threshold in [0, 1] pixel space
    active  = std >= min_std             # True for ~400 informative pixels
    X_np    = X_np - mean                # mean-centre ALL features
    X_np[:, active] /= std[active]       # z-score only the active features
    # Active pixels now have std ≈ 1 and values in roughly [-3, 3].
    # Background pixels have values in roughly [-0.05, 0.05] (near zero).

    # Use centre-column mean intensity as a continuous colour proxy
    # (captures left–right style variation across handwriting samples).
    col_center = X_np[:, 14].copy()   # pixel column 14 (centre of 28-wide img)

    return NumpyDataset(data=X_np.astype(np.float32), labels=col_center)


def get_dataset(name: str, **kwargs) -> NumpyDataset:
    """Unified entry point. name: 'swiss_roll', 'linear', 'mnist', or 'mnist_digit'."""
    registry = {
        "swiss_roll":   make_swiss_roll_dataset,
        "linear":       make_linear_manifold_dataset,
        "mnist":        make_mnist_dataset,
        "mnist_digit":  make_mnist_digit_dataset,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(registry.keys())}")
    return registry[name](**kwargs)


def get_dataloader(name: str, batch_size: int = 256, shuffle: bool = True, **kwargs) -> DataLoader:
    """Convenience wrapper that returns a DataLoader directly."""
    dataset = get_dataset(name, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Unified metadata interface
# ─────────────────────────────────────────────────────────────────────────────
# The "expected intrinsic dimension" is the dimension the manifold hypothesis
# predicts for each dataset. It is KNOWN exactly for the synthetic datasets
# (by construction) and only APPROXIMATE for the real ones (from the
# literature / our own Step-3 estimates). It is used as a reference against
# which the Step-3 estimators and the Step-7 Jacobian estimate are compared;
# the value actually used as the model latent dimension d is the one fixed in
# Step 3, not this reference.
_EXPECTED_INTRINSIC_DIM = {
    "swiss_roll":  2,      # 2D sheet curved into R^3 (exact)
    "linear":      None,   # equals intrinsic_dim kwarg (filled in below)
    "mnist":       12,     # approximate, full 10-digit MNIST (literature ~12-14)
    "mnist_digit": None,   # single digit, lower; left unspecified
    "coil20":      None,   # ~1 per object (pure rotation); filled by its loader
}


def get_dataset_with_meta(name: str, **kwargs):
    """
    Unified loader returning both the dataset and a metadata dict, so every
    dataset is consumed through the same interface downstream.

    Returns:
        (dataset, meta) where meta has keys:
            'name'                   : dataset key
            'ambient_dim'            : D (features live in R^D)
            'n_samples'              : N actually loaded
            'expected_intrinsic_dim' : reference intrinsic dim (int or None)
    """
    dataset = get_dataset(name, **kwargs)

    expected = _EXPECTED_INTRINSIC_DIM.get(name)
    if name == "linear":
        expected = kwargs.get("intrinsic_dim", 3)

    meta = {
        "name": name,
        "ambient_dim": int(dataset.data.shape[1]),
        "n_samples": int(dataset.data.shape[0]),
        "expected_intrinsic_dim": expected,
    }
    return dataset, meta


if __name__ == "__main__":
    print("== Swiss Roll ==")
    sr = get_dataset("swiss_roll", n_samples=2000)
    print(f"  Shape: {sr.data.shape}, Range: [{sr.data.min():.2f}, {sr.data.max():.2f}]")

    print("== Linear H_3,30 ==")
    h = get_dataset("linear", n_samples=10000, intrinsic_dim=3, ambient_dim=30)
    print(f"  Shape: {h.data.shape}, Range: [{h.data.min():.2f}, {h.data.max():.2f}]")

    print("== MNIST ==")
    mn = get_dataset("mnist", n_samples=1000)
    print(f"  Shape: {mn.data.shape}, Labels: {np.unique(mn.labels)}")

    print("== MNIST digit 3 ==")
    d3 = get_dataset("mnist_digit", digit=3, n_samples=5000)
    print(f"  Shape: {d3.data.shape}")
    print(f"  Mean: {d3.data.numpy().mean():.4f} (expected ~0)")
    print(f"  Std:  {d3.data.numpy().std():.4f}  (expected ~1)")

    print("== MNIST digit 7 ==")
    d7 = get_dataset("mnist_digit", digit=7, n_samples=5000)
    print(f"  Shape: {d7.data.shape}")
