"""
noise.py
--------
Corruption functions for Denoising Autoencoders.

Three schemes implemented:
    - Gaussian:       additive N(0, sigma^2) noise
    - Masking:        random zeroing of dimensions with probability p
    - Salt & Pepper:  random replacement with min/max values

Each function takes a clean batch tensor and returns a corrupted copy.
The original tensor is never modified in-place.
"""

import torch


def corrupt_gaussian(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """
    Additive Gaussian noise: x_tilde = x + N(0, sigma^2 I).

    Args:
        x:     Clean input tensor of any shape.
        sigma: Standard deviation of the noise.

    Returns:
        Corrupted copy of x (same shape).
    """
    return x + sigma * torch.randn_like(x)


def corrupt_masking(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """
    Masking noise: each element is zeroed independently with probability p.

    Args:
        x: Clean input tensor.
        p: Probability of zeroing each element.

    Returns:
        Corrupted copy of x (same shape).
    """
    mask = torch.bernoulli(torch.full_like(x, 1.0 - p))
    return x * mask


def corrupt_salt_pepper(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """
    Salt-and-pepper noise: each element is independently replaced with
    the min or max value of the batch with probability p.

    Args:
        x: Clean input tensor.
        p: Probability of corruption per element (half salt, half pepper).

    Returns:
        Corrupted copy of x (same shape).
    """
    x_out = x.clone()
    x_min = x.min().item()
    x_max = x.max().item()

    # Salt: set to max
    salt_mask = torch.rand_like(x) < (p / 2)
    x_out[salt_mask] = x_max

    # Pepper: set to min
    pepper_mask = torch.rand_like(x) < (p / 2)
    x_out[pepper_mask] = x_min

    return x_out


def get_noise_fn(noise_type: str, **kwargs):
    """
    Factory function that returns a noise callable.

    Args:
        noise_type: One of 'gaussian', 'masking', 'salt_pepper'.
        kwargs:     Passed to the noise function (sigma, p, etc.).

    Returns:
        A callable: noise_fn(x) -> x_corrupted.

    Example:
        noise_fn = get_noise_fn('gaussian', sigma=0.3)
        x_noisy = noise_fn(x_clean)
    """
    registry = {
        "gaussian": corrupt_gaussian,
        "masking": corrupt_masking,
        "salt_pepper": corrupt_salt_pepper,
    }
    if noise_type not in registry:
        raise ValueError(f"Unknown noise type '{noise_type}'. Available: {list(registry.keys())}")

    fn = registry[noise_type]
    return lambda x: fn(x, **kwargs)


def suggest_dae_sigma(X, k: int = 10, fraction: float = 0.1) -> float:
    """
    Provisional heuristic for the Gaussian corruption level of a DAE.

    Returns a small fraction of the median distance to the k-th nearest
    neighbour of the training data:

        sigma ~= fraction * median_i( dist to k-th NN of x_i )

    Rationale (provisional, see the notebook markdown): the corruption
    should stay small relative to the local inter-point spacing so that
    the noisy sample x_tilde remains within the "tube" around the manifold
    that the denoiser must project back onto. The k-th NN distance is a
    robust proxy for that local scale. This value is a PLACEHOLDER pending
    a formal justification tied to the reach of the manifold, still being
    developed in the methodology chapter of the thesis.

    Args:
        X:        Data array (N, D) — numpy array or tensor.
        k:        Which neighbour to use for the local-scale estimate.
        fraction: Fraction of the median k-th NN distance (e.g. 0.05-0.10).

    Returns:
        Suggested sigma (float).
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    k = min(k, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    median_kth = float(np.median(dists[:, k]))
    return fraction * median_kth


if __name__ == "__main__":
    x = torch.randn(5, 3)
    print("Original:\n", x)
    print("\nGaussian (sigma=0.5):\n", corrupt_gaussian(x, sigma=0.5))
    print("\nMasking (p=0.5):\n", corrupt_masking(x, p=0.5))
    print("\nSalt&Pepper (p=0.3):\n", corrupt_salt_pepper(x, p=0.3))
