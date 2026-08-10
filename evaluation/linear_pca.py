"""
linear_pca.py
-------------
Step 4 utilities: quantify how close the subspace recovered by a trained
linear autoencoder is to the PCA subspace of the same dimension.

Theory (Baldi & Hornik, 1989): a linear AE trained with MSE converges to
a solution whose decoder columns span the principal subspace — the same
subspace as the top-k PCA components. It need not recover the individual
principal *axes* (any rotation within the subspace is an equally good
optimum), so the right comparison is between SUBSPACES, via principal
angles (scipy.linalg.subspace_angles), not a coefficient-by-coefficient
match. Small angles => same subspace.

We also report the reconstruction MSE of both methods as a direct,
subspace-free check that they explain the data equally well.
"""

import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA


def compare_ae_pca(model, X: np.ndarray, latent_dim: int) -> dict:
    """
    Compare a trained linear AE against PCA on the same data.

    Args:
        model:      Trained LinearAE (uses model.decoder.weight and
                    model.encode / model.decode).
        X:          Data array (N, D).
        latent_dim: Subspace dimension k (the fixed d from Step 3).

    Returns:
        Dict with:
            'principal_angles_deg' : k principal angles in degrees
            'max_angle_deg'        : largest principal angle (worst case)
            'ae_mse'               : linear-AE reconstruction MSE
            'pca_mse'              : PCA reconstruction MSE
    """
    N, D = X.shape

    # ── PCA subspace (mean-centred) ─────────────────────────────────────────
    pca = PCA(n_components=latent_dim)
    Z_pca = pca.fit_transform(X)
    X_pca_rec = pca.inverse_transform(Z_pca)
    pca_mse = float(np.mean((X - X_pca_rec) ** 2))
    pca_basis = pca.components_.T                  # (D, k), columns span it

    # ── Linear-AE subspace = column space of the decoder weight ─────────────
    W_dec = model.decoder.weight.detach().cpu().numpy()   # (D, k)
    # Orthonormalise for a well-conditioned angle computation.
    ae_basis, _ = np.linalg.qr(W_dec)

    angles_rad = subspace_angles(ae_basis, pca_basis)
    angles_deg = np.degrees(angles_rad)

    # ── AE reconstruction MSE ───────────────────────────────────────────────
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        X_ae_rec = model(X_t)["x_rec"].cpu().numpy()
    ae_mse = float(np.mean((X - X_ae_rec) ** 2))

    return {
        "principal_angles_deg": angles_deg,
        "max_angle_deg": float(np.max(angles_deg)),
        "ae_mse": ae_mse,
        "pca_mse": pca_mse,
    }
