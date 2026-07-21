# Manifold Learning via Denoising Autoencoders

**DLAI 2025/2026 — Sapienza University of Rome**

Representation learning project investigating the manifold hypothesis through Denoising Autoencoders (DAE) and Variational Autoencoders (VAE), with quantitative evaluation via intrinsic dimension estimation.

## Project Structure

```
manifold-learning-dae/
├── data/
│   ├── datasets.py              # Swiss Roll, H_{d,D}, MNIST loaders
│   └── noise.py                 # Gaussian, masking, salt & pepper corruption
├── models/
│   ├── autoencoder.py           # Denoising Autoencoder (DAE)
│   ├── vae.py                   # Variational Autoencoder (VAE)
│   └── losses.py                # MSE and ELBO losses
├── dimension_estimation/
│   ├── minkowski.py             # Box-counting Minkowski dimension
│   ├── geometric_prob.py        # Geometrically-probabilistic method (Ivanov et al.)
│   ├── flattening.py            # CDF flattening transformation
│   └── metrics.py               # Shared utilities (kNN, volume, KS test)
├── evaluation/
│   ├── topology.py              # Trustworthiness and continuity metrics
│   ├── latent_analysis.py       # Interpolations, Jacobian, distortion
│   └── visualization.py         # All plotting functions
├── experiments/
│   ├── train.py                 # General training script (CLI)
│   ├── ablation_noise.py        # Experiment 4.2: noise ablation
│   ├── dae_vs_vae.py            # Experiment 4.3: DAE vs VAE comparison
│   └── dimension_pipeline.py    # Experiment 4.4: dimension estimation pipeline
└── results/                     # Auto-generated figures and metrics
```

## Setup

```bash
pip install -r requirements.txt
```

Required packages: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## Running Experiments

All experiments are run from the **project root directory**.

### Train a single model
```bash
# DAE on Swiss Roll
python -m experiments.train --model dae --dataset swiss_roll --noise gaussian --sigma 0.2

# VAE on Swiss Roll
python -m experiments.train --model vae --dataset swiss_roll --beta 1.0

# DAE on linear manifold H_{3,30}
python -m experiments.train --model dae --dataset linear --intrinsic_dim 3 --ambient_dim 30 --latent_dim 3

# DAE on MNIST
python -m experiments.train --model dae --dataset mnist --latent_dim 12 --epochs 50
```

### Noise ablation (Experiment 4.2)
```bash
python -m experiments.ablation_noise
```

### DAE vs VAE comparison (Experiment 4.3)
```bash
python -m experiments.dae_vs_vae
```

### Full dimension estimation pipeline (Experiment 4.4)
```bash
# Swiss Roll
python -m experiments.dimension_pipeline --dataset swiss_roll --n_samples 2000 --n_max 7

# Linear manifold
python -m experiments.dimension_pipeline --dataset linear --n_samples 10000 --intrinsic_dim 3 --ambient_dim 30 --latent_dim 3 --n_max 7
```

## References

- Vincent et al. (2010) — Stacked Denoising Autoencoders
- Kingma & Welling (2014) — Auto-Encoding Variational Bayes
- Fefferman et al. (2016) — Testing the Manifold Hypothesis
- Ivanov et al. (2021) — Manifold Hypothesis in Data Analysis: Double Geometrically-Probabilistic Approach to Manifold Dimension Estimation
