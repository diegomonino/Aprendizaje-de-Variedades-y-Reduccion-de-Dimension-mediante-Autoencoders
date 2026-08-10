"""
config.py
---------
Tiny JSON persistence for the experiment configuration, so the latent
dimension d fixed in Step 3 is written once and read back by every later
step (Steps 4-8) instead of being hard-coded in multiple notebook cells.

Layout of results/experiment_config.json:

    {
      "swiss_roll": {
        "ambient_dim": 3,
        "expected_intrinsic_dim": 2,
        "estimates": {...},          # estimate_all_dimensions output
        "latent_dim": 2              # the d used by all models
      },
      ...
    }
"""

import os
import json

DEFAULT_PATH = os.path.join("results", "experiment_config.json")


def save_experiment_config(config: dict, path: str = DEFAULT_PATH):
    """Write the full experiment config dict to JSON (creates results/)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_experiment_config(path: str = DEFAULT_PATH) -> dict:
    """Read the experiment config back. Raises if it does not exist yet."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latent_dim(dataset_name: str, path: str = DEFAULT_PATH) -> int:
    """Return the fixed latent dimension d for a dataset (from Step 3)."""
    return int(load_experiment_config(path)[dataset_name]["latent_dim"])
