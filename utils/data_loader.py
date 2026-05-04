import os
import glob
import pandas as pd
import numpy as np
from utils.feature_extractor import extract_features


def load_dataset(data_dir):
    """
    Loads all pressure CSVs, extracts features, returns a labelled DataFrame.
    Expected filename pattern: subjectX_label_trialY_pressure.csv
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    raw_data = []

    for file_path in all_csvs:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 4 and parts[-1].endswith('.csv'):
            label = parts[1]
            try:
                matrix = np.loadtxt(file_path, delimiter=',')
                if matrix.ndim == 2:
                    features = extract_features(matrix)
                    features['label'] = label
                    raw_data.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not raw_data:
        print("No valid CSV data found.")
        return pd.DataFrame()
    return pd.DataFrame(raw_data)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalisation to [0, 1]."""
    mn, mx = matrix.min(), matrix.max()
    if mx - mn < 1e-8:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((matrix - mn) / (mx - mn)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rich offline augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augment(matrix: np.ndarray):
    """
    Returns 3 augmented variants (fast mode — good accuracy/speed balance):
      • Horizontal flip   — simulates left/right foot swap
      • Gaussian noise    — σ=0.02
      • Brightness scale  — random ±15% intensity variation
    """
    variants = []

    # H-flip
    variants.append(np.fliplr(matrix).copy())

    # Noise σ=0.02
    noisy = np.clip(matrix + np.random.normal(0, 0.02, matrix.shape), 0, 1)
    variants.append(noisy.astype(np.float32))

    # Brightness scale
    scale  = 1.0 + (np.random.rand() - 0.5) * 0.30
    bright = np.clip(matrix * scale, 0, 1).astype(np.float32)
    variants.append(bright)

    return variants



# ─────────────────────────────────────────────────────────────────────────────
# Deep-learning dataset loader
# ─────────────────────────────────────────────────────────────────────────────
def load_dl_dataset(data_dir, augment=False):
    """
    Loads the pressure-plate dataset for deep learning training.

    Each 128×48 matrix is:
      1. Loaded from CSV
      2. Per-sample min-max normalised to [0, 1]
      3. Optionally augmented (7 extra variants per sample when augment=True)
      4. Stacked into a (N, 1, 128, 48) tensor

    Args:
        data_dir (str) : Directory containing CSV files.
        augment  (bool): If True, offline augmented copies are added.

    Returns:
        Tuple[Tensor, Tensor, dict]: (X_tensor, y_tensor, label_map)
    """
    import torch

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    label_map = {
        'normal':        0,
        'antalgic':      1,
        'lurching':      2,
        'steppage':      3,
        'stiff-legged':  4,
        'trendelenburg': 5,
    }

    all_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    X_list, y_list = [], []

    for file_path in all_csvs:
        filename  = os.path.basename(file_path)
        parts     = filename.split('_')
        if len(parts) < 4 or not parts[-1].endswith('.csv'):
            continue

        label_name = parts[1]
        if label_name not in label_map:
            continue

        try:
            matrix = np.loadtxt(file_path, delimiter=',')
        except Exception:
            continue

        if matrix.shape != (128, 48):
            continue

        matrix    = _normalize(matrix)
        label_idx = label_map[label_name]

        # Original
        X_list.append(torch.tensor(matrix).unsqueeze(0))
        y_list.append(label_idx)

        if augment:
            for aug in _augment(matrix):
                X_list.append(torch.tensor(aug).unsqueeze(0))
                y_list.append(label_idx)

    if not X_list:
        print("No DL data found.")
        return None, None, label_map

    X_tensor = torch.stack(X_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return X_tensor, y_tensor, label_map
