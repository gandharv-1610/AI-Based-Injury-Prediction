import numpy as np


def extract_features(pressure_matrix: np.ndarray) -> dict:
    """
    Extracts rich biomechanical features from a 128×48 plantar pressure matrix.

    Foot layout assumption:
        Rows    0 – 42   → Forefoot (toes & metatarsal heads)
        Rows   43 – 84   → Midfoot  (arch region)
        Rows   85 – 127  → Heel     (calcaneus)
        Cols    0 – 23   → Left foot half
        Cols   24 – 47   → Right foot half

    Returns:
        dict: 16 extracted features ready for ML / display use.
    """
    pressure_matrix = np.asarray(pressure_matrix, dtype=np.float64)
    total_pressure = np.sum(pressure_matrix)

    _ZERO = {
        'total_pressure': 0.0,
        'balance': 0.0,
        'cop_x': 0.0,
        'cop_y': 0.0,
        'max_pressure': 0.0,
        'variance': 0.0,
        'std_dev': 0.0,
        'skewness': 0.0,
        'kurtosis': 0.0,
        'forefoot_ratio': 0.0,
        'midfoot_ratio': 0.0,
        'heel_ratio': 0.0,
        'left_ratio': 0.0,
        'right_ratio': 0.0,
        'peak_pressure_index': 0.0,
        'contact_area': 0.0,
    }

    if total_pressure < 1e-8:
        return _ZERO

    rows, cols = pressure_matrix.shape
    midcol = cols // 2

    # ── 1. Left / Right balance ──────────────────────────────────────────────
    left_sum  = float(np.sum(pressure_matrix[:, :midcol]))
    right_sum = float(np.sum(pressure_matrix[:, midcol:]))
    balance   = abs(left_sum - right_sum) / total_pressure
    left_ratio  = left_sum  / total_pressure
    right_ratio = right_sum / total_pressure

    # ── 2. Centre of Pressure (COP) ──────────────────────────────────────────
    y_idx, x_idx = np.indices(pressure_matrix.shape)
    cop_x = float(np.sum(pressure_matrix * x_idx) / total_pressure)
    cop_y = float(np.sum(pressure_matrix * y_idx) / total_pressure)

    # ── 3. Basic statistics ───────────────────────────────────────────────────
    flat = pressure_matrix.flatten()
    mean_p   = float(np.mean(flat))
    std_dev  = float(np.std(flat))
    variance = float(np.var(flat))

    # Skewness (manual, avoids scipy dependency)
    skewness = float(np.mean(((flat - mean_p) / (std_dev + 1e-8)) ** 3))
    # Kurtosis (excess)
    kurtosis = float(np.mean(((flat - mean_p) / (std_dev + 1e-8)) ** 4) - 3.0)

    # ── 4. Peak pressure ──────────────────────────────────────────────────────
    max_pressure = float(np.max(pressure_matrix))
    # Peak Pressure Index: ratio of peak to mean (non-uniformity measure)
    peak_pressure_index = max_pressure / (mean_p + 1e-8)

    # ── 5. Foot-zone pressure ratios ─────────────────────────────────────────
    # Divide the 128-row matrix into three roughly equal thirds
    zone_h = rows // 3
    forefoot = pressure_matrix[:zone_h, :]           # rows 0 – 42
    midfoot  = pressure_matrix[zone_h: 2*zone_h, :]  # rows 43 – 84
    heel     = pressure_matrix[2*zone_h:, :]          # rows 85 – 127

    forefoot_ratio = float(np.sum(forefoot) / total_pressure)
    midfoot_ratio  = float(np.sum(midfoot)  / total_pressure)
    heel_ratio     = float(np.sum(heel)     / total_pressure)

    # ── 6. Contact area (fraction of non-zero cells) ─────────────────────────
    contact_area = float(np.count_nonzero(pressure_matrix) / pressure_matrix.size)

    return {
        'total_pressure':     float(total_pressure),
        'balance':            balance,
        'cop_x':              cop_x,
        'cop_y':              cop_y,
        'max_pressure':       max_pressure,
        'variance':           variance,
        'std_dev':            std_dev,
        'skewness':           skewness,
        'kurtosis':           kurtosis,
        'forefoot_ratio':     forefoot_ratio,
        'midfoot_ratio':      midfoot_ratio,
        'heel_ratio':         heel_ratio,
        'left_ratio':         left_ratio,
        'right_ratio':        right_ratio,
        'peak_pressure_index': peak_pressure_index,
        'contact_area':       contact_area,
    }
