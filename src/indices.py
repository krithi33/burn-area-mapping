"""
indices.py
==========
Local (rasterio-based) spectral index computation.
Used after GeoTIFFs are downloaded from Google Drive.

Band order in exported GeoTIFF (analysis_stack_dixie2021.tif):
  1  pre_NBR    9  post_NBR
  2  pre_NDVI   10 post_NDVI
  3  pre_NDWI   11 post_NDWI
  4  pre_BAI    12 post_BAI
  5  pre_B4     13 post_B4
  6  pre_B8     14 post_B8
  7  pre_B11    15 post_B11
  8  pre_B12    16 post_B12
  17 dNBR
  18 RdNBR
"""

import numpy as np
import rasterio
from rasterio.plot import show
from pathlib import Path


# ── Band index map (1-based → 0-based for numpy) ──────────────────────────

BAND_MAP = {
    "pre_NBR":   0,  "post_NBR":   8,
    "pre_NDVI":  1,  "post_NDVI":  9,
    "pre_NDWI":  2,  "post_NDWI":  10,
    "pre_BAI":   3,  "post_BAI":   11,
    "pre_B4":    4,  "post_B4":    12,
    "pre_B8":    5,  "post_B8":    13,
    "pre_B11":   6,  "post_B11":   14,
    "pre_B12":   7,  "post_B12":   15,
    "dNBR":      16,
    "RdNBR":     17,
}

# USGS burn severity thresholds for dNBR
# Source: Key & Benson (2006), FIREMON Landscape Assessment
DNBR_THRESHOLDS = {
    "Enhanced Regrowth":      (-np.inf, -0.25),
    "Unburned":               (-0.25,    0.10),
    "Low Severity":           ( 0.10,    0.27),
    "Moderate-Low Severity":  ( 0.27,    0.44),
    "Moderate-High Severity": ( 0.44,    0.66),
    "High Severity":          ( 0.66,    np.inf),
}

SEVERITY_COLORS = {
    "Enhanced Regrowth":      "#1a9641",
    "Unburned":               "#a6d96a",
    "Low Severity":           "#ffffbf",
    "Moderate-Low Severity":  "#fdae61",
    "Moderate-High Severity": "#f46d43",
    "High Severity":          "#d73027",
}

# For binary classification
BURN_THRESHOLD = 0.27   # dNBR ≥ 0.27 → burned (moderate+ severity)


# ── I/O Helpers ────────────────────────────────────────────────────────────

def load_stack(tif_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load analysis stack GeoTIFF.
    Returns:
        data   : ndarray shape (n_bands, height, width)
        profile: rasterio profile dict (CRS, transform, nodata, etc.)
    """
    with rasterio.open(tif_path) as src:
        data    = src.read().astype(np.float32)
        profile = src.profile
    print(f"Loaded: {Path(tif_path).name}  shape={data.shape}  CRS={profile['crs']}")
    return data, profile


def get_band(data: np.ndarray, name: str) -> np.ndarray:
    """Extract a named band from the stack array."""
    return data[BAND_MAP[name]]


def save_raster(
    array: np.ndarray,
    ref_profile: dict,
    out_path: str | Path,
    nodata: float = -9999.0
) -> None:
    """Save a single-band float32 array as GeoTIFF, inheriting CRS/transform."""
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=nodata)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
    print(f"Saved: {out_path}")


# ── Severity Classification ────────────────────────────────────────────────

def classify_dnbr(dNBR: np.ndarray) -> np.ndarray:
    """
    Apply USGS thresholds to dNBR array.
    Returns integer array:
        0 = Enhanced Regrowth
        1 = Unburned
        2 = Low Severity
        3 = Moderate-Low
        4 = Moderate-High
        5 = High Severity
    """
    out = np.zeros_like(dNBR, dtype=np.uint8)
    thresholds = list(DNBR_THRESHOLDS.values())
    for cls_id, (lo, hi) in enumerate(thresholds):
        out[(dNBR > lo) & (dNBR <= hi)] = cls_id
    return out


def classify_binary(dNBR: np.ndarray, threshold: float = BURN_THRESHOLD) -> np.ndarray:
    """Binary: 1 = burned (dNBR ≥ threshold), 0 = unburned."""
    return (dNBR >= threshold).astype(np.uint8)


# ── Feature Matrix for ML ──────────────────────────────────────────────────

def build_feature_matrix(
    data: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Flatten the stack into a 2D feature matrix (n_pixels, n_features).

    Features used:
      Pre-fire:  NBR, NDVI, NDWI, BAI, B4, B8, B12
      Post-fire: NBR, NDVI, NDWI, BAI, B4, B8, B12
      Change:    dNBR, RdNBR

    Includes both pre AND post features because the RF needs to distinguish
    *why* post-fire values are low (fire vs naturally sparse vegetation).
    The pre-fire context is the key signal that separates them.
    """
    feature_names = [
        # Raw spectral bands only — no derived indices that encode the label
        "pre_NBR",  "pre_NDVI", "pre_NDWI", "pre_BAI",
        "pre_B4",   "pre_B8",   "pre_B11",  "pre_B12",
        "post_NBR", "post_NDVI","post_NDWI","post_BAI",
        "post_B4",  "post_B8",  "post_B11", "post_B12",
        # NOTE: dNBR and RdNBR intentionally excluded —
        # labels are derived from dNBR so including it is data leakage.
        # The RF must learn from raw bands, not the label-generating rule.
    ]

    bands = np.stack([get_band(data, f) for f in feature_names], axis=0)
    # Shape: (n_features, H, W) → flatten to (H*W, n_features)
    n_features, H, W = bands.shape
    X = bands.reshape(n_features, -1).T   # (H*W, n_features)

    if mask is not None:
        flat_mask = mask.reshape(-1).astype(bool)
        X = X[flat_mask]

    # Replace NaN/inf from masked/edge pixels
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"Feature matrix: {X.shape}  features: {feature_names}")
    return X, feature_names


# ── Summary Statistics ─────────────────────────────────────────────────────

def burn_severity_summary(severity_map: np.ndarray) -> dict:
    """
    Compute area statistics per severity class.
    Assumes 20m pixels → each pixel = 0.04 ha = 400 m²
    """
    pixel_area_ha = (EXPORT_SCALE := 20) ** 2 / 10_000   # noqa: F841
    pixel_area_ha = 400 / 10_000  # 0.04 ha

    classes = list(DNBR_THRESHOLDS.keys())
    total_pixels = severity_map.size
    summary = {}
    for cls_id, cls_name in enumerate(classes):
        n = np.sum(severity_map == cls_id)
        summary[cls_name] = {
            "pixels": int(n),
            "area_ha": round(n * pixel_area_ha, 1),
            "pct": round(100 * n / total_pixels, 2)
        }
    return summary
