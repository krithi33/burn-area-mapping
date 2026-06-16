"""
pipeline.py
===========
End-to-end burn detection pipeline (post-GEE-export).
Run this after downloading GeoTIFFs from Google Drive.

Usage:
    python -m src.pipeline \
        --stack  data/processed/analysis_stack_dixie2021.tif \
        --samples data/samples/training_samples_dixie2021.csv \
        --out    outputs/

Pipeline stages:
    1. Load raster stack
    2. Threshold classification (dNBR)
    3. Train Random Forest on labeled samples
    4. RF inference on full raster
    5. Method comparison metrics
    6. Export maps + figures
    7. Save interactive Folium map
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .indices import (
    load_stack, get_band, save_raster,
    classify_binary, classify_dnbr,
    build_feature_matrix, burn_severity_summary
)
from .classifier import ThresholdClassifier, BurnRFClassifier, compare_methods
from .visualise import (
    plot_dnbr_distribution, plot_severity_map,
    plot_method_comparison, plot_feature_importance,
    plot_confusion_matrix, build_folium_map
)


def run_pipeline(
    stack_path: str,
    samples_path: str,
    out_dir: str = "outputs/"
) -> dict:

    out = Path(out_dir)
    figures_dir = out / "figures"
    maps_dir    = out / "maps"
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BURN AREA DETECTION PIPELINE — DIXIE FIRE 2021")
    print("=" * 60)

    # ── 1. Load raster ─────────────────────────────────────────────────────
    print("\n[1/7] Loading raster stack ...")
    data, profile = load_stack(stack_path)
    dNBR    = get_band(data, "dNBR")
    rdNBR   = get_band(data, "RdNBR")
    H, W    = data.shape[1], data.shape[2]

    # ── 2. Threshold classification ────────────────────────────────────────
    print("\n[2/7] Threshold classification ...")
    thresh_clf    = ThresholdClassifier(threshold=0.27)
    threshold_map = thresh_clf.predict(data)
    severity_map  = classify_dnbr(dNBR)

    severity_summary = burn_severity_summary(severity_map)
    print("\n  Burn Severity Summary (threshold method):")
    for cls, stats in severity_summary.items():
        print(f"    {cls:<30} {stats['area_ha']:>10,.1f} ha  ({stats['pct']:.1f}%)")

    save_raster(threshold_map.astype(np.float32), profile,
                out / "threshold_burn_mask.tif")

    # ── 3. Load training samples + train RF ───────────────────────────────
    print("\n[3/7] Loading training samples ...")
    samples_df = pd.read_csv(samples_path)
    print(f"  Samples: {len(samples_df):,} rows, "
          f"burned={samples_df['burned_label'].sum():,}, "
          f"unburned={(samples_df['burned_label'] == 0).sum():,}")

    # Exclude label, GEE metadata, coordinates, AND derived change indices
    # dNBR/RdNBR excluded because labels are dNBR-derived → data leakage
    exclude_cols = {"burned_label", "system:index", ".geo", "longitude", "latitude",
                    "dNBR", "RdNBR"}
    feature_cols = [c for c in samples_df.columns if c not in exclude_cols]
    X = samples_df[feature_cols].values
    y = samples_df["burned_label"].values

    print("\n[4/7] Training Random Forest ...")
    rf_clf = BurnRFClassifier(n_estimators=200, max_depth=20)
    train_metrics = rf_clf.fit(X, y, feature_names=feature_cols)

    print("\n  Cross-validation (5-fold) ...")
    cv_metrics = rf_clf.cross_validate(X, y, n_splits=5)

    # ── 5. RF inference on full raster ────────────────────────────────────
    print("\n[5/7] RF inference on full raster ...")
    X_full, feat_names = build_feature_matrix(data)
    y_proba   = rf_clf.predict_proba(X_full)
    proba_map = y_proba.reshape(H, W).astype(np.float32)

    # Calibrate decision threshold to match threshold classifier's burn fraction.
    # Default 0.5 fails here because training was 50/50 balanced but the real
    # raster is ~17% burned — so uncertain pixels score 0.3-0.4 and get
    # clipped to unburned at 0.5. We find the probability percentile that
    # reproduces the same burned pixel count as the threshold method, giving
    # a fair like-for-like comparison.
    thresh_burn_frac = threshold_map.mean()   # e.g. 0.173
    rf_threshold = float(np.percentile(y_proba, (1 - thresh_burn_frac) * 100))
    print(f"  Threshold burn fraction : {thresh_burn_frac:.3f}")
    print(f"  Calibrated RF threshold : {rf_threshold:.4f}  (was 0.5)")

    y_pred    = (y_proba >= rf_threshold).astype(np.uint8)
    rf_map    = y_pred.reshape(H, W)

    save_raster(rf_map.astype(np.float32), profile, out / "rf_burn_mask.tif")
    save_raster(proba_map,                 profile, out / "rf_burn_probability.tif")

    # ── 6. Method comparison ───────────────────────────────────────────────
    print("\n[6/7] Method comparison ...")
    comparison_df = compare_methods(threshold_map, rf_map)

    # ── 7. Figures ─────────────────────────────────────────────────────────
    print("\n[7/7] Generating figures ...")

    plot_dnbr_distribution(dNBR,
        out_path=figures_dir / "dnbr_distribution.png")

    plot_severity_map(dNBR,
        out_path=figures_dir / "burn_severity_map.png")

    plot_method_comparison(threshold_map, rf_map, proba_map,
        out_path=figures_dir / "method_comparison.png")

    importance_df = rf_clf.feature_importance_df()
    plot_feature_importance(importance_df,
        out_path=figures_dir / "feature_importance.png")

    if "confusion_matrix" in train_metrics:
        plot_confusion_matrix(
            train_metrics["confusion_matrix"],
            title="RF Confusion Matrix (Test Set)",
            out_path=figures_dir / "confusion_matrix.png"
        )

    # Interactive Folium map
    build_folium_map(
        tif_path=stack_path,
        threshold_map=threshold_map,
        rf_map=rf_map,
        proba_map=proba_map,
        out_path=maps_dir / "burn_map_interactive.html",
        perimeter_path="data/raw/dixie_fire_perimeter.geojson"
    )

    # ── Save model + metrics ───────────────────────────────────────────────
    rf_clf.save(out / "rf_model.joblib")
    importance_df.to_csv(out / "feature_importance.csv", index=False)
    comparison_df.to_csv(out / "method_comparison.csv",  index=False)

    metrics_out = {
        "train_metrics": {
            k: v for k, v in train_metrics.items()
            if k not in ("report", "confusion_matrix")
        },
        "cv_metrics":        cv_metrics,
        "severity_summary":  severity_summary,
        "rf_decision_threshold": round(rf_threshold, 4),
        "threshold_burn_fraction": round(float(thresh_burn_frac), 4),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {out / 'metrics.json'}")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE                                           ║
╠══════════════════════════════════════════════════════════════╣
║  outputs/                                                    ║
║    threshold_burn_mask.tif    (raster, binary)               ║
║    rf_burn_mask.tif           (raster, binary)               ║
║    rf_burn_probability.tif    (raster, continuous)           ║
║    rf_model.joblib            (trained model)                ║
║    metrics.json               (all evaluation metrics)       ║
║    feature_importance.csv                                    ║
║    method_comparison.csv                                     ║
║    figures/                                                  ║
║      dnbr_distribution.png                                   ║
║      burn_severity_map.png                                   ║
║      method_comparison.png   ← use this in README            ║
║      feature_importance.png                                  ║
║      confusion_matrix.png                                    ║
║    maps/                                                     ║
║      burn_map_interactive.html ← open in browser             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    return metrics_out


def main():
    parser = argparse.ArgumentParser(
        description="Burn Area Detection Pipeline — Dixie Fire 2021"
    )
    parser.add_argument("--stack",   required=True,
                        help="Path to analysis_stack GeoTIFF")
    parser.add_argument("--samples", required=True,
                        help="Path to training samples CSV")
    parser.add_argument("--out",     default="outputs/",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()
    run_pipeline(args.stack, args.samples, args.out)


if __name__ == "__main__":
    main()
