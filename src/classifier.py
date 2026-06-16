"""
classifier.py
=============
Two burn classification approaches, side-by-side:

  1. Threshold Classifier  — USGS dNBR thresholds (deterministic, no training)
  2. Random Forest         — supervised ML on spectral features

Both produce a burned/unburned binary map. The comparison between them
is the analytical centrepiece of this project.

WHY compare two methods?
  Threshold is fast, interpretable, and the industry standard for rapid
  post-fire assessment (used by CAL FIRE, USFS, USGS BAER teams).
  RF captures non-linear feature interactions and handles mixed pixels
  better in heterogeneous landscapes. Understanding where they disagree
  is operationally valuable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from .indices import (
    classify_binary, build_feature_matrix,
    BAND_MAP, BURN_THRESHOLD
)


# ── 1. Threshold Classifier ────────────────────────────────────────────────

class ThresholdClassifier:
    """
    Rule-based burn detection using dNBR threshold.
    No training required. Fully deterministic and reproducible.

    This is the baseline every remote sensing paper uses. If your RF
    can't beat it on precision/recall, something is wrong with your labels.
    """

    def __init__(self, threshold: float = BURN_THRESHOLD):
        self.threshold = threshold

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        data: full stack array (n_bands, H, W)
        Returns: binary map (H, W), dtype uint8
        """
        dNBR = data[BAND_MAP["dNBR"]]
        return classify_binary(dNBR, self.threshold)

    def predict_flat(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Predict on flat feature matrix (n_pixels, n_features)."""
        dNBR_idx = feature_names.index("dNBR")
        return (X[:, dNBR_idx] >= self.threshold).astype(np.uint8)

    def __repr__(self):
        return f"ThresholdClassifier(threshold={self.threshold})"


# ── 2. Random Forest Classifier ────────────────────────────────────────────

class BurnRFClassifier:
    """
    Random Forest burn classifier with cross-validation and feature importance.

    Design choices explained:
      n_estimators=200  : Enough trees for stable OOB error; diminishing returns after ~300
      max_depth=20      : Prevents overfitting to noisy cloud-edge pixels
      class_weight='balanced': Burned pixels are minority class (~15-30% of AOI)
      n_jobs=-1         : Parallelise across all CPU cores
      oob_score=True    : Free internal validation without touching test set
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.feature_names_: list[str] = []
        self.model = Pipeline([
            ("scaler", StandardScaler()),    # RF doesn't need scaling, but helps
            ("rf", RandomForestClassifier(   # SVM/LogReg comparisons later
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                oob_score=True,
                n_jobs=-1,
                random_state=random_state,
                verbose=0
            ))
        ])

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        test_size: float = 0.2
    ) -> dict:
        """
        Train RF with stratified train/test split.
        Returns dict of evaluation metrics.

        WHY stratified split?
          Ensures burned/unburned class ratio is the same in train and test.
          Without stratification, a lucky split could put all burned pixels
          in train → inflated test accuracy.
        """
        self.feature_names_ = feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )

        print(f"Training RF on {X_train.shape[0]:,} samples ...")
        self.model.fit(X_train, y_train)

        rf = self.model.named_steps["rf"]
        print(f"  OOB score: {rf.oob_score_:.4f}")

        metrics = self._evaluate(X_test, y_test, split="test")
        return metrics

    # ── Cross-Validation ───────────────────────────────────────────────────

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        """
        Stratified k-fold CV for robust performance estimates.
        Reports mean ± std for F1, ROC-AUC, Average Precision.

        WHY CV on top of train/test split?
          With spatial data, a single split can be spatially autocorrelated
          (nearby pixels in train AND test → optimistic estimates).
          CV over multiple folds gives a more honest performance range.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = {}
        for metric in ["f1", "roc_auc", "average_precision"]:
            cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            scores[metric] = {"mean": cv_scores.mean(), "std": cv_scores.std()}
            print(f"  CV {metric}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return scores

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of burned class — useful for risk gradient maps."""
        return self.model.predict_proba(X)[:, 1]

    def predict_map(self, data: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """
        Run inference on full raster stack.
        data: (n_bands, H, W)
        Returns: binary map (H, W)
        """
        X, _ = build_feature_matrix(data)
        H, W = data.shape[1], data.shape[2]
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.uint8).reshape(H, W)

    def predict_proba_map(self, data: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Return continuous burn probability map (H, W), values in [0, 1]."""
        X, _ = build_feature_matrix(data)
        H, W = data.shape[1], data.shape[2]
        return self.predict_proba(X).reshape(H, W).astype(np.float32)

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _evaluate(self, X: np.ndarray, y: np.ndarray, split: str = "test") -> dict:
        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "split":             split,
            "f1":                f1_score(y, y_pred),
            "roc_auc":           roc_auc_score(y, y_proba),
            "avg_precision":     average_precision_score(y, y_proba),
            "confusion_matrix":  confusion_matrix(y, y_pred).tolist(),
            "report":            classification_report(y, y_pred, target_names=["Unburned", "Burned"])
        }
        print(f"\n── {split.upper()} Performance ──")
        print(metrics["report"])
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['avg_precision']:.4f}")
        return metrics

    # ── Feature Importance ─────────────────────────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        """
        Return feature importances as a sorted DataFrame.
        Uses Gini impurity reduction (MDI) from the RF.

        Note: MDI can overestimate importance of high-cardinality features.
        For a more robust estimate, use permutation importance (sklearn.inspection)
        — which we do in the analysis notebook.
        """
        rf = self.model.named_steps["rf"]
        importances = rf.feature_importances_
        df = pd.DataFrame({
            "feature":    self.feature_names_,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        print(f"Model loaded ← {path}")


# ── Method Comparison ──────────────────────────────────────────────────────

def compare_methods(
    threshold_pred: np.ndarray,
    rf_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Agreement analysis between threshold and RF predictions.
    Computes pixel-level agreement and optionally scores against ground truth.
    """
    agreement = (threshold_pred == rf_pred)
    pct_agree = agreement.mean() * 100

    rows = [
        {"method": "Agreement",            "value": f"{pct_agree:.1f}%"},
        {"method": "Threshold burned px",  "value": f"{threshold_pred.sum():,}"},
        {"method": "RF burned px",         "value": f"{rf_pred.sum():,}"},
    ]

    if y_true is not None:
        for name, pred in [("Threshold", threshold_pred), ("RF", rf_pred)]:
            f1  = f1_score(y_true.ravel(), pred.ravel())
            auc = roc_auc_score(y_true.ravel(), pred.ravel())
            rows += [
                {"method": f"{name} F1",      "value": f"{f1:.4f}"},
                {"method": f"{name} ROC-AUC", "value": f"{auc:.4f}"},
            ]

    df = pd.DataFrame(rows)
    print("\n── Method Comparison ──")
    print(df.to_string(index=False))
    return df
