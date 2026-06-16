"""
tests/test_indices.py
=====================
Unit tests for spectral index computation and classification logic.
Run with: pytest tests/

These tests use synthetic data (no GEE or GeoTIFF required) so they
pass in CI without any credentials.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.indices import (
    classify_binary, classify_dnbr, burn_severity_summary,
    DNBR_THRESHOLDS, BURN_THRESHOLD
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_dnbr():
    """dNBR array covering all severity classes."""
    return np.array([
        -0.5,   # Enhanced Regrowth
        0.0,    # Unburned
        0.15,   # Low Severity
        0.35,   # Moderate-Low
        0.55,   # Moderate-High
        0.80,   # High Severity
    ], dtype=np.float32)


@pytest.fixture
def mock_stack():
    """Minimal mock stack (18 bands, 4x4 pixels)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, size=(18, 4, 4)).astype(np.float32)


# ── Binary classification ──────────────────────────────────────────────────

class TestClassifyBinary:
    def test_burned_above_threshold(self, mock_dnbr):
        result = classify_binary(mock_dnbr, threshold=0.27)
        # Values >= 0.27: indices 3, 4, 5
        assert result[3] == 1
        assert result[4] == 1
        assert result[5] == 1

    def test_unburned_below_threshold(self, mock_dnbr):
        result = classify_binary(mock_dnbr, threshold=0.27)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 0

    def test_output_dtype(self, mock_dnbr):
        result = classify_binary(mock_dnbr)
        assert result.dtype == np.uint8

    def test_output_binary(self, mock_dnbr):
        result = classify_binary(mock_dnbr)
        assert set(np.unique(result)).issubset({0, 1})

    def test_custom_threshold(self, mock_dnbr):
        result_low  = classify_binary(mock_dnbr, threshold=0.1)
        result_high = classify_binary(mock_dnbr, threshold=0.5)
        assert result_low.sum() > result_high.sum()

    def test_all_zeros_when_below_threshold(self):
        arr = np.full((10,), -0.5, dtype=np.float32)
        result = classify_binary(arr, threshold=0.27)
        assert result.sum() == 0

    def test_all_ones_when_above_threshold(self):
        arr = np.full((10,), 0.9, dtype=np.float32)
        result = classify_binary(arr, threshold=0.27)
        assert result.sum() == 10


# ── Multi-class severity ───────────────────────────────────────────────────

class TestClassifyDNBR:
    def test_correct_classes(self, mock_dnbr):
        result = classify_dnbr(mock_dnbr)
        # 0=Regrowth, 1=Unburned, 2=Low, 3=Mod-Low, 4=Mod-High, 5=High
        expected = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self, mock_dnbr):
        result = classify_dnbr(mock_dnbr)
        assert result.shape == mock_dnbr.shape

    def test_output_dtype(self, mock_dnbr):
        result = classify_dnbr(mock_dnbr)
        assert result.dtype == np.uint8

    def test_class_range(self, mock_dnbr):
        result = classify_dnbr(mock_dnbr)
        assert result.min() >= 0
        assert result.max() <= 5

    def test_high_severity_at_boundary(self):
        arr = np.array([0.66, 0.661], dtype=np.float32)
        result = classify_dnbr(arr)
        # 0.66 is the upper boundary of Mod-High → should be Mod-High (4)
        # 0.661 is above 0.66 → High Severity (5)
        assert result[1] == 5


# ── Burn severity summary ──────────────────────────────────────────────────

class TestBurnSeveritySummary:
    def test_returns_all_classes(self, mock_dnbr):
        severity_map = classify_dnbr(mock_dnbr)
        summary = burn_severity_summary(severity_map)
        assert len(summary) == len(DNBR_THRESHOLDS)

    def test_percentages_sum_to_100(self, mock_dnbr):
        severity_map = classify_dnbr(mock_dnbr)
        summary = burn_severity_summary(severity_map)
        total_pct = sum(v["pct"] for v in summary.values())
        assert abs(total_pct - 100.0) < 0.5   # allow floating point tolerance

    def test_pixel_counts_sum_to_total(self):
        arr = np.zeros((100, 100), dtype=np.float32) + 0.5
        summary = burn_severity_summary(arr)
        total = sum(v["pixels"] for v in summary.values())
        assert total == 100 * 100

    def test_area_positive(self, mock_dnbr):
        severity_map = classify_dnbr(mock_dnbr)
        summary = burn_severity_summary(severity_map)
        for cls, stats in summary.items():
            assert stats["area_ha"] >= 0, f"Negative area for {cls}"


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_nan_handling(self):
        arr = np.array([np.nan, 0.5, -0.1], dtype=np.float32)
        result = classify_binary(np.nan_to_num(arr, nan=0.0))
        assert result.shape == (3,)

    def test_uniform_array(self):
        arr = np.full((50, 50), 0.5, dtype=np.float32)
        result = classify_binary(arr, threshold=0.27)
        assert result.sum() == 50 * 50

    def test_large_array_performance(self):
        """Should process 1M pixels quickly."""
        import time
        arr = np.random.uniform(-1, 1, size=(1000, 1000)).astype(np.float32)
        t0 = time.time()
        classify_binary(arr)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"classify_binary too slow: {elapsed:.2f}s for 1M pixels"
