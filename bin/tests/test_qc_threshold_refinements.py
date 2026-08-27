"""Unit tests for the 2026-06-22 QC threshold/metric refinements.

Covers the new code paths:
1. ``read_xenium_major_version`` — XOA version detection for per-version
   intensity floors (inlined helper block in ``image_qc``).
2. ``read_bundle_metrics`` / ``_csv_float`` / ``_csv_str`` — the three
   10x-defined gates read from metrics_summary.csv, with the critical
   absent-vs-zero distinction (``transcript_qc_processing``).
3. ``assess_raw_intensity_quality`` — WARN-only intensity (no FAIL tier).

ADAPTED FROM UPSTREAM (nf-xenium-processing dev HEAD 5e35cae,
``tests/test_qc_threshold_refinements.py``):
- ``xenium_helpers`` is not shipped by this pipeline; its helpers are inlined
  into the scripts, so ``read_xenium_major_version`` is an attribute of the
  ``image_qc`` module (``transcript_qc_processing`` deliberately omits it).
  The ``utils = importlib.import_module("xenium_helpers.utils")`` handle and its
  ``sys.path`` entry are therefore gone, and ``utils.read_xenium_major_version``
  is now ``image_qc.read_xenium_major_version``.
- ``molecule_qc_processing`` is named ``transcript_qc_processing`` here; the
  ``mqc`` alias is kept and repointed.
- Scripts live in the pipeline-level ``bin/`` rather than
  ``modules/local/*/resources/usr/bin/``.
No assertion was changed.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# bin modules have heavy optional top-level imports (napari, snr_metrics) that
# are not needed here. Stub them before importing, matching test_focus_score_compute.
_bin_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_bin_dir))

for _mod in (
    "napari_skimage_regionprops",
    "napari_simpleitk_image_processing",
    "snr_metrics",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["napari_skimage_regionprops"].regionprops_table = lambda *a, **kw: None  # type: ignore[attr-defined]

mqc = importlib.import_module("transcript_qc_processing")
image_qc = importlib.import_module("image_qc")


# ---------------------------------------------------------------------------
# read_xenium_major_version
# ---------------------------------------------------------------------------
def _write_experiment(tmp_path: Path, version) -> Path:
    import json

    d = tmp_path
    payload = {} if version is None else {"analysis_sw_version": version}
    (d / "experiment.xenium").write_text(json.dumps(payload))
    return d


@pytest.mark.parametrize(
    "version, expected",
    [
        ("xenium-3.2.0.7", 3),
        ("xenium-4.0.1.0", 4),
        ("xenium-2.0.0", 2),
        (None, None),  # key absent
        ("garbage", None),  # unparseable
    ],
)
def test_read_xenium_major_version(tmp_path, version, expected):
    d = _write_experiment(tmp_path, version)
    assert image_qc.read_xenium_major_version(d) == expected


def test_read_xenium_major_version_missing_file(tmp_path):
    # No experiment.xenium present
    assert image_qc.read_xenium_major_version(tmp_path) is None


# ---------------------------------------------------------------------------
# _csv_float / _csv_str — absent (None) vs zero (0.0) buckets
# ---------------------------------------------------------------------------
def test_csv_float_distinguishes_absent_from_zero():
    assert mqc._csv_float(0.0) == 0.0  # real zero is kept
    assert mqc._csv_float("0") == 0.0
    assert mqc._csv_float("") is None  # blank -> absent
    assert mqc._csv_float(None) is None
    assert mqc._csv_float(float("nan")) is None  # pandas blank -> NaN -> absent
    assert mqc._csv_float("nan") is None
    assert mqc._csv_float("abc") is None


def test_csv_str_blank_is_none():
    assert (
        mqc._csv_str("xenium_cell_segmentation_stains_v1")
        == "xenium_cell_segmentation_stains_v1"
    )
    assert mqc._csv_str("") is None
    assert mqc._csv_str(None) is None
    assert mqc._csv_str(float("nan")) is None  # blank stain_definition
    assert mqc._csv_str("nan") is None


# ---------------------------------------------------------------------------
# read_bundle_metrics
# ---------------------------------------------------------------------------
def _write_metrics_csv(tmp_path: Path, **cols) -> Path:
    pd.DataFrame([cols]).to_csv(tmp_path / "metrics_summary.csv", index=False)
    return tmp_path


def test_read_bundle_metrics_stain_segmentation_run(tmp_path):
    d = _write_metrics_csv(
        tmp_path,
        nuclear_transcripts_per_100um2=180.5,
        fraction_empty_cells=0.00163,
        segmented_cell_stain_frac=0.987,
        stain_definition="xenium_cell_segmentation_stains_v1",
    )
    out = mqc.read_bundle_metrics(d, is_resegmented=False)
    assert out["nuclear_transcripts_per_100um2"] == pytest.approx(180.5)
    assert out["fraction_empty_cells"] == pytest.approx(0.00163)
    assert out["segmented_cell_stain_frac"] == pytest.approx(0.987)
    assert out["stain_definition"] == "xenium_cell_segmentation_stains_v1"
    assert out["source"] == "metrics_summary.csv"


def test_read_bundle_metrics_nuc_expansion_stain_def_blank(tmp_path):
    # Nucleus-expansion run: stain_frac is a real 0.0, stain_definition blank.
    # The 0.0 must be kept (real state), stain_definition must be None (absent).
    d = _write_metrics_csv(
        tmp_path,
        nuclear_transcripts_per_100um2=35.2,
        fraction_empty_cells=0.0056,
        segmented_cell_stain_frac=0.0,
        stain_definition="",
    )
    out = mqc.read_bundle_metrics(d, is_resegmented=False)
    assert out["segmented_cell_stain_frac"] == 0.0  # real zero, not None
    assert out["stain_definition"] is None  # absent, so gate stays informational


def test_read_bundle_metrics_resegmented_is_na(tmp_path):
    _write_metrics_csv(
        tmp_path,
        nuclear_transcripts_per_100um2=180.5,
        fraction_empty_cells=0.0016,
        segmented_cell_stain_frac=0.99,
        stain_definition="stains_v1",
    )
    out = mqc.read_bundle_metrics(tmp_path, is_resegmented=True)
    assert out["nuclear_transcripts_per_100um2"] is None
    assert out["fraction_empty_cells"] is None
    assert out["segmented_cell_stain_frac"] is None
    assert "resegmented" in out["source"]


def test_read_bundle_metrics_missing_csv(tmp_path):
    out = mqc.read_bundle_metrics(tmp_path, is_resegmented=False)
    assert out["nuclear_transcripts_per_100um2"] is None
    assert out["source"] is None


def test_read_bundle_metrics_missing_column(tmp_path):
    d = _write_metrics_csv(tmp_path, some_other_col=1.0)
    out = mqc.read_bundle_metrics(d, is_resegmented=False)
    assert out["nuclear_transcripts_per_100um2"] is None
    assert out["source"] == "metrics_summary.csv"  # CSV present, column absent


# ---------------------------------------------------------------------------
# assess_raw_intensity_quality — WARN-only (no FAIL tier)
# ---------------------------------------------------------------------------
def test_intensity_quality_never_fails():
    # All tiles far below the critical floor — under the old logic this FAILed.
    df = pd.DataFrame(
        {
            "dapi_intensity": [10.0] * 100,
            "boundary_intensity": [5.0] * 100,
            "intrna_intensity": [5.0] * 100,
            "tissue_coverage": [1.0] * 100,
        }
    )
    stats = image_qc.assess_raw_intensity_quality(
        df,
        dapi_threshold_critical=500,
        boundary_threshold_critical=100,
        intrna_threshold_critical=300,
    )
    for ch in ("dapi", "boundary", "intrna"):
        assert stats[ch]["quality_status"] in ("warn", "pass", "not_available")
        assert stats[ch]["quality_status"] != "fail"
        assert stats[ch]["pct_fail_threshold"] is None
    assert stats["overall_quality"] in ("warn", "pass", "not_available")
    assert stats["overall_quality"] != "fail"


# ---------------------------------------------------------------------------
# compute_whole_grid_stain_percentiles — mask-independent diagnostic (§5.5)
# ---------------------------------------------------------------------------
def test_whole_grid_percentiles_all_channels():
    # 100 tiles 1..100; p95/p99 are computed over the WHOLE grid (no mask).
    df = pd.DataFrame(
        {
            "dapi_intensity": list(range(1, 101)),
            "boundary_intensity": list(range(1, 101)),
            "intrna_intensity": list(range(1, 101)),
            "tissue_coverage": [0.0] * 100,  # no tissue — must be ignored
        }
    )
    out = image_qc.compute_whole_grid_stain_percentiles(df)
    # Mask-independent: zero tissue coverage must NOT zero the percentiles.
    assert out["dapi"]["p99"] == pytest.approx(99.01, abs=0.5)
    assert out["dapi"]["p95"] == pytest.approx(95.05, abs=0.5)
    for ch in ("dapi", "boundary", "intrna"):
        assert out[ch]["p99"] is not None


def test_whole_grid_percentiles_missing_channel():
    df = pd.DataFrame({"dapi_intensity": [100.0, 200.0, 300.0]})
    out = image_qc.compute_whole_grid_stain_percentiles(df)
    assert out["dapi"]["p99"] is not None
    assert out["boundary"] == {"p95": None, "p99": None}
    assert out["intrna"] == {"p95": None, "p99": None}


def test_whole_grid_percentiles_all_nan_channel():
    df = pd.DataFrame(
        {
            "dapi_intensity": [float("nan")] * 5,
            "boundary_intensity": [10.0] * 5,
        }
    )
    out = image_qc.compute_whole_grid_stain_percentiles(df)
    assert out["dapi"] == {"p95": None, "p99": None}  # all-NaN -> None
    assert out["boundary"]["p99"] == pytest.approx(10.0)


def test_whole_grid_percentiles_empty_df():
    out = image_qc.compute_whole_grid_stain_percentiles(pd.DataFrame())
    for ch in ("dapi", "boundary", "intrna"):
        assert out[ch] == {"p95": None, "p99": None}
