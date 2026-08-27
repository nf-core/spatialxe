"""Tests for SNR verdicts that must not read PASS when nothing was measured.

Three defects, all of which produced a green report rather than an error, which is
worse than a crash because nothing about the run looks wrong:

* ``aggregate_snr_verdict`` started at ``"PASS"`` and only moved on an explicit
  FAIL or WARN. Components that could not run return ``{"status": "skipped"}`` with
  no ``verdict`` key, so they matched neither branch and the overall stayed PASS.
* ``compute_roi_snr`` compared an all-NaN median against its thresholds. Both
  ``NaN < warn`` and ``NaN > warn`` are False, so a bundle with no negative-control
  probes reported PASS and wrote a bare ``NaN`` token into the JSON.
* ``compute_image_snr_from_pixel_maps`` took its median over every tile on the
  slide while its sibling ``compute_image_snr_from_roi_df`` filtered to tissue
  first, though both grade against the same ``image_snr_db`` warn/fail pair. An
  empty tile splits its own noise and can score higher than real tissue, so a
  slide with a small tissue footprint was graded on the coverslip.

The image-SNR tests assert the two sibling functions agree on which population
they measure, rather than pinning dB values, so they cannot enshrine a scope error.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import json

import numpy as np
import pandas as pd

# snr_metrics is stubbed by other test modules (they share sys.modules), so a plain
# import can hand back an empty ModuleType depending on collection order. Load the
# real module by path, the same way test_tile_consumers.py does, so this file works
# whatever ran first.
_snr_path = (
    Path(__file__).resolve().parent.parent / "snr_metrics.py"
)
_spec = importlib.util.spec_from_file_location("_real_snr_metrics", _snr_path)
assert _spec is not None and _spec.loader is not None  # narrow for mypy; path is checked above
snr_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snr_metrics)


# ---------------------------------------------------------------------------
# aggregate_snr_verdict
# ---------------------------------------------------------------------------


def test_no_component_reported_a_verdict_is_not_pass():
    """Every component skipped: the module measured nothing, so PASS is a lie."""
    parts = {
        "SNR_image_roi_quartile_db": {"status": "skipped", "reason": "missing column"},
        "SNR_image_otsu": {"status": "error", "error": "unreadable input"},
        "SNR_roi_tx": {"status": "skipped", "reason": "no transcripts file"},
    }
    out = snr_metrics.aggregate_snr_verdict(parts)
    assert out["overall_snr_verdict"] == "NOT_COMPUTED"
    assert out["components_not_computed"] == sorted(parts)


def test_partial_skip_keeps_the_real_verdict_but_names_the_gaps():
    """One component reported, others did not. The verdict stands and says so."""
    parts = {
        "SNR_image_otsu": {"status": "ok", "verdict": "WARN"},
        "SNR_roi_tx": {"status": "skipped", "reason": "no transcripts file"},
    }
    out = snr_metrics.aggregate_snr_verdict(parts)
    assert out["overall_snr_verdict"] == "WARN"
    assert out["components_not_computed"] == ["SNR_roi_tx"]


def test_all_components_pass_reports_pass_with_no_gap_list():
    parts = {
        "SNR_image_otsu": {"status": "ok", "verdict": "PASS"},
        "SNR_roi_tx": {"status": "ok", "verdict": "PASS"},
    }
    out = snr_metrics.aggregate_snr_verdict(parts)
    assert out["overall_snr_verdict"] == "PASS"
    assert "components_not_computed" not in out


def test_fail_still_wins_over_warn():
    """Pre-existing precedence must survive the NOT_COMPUTED change."""
    parts = {
        "a": {"verdict": "WARN"},
        "b": {"verdict": "FAIL"},
        "c": {"status": "skipped"},
    }
    assert snr_metrics.aggregate_snr_verdict(parts)["overall_snr_verdict"] == "FAIL"


def test_non_dict_part_counts_as_not_computed():
    parts = {"a": None, "b": "unexpected"}
    out = snr_metrics.aggregate_snr_verdict(parts)
    assert out["overall_snr_verdict"] == "NOT_COMPUTED"
    assert out["components_not_computed"] == ["a", "b"]


# ---------------------------------------------------------------------------
# _write_snr_json
# ---------------------------------------------------------------------------


def test_json_writer_never_emits_a_bare_nan_token(tmp_path):
    """`json.dump(default=str)` does not catch float NaN; it writes `NaN`."""
    summary = {
        "median_roi_tx_snr_ratio": float("nan"),
        "nested": {"also_bad": float("inf"), "fine": 1.5},
        "listed": [float("nan"), 2.0],
    }
    snr_metrics._write_snr_json(summary, tmp_path)

    raw = (tmp_path / "snr_metrics.json").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # strict parsers must accept it
    parsed = json.loads(raw, parse_constant=_reject_constant)
    assert parsed["median_roi_tx_snr_ratio"] is None
    assert parsed["nested"]["also_bad"] is None
    assert parsed["nested"]["fine"] == 1.5
    assert parsed["listed"] == [None, 2.0]


def _reject_constant(name):  # pragma: no cover - only runs if the file is bad
    raise AssertionError(f"non-standard JSON constant written: {name}")


# ---------------------------------------------------------------------------
# compute_image_snr_from_pixel_maps: tissue scoping
# ---------------------------------------------------------------------------

_THRESH = {"image_snr_db": {"warn": 15.0, "fail": 10.0}}


def _tiled_frame(n: int, tissue_flags: list[bool], stride: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [i * stride for i in range(n)],
            "x2": [(i + 1) * stride for i in range(n)],
            "y1": [0] * n,
            "y2": [stride] * n,
            "overlaps_tissue": tissue_flags,
            "tissue_coverage": [1.0 if t else 0.0 for t in tissue_flags],
        }
    )


def test_median_ignores_non_tissue_tiles():
    """Background tiles must not enter the median, however many there are.

    Supplying per-tile dB directly isolates the scoping decision from the Otsu
    measurement: tissue tiles are poor (5 dB) and the many background tiles are
    excellent (40 dB). Scoped to tissue this must FAIL. Unscoped, the background
    median would carry it to PASS.
    """
    tissue = [True, True, True]
    background = [False] * 12
    df = _tiled_frame(15, tissue + background)
    per_tile = [5.0, 5.0, 5.0] + [40.0] * 12

    out = snr_metrics.compute_image_snr_from_pixel_maps(
        None,  # focus_maps unused: precomputed_db supplies the per-tile dB
        df,
        map_key="dapi",
        snr_thresholds=_THRESH,
        precomputed_db=per_tile,
    )

    assert out["status"] == "ok"
    assert out["scope"] == "tissue_tiles"
    assert out["snr_db_median"] == 5.0
    assert out["verdict"] == "FAIL"
    assert out["n_rois_computed"] == 3
    assert out["n_rois_all_tiles"] == 15


def test_per_tile_column_stays_unfiltered_for_the_concordance_figure():
    """Scoping the verdict must not silently empty the exposed per-tile column."""
    df = _tiled_frame(4, [True, False, True, False])
    per_tile = [20.0, 30.0, 22.0, 31.0]

    snr_metrics.compute_image_snr_from_pixel_maps(
        None,  # focus_maps unused: precomputed_db supplies the per-tile dB
        df,
        map_key="dapi",
        snr_thresholds=_THRESH,
        precomputed_db=per_tile,
    )

    written = df["snr_image_otsu_db"].to_numpy()
    assert not np.isnan(written).any(), "background tiles lost their per-tile dB"
    np.testing.assert_allclose(written, per_tile)


def test_declines_rather_than_grading_a_slide_with_no_tissue():
    df = _tiled_frame(4, [False] * 4)
    out = snr_metrics.compute_image_snr_from_pixel_maps(
        None,  # focus_maps unused: precomputed_db supplies the per-tile dB
        df,
        map_key="dapi",
        snr_thresholds=_THRESH,
        precomputed_db=[40.0] * 4,
    )
    assert out["status"] == "skipped"
    assert out["reason"] == "no_valid_tissue_roi_tiles"
    assert "verdict" not in out, "a verdict here would be graded on background"
    # and the aggregate must therefore not call it PASS
    assert (
        snr_metrics.aggregate_snr_verdict({"SNR_image_otsu": out})[
            "overall_snr_verdict"
        ]
        == "NOT_COMPUTED"
    )


def test_the_two_siblings_agree_on_which_population_they_measure():
    """The reference: both functions grade the same tissue tiles.

    ``compute_image_snr_from_roi_df`` has always filtered to tissue. Giving both
    the same frame, where the tissue tiles carry one dB value and the background
    another, the pixel-maps median must match the tissue value rather than land
    between the two.
    """
    tissue = [True] * 10
    background = [False] * 10
    df = _tiled_frame(20, tissue + background)
    per_tile = [18.0] * 10 + [45.0] * 10

    out = snr_metrics.compute_image_snr_from_pixel_maps(
        None,  # focus_maps unused: precomputed_db supplies the per-tile dB
        df.copy(),
        map_key="dapi",
        snr_thresholds=_THRESH,
        precomputed_db=per_tile,
    )
    assert out["snr_db_median"] == 18.0
    assert out["n_rois_computed"] == 10


def test_frame_without_tissue_columns_still_grades_every_tile():
    """No tissue information available: fall back to all tiles, do not decline."""
    df = pd.DataFrame(
        {
            "x1": [0, 8],
            "x2": [8, 16],
            "y1": [0, 0],
            "y2": [8, 8],
        }
    )
    out = snr_metrics.compute_image_snr_from_pixel_maps(
        None,  # focus_maps unused: precomputed_db supplies the per-tile dB
        df,
        map_key="dapi",
        snr_thresholds=_THRESH,
        precomputed_db=[20.0, 22.0],
    )
    assert out["status"] == "ok"
    assert out["n_rois_computed"] == 2
