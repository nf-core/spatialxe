"""Unit tests for the cell -> grid-tile mapping (map_grid_roi_to_cells).

`map_grid_roi_to_cells` has two implementations of the same operation:

* a fast vectorised path, taken when the grid is non-overlapping and complete,
  which converts a centroid to a grid index arithmetically; and
* a slow containment path (`roi_x1 <= x < roi_x2`), taken otherwise, which is the
  reference definition of "which tile is this cell in".

The fast path used `np.round` on centroids where it must use `np.floor`. Rounding
is correct for the tile *edges* used to populate the lookup grid (they are exact
multiples of the stride) but wrong for centroids, which fall anywhere inside a
tile: every cell past the half-stride mark was pushed into the next tile.
Measured on four calibration samples, that misassigned 73.5-73.9% of cells and
inflated `pct_blurred_gmm_2d_roi` by 2x to 14x while depressing
`ccfs_gmm_agreement_pct` by 10-19 points. See
`docs/plans/2026-08-24_SPIKE_tile-mapping-impact.md`.

These tests assert the two paths agree rather than pinning tile IDs the tests
themselves compute, so they cannot enshrine the same off-by-half error, and they
fail loudly if `round` is ever reintroduced.

`overlapping=True` is what forces the slow path; the two calls are otherwise
identical.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

# image_qc.py has heavy top-level imports (napari, snr_metrics, etc.) not needed
# for the pure function under test. Stub them before importing, matching
# test_cluster_outliers.py. Import paths come from tests/conftest.py.
_stubs = [
    "napari_skimage_regionprops",
    "napari_simpleitk_image_processing",
    "snr_metrics",
    "scanpy",
]
for _mod in _stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["napari_skimage_regionprops"].regionprops_table = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["scanpy"].AnnData = object  # type: ignore[attr-defined]

image_qc = importlib.import_module("image_qc")
map_grid_roi_to_cells = image_qc.map_grid_roi_to_cells

STRIDE = 35


def _grid(n_cols: int, n_rows: int, stride: int = STRIDE) -> pd.DataFrame:
    """A complete non-overlapping tile grid, one row per tile."""
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            rows.append(
                {
                    "roi_id": r * n_cols + c,
                    "x1": c * stride,
                    "x2": (c + 1) * stride,
                    "y1": r * stride,
                    "y2": (r + 1) * stride,
                    # columns the function joins onto cells; distinct per tile so a
                    # misassignment is visible, not masked by equal values
                    "focus_score": float(r * n_cols + c),
                    "focus_score_norm": float(r * n_cols + c) / 100.0,
                    "raw_intensity": 1000.0 + (r * n_cols + c),
                    "tissue_coverage": ((r * n_cols + c) % 11) / 10.0,
                }
            )
    return pd.DataFrame(rows)


def _cells(n_cols: int, n_rows: int, fracs, stride: int = STRIDE) -> pd.DataFrame:
    """One cell per (tile, fraction) pair, placed at `frac` of the way into the tile."""
    rows = []
    cid = 0
    for r in range(n_rows):
        for c in range(n_cols):
            for fx in fracs:
                for fy in fracs:
                    rows.append(
                        {
                            "cell_id": f"c{cid}",
                            "x": c * stride + fx * stride,
                            "y": r * stride + fy * stride,
                        }
                    )
                    cid += 1
    return pd.DataFrame(rows)


def _map(grid, cells, *, slow: bool):
    return map_grid_roi_to_cells(grid, cells, overlapping=slow)


# The three fractions that matter: below the halfway point (round and floor agree),
# exactly at it, and above it (round and floor disagree).
_FRACS = (0.25, 0.5, 0.75)


def test_fast_path_matches_containment_path():
    """The vectorised lookup must agree with `roi_x1 <= x < roi_x2` exactly."""
    grid = _grid(6, 5)
    cells = _cells(6, 5, _FRACS)

    fast = _map(grid, cells, slow=False)
    slow = _map(grid, cells, slow=True)

    assert len(fast) == len(cells)
    pd.testing.assert_series_equal(
        fast["roi_id"].reset_index(drop=True),
        slow["roi_id"].reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("frac", _FRACS)
def test_each_offset_within_a_tile_maps_to_that_tile(frac):
    """A cell anywhere inside a tile belongs to that tile, not its neighbour.

    Parametrised so a failure names the offending offset. `round` fails at 0.5 and
    0.75; `floor` passes all three.
    """
    grid = _grid(4, 4)
    cells = _cells(4, 4, (frac,))

    fast = _map(grid, cells, slow=False)

    # the tile each cell sits in, computed from the tile bounds rather than from
    # the same arithmetic the function under test uses
    expected = []
    for _, cell in cells.iterrows():
        hit = grid[
            (grid["x1"] <= cell["x"])
            & (cell["x"] < grid["x2"])
            & (grid["y1"] <= cell["y"])
            & (cell["y"] < grid["y2"])
        ]
        expected.append(hit["roi_id"].iloc[0])

    assert list(fast["roi_id"]) == expected


def test_joined_tile_attributes_follow_the_correct_tile():
    """The joined columns, not just roi_id, must come from the containing tile."""
    grid = _grid(5, 5)
    cells = _cells(5, 5, _FRACS)

    fast = _map(grid, cells, slow=False)
    slow = _map(grid, cells, slow=True)

    for col in ("roi_tissue_coverage", "DAPI_RFS_roi"):
        if col in fast.columns and col in slow.columns:
            np.testing.assert_allclose(
                fast[col].to_numpy(dtype=float),
                slow[col].to_numpy(dtype=float),
                equal_nan=True,
            )


def test_cells_on_the_far_edge_are_not_pushed_off_the_grid():
    """A cell just inside the last tile stays in it rather than being clipped."""
    n = 4
    grid = _grid(n, n)
    last = grid["roi_id"].max()
    eps = 0.01
    cells = pd.DataFrame(
        [
            {
                "cell_id": "edge",
                "x": n * STRIDE - eps,
                "y": n * STRIDE - eps,
            }
        ]
    )

    fast = _map(grid, cells, slow=False)
    slow = _map(grid, cells, slow=True)

    assert fast["roi_id"].iloc[0] == last
    assert slow["roi_id"].iloc[0] == last
