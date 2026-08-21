"""Tests for the bounded-host-memory IMAGE_QC paths.

The full-resolution focus/mean/Laplacian planes are ~22 GB each on a 5.5
gigapixel sample.  Holding them in host RAM, and reducing over them with
whole-image ``scipy.ndimage`` calls, is what forced the 180 -> 720 GB retry
ladder.  These tests pin the replacements:

1. ``_labeled_sums_chunked`` reduces in row blocks and must agree exactly with
   ``scipy.ndimage.mean`` / ``skimage.measure.regionprops``, including when a
   label straddles a block boundary.
2. ``calculate_ccfs_from_focus_maps`` must reproduce the previous
   ndimage.mean + regionprops + bincount algorithm value-for-value.
3. ``_PlaneStore`` must spill large planes to disk, expose them as ordinary
   arrays, and refuse to start when the scratch filesystem is too small.
4. ``_figure_worker_limit`` must cap the forked figure fan-out.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import mean as ndimage_mean
from skimage.measure import regionprops

# Same stub-then-import dance as upstream: image_qc.py has heavy top-level
# imports that the functions under test do not need.
# ADAPTED FROM UPSTREAM: upstream also inserted
# `modules/local/image_qc/resources/usr/bin`, which does not exist in this
# pipeline -- `image_qc.py` lives in the pipeline-level `bin/`. This file lives
# in `bin/tests/`, so parent.parent is that `bin/`.
_bin_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_bin_dir))
# spawn passes sys.path to children, so this makes the worker helper importable there
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _mod in (
    "napari_skimage_regionprops",
    "napari_simpleitk_image_processing",
    "snr_metrics",
    "scanpy",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["napari_skimage_regionprops"].regionprops_table = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["scanpy"].AnnData = object  # type: ignore[attr-defined]

image_qc = importlib.import_module("image_qc")
import helpers_memmap_worker  # noqa: E402  (needs the sys.path insert above)

# /proc is Linux-only. Tests that read it directly -- via _process_rss,
# _tree_rss, or os.listdir("/proc") -- get None or FileNotFoundError off-Linux,
# so a Mac reviewer would otherwise see spurious red. Skip them there; the
# cgroup tests below use monkeypatched sources and run everywhere.
requires_proc = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="reads /proc, which exists only on Linux",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RowSliceOnly:
    """Array handle that only supports ``.shape`` and ``[y0:y1]`` row slicing.

    Stands in for a zarr array or memmap: if ``_labeled_sums_chunked`` ever
    materialised the whole plane (``np.asarray(handle)``) this would fail, so the
    test proves the reduction really is block-wise.
    """

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self.shape = array.shape

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError(f"expected a row slice, got {key!r}")
        return self._array[key]


class _MasksGroup:
    """Minimal stand-in for the zarr group passed as ``cell_masks_zarr``."""

    def __init__(self, nuclear: np.ndarray, cell: np.ndarray) -> None:
        self._masks = {"0": nuclear, "1": cell}

    def get(self, name):
        if name == "masks":
            return self
        return self._masks[name]


@pytest.fixture
def labelled_scene():
    """Deterministic label planes plus aligned value planes.

    Labels are non-contiguous and deliberately span many rows so that any
    row-block size splits several of them.
    """
    rng = np.random.RandomState(1234)
    height, width = 64, 48
    nuclear = np.zeros((height, width), dtype=np.int32)
    # Tall blobs that straddle row blocks; label values are non-consecutive.
    for idx, label in enumerate((3, 7, 8, 15, 42)):
        y0 = idx * 12
        nuclear[y0 : y0 + 14, idx * 9 : idx * 9 + 8] = label

    cell = np.zeros((height, width), dtype=np.int32)
    for idx, label in enumerate((5, 9, 11, 20, 41)):
        y0 = idx * 12
        cell[y0 : y0 + 16, idx * 9 : idx * 9 + 10] = label

    focus = rng.rand(height, width).astype(np.float32) * 100.0
    mean = rng.rand(height, width).astype(np.float32) * 5000.0
    boundary = rng.rand(height, width).astype(np.float32) * 900.0
    intrna = rng.rand(height, width).astype(np.float32) * 700.0
    return nuclear, cell, focus, mean, boundary, intrna


def _reference_ccfs(focus_maps, nuclear_mask, cellseg_mask):
    """The pre-change algorithm: np.unique + ndimage.mean + regionprops + bincount."""
    import pandas as pd

    labels = np.unique(nuclear_mask)
    labels = labels[labels > 0]
    focus_map = focus_maps["dapi_focus_map"]
    mean_map = focus_maps["dapi_mean_map"]

    cell_focus = ndimage_mean(focus_map, nuclear_mask, labels)
    cell_intensity = ndimage_mean(mean_map, nuclear_mask, labels)

    dapi_norm = np.percentile(cell_intensity, 99)
    if dapi_norm == 0:
        dapi_norm = 1.0
    ccfs_dapi = np.asarray(cell_focus) / dapi_norm

    props = regionprops(nuclear_mask)
    label_to_ccfs = dict(zip(labels, ccfs_dapi))
    label_to_intensity = dict(zip(labels, cell_intensity))

    rows = []
    for p in props:
        cy, cx = p.centroid
        iy = min(int(cy), cellseg_mask.shape[0] - 1)
        ix = min(int(cx), cellseg_mask.shape[1] - 1)
        rows.append(
            {
                "label": p.label,
                "centroid-0": cy,
                "centroid-1": cx,
                "area_nucleus": p.area,
                "CCFS_DAPI": label_to_ccfs.get(p.label, np.nan),
                "mean_intensity": label_to_intensity.get(p.label, np.nan),
                "CellID": int(cellseg_mask[iy, ix]),
            }
        )
    out = pd.DataFrame(rows)

    cell_areas = np.bincount(cellseg_mask.ravel())
    out["area_cell"] = out["CellID"].map(
        lambda cid: int(cell_areas[cid]) if cid < len(cell_areas) else 0
    )

    for key, column in (
        ("boundary_mean_map", "mean_intensity_Boundary"),
        ("intrna_mean_map", "mean_intensity_IntRNA"),
    ):
        plane = focus_maps.get(key)
        if plane is None:
            out[column] = np.nan
            continue
        uniq = out["CellID"].unique()
        uniq = uniq[uniq > 0]
        per_cell = ndimage_mean(plane, cellseg_mask, uniq)
        out[column] = out["CellID"].map(dict(zip(uniq, per_cell)))

    return out


# ---------------------------------------------------------------------------
# _labeled_sums_chunked
# ---------------------------------------------------------------------------


class TestLabeledSumsChunked:
    """Row-blocked per-label reduction must equal the whole-image reduction."""

    def test_mean_matches_scipy_ndimage(self, labelled_scene):
        nuclear, _, focus, mean, _, _ = labelled_scene
        counts, sums = image_qc._labeled_sums_chunked(
            nuclear, {"focus": focus, "intensity": mean}
        )

        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]
        got_focus = sums["focus"][labels] / counts[labels]
        got_mean = sums["intensity"][labels] / counts[labels]

        expected_focus = ndimage_mean(focus, nuclear, labels)
        expected_mean = ndimage_mean(mean, nuclear, labels)

        np.testing.assert_allclose(got_focus, expected_focus, rtol=1e-12, atol=0)
        np.testing.assert_allclose(got_mean, expected_mean, rtol=1e-12, atol=0)

    def test_label_set_matches_np_unique(self, labelled_scene):
        nuclear = labelled_scene[0]
        counts, _ = image_qc._labeled_sums_chunked(nuclear, {})
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]
        expected = np.unique(nuclear)
        np.testing.assert_array_equal(labels, expected[expected > 0])

    def test_counts_match_regionprops_area(self, labelled_scene):
        nuclear = labelled_scene[0]
        counts, _ = image_qc._labeled_sums_chunked(nuclear, {})
        for p in regionprops(nuclear):
            assert counts[p.label] == p.area, f"area mismatch for label {p.label}"

    def test_centroids_match_regionprops(self, labelled_scene):
        nuclear = labelled_scene[0]
        counts, sums = image_qc._labeled_sums_chunked(nuclear, {}, include_coords=True)
        for p in regionprops(nuclear):
            cy = sums["centroid_y_sum"][p.label] / counts[p.label]
            cx = sums["centroid_x_sum"][p.label] / counts[p.label]
            np.testing.assert_allclose((cy, cx), p.centroid, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("rows_per_chunk", [1, 2, 3, 7, 13, 64, 4096])
    def test_block_size_does_not_change_result(self, labelled_scene, rows_per_chunk):
        """Every label straddles a boundary at some block size; sums stay additive."""
        nuclear, _, focus, _, _, _ = labelled_scene
        single = image_qc._labeled_sums_chunked(
            nuclear, {"focus": focus}, include_coords=True, rows_per_chunk=10_000
        )
        blocked = image_qc._labeled_sums_chunked(
            nuclear,
            {"focus": focus},
            include_coords=True,
            rows_per_chunk=rows_per_chunk,
        )
        np.testing.assert_array_equal(single[0], blocked[0])
        for key in single[1]:
            np.testing.assert_allclose(
                single[1][key], blocked[1][key], rtol=1e-12, atol=0
            )

    def test_accepts_row_sliceable_handle(self, labelled_scene):
        """Must never materialise the whole plane — proves zarr/memmap support."""
        nuclear, _, focus, _, _, _ = labelled_scene
        handle = _RowSliceOnly(nuclear)
        counts, sums = image_qc._labeled_sums_chunked(
            handle, {"focus": _RowSliceOnly(focus)}, rows_per_chunk=5
        )
        expected_counts, expected_sums = image_qc._labeled_sums_chunked(
            nuclear, {"focus": focus}
        )
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(
            sums["focus"], expected_sums["focus"], rtol=1e-12, atol=0
        )

    def test_empty_value_planes_gives_counts_only(self, labelled_scene):
        nuclear = labelled_scene[0]
        counts, sums = image_qc._labeled_sums_chunked(nuclear, {})
        assert sums == {}
        assert counts.sum() == nuclear.size


# ---------------------------------------------------------------------------
# calculate_ccfs_from_focus_maps
# ---------------------------------------------------------------------------


class TestCalculateCcfsFromFocusMaps:
    """The blocked rewrite must reproduce the previous algorithm exactly."""

    def _run(self, labelled_scene):
        nuclear, cell, focus, mean, boundary, intrna = labelled_scene
        focus_maps = {
            "dapi_focus_map": focus,
            "dapi_mean_map": mean,
            "boundary_mean_map": boundary,
            "intrna_mean_map": intrna,
        }
        got = image_qc.calculate_ccfs_from_focus_maps(
            focus_maps, _MasksGroup(nuclear, cell), cell, []
        )
        expected = _reference_ccfs(focus_maps, nuclear, cell)
        return got, expected

    def test_same_rows_and_labels(self, labelled_scene):
        got, expected = self._run(labelled_scene)
        assert len(got) == len(expected)
        np.testing.assert_array_equal(
            got["label"].to_numpy(), expected["label"].to_numpy()
        )

    @pytest.mark.parametrize(
        "column",
        [
            "centroid-0",
            "centroid-1",
            "CCFS_DAPI",
            "mean_intensity",
            "mean_intensity_Boundary",
            "mean_intensity_IntRNA",
        ],
    )
    def test_float_columns_match_reference(self, labelled_scene, column):
        got, expected = self._run(labelled_scene)
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=np.float64),
            expected[column].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=0,
        )

    @pytest.mark.parametrize("column", ["area_nucleus", "area_cell", "CellID"])
    def test_integer_columns_match_reference(self, labelled_scene, column):
        got, expected = self._run(labelled_scene)
        np.testing.assert_array_equal(
            got[column].to_numpy(dtype=np.int64),
            expected[column].to_numpy(dtype=np.int64),
        )

    def test_missing_optional_channels_give_nan(self, labelled_scene):
        nuclear, cell, focus, mean, _, _ = labelled_scene
        got = image_qc.calculate_ccfs_from_focus_maps(
            {"dapi_focus_map": focus, "dapi_mean_map": mean},
            _MasksGroup(nuclear, cell),
            cell,
            [],
        )
        assert got["mean_intensity_Boundary"].isna().all()
        assert got["mean_intensity_IntRNA"].isna().all()

    def test_requires_dapi_maps(self, labelled_scene):
        nuclear, cell, _, mean, _, _ = labelled_scene
        with pytest.raises(ValueError, match="dapi_focus_map"):
            image_qc.calculate_ccfs_from_focus_maps(
                {"dapi_mean_map": mean}, _MasksGroup(nuclear, cell), cell, []
            )

    def test_reads_nuclear_mask_lazily(self, labelled_scene):
        """The nuclear plane must be row-sliced, never materialised whole."""
        nuclear, cell, focus, mean, _, _ = labelled_scene

        class _LazyMasks:
            def get(self, name):
                if name == "masks":
                    return self
                return _RowSliceOnly(nuclear)

        got = image_qc.calculate_ccfs_from_focus_maps(
            {"dapi_focus_map": focus, "dapi_mean_map": mean},
            _LazyMasks(),
            cell,
            [],
        )
        expected = _reference_ccfs(
            {"dapi_focus_map": focus, "dapi_mean_map": mean}, nuclear, cell
        )
        np.testing.assert_allclose(
            got["CCFS_DAPI"].to_numpy(dtype=np.float64),
            expected["CCFS_DAPI"].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=0,
        )


# ---------------------------------------------------------------------------
# _PlaneStore
# ---------------------------------------------------------------------------


class TestPlaneStore:
    """Large planes must land on disk; small ones may stay in RAM."""

    def test_small_plane_stays_in_ram(self, tmp_path):
        """Below the spill threshold a file would be pure overhead."""
        store = image_qc._PlaneStore((64, 64), ["focus_map"], plane_dir=tmp_path)
        assert store.on_disk is False
        assert store.descriptors() is None
        assert not isinstance(store.arrays()["focus_map"], np.memmap)

    def test_defaults_to_cwd(self, tmp_path, monkeypatch):
        """No plane_dir means the task working directory, per Nextflow convention."""
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        monkeypatch.chdir(tmp_path)
        store = image_qc._PlaneStore((128, 128), ["focus_map"], prefix="dapi")
        assert store.on_disk is True
        path = Path(store.descriptors()["focus_map"])
        assert path.parent == tmp_path.resolve()

    def test_large_plane_spills_to_disk(self, tmp_path, monkeypatch):
        # Shrink the spill threshold rather than allocating a real 2 GB plane.
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        store = image_qc._PlaneStore(
            (128, 128), ["focus_map", "mean_map"], plane_dir=tmp_path, prefix="dapi"
        )
        assert store.on_disk is True
        arrays = store.arrays()
        assert isinstance(arrays["focus_map"], np.memmap)
        descriptors = store.descriptors()
        assert descriptors is not None
        assert set(descriptors) == {"focus_map", "mean_map"}
        for path in descriptors.values():
            assert Path(path).exists()

    def test_disk_plane_round_trips_through_the_file(self, tmp_path, monkeypatch):
        """A second mapping of the same file sees the first mapping's writes.

        This is the property the process-per-GPU pool relies on: workers share
        planes by filename, so no pixel data is ever pickled.
        """
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        store = image_qc._PlaneStore(
            (32, 48), ["focus_map"], plane_dir=tmp_path, prefix="dapi"
        )
        plane = store.arrays()["focus_map"]
        plane[4:9, 6:11] = 3.5
        store.flush()

        path = store.descriptors()["focus_map"]
        reopened = np.memmap(path, dtype=np.float32, mode="r", shape=(32, 48))
        assert reopened[4:9, 6:11].min() == pytest.approx(3.5)
        assert reopened[0, 0] == pytest.approx(0.0)

    def test_insufficient_scratch_space_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)

        class _TinyUsage:
            free = 16

        monkeypatch.setattr(image_qc.shutil, "disk_usage", lambda _p: _TinyUsage())
        with pytest.raises(RuntimeError, match="free"):
            image_qc._PlaneStore(
                (128, 128), ["focus_map"], plane_dir=tmp_path, prefix="dapi"
            )

    def test_release_deletes_backing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        store = image_qc._PlaneStore(
            (64, 64), ["focus_map"], plane_dir=tmp_path, prefix="dapi"
        )
        paths = [Path(p) for p in store.descriptors().values()]
        assert all(p.exists() for p in paths)
        store.release()
        assert not any(p.exists() for p in paths)


# ---------------------------------------------------------------------------
# _figure_worker_limit
# ---------------------------------------------------------------------------


class TestCrossProcessPlaneSharing:
    """Separate processes must share a plane by filename, with no data pickled.

    This is the property the process-per-GPU dispatch rests on, and the reason
    disk-backing and multi-GPU are the same change: each worker maps the plane
    file itself and writes its own disjoint tile region.  Exercised here with a
    trivial write instead of a CuPy kernel, so it runs without a GPU.
    """

    def test_spawned_workers_writes_are_visible_to_the_parent(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        height, width = 200, 160

        store = image_qc._PlaneStore(
            (height, width), ["focus_map"], plane_dir=tmp_path, prefix="dapi"
        )
        plane = store.arrays()["focus_map"]
        plane[:] = 0.0
        store.flush()

        tiles = image_qc._compute_tile_grid(height, width, tile_size=64, overlap=8)
        assert len(tiles) > 1, "need several tiles for this to mean anything"
        tasks = [(spec, float(i + 1)) for i, spec in enumerate(tiles)]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=2,
            initializer=helpers_memmap_worker.init,
            initargs=(
                store.descriptors()["focus_map"],
                (height, width),
                np.dtype(np.float32).str,
            ),
        ) as pool:
            written = list(pool.imap_unordered(helpers_memmap_worker.write_tile, tasks))

        assert sorted(written) == sorted(v for _, v in tasks)

        expected = np.zeros((height, width), dtype=np.float32)
        for spec, value in tasks:
            expected[
                spec["write_y0"] : spec["write_y1"],
                spec["write_x0"] : spec["write_x1"],
            ] = value

        # Every pixel covered: the write regions tile the image exactly.
        assert (expected != 0).all(), "tile write regions do not cover the plane"
        # MAP_SHARED: the parent's pre-existing mapping sees the children's writes.
        np.testing.assert_array_equal(np.asarray(plane), expected)

    def test_worker_writes_land_in_the_backing_file(self, tmp_path, monkeypatch):
        """Not just the parent's mapping — the bytes are really on disk."""
        monkeypatch.setattr(image_qc, "_PLANE_SPILL_BYTES", 1024)
        height, width = 96, 96
        store = image_qc._PlaneStore(
            (height, width), ["mean_map"], plane_dir=tmp_path, prefix="boundary"
        )
        store.arrays()["mean_map"][:] = 0.0
        store.flush()
        path = store.descriptors()["mean_map"]

        spec = {"write_y0": 10, "write_y1": 20, "write_x0": 30, "write_x1": 40}
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=1,
            initializer=helpers_memmap_worker.init,
            initargs=(path, (height, width), np.dtype(np.float32).str),
        ) as pool:
            pool.apply(helpers_memmap_worker.write_tile, ((spec, 7.5),))

        fresh = np.memmap(path, dtype=np.float32, mode="r", shape=(height, width))
        assert fresh[10:20, 30:40].min() == pytest.approx(7.5)
        assert fresh[0, 0] == pytest.approx(0.0)


class TestMemoryInstrumentation:
    """working_set must exclude reclaimable page cache.

    This matters specifically because disk-backed planes deliberately generate
    page cache: reading the raw usage counter would show a large number and
    wrongly suggest the memory fix did nothing.
    """

    def _write_cgroup(self, root, usage, inactive, active, v2=True):
        if v2:
            (root / "memory.current").write_text(f"{usage}\n")
            (root / "memory.stat").write_text(
                f"anon 1234\ninactive_file {inactive}\nactive_file {active}\n"
            )
            return (
                str(root / "memory.current"),
                str(root / "memory.stat"),
                "inactive_file",
                "active_file",
            )
        (root / "memory.usage_in_bytes").write_text(f"{usage}\n")
        (root / "memory.stat").write_text(
            f"total_inactive_file {inactive}\ntotal_active_file {active}\n"
        )
        return (
            str(root / "memory.usage_in_bytes"),
            str(root / "memory.stat"),
            "total_inactive_file",
            "total_active_file",
        )

    def test_subtracts_inactive_file_v2(self, tmp_path, monkeypatch):
        source = self._write_cgroup(tmp_path, usage=200, inactive=150, active=10)
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (source,))
        working_set, page_cache = image_qc._cgroup_memory()
        assert working_set == 50  # 200 usage - 150 reclaimable
        assert page_cache == 160

    def test_supports_cgroup_v1(self, tmp_path, monkeypatch):
        source = self._write_cgroup(
            tmp_path, usage=500, inactive=400, active=25, v2=False
        )
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (source,))
        working_set, page_cache = image_qc._cgroup_memory()
        assert working_set == 100
        assert page_cache == 425

    def test_falls_through_to_second_source(self, tmp_path, monkeypatch):
        missing = ("/nonexistent/current", "/nonexistent/stat", "inactive_file", "a")
        source = self._write_cgroup(tmp_path, usage=90, inactive=40, active=0)
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (missing, source))
        assert image_qc._cgroup_memory()[0] == 50

    def test_returns_none_when_no_cgroup(self, monkeypatch):
        monkeypatch.setattr(
            image_qc,
            "_CGROUP_SOURCES",
            (("/nonexistent/a", "/nonexistent/b", "inactive_file", "active_file"),),
        )
        assert image_qc._cgroup_memory() is None

    def test_never_reports_negative_working_set(self, tmp_path, monkeypatch):
        source = self._write_cgroup(tmp_path, usage=10, inactive=999, active=0)
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (source,))
        assert image_qc._cgroup_memory()[0] == 0.0

    @requires_proc
    def test_process_rss_is_positive(self):
        assert image_qc._process_rss() > 0

    def test_log_mem_tracks_high_water_mark(self, tmp_path, monkeypatch):
        monkeypatch.setitem(image_qc._MEM_PEAK, "working_set", 0.0)
        (tmp_path / "lo").mkdir(parents=True, exist_ok=True)
        low = self._write_cgroup(tmp_path / "lo", usage=0, inactive=0, active=0)
        (tmp_path / "hi").mkdir(parents=True, exist_ok=True)
        high = self._write_cgroup(tmp_path / "hi", usage=800, inactive=100, active=0)

        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (high,))
        image_qc._log_mem("peak stage")
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", (low,))
        image_qc._log_mem("later quiet stage")

        assert image_qc._MEM_PEAK["working_set"] == 700

    def test_log_mem_survives_missing_cgroup(self, monkeypatch):
        monkeypatch.setattr(image_qc, "_CGROUP_SOURCES", ())
        image_qc._log_mem("no cgroup")  # must not raise
        image_qc._log_mem_summary()


class TestFigureWorkerLimit:
    """The forked figure fan-out must be capped, not os.cpu_count()."""

    def test_env_override_wins(self, monkeypatch):
        # Override is clamped to n_tasks; with plenty of tasks it wins outright.
        monkeypatch.setenv("IMAGE_QC_FIGURE_WORKERS", "2")
        assert image_qc._figure_worker_limit(11) == 2

    def test_capped_at_max(self, monkeypatch):
        monkeypatch.delenv("IMAGE_QC_FIGURE_WORKERS", raising=False)
        monkeypatch.setattr(image_qc.os, "cpu_count", lambda: 96)
        assert image_qc._figure_worker_limit(50) <= image_qc._FIGURE_WORKERS_MAX

    def test_always_at_least_one(self, monkeypatch):
        monkeypatch.delenv("IMAGE_QC_FIGURE_WORKERS", raising=False)
        monkeypatch.setattr(image_qc.os, "cpu_count", lambda: None)
        assert image_qc._figure_worker_limit(5) >= 1


def _hold_200mb(_):
    """Worker for the tree-RSS test. Module level because Pool pickles the callable."""
    import numpy as np

    buf = np.ones(25_000_000, dtype=np.float64)  # ~200 MB resident
    time.sleep(3.0)
    return float(buf[0])


@requires_proc
class TestTreeRss:
    """`_tree_rss` must count forked children, and only our own.

    The memory summary previously reported `VmHWM` from /proc/self/status, which
    excludes children — and the figure phase forks up to `_FIGURE_WORKERS_MAX` of them.
    On run 3V53J4ewZt1vsU that module reported 33.4 GB while Tower measured 100.6 GB for
    the same task, and the summary line told the reader to size the memory request from
    the smaller figure. Following it would under-provision by 3x.

    It sums descendants rather than all of /proc, because these modules also run
    directly on shared servers where the latter would add other users' processes.
    """

    def test_counts_forked_children(self):
        import multiprocessing as mp

        alone = image_qc._tree_rss()
        assert alone is not None and alone > 0

        ctx = mp.get_context("fork")
        with ctx.Pool(3) as pool:
            pending = pool.map_async(_hold_200mb, range(3))
            time.sleep(1.5)
            with_children = image_qc._tree_rss()
            pending.get(timeout=60)

        assert with_children > alone + 300 * 1024**2, (
            f"{(with_children - alone) / 1024**3:.2f} GB delta for three ~200 MB "
            "children — forked memory is not being counted"
        )

    def test_excludes_processes_outside_our_tree(self):
        """Our tree must be a strict subset of everything readable in /proc.

        The earlier version of this test asserted only that the result was under 64 GB,
        which its name did not justify. This compares against the sum over every
        readable process: if `_tree_rss` summed all of /proc — correct in a container,
        wrong on a shared server — the two would be equal.
        """
        everything = 0.0
        others = 0
        me = os.getpid()
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            everything += float(line.split()[1]) * 1024.0
                            if int(entry) != me:
                                others += 1
                            break
            except (OSError, ValueError, IndexError):
                continue
        if others == 0:
            pytest.skip("no other readable processes — nothing to exclude")
        ours = image_qc._tree_rss()
        assert ours is not None
        assert ours <= everything, "our tree cannot exceed all of /proc"
        assert ours < everything, (
            f"tree RSS {ours / 1024**3:.2f} GB equals the total over "
            f"{others} other processes — unrelated processes are being counted"
        )

    def test_returns_none_when_proc_is_unreadable(self, monkeypatch):
        monkeypatch.setattr(
            image_qc.os, "listdir", lambda *_: (_ for _ in ()).throw(OSError())
        )
        assert image_qc._tree_rss() is None


class TestCgroupPeak:
    """`_cgroup_peak` reads the kernel's continuous all-process high-water mark.

    The other three figures in the summary each understate the total in a different way,
    which is why this one exists and why the summary no longer points at a single number:

      tree_rss          all processes, but sampled at stage boundaries -- on run
                        1AjX0aBBfdbAFQ it reported 16.2 GB against a main_process_rss of
                        19.6 GB, i.e. below a figure it is supposed to contain, because
                        the main process peaked between samples
      working_set       OOM-relevant, also sampled
      main_process_rss  continuous but excludes the forked figure workers
    """

    def test_reads_v2_then_v1(self, tmp_path, monkeypatch):
        v2 = tmp_path / "memory.peak"
        v1 = tmp_path / "max_usage_in_bytes"
        v1.write_text("111\n")
        monkeypatch.setattr(image_qc, "_CGROUP_PEAK_PATHS", (str(v2), str(v1)))
        assert image_qc._cgroup_peak() == 111.0  # falls through to v1
        v2.write_text("222\n")
        assert image_qc._cgroup_peak() == 222.0  # prefers v2

    def test_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            image_qc,
            "_CGROUP_PEAK_PATHS",
            (str(tmp_path / "nope"), str(tmp_path / "also-nope")),
        )
        assert image_qc._cgroup_peak() is None

    def test_none_on_unparseable(self, tmp_path, monkeypatch):
        bad = tmp_path / "memory.peak"
        bad.write_text("max\n")
        monkeypatch.setattr(image_qc, "_CGROUP_PEAK_PATHS", (str(bad),))
        assert image_qc._cgroup_peak() is None

    def test_summary_reports_every_figure(self, caplog, monkeypatch):
        """All four must appear, so no reader takes one for the whole story."""
        monkeypatch.setattr(image_qc, "_cgroup_peak", lambda: 123 * 1024**3)
        monkeypatch.setitem(image_qc._MEM_PEAK, "tree_rss", 40 * 1024**3)
        monkeypatch.setitem(image_qc._MEM_PEAK, "working_set", 30 * 1024**3)
        monkeypatch.setitem(image_qc._MEM_PEAK, "rss", 50 * 1024**3)
        with caplog.at_level("INFO"):
            image_qc._log_mem_summary()
        text = caplog.text
        for token in (
            "cgroup_peak=123.0GB",
            "tree_rss=40.0GB",
            "working_set=30.0GB",
            "main_process_rss=50.0GB",
        ):
            assert token in text, token
        assert "peakRss" in text, "must point at the authoritative source for sizing"
