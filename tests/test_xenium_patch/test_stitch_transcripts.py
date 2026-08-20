"""Tests for stitch_transcripts.py — sopa-based stitching."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Import the standalone script from module resources
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[2] / "bin/xenium_patch_stitch_transcripts.py"
_spec = importlib.util.spec_from_file_location(
    "xenium_patch_stitch_transcripts", _SCRIPT
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["xenium_patch_stitch_transcripts"] = _mod
_spec.loader.exec_module(_mod)

from xenium_patch_stitch_transcripts import (  # noqa: E402
    Bounds,
    PatchGridMetadata,
    PatchInfo,
    _normalize_geometry_collection,
    read_geojson,
    stitch_transcript_assignments,
    transform_polygons,
)

PIXEL_SIZE = 0.2125


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patch_info(
    patch_id: str,
    row: int,
    col: int,
    global_x: tuple[float, float],
    global_y: tuple[float, float],
    core_x: tuple[float, float],
    core_y: tuple[float, float],
) -> PatchInfo:
    """Create a PatchInfo with bounds in both pixel and micron space."""
    return PatchInfo(
        patch_id=patch_id,
        row=row,
        col=col,
        global_bounds_px=Bounds(
            global_x[0] / PIXEL_SIZE,
            global_x[1] / PIXEL_SIZE,
            global_y[0] / PIXEL_SIZE,
            global_y[1] / PIXEL_SIZE,
        ),
        global_bounds_um=Bounds(global_x[0], global_x[1], global_y[0], global_y[1]),
        core_bounds_px=Bounds(
            core_x[0] / PIXEL_SIZE,
            core_x[1] / PIXEL_SIZE,
            core_y[0] / PIXEL_SIZE,
            core_y[1] / PIXEL_SIZE,
        ),
        core_bounds_um=Bounds(core_x[0], core_x[1], core_y[0], core_y[1]),
    )


def _make_metadata(patches: list[PatchInfo]) -> PatchGridMetadata:
    """Create minimal PatchGridMetadata."""
    return PatchGridMetadata(
        version="1.0",
        bundle_path="",
        image_height_px=10000,
        image_width_px=10000,
        pixel_size_um=PIXEL_SIZE,
        transcript_extent_um=Bounds(0.0, 2125.0, 0.0, 2125.0),
        grid_rows=1,
        grid_cols=2,
        overlap_um=50.0,
        overlap_px=236,
        patches=patches,
    )


def _write_grid_json(metadata: PatchGridMetadata, output_path: Path) -> None:
    """Serialize PatchGridMetadata to JSON (matching the format load_grid_metadata expects)."""

    def bounds_dict(b: Bounds) -> dict:
        return {"x_min": b.x_min, "x_max": b.x_max, "y_min": b.y_min, "y_max": b.y_max}

    data = {
        "version": metadata.version,
        "bundle_path": metadata.bundle_path,
        "image_height_px": metadata.image_height_px,
        "image_width_px": metadata.image_width_px,
        "pixel_size_um": metadata.pixel_size_um,
        "transcript_extent_um": bounds_dict(metadata.transcript_extent_um),
        "grid_rows": metadata.grid_rows,
        "grid_cols": metadata.grid_cols,
        "overlap_um": metadata.overlap_um,
        "overlap_px": metadata.overlap_px,
        "grid_type": metadata.grid_type,
        "patches": [
            {
                "patch_id": p.patch_id,
                "row": p.row,
                "col": p.col,
                "global_bounds_px": bounds_dict(p.global_bounds_px),
                "global_bounds_um": bounds_dict(p.global_bounds_um),
                "core_bounds_px": bounds_dict(p.core_bounds_px),
                "core_bounds_um": bounds_dict(p.core_bounds_um),
            }
            for p in metadata.patches
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _write_patch_csv(
    patch_dir: Path, rows: list[dict], filename: str = "segmentation.csv"
) -> None:
    """Write a Baysor-style CSV."""
    if not rows:
        return
    cols = list(rows[0].keys())
    arrays = {
        col: pa.array([str(r[col]) for r in rows], type=pa.string()) for col in cols
    }
    table = pa.table(arrays)
    patch_dir.mkdir(parents=True, exist_ok=True)
    pa_csv.write_csv(table, patch_dir / filename)


def _write_patch_geojson(
    patch_dir: Path,
    cell_polygons: dict[str, Polygon],
    filename: str = "segmentation_polygons.json",
) -> None:
    """Write a GeoJSON FeatureCollection with cell polygons in local coordinates."""
    features = []
    for cell_id, poly in cell_polygons.items():
        features.append(
            {
                "type": "Feature",
                "id": cell_id,
                "geometry": mapping(poly),
                "properties": {"cell_id": cell_id},
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    patch_dir.mkdir(parents=True, exist_ok=True)
    with open(patch_dir / filename, "w") as f:
        json.dump(geojson, f)


# ---------------------------------------------------------------------------
# Stitch tests
# ---------------------------------------------------------------------------


class TestStitchBasic:
    def test_stitch_basic(self, tmp_path: Path):
        """Create 2 patches with non-overlapping cells, verify merged output."""
        # Patch 0: core [0,500) x [0,1000), global [0,525) x [0,1000)
        # Patch 1: core [500,1000) x [0,1000), global [475,1000) x [0,1000)
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 525.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 500.0),
            core_y=(0.0, 1000.0),
        )
        p1 = _make_patch_info(
            "patch_1",
            0,
            1,
            global_x=(475.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(500.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0, p1])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # Patch 0: cell at (100, 100) local -> (100, 100) global
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
                {
                    "transcript_id": "tx_2",
                    "x": "200.0",
                    "y": "200.0",
                    "gene": "B",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_0",
            {"cell_1": Polygon([(50, 50), (250, 50), (250, 250), (50, 250)])},
        )

        # Patch 1: cell at (100, 100) local -> (575, 100) global
        _write_patch_csv(
            patches_dir / "patch_1",
            [
                {
                    "transcript_id": "tx_3",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "C",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_1",
            {"cell_1": Polygon([(50, 50), (200, 50), (200, 200), (50, 200)])},
        )

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        csv_out = output_dir / "xr-transcript-metadata.csv"
        assert csv_out.exists()

        geo_out = output_dir / "xr-cell-polygons.geojson"
        assert geo_out.exists()

        # Read CSV and verify transcripts present
        merged = pa_csv.read_csv(csv_out)
        assert merged.num_rows >= 3

    def test_stitch_cell_id_sequential(self, tmp_path: Path):
        """Verify global IDs are cell-1, cell-2, ..."""
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 525.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 500.0),
            core_y=(0.0, 1000.0),
        )
        p1 = _make_patch_info(
            "patch_1",
            0,
            1,
            global_x=(475.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(500.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0, p1])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_0",
            {"cell_1": Polygon([(50, 50), (250, 50), (250, 250), (50, 250)])},
        )

        _write_patch_csv(
            patches_dir / "patch_1",
            [
                {
                    "transcript_id": "tx_2",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "B",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_1",
            {"cell_1": Polygon([(50, 50), (200, 50), (200, 200), (50, 200)])},
        )

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        geo_out = output_dir / "xr-cell-polygons.geojson"
        with open(geo_out) as f:
            geo = json.load(f)

        cell_ids = [feat["id"] for feat in geo["features"]]
        for cid in cell_ids:
            assert cid.startswith("cell-"), f"Cell ID {cid} not in cell-N format"

        # IDs should be sequential starting at 1
        numbers = sorted(int(cid.split("-")[1]) for cid in cell_ids)
        assert numbers == list(range(1, len(cell_ids) + 1))

    def test_stitch_transcript_dedup(self, tmp_path: Path):
        """Same transcript in 2 patches: assigned wins over noise."""
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 600.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 500.0),
            core_y=(0.0, 1000.0),
        )
        p1 = _make_patch_info(
            "patch_1",
            0,
            1,
            global_x=(400.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(500.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0, p1])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # tx_dup appears in both patches. In patch_0 it's assigned, in patch_1 it's noise.
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_dup",
                    "x": "450.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
                {
                    "transcript_id": "tx_only0",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "B",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_0",
            {"cell_1": Polygon([(50, 50), (500, 50), (500, 250), (50, 250)])},
        )

        _write_patch_csv(
            patches_dir / "patch_1",
            [
                {
                    "transcript_id": "tx_dup",
                    "x": "50.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "",
                    "is_noise": "1",
                },
                {
                    "transcript_id": "tx_only1",
                    "x": "200.0",
                    "y": "200.0",
                    "gene": "C",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_1",
            {"cell_1": Polygon([(150, 50), (350, 50), (350, 350), (150, 350)])},
        )

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        csv_out = output_dir / "xr-transcript-metadata.csv"
        merged = pa_csv.read_csv(csv_out)
        tid_col = merged.column("transcript_id").to_pylist()
        cell_col = merged.column("cell").to_pylist()

        # tx_dup should appear exactly once
        dup_count = tid_col.count("tx_dup")
        assert dup_count == 1, f"tx_dup appears {dup_count} times, expected 1"

        # The kept version should be assigned (non-empty cell)
        dup_idx = tid_col.index("tx_dup")
        assert cell_col[dup_idx] != "", "tx_dup should be assigned, not noise"

    def test_stitch_noise_spatial_reassignment(self, tmp_path: Path):
        """Noise transcript inside a resolved cell polygon gets assigned."""
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 600.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 600.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # tx_noise is at (150, 150) local -> (150, 150) global, inside the cell polygon
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_assigned",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
                {
                    "transcript_id": "tx_noise",
                    "x": "150.0",
                    "y": "150.0",
                    "gene": "B",
                    "cell": "",
                    "is_noise": "1",
                },
            ],
        )
        # Cell polygon covers (50,50) to (250,250) in local coords -> global same since origin is 0
        _write_patch_geojson(
            patches_dir / "patch_0",
            {"cell_1": Polygon([(50, 50), (250, 50), (250, 250), (50, 250)])},
        )

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        csv_out = output_dir / "xr-transcript-metadata.csv"
        merged = pa_csv.read_csv(csv_out)
        tid_col = merged.column("transcript_id").to_pylist()
        cell_col = merged.column("cell").to_pylist()

        noise_idx = tid_col.index("tx_noise")
        assert cell_col[noise_idx] != "", (
            "tx_noise should be spatially reassigned to a cell"
        )

    def test_stitch_geojson_not_found(self, tmp_path: Path):
        """When GeoJSON doesn't exist, stitch should still work (transcript-only)."""
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # Write CSV but no GeoJSON
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "cell_1",
                    "is_noise": "0",
                },
            ],
        )

        output_dir = tmp_path / "output"
        # Should not raise
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        # No geojson output (no polygons to write)
        geo_out = output_dir / "xr-cell-polygons.geojson"
        assert not geo_out.exists()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestReadGeoJSON:
    def test_read_geojson_feature_collection(self, tmp_path: Path):
        """Standard FeatureCollection is returned as-is."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "cell_1",
                    "geometry": mapping(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])),
                    "properties": {"cell_id": "cell_1"},
                }
            ],
        }
        path = tmp_path / "test.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f)

        result = read_geojson(path)
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 1
        assert result["features"][0]["id"] == "cell_1"

    def test_read_geojson_geometry_collection(self, tmp_path: Path):
        """proseg's GeometryCollection format is normalized to FeatureCollection."""
        geojson = {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                    "cell": 1,
                },
                {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]],
                    "cell": 2,
                },
            ],
        }
        path = tmp_path / "proseg.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f)

        result = read_geojson(path)
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 2
        assert result["features"][0]["id"] == "1"
        assert result["features"][1]["id"] == "2"
        # geometry should not contain the 'cell' key
        assert "cell" not in result["features"][0]["geometry"]


class TestTransformPolygons:
    def test_transform_polygons_offset(self):
        """Verify coordinate shift by (offset_x, offset_y)."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "cell_1",
                    "geometry": mapping(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])),
                    "properties": {"cell_id": "cell_1"},
                }
            ],
        }

        shifted = transform_polygons(geojson, offset_x=100.0, offset_y=200.0)

        assert shifted["type"] == "FeatureCollection"
        assert len(shifted["features"]) == 1

        coords = shifted["features"][0]["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        assert min(xs) == pytest.approx(100.0)
        assert max(xs) == pytest.approx(110.0)
        assert min(ys) == pytest.approx(200.0)
        assert max(ys) == pytest.approx(210.0)


class TestNormalizeGeometryCollection:
    def test_empty_geometry_collection(self):
        """Empty GeometryCollection returns empty FeatureCollection."""
        result = _normalize_geometry_collection(
            {"type": "GeometryCollection", "geometries": []}
        )
        assert result["type"] == "FeatureCollection"
        assert result["features"] == []

    def test_string_cell_id_passthrough(self):
        """Non-integer cell key is passed through as string."""
        geojson = {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    "cell": "custom-id",
                }
            ],
        }
        result = _normalize_geometry_collection(geojson)
        assert result["features"][0]["id"] == "custom-id"


# ---------------------------------------------------------------------------
# Baysor native format tests (empty cell column, mismatched IDs)
# ---------------------------------------------------------------------------


class TestBaysorNativeFormat:
    def test_baysor_empty_cell_column(self, tmp_path: Path):
        """Baysor native output: cell column is empty, GeoJSON has integer IDs.

        This is the core bug that spatial containment fixes. Previously, the
        ID-matching approach would skip all polygons because no CSV cell values
        matched the GeoJSON cell IDs.
        """
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # Baysor CSV: cell column is EMPTY, cell_id has string labels
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_1",
                    "cell_id": "higeahke-1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "GeneA",
                    "cell": "",
                    "is_noise": "0",
                },
                {
                    "transcript_id": "tx_2",
                    "cell_id": "higeahke-1",
                    "x": "150.0",
                    "y": "150.0",
                    "gene": "GeneB",
                    "cell": "",
                    "is_noise": "0",
                },
                {
                    "transcript_id": "tx_3",
                    "cell_id": "",
                    "x": "800.0",
                    "y": "800.0",
                    "gene": "GeneC",
                    "cell": "",
                    "is_noise": "1",
                },
            ],
        )

        # GeoJSON: GeometryCollection with integer cell keys (proseg format)
        # Polygon covers (50,50)-(250,250), so tx_1 and tx_2 are inside, tx_3 is outside
        geojson = {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[50, 50], [250, 50], [250, 250], [50, 250], [50, 50]]
                    ],
                    "cell": 4986,
                }
            ],
        }
        patch_dir = patches_dir / "patch_0"
        patch_dir.mkdir(parents=True, exist_ok=True)
        with open(patch_dir / "segmentation_polygons.json", "w") as f:
            json.dump(geojson, f)

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        csv_out = output_dir / "xr-transcript-metadata.csv"
        assert csv_out.exists(), "CSV output should be written"

        geo_out = output_dir / "xr-cell-polygons.geojson"
        assert geo_out.exists(), "GeoJSON output should be written"

        merged = pa_csv.read_csv(csv_out)
        tid_col = merged.column("transcript_id").to_pylist()
        cell_col = merged.column("cell").to_pylist()

        # tx_1 and tx_2 should be assigned to a cell (spatially inside polygon)
        for tx_id in ["tx_1", "tx_2"]:
            idx = tid_col.index(tx_id)
            assert cell_col[idx] != "", (
                f"{tx_id} should be assigned via spatial containment"
            )
            assert cell_col[idx].startswith("cell-"), (
                f"{tx_id} should have global ID format"
            )

        # tx_3 should be noise (outside polygon)
        tx3_idx = tid_col.index("tx_3")
        assert cell_col[tx3_idx] == "", "tx_3 should remain noise (outside polygon)"

    def test_baysor_two_patches_empty_cell(self, tmp_path: Path):
        """Two patches with Baysor native format: spatial assignment across patches."""
        p0 = _make_patch_info(
            "patch_0",
            0,
            0,
            global_x=(0.0, 525.0),
            global_y=(0.0, 1000.0),
            core_x=(0.0, 500.0),
            core_y=(0.0, 1000.0),
        )
        p1 = _make_patch_info(
            "patch_1",
            0,
            1,
            global_x=(475.0, 1000.0),
            global_y=(0.0, 1000.0),
            core_x=(500.0, 1000.0),
            core_y=(0.0, 1000.0),
        )
        metadata = _make_metadata([p0, p1])

        patches_dir = tmp_path / "patches"
        _write_grid_json(metadata, patches_dir / "patch_grid.json")

        # Patch 0: cell column empty, transcript at (100,100)
        _write_patch_csv(
            patches_dir / "patch_0",
            [
                {
                    "transcript_id": "tx_1",
                    "cell_id": "abc-1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "A",
                    "cell": "",
                    "is_noise": "0",
                },
            ],
        )
        # Polygon at (50,50)-(250,250) in local coords
        _write_patch_geojson(
            patches_dir / "patch_0",
            {"anything": Polygon([(50, 50), (250, 50), (250, 250), (50, 250)])},
        )

        # Patch 1: cell column empty, transcript at (100,100) local -> (575,100) global
        _write_patch_csv(
            patches_dir / "patch_1",
            [
                {
                    "transcript_id": "tx_2",
                    "cell_id": "xyz-1",
                    "x": "100.0",
                    "y": "100.0",
                    "gene": "B",
                    "cell": "",
                    "is_noise": "0",
                },
            ],
        )
        _write_patch_geojson(
            patches_dir / "patch_1",
            {"whatever": Polygon([(50, 50), (200, 50), (200, 200), (50, 200)])},
        )

        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
        )

        csv_out = output_dir / "xr-transcript-metadata.csv"
        assert csv_out.exists()

        merged = pa_csv.read_csv(csv_out)
        tid_col = merged.column("transcript_id").to_pylist()
        cell_col = merged.column("cell").to_pylist()

        # Both transcripts should be assigned
        for tx_id in ["tx_1", "tx_2"]:
            idx = tid_col.index(tx_id)
            assert cell_col[idx] != "", f"{tx_id} should be assigned"
            assert cell_col[idx].startswith("cell-")

        # They should be in different cells
        tx1_cell = cell_col[tid_col.index("tx_1")]
        tx2_cell = cell_col[tid_col.index("tx_2")]
        assert tx1_cell != tx2_cell, (
            "Transcripts in different patches should have different cells"
        )

        geo_out = output_dir / "xr-cell-polygons.geojson"
        assert geo_out.exists()
        with open(geo_out) as f:
            geo = json.load(f)
        assert len(geo["features"]) == 2


# ---------------------------------------------------------------------------
# min-transcripts-per-cell filter — regression for the dropped CLI argument
# (XENIUM_PATCH_STITCH failed with "unrecognized arguments:
#  --min-transcripts-per-cell 50" after the script was split; see
#  docs/failures/2026-06-27_xenium-patch-stitch-min-transcripts-arg.md)
# ---------------------------------------------------------------------------


def _write_one_patch_two_cells(tmp_path: Path) -> Path:
    """One full-extent patch: cell_big (3 transcripts) + cell_small (1)."""
    p0 = _make_patch_info(
        "patch_0",
        0,
        0,
        global_x=(0.0, 1000.0),
        global_y=(0.0, 1000.0),
        core_x=(0.0, 1000.0),
        core_y=(0.0, 1000.0),
    )
    metadata = _make_metadata([p0])
    patches_dir = tmp_path / "patches"
    _write_grid_json(metadata, patches_dir / "patch_grid.json")
    _write_patch_csv(
        patches_dir / "patch_0",
        [
            {
                "transcript_id": "tx_1",
                "x": "100.0",
                "y": "100.0",
                "gene": "A",
                "cell": "cell_big",
                "is_noise": "0",
            },
            {
                "transcript_id": "tx_2",
                "x": "110.0",
                "y": "110.0",
                "gene": "B",
                "cell": "cell_big",
                "is_noise": "0",
            },
            {
                "transcript_id": "tx_3",
                "x": "120.0",
                "y": "120.0",
                "gene": "C",
                "cell": "cell_big",
                "is_noise": "0",
            },
            {
                "transcript_id": "tx_small",
                "x": "800.0",
                "y": "800.0",
                "gene": "D",
                "cell": "cell_small",
                "is_noise": "0",
            },
        ],
    )
    _write_patch_geojson(
        patches_dir / "patch_0",
        {
            "cell_big": Polygon([(50, 50), (200, 50), (200, 200), (50, 200)]),
            "cell_small": Polygon([(750, 750), (850, 750), (850, 850), (750, 850)]),
        },
    )
    return patches_dir


class TestMinTranscriptsFilter:
    def test_filter_drops_small_cells(self, tmp_path: Path):
        """min_transcripts_per_cell=2 reassigns the 1-transcript cell to noise
        and removes its polygon, keeping the 3-transcript cell."""
        patches_dir = _write_one_patch_two_cells(tmp_path)
        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
            min_transcripts_per_cell=2,
        )

        merged = pa_csv.read_csv(output_dir / "xr-transcript-metadata.csv")
        tid = merged.column("transcript_id").to_pylist()
        cell = merged.column("cell").to_pylist()

        # The lone transcript of the small cell is demoted to noise (empty cell)
        assert cell[tid.index("tx_small")] == "", (
            "small cell's transcript should be reassigned to noise"
        )
        # The 3-transcript cell survives
        assert cell[tid.index("tx_1")] != "", "big cell should be retained"

        # Its polygon is dropped — only the surviving cell remains
        with open(output_dir / "xr-cell-polygons.geojson") as f:
            geo = json.load(f)
        assert len(geo["features"]) == 1

    def test_no_filter_keeps_all_cells(self, tmp_path: Path):
        """min_transcripts_per_cell=0 (default) keeps both cells — control."""
        patches_dir = _write_one_patch_two_cells(tmp_path)
        output_dir = tmp_path / "output"
        stitch_transcript_assignments(
            patches_dir=patches_dir,
            output_dir=output_dir,
            max_workers=1,
            min_transcripts_per_cell=0,
        )
        with open(output_dir / "xr-cell-polygons.geojson") as f:
            geo = json.load(f)
        assert len(geo["features"]) == 2

    def test_cli_accepts_min_transcripts_per_cell(self, tmp_path: Path):
        """Reproduces the production failure: invoke the script exactly as the
        module does, including --min-transcripts-per-cell, via the real CLI.
        The argparse contract must accept the flag (no 'unrecognized arguments')."""
        patches_dir = _write_one_patch_two_cells(tmp_path)
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--patches",
                str(patches_dir),
                "--output",
                str(output_dir),
                "--min-transcripts-per-cell",
                "50",
            ],
            capture_output=True,
            text=True,
        )
        assert "unrecognized arguments" not in result.stderr, result.stderr
        assert result.returncode == 0, (
            f"script exited {result.returncode}\nstderr:\n{result.stderr}"
        )
