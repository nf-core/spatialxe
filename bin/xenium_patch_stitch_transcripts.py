#!/usr/bin/env python3
"""Stitch per-patch Baysor segmentation results into unified output.

Standalone script that replaces the xenium_patch CLI package's stitch
functionality. Uses sopa's solve_conflicts() for overlap resolution.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import shapely
from shapely.affinity import translate
from shapely.geometry import mapping, shape
from sopa.segmentation.resolve import solve_conflicts

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ensure_polygon(geom) -> "shapely.Polygon | None":
    """Extract a single Polygon from any geometry, or return None.

    XeniumRanger only accepts Polygon. make_valid() and solve_conflicts
    can produce MultiPolygon, GeometryCollection, MultiLineString, etc.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        return max(polys, key=lambda g: g.area) if polys else None
    # LineString, MultiLineString, Point, etc. — not a polygon
    return None


# ---------------------------------------------------------------------------
# Inline types (from _types.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned bounding box in either pixel or micron coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class PatchInfo:
    """Metadata for a single patch in the grid."""

    patch_id: str
    row: int
    col: int
    global_bounds_px: Bounds
    global_bounds_um: Bounds
    core_bounds_px: Bounds
    core_bounds_um: Bounds


@dataclass
class PatchGridMetadata:
    """Full grid metadata, serializable to JSON."""

    version: str
    bundle_path: str
    image_height_px: int
    image_width_px: int
    pixel_size_um: float
    transcript_extent_um: Bounds
    grid_rows: int
    grid_cols: int
    overlap_um: float
    overlap_px: int
    patches: list[PatchInfo]
    grid_type: str = "uniform"


# ---------------------------------------------------------------------------
# Internal result containers
# ---------------------------------------------------------------------------


@dataclass
class _PatchGeoResult:
    """Result of parallel GeoJSON processing for a single patch."""

    features: list[dict]
    cell_ids: list[str]


@dataclass
class _PatchCsvResult:
    """Result of parallel CSV reading for a single patch."""

    table: pa.Table
    has_cell_col: bool
    has_x_col: bool
    has_y_col: bool
    has_gene_col: bool = False
    has_feature_name_col: bool = False


# ---------------------------------------------------------------------------
# Grid metadata I/O (from grid.py)
# ---------------------------------------------------------------------------


def _dict_to_bounds(d: dict) -> Bounds:
    return Bounds(d["x_min"], d["x_max"], d["y_min"], d["y_max"])


def load_grid_metadata(input_path: Path) -> PatchGridMetadata:
    """Deserialize PatchGridMetadata from JSON.

    Args:
        input_path: Path to JSON file to read.

    Returns:
        Reconstructed PatchGridMetadata.
    """
    with open(input_path) as f:
        data = json.load(f)

    patches = [
        PatchInfo(
            patch_id=p["patch_id"],
            row=p["row"],
            col=p["col"],
            global_bounds_px=_dict_to_bounds(p["global_bounds_px"]),
            global_bounds_um=_dict_to_bounds(p["global_bounds_um"]),
            core_bounds_px=_dict_to_bounds(p["core_bounds_px"]),
            core_bounds_um=_dict_to_bounds(p["core_bounds_um"]),
        )
        for p in data["patches"]
    ]

    return PatchGridMetadata(
        version=data["version"],
        bundle_path=data["bundle_path"],
        image_height_px=data["image_height_px"],
        image_width_px=data["image_width_px"],
        pixel_size_um=data["pixel_size_um"],
        transcript_extent_um=_dict_to_bounds(data["transcript_extent_um"]),
        grid_rows=data["grid_rows"],
        grid_cols=data["grid_cols"],
        overlap_um=data["overlap_um"],
        overlap_px=data["overlap_px"],
        grid_type=data.get("grid_type", "uniform"),
        patches=patches,
    )


# ---------------------------------------------------------------------------
# GeoJSON I/O (from polygon_io.py)
# ---------------------------------------------------------------------------


def _normalize_geometry_collection(geojson: dict) -> dict:
    """Convert a GeometryCollection to a FeatureCollection.

    proseg-to-baysor produces a non-standard GeoJSON GeometryCollection where
    each geometry object has a custom ``cell`` key (bare integer) instead of
    using Feature wrappers. This normalises it to a standard FeatureCollection
    with ``id`` and ``properties.cell_id`` on each feature, using the
    ``"cell-{N}"`` format that matches the companion CSV.

    Args:
        geojson: Parsed GeoJSON dict with type GeometryCollection.

    Returns:
        Standard FeatureCollection dict.
    """
    features = []
    for geom in geojson.get("geometries", []):
        cell_raw = geom.get("cell", "")
        cell_id = str(cell_raw)
        clean_geom = {k: v for k, v in geom.items() if k != "cell"}
        feature = {
            "type": "Feature",
            "id": cell_id,
            "geometry": clean_geom,
            "properties": {"cell_id": cell_id},
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def read_geojson(geojson_path: Path) -> dict:
    """Read a GeoJSON file and normalise to FeatureCollection.

    Handles both standard FeatureCollections and the GeometryCollection
    format produced by proseg-to-baysor.

    Args:
        geojson_path: Path to the GeoJSON file.

    Returns:
        Parsed GeoJSON dict (always a FeatureCollection).
    """
    with open(geojson_path) as f:
        data = json.load(f)
    if data.get("type") == "GeometryCollection":
        return _normalize_geometry_collection(data)
    return data


def transform_polygons(geojson: dict, offset_x: float, offset_y: float) -> dict:
    """Shift all polygon coordinates by (offset_x, offset_y).

    Args:
        geojson: Input FeatureCollection.
        offset_x: Translation in x.
        offset_y: Translation in y.

    Returns:
        New FeatureCollection with shifted geometries.
    """
    features = []
    for feat in geojson.get("features", []):
        geom = shape(feat["geometry"])
        shifted = translate(geom, xoff=offset_x, yoff=offset_y)
        new_feat = {**feat, "geometry": mapping(shifted)}
        features.append(new_feat)
    return {"type": "FeatureCollection", "features": features}


def write_geojson(geojson: dict, output_path: Path) -> None:
    """Write a GeoJSON FeatureCollection.

    Args:
        geojson: GeoJSON dict to write.
        output_path: Destination path (parent dirs created automatically).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)


# ---------------------------------------------------------------------------
# Arrow utilities (from _arrow_utils.py)
# ---------------------------------------------------------------------------


def float_str_array(f64_array: pa.Array) -> pa.Array:
    """Convert a float64 pyarrow array to string using Python's str(float) format.

    pyarrow's built-in cast omits trailing '.0' for whole numbers. This
    function ensures output matches str(float(...)) for CSV compatibility.

    Args:
        f64_array: Float64 pyarrow array to convert.

    Returns:
        String pyarrow array with Python-formatted float values.
    """
    return pa.array(
        [str(v) if v is not None else None for v in f64_array.to_pylist()],
        type=pa.string(),
    )


# ---------------------------------------------------------------------------
# Parallel I/O
# ---------------------------------------------------------------------------


def _read_and_transform_geojson(
    patch: PatchInfo,
    patches_dir: Path,
    geojson_filename: str,
) -> _PatchGeoResult | None:
    """Read, transform GeoJSON for a single patch (no core clipping).

    Args:
        patch: Patch metadata.
        patches_dir: Root patches directory.
        geojson_filename: GeoJSON filename within each patch directory.

    Returns:
        _PatchGeoResult with features and cell IDs, or None if no GeoJSON.
    """
    geojson_path = patches_dir / patch.patch_id / geojson_filename
    if not geojson_path.exists():
        return None

    geojson = read_geojson(geojson_path)

    offset_x = patch.global_bounds_um.x_min
    offset_y = patch.global_bounds_um.y_min
    geojson = transform_polygons(geojson, offset_x, offset_y)

    features = geojson.get("features", [])
    seen: set[str] = set()
    cell_ids: list[str] = []
    for feat in features:
        old_id = str(feat.get("id", feat.get("properties", {}).get("cell_id", "")))
        if old_id not in seen:
            seen.add(old_id)
            cell_ids.append(old_id)

    return _PatchGeoResult(features=features, cell_ids=cell_ids)


def _read_patch_csv(
    patch: PatchInfo,
    patches_dir: Path,
    csv_filename: str,
) -> _PatchCsvResult | None:
    """Read a patch CSV into a pyarrow Table.

    All columns are read as strings to preserve exact formatting.

    Args:
        patch: Patch metadata.
        patches_dir: Root patches directory.
        csv_filename: CSV filename within each patch directory.

    Returns:
        _PatchCsvResult with the table and column presence flags, or None.
    """
    csv_path = patches_dir / patch.patch_id / csv_filename
    if not csv_path.exists():
        return None

    with open(csv_path) as fh:
        header_line = fh.readline().strip()
    col_names = header_line.split(",")
    all_string_types = {name: pa.string() for name in col_names}

    table = pa_csv.read_csv(
        csv_path,
        convert_options=pa_csv.ConvertOptions(
            column_types=all_string_types,
            strings_can_be_null=False,
        ),
        read_options=pa_csv.ReadOptions(use_threads=True),
    )

    return _PatchCsvResult(
        table=table,
        has_cell_col="cell" in table.column_names,
        has_x_col="x" in table.column_names,
        has_y_col="y" in table.column_names,
        has_gene_col="gene" in table.column_names,
        has_feature_name_col="feature_name" in table.column_names,
    )


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------


def _transform_patch_coords(
    csv_result: _PatchCsvResult,
    offset_x: float,
    offset_y: float,
) -> pa.Table:
    """Shift transcript coordinates from local patch space to global space.

    Args:
        csv_result: The raw CSV table and column flags.
        offset_x: X offset for coordinate transform (microns).
        offset_y: Y offset for coordinate transform (microns).

    Returns:
        Table with x, y columns shifted to global coordinates.
    """
    table = csv_result.table

    if table.num_rows == 0:
        return table

    if csv_result.has_x_col:
        x_f64 = pc.add(
            table.column("x").cast(pa.float64()),
            pa.scalar(offset_x, type=pa.float64()),
        )
        table = table.set_column(
            table.schema.get_field_index("x"),
            "x",
            float_str_array(x_f64),
        )
    if csv_result.has_y_col:
        y_f64 = pc.add(
            table.column("y").cast(pa.float64()),
            pa.scalar(offset_y, type=pa.float64()),
        )
        table = table.set_column(
            table.schema.get_field_index("y"),
            "y",
            float_str_array(y_f64),
        )

    return table


# ---------------------------------------------------------------------------
# Sopa conflict resolution
# ---------------------------------------------------------------------------


def _stitch_sopa_resolve(
    metadata: PatchGridMetadata,
    geo_results: list[_PatchGeoResult | None],
    csv_results: list[_PatchCsvResult | None],
    all_geojson_features: list[dict],
    all_tables: list[pa.Table],
    threshold: float = 0.5,
) -> set[str]:
    """Stitch per-patch segmentation using spatial containment assignment.

    1. Collect ALL non-empty polygons from all patches (no transcript filtering).
    2. Resolve overlapping polygons via sopa's solve_conflicts().
    3. Assign sequential global cell IDs (cell-1, cell-2, ...).
    4. Spatially assign transcripts to resolved polygons using STRtree.
    5. Noise transcripts (outside all polygons) kept only from their core patch.

    This approach works regardless of whether Baysor's CSV ``cell`` column
    matches GeoJSON cell IDs -- all assignment is done by spatial containment.

    Args:
        metadata: Grid metadata with patch list.
        geo_results: Per-patch GeoJSON results (already in global coords).
        csv_results: Per-patch CSV results.
        all_geojson_features: Output list to append resolved GeoJSON features.
        all_tables: Output list to append processed CSV tables.
        threshold: Overlap threshold for sopa's solve_conflicts (0-1).

    Returns:
        Set of global cell IDs created by merging overlapping cells.
    """
    # --- Phase 1: Collect all polygons from all patches ---
    all_polygons: list = []
    patch_indices_list: list[int] = []

    for i, patch in enumerate(metadata.patches):
        geo_result = geo_results[i]
        if geo_result is None:
            continue

        for feat in geo_result.features:
            polygon = shape(feat["geometry"])
            if polygon.is_empty:
                continue
            if not polygon.is_valid:
                polygon = shapely.make_valid(polygon)
            # Ensure we have a single Polygon (xeniumranger rejects all else)
            polygon = _ensure_polygon(polygon)
            if polygon is None:
                continue

            all_polygons.append(polygon)
            patch_indices_list.append(i)

    if not all_polygons:
        print("[stitch] No polygons found in any patch")
        # Still transform and collect CSVs as noise-only
        for i, patch in enumerate(metadata.patches):
            csv_result = csv_results[i]
            if csv_result is None:
                continue
            offset_x = patch.global_bounds_um.x_min
            offset_y = patch.global_bounds_um.y_min
            transformed = _transform_patch_coords(csv_result, offset_x, offset_y)
            if transformed.num_rows > 0:
                all_tables.append(transformed)
        return set()

    # --- Phase 2: Resolve overlapping polygons via sopa ---
    patch_idx_array = np.array(patch_indices_list, dtype=np.int64)
    input_gdf = gpd.GeoDataFrame(geometry=all_polygons)
    resolved_gdf, kept_indices = solve_conflicts(
        input_gdf,
        threshold=threshold,
        patch_indices=patch_idx_array,
        return_indices=True,
    )

    # --- Phase 3: Assign global cell IDs to resolved polygons ---
    merged_cell_ids: set[str] = set()
    kept_arr = np.asarray(kept_indices)
    resolved_polys: list = []
    resolved_ids: list[str] = []

    for rank, orig_idx in enumerate(kept_arr, start=1):
        global_id = f"cell-{rank}"
        geom = resolved_gdf.geometry.iloc[rank - 1]

        # Ensure single Polygon after solve_conflicts union
        geom = _ensure_polygon(geom)
        if geom is None:
            continue

        if orig_idx < 0:
            merged_cell_ids.add(global_id)

        resolved_polys.append(geom)
        resolved_ids.append(global_id)

        all_geojson_features.append(
            {
                "type": "Feature",
                "id": global_id,
                "geometry": mapping(geom),
                "properties": {"cell_id": global_id},
            }
        )

    print(
        f"[stitch] Resolved {len(all_polygons)} input polygons to "
        f"{len(resolved_polys)} cells ({len(merged_cell_ids)} merged)"
    )

    # --- Phase 4: Spatial transcript assignment via STRtree ---
    poly_tree = shapely.STRtree(resolved_polys)

    for i, patch in enumerate(metadata.patches):
        csv_result = csv_results[i]
        if csv_result is None:
            continue

        offset_x = patch.global_bounds_um.x_min
        offset_y = patch.global_bounds_um.y_min
        core = patch.core_bounds_um

        transformed = _transform_patch_coords(csv_result, offset_x, offset_y)
        if transformed.num_rows == 0:
            continue

        if not csv_result.has_x_col or not csv_result.has_y_col:
            all_tables.append(transformed)
            continue

        # Get global coordinates for spatial query
        gx = transformed.column("x").cast(pa.float64()).to_numpy(zero_copy_only=False)
        gy = transformed.column("y").cast(pa.float64()).to_numpy(zero_copy_only=False)
        points = shapely.points(gx, gy)

        # Query STRtree: returns (input_indices, tree_indices)
        point_hits, poly_hits = poly_tree.query(points, predicate="intersects")

        # Build point -> cell_id mapping (first hit wins)
        point_to_cell: dict[int, str] = {}
        for pt_idx, poly_idx in zip(point_hits, poly_hits):
            if pt_idx not in point_to_cell:
                point_to_cell[pt_idx] = resolved_ids[poly_idx]

        # Build cell and is_noise columns
        n_rows = transformed.num_rows
        cell_arr = [""] * n_rows
        is_noise_arr = ["true"] * n_rows
        for pt_idx, cell_id in point_to_cell.items():
            cell_arr[pt_idx] = cell_id
            is_noise_arr[pt_idx] = "false"

        # Filter noise transcripts to core bounds only
        # Assigned transcripts are kept from all patches (dedup later by transcript_id)
        in_core = (
            (gx >= core.x_min)
            & (gx < core.x_max)
            & (gy >= core.y_min)
            & (gy < core.y_max)
        )
        is_assigned = np.array([c != "" for c in cell_arr])
        keep_mask = pa.array(is_assigned | in_core, type=pa.bool_())

        filtered = transformed.filter(keep_mask)
        cell_arr_filtered = [c for c, k in zip(cell_arr, (is_assigned | in_core)) if k]
        is_noise_filtered = [
            n for n, k in zip(is_noise_arr, (is_assigned | in_core)) if k
        ]

        if filtered.num_rows == 0:
            continue

        # Set cell and is_noise columns
        cell_idx = (
            filtered.schema.get_field_index("cell")
            if "cell" in filtered.column_names
            else None
        )
        if cell_idx is not None:
            filtered = filtered.set_column(
                cell_idx, "cell", pa.array(cell_arr_filtered, type=pa.string())
            )
        else:
            filtered = filtered.append_column(
                "cell", pa.array(cell_arr_filtered, type=pa.string())
            )

        noise_idx = (
            filtered.schema.get_field_index("is_noise")
            if "is_noise" in filtered.column_names
            else None
        )
        if noise_idx is not None:
            filtered = filtered.set_column(
                noise_idx,
                "is_noise",
                pa.array(is_noise_filtered, type=pa.string()),
            )
        else:
            filtered = filtered.append_column(
                "is_noise", pa.array(is_noise_filtered, type=pa.string())
            )

        all_tables.append(filtered)

    return merged_cell_ids


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def stitch_transcript_assignments(
    patches_dir: Path,
    output_dir: Path,
    csv_filename: str = "segmentation.csv",
    geojson_filename: str = "segmentation_polygons.json",
    max_workers: int | None = None,
    min_transcripts_per_cell: int = 0,
) -> None:
    """Stitch per-patch transcript assignments and polygons into unified output.

    For each patch, reads the transcript assignment CSV and polygon GeoJSON.
    Cells are deduplicated using sopa's solve_conflicts() which resolves
    overlapping cells at patch boundaries based on area overlap ratio.

    Processing is split into a parallel I/O phase (reading GeoJSON and CSV
    files via thread pool) and a sequential phase (dedup, global cell ID
    assignment, remapping, and concatenation).

    Args:
        patches_dir: Directory containing patch subdirectories and patch_grid.json.
        output_dir: Output directory for stitched CSV and GeoJSON.
        csv_filename: CSV filename within each patch directory.
        geojson_filename: GeoJSON filename within each patch directory.
        max_workers: Maximum number of threads for parallel I/O.
        min_transcripts_per_cell: Drop cells with fewer transcripts after
            stitching; their transcripts are reassigned to noise and their
            polygons removed (0 = no filter).
    """
    patches_dir = Path(patches_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_grid_metadata(patches_dir / "patch_grid.json")

    n_patches = len(metadata.patches)
    if max_workers is None:
        max_workers = min(n_patches, os.cpu_count() or 1)

    # ---- Parallel phase: read GeoJSON and CSV files concurrently ----
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        geo_futures = [
            executor.submit(
                _read_and_transform_geojson, p, patches_dir, geojson_filename
            )
            for p in metadata.patches
        ]
        csv_futures = [
            executor.submit(_read_patch_csv, p, patches_dir, csv_filename)
            for p in metadata.patches
        ]
    geo_results = [f.result() for f in geo_futures]
    csv_results = [f.result() for f in csv_futures]

    # ---- Sequential phase: assign global cell IDs, remap, concatenate ----
    all_tables: list[pa.Table] = []
    all_geojson_features: list[dict] = []

    _stitch_sopa_resolve(
        metadata,
        geo_results,
        csv_results,
        all_geojson_features,
        all_tables,
        threshold=0.5,
    )

    # Concatenate all patch tables
    if all_tables:
        merged = pa.concat_tables(all_tables)

        # Deduplicate by transcript_id: prefer assigned over noise
        if "transcript_id" in merged.column_names:
            if "cell" in merged.column_names:
                is_noise = pc.equal(merged.column("cell"), "").cast(pa.int8())
                row_order = pa.array(np.arange(merged.num_rows), type=pa.int64())
                sort_table = pa.table({"_noise": is_noise, "_row": row_order})
                sort_indices = pc.sort_indices(
                    sort_table,
                    sort_keys=[("_noise", "ascending"), ("_row", "ascending")],
                )
                merged = merged.take(sort_indices)

            tid_np = merged.column("transcript_id").to_numpy(zero_copy_only=False)
            _, first_indices = np.unique(tid_np, return_index=True)
            first_indices.sort()
            merged = merged.take(first_indices)

        # Post-stitch cell filter: drop cells below min_transcripts_per_cell
        if min_transcripts_per_cell > 0 and "cell" in merged.column_names:
            cell_col = merged.column("cell")
            cell_counts: dict[str, int] = {}
            for c in cell_col.to_pylist():
                if c:
                    cell_counts[c] = cell_counts.get(c, 0) + 1
            small_cells = {
                cid
                for cid, cnt in cell_counts.items()
                if cnt < min_transcripts_per_cell
            }
            if small_cells:
                # Reassign transcripts from small cells to noise
                new_cell = ["" if c in small_cells else c for c in cell_col.to_pylist()]
                new_noise = [
                    "true" if c in small_cells else n
                    for c, n in zip(
                        cell_col.to_pylist(),
                        merged.column("is_noise").to_pylist()
                        if "is_noise" in merged.column_names
                        else ["false"] * merged.num_rows,
                    )
                ]
                cidx = merged.column_names.index("cell")
                merged = merged.set_column(
                    cidx, "cell", pa.array(new_cell, type=pa.string())
                )
                if "is_noise" in merged.column_names:
                    nidx = merged.column_names.index("is_noise")
                    merged = merged.set_column(
                        nidx, "is_noise", pa.array(new_noise, type=pa.string())
                    )
                # Remove filtered cells from GeoJSON
                all_geojson_features[:] = [
                    f
                    for f in all_geojson_features
                    if str(f.get("id", f.get("properties", {}).get("cell_id", "")))
                    not in small_cells
                ]
                print(
                    f"[stitch] Filtered {len(small_cells)} cells with "
                    f"<{min_transcripts_per_cell} transcripts"
                )

        # Log assignment stats
        if "cell" in merged.column_names:
            cell_vals = merged.column("cell").to_pylist()
            n_assigned = sum(1 for c in cell_vals if c)
            n_noise = sum(1 for c in cell_vals if not c)
            print(
                f"[stitch] Final: {merged.num_rows} transcripts, "
                f"{n_assigned} assigned, {n_noise} noise"
            )

        # Cast is_noise to integer for xeniumranger compatibility
        if "is_noise" in merged.column_names:
            noise_col = merged.column("is_noise")
            if noise_col.type == pa.string():
                lower = pc.utf8_lower(noise_col)
                is_true = pc.or_(pc.equal(lower, "true"), pc.equal(lower, "1"))
                idx = merged.column_names.index("is_noise")
                merged = merged.set_column(idx, "is_noise", is_true.cast(pa.int8()))

        # Write CSV
        if merged.num_rows > 0:
            csv_out = output_dir / "xr-transcript-metadata.csv"
            pa_csv.write_csv(
                merged,
                csv_out,
                write_options=pa_csv.WriteOptions(quoting_style="needed"),
            )

    # Safety net: remove orphan polygons with zero transcripts
    if all_geojson_features and all_tables:
        csv_cell_ids: set[str] = set()
        if "cell" in merged.column_names:
            csv_cell_ids = set(c for c in merged.column("cell").to_pylist() if c)
        all_geojson_features = [
            f
            for f in all_geojson_features
            if str(f.get("id", f.get("properties", {}).get("cell_id", "")))
            in csv_cell_ids
        ]

    # Write merged GeoJSON
    if all_geojson_features:
        merged_geo = {"type": "FeatureCollection", "features": all_geojson_features}
        write_geojson(merged_geo, output_dir / "xr-cell-polygons.geojson")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch per-patch Baysor segmentation results into unified output."
    )
    parser.add_argument(
        "--patches",
        type=Path,
        required=True,
        help="Directory containing patch subdirectories and patch_grid.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for stitched CSV and GeoJSON",
    )
    parser.add_argument(
        "--csv-filename",
        default="segmentation.csv",
        help="CSV filename within each patch (default: segmentation.csv)",
    )
    parser.add_argument(
        "--geojson-filename",
        default="segmentation_polygons.json",
        help="GeoJSON filename within each patch (default: segmentation_polygons.json)",
    )
    parser.add_argument(
        "--min-transcripts-per-cell",
        type=int,
        default=0,
        help="Drop cells with fewer transcripts (0 = no filter, default: 0)",
    )
    args = parser.parse_args()

    stitch_transcript_assignments(
        patches_dir=args.patches,
        output_dir=args.output,
        csv_filename=args.csv_filename,
        geojson_filename=args.geojson_filename,
        min_transcripts_per_cell=args.min_transcripts_per_cell,
    )


if __name__ == "__main__":
    main()
