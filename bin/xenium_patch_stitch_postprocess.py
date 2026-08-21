#!/usr/bin/env python3
"""
Post-process stitched per-patch segmentation outputs.

Ensures every GeoJSON feature is a single Polygon: make_valid() and
sopa.solve_conflicts() can produce MultiPolygon, MultiLineString, or
GeometryCollection geometries that XeniumRanger rejects. Cells dropped
during cleanup are also reassigned to UNASSIGNED in the transcript CSV
so the two outputs stay consistent.
"""

import argparse
import csv
import json

import shapely
from shapely.geometry import mapping, shape


def clean_geojson(geojson_path: str) -> set:
    """
    Force every feature to a single valid Polygon.

    Returns the set of cell ids whose features were dropped.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    clean = []
    dropped_cells = set()
    for feat in data["features"]:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        poly = None
        if geom.geom_type == "Polygon":
            poly = geom
        elif geom.geom_type == "MultiPolygon":
            poly = max(geom.geoms, key=lambda g: g.area)
        elif geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
            if polys:
                poly = max(polys, key=lambda g: g.area)
        if poly is not None and not poly.is_empty:
            feat["geometry"] = mapping(poly)
            clean.append(feat)
        else:
            cell_id = feat.get("id") or feat.get("properties", {}).get("cell_id", "")
            dropped_cells.add(str(cell_id))

    print(f"GeoJSON: {len(clean)} kept, {len(dropped_cells)} dropped: {dropped_cells}")
    data["features"] = clean
    with open(geojson_path, "w") as f:
        json.dump(data, f)

    return dropped_cells


def reassign_dropped(csv_path: str, dropped_cells: set) -> None:
    """
    Reassign transcripts of dropped cells to UNASSIGNED in the CSV.
    """
    if not dropped_cells:
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        raise ValueError(
            f"No header row in {csv_path}; cannot rewrite transcript assignments"
        )

    reassigned = 0
    for row in rows:
        if row["cell"] in dropped_cells:
            row["cell"] = ""
            row["is_noise"] = "1"
            reassigned += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {reassigned} transcripts reassigned to UNASSIGNED")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean stitched GeoJSON polygons and reconcile transcript CSV."
    )
    parser.add_argument(
        "--geojson", required=True, help="Path to xr-cell-polygons.geojson"
    )
    parser.add_argument(
        "--csv", required=True, help="Path to xr-transcript-metadata.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dropped = clean_geojson(args.geojson)
    reassign_dropped(args.csv, dropped)
