#!/usr/bin/env python3

"""
Estimate a recommended Baysor --scale parameter from Xenium transcripts.
"""

import argparse
import math
from pathlib import Path

import numpy as np # type: ignore
import pandas as pd # type: ignore
from numpy.typing import NDArray # type: ignore
from pandas import DataFrame, Series # type: ignore
from scipy.spatial import cKDTree # type: ignore

from utils.utils_logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Baysor --scale from Xenium transcript coordinates."
    )

    parser.add_argument(
        "--transcripts",
        required=True,
        type=Path,
        help="Path to transcripts file (.parquet or .csv)",
    )
    parser.add_argument(
        "--cell-column",
        default=None,
        help=(
            "Column name for cell IDs. Omit when no prior segmentation is "
            "available; the scale will instead be estimated from local "
            "transcript density (k-nearest-neighbor distances)."
        ),
    )
    parser.add_argument("--prefix", default="", help="Prefix for output files")
    parser.add_argument("--x-column", default="x_location", help="Column name for x coordinates")
    parser.add_argument("--y-column", default="y_location", help="Column name for y coordinates")

    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile for cell radius calculation")
    parser.add_argument("--min-transcripts", type=int, default=10, help="Minimum number of transcripts per cell")

    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors to use for density-based radius estimation (no-prior case)",
    )

    parser.add_argument("--max-scale", type=float, default=30.0, help="Maximum scale value")
    parser.add_argument("--min-scale", type=float, default=3.0, help="Minimum scale value")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def median_absolute_deviation(values: NDArray[np.floating]) -> float:
    median: float = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def trim_outliers_hybrid(
    values: np.ndarray,
    n_mad: float = 5.0,
    q: float = 0.99,
) -> np.ndarray:
    """
    Robust trimming:
    1. MAD-based filtering (removes extreme structural outliers)
    2. Quantile cap (removes residual tail)
    """

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad > 0:
        values = values[np.abs(values - median) <= n_mad * mad]

    if len(values) == 0:
        return values

    upper = np.quantile(values, q)
    values = values[values <= upper]

    return values


def compute_cell_radius(
    cell: DataFrame,
    x_col: str,
    y_col: str,
    percentile: float,
) -> float:
    """
    Compute the radius of a cell based on the specified percentile of distances
    from the cell centroid.
    """
    centroid_x: float = float(cell[x_col].mean())
    centroid_y: float = float(cell[y_col].mean())

    dx: Series = cell[x_col] - centroid_x
    dy: Series = cell[y_col] - centroid_y

    distances: NDArray[np.floating] = np.sqrt(dx.to_numpy() ** 2 + dy.to_numpy() ** 2)

    return float(np.percentile(distances, percentile))


def clean_cell_ids(df: DataFrame, col: str) -> DataFrame:
    """
    Clean cell IDs in the specified column of the DataFrame.
    Removes invalid or missing cell IDs, and converts numeric-like IDs to integers."""
    df = df.copy()

    sample = df[col].dropna().astype(str).head(100)
    is_numeric_like = sample.str.fullmatch(r"-?\d+").mean() > 0.8

    if is_numeric_like:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].notna()].copy() # type: ignore
        df = df[df[col] >= 0].copy() # type: ignore

    else:
        df[col] = df[col].astype(str)

        df = df[
            (df[col] != "-1") &
            (df[col].str.lower() != "nan") &
            (df[col].str.lower() != "none")
        ].copy() # type: ignore

    return df


def load_transcripts(path: Path) -> DataFrame:
    """
    Load transcripts from a Parquet or CSV file.
    """

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported transcript file format '{suffix}'. "
        "Supported formats are .parquet and .csv."
    )


def estimate_radii_from_cells(
    df: DataFrame,
    cell_col: str,
    x_col: str,
    y_col: str,
    percentile: float,
    min_transcripts: int,
) -> tuple[list[float], int, int]:
    """
    Per-cell radius estimation using prior segmentation (column-based).
    """

    df = clean_cell_ids(df, cell_col)

    if df.empty:
        raise RuntimeError("No assigned transcripts found after cleaning.")

    radii: list[float] = []
    total_cells = 0
    used_cells = 0

    grouped = df.groupby(cell_col)

    for _, cell in grouped:

        total_cells += 1

        if len(cell) < min_transcripts:
            continue

        used_cells += 1

        radius: float = compute_cell_radius(cell, x_col, y_col, percentile)
        radii.append(radius)

    return radii, total_cells, used_cells


def estimate_radii_from_density(
    df: DataFrame,
    x_col: str,
    y_col: str,
    percentile: float,
    k_neighbors: int,
) -> tuple[list[float], int, int]:
    """
    Density-based radius estimation for the no-prior case.

    Uses the distance to the k-th nearest neighbor of each transcript as a
    proxy for local point density, then converts that to a proxy "cell
    radius" via the requested percentile. This avoids requiring per-cell
    grouping when no prior segmentation (cell assignment) is available.
    """

    coords = df[[x_col, y_col]].to_numpy()

    if len(coords) <= k_neighbors:
        raise RuntimeError(
            f"Not enough transcripts ({len(coords)}) for k={k_neighbors} "
            "nearest-neighbor density estimation."
        )

    tree = cKDTree(coords)
    # k_neighbors + 1 because the nearest neighbor of a point is itself (distance 0)
    distances, _ = tree.query(coords, k=k_neighbors + 1)
    knn_distances = distances[:, -1]

    radius = float(np.percentile(knn_distances, percentile))

    total_cells = len(coords)
    used_cells = len(coords)

    return [radius], total_cells, used_cells


def main() -> None:

    args = parse_args()

    df: DataFrame = load_transcripts(args.transcripts)

    required_cols: list[str] = [args.x_column, args.y_column]
    if args.cell_column:
        required_cols.append(args.cell_column)

    missing: list[str] = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.cell_column:
        radii, total_cells, used_cells = estimate_radii_from_cells(
            df,
            args.cell_column,
            args.x_column,
            args.y_column,
            args.percentile,
            args.min_transcripts,
        )
    else:
        radii, total_cells, used_cells = estimate_radii_from_density(
            df,
            args.x_column,
            args.y_column,
            args.percentile,
            args.knn_neighbors,
        )

    if len(radii) == 0:
        raise RuntimeError(
            "No cells passed filtering. Try lowering --min-transcripts."
        )

    radii_arr: NDArray[np.floating] = np.asarray(radii)

    radii_arr = trim_outliers_hybrid(radii_arr)

    if len(radii_arr) == 0:
        raise RuntimeError("All radii removed by outlier filtering.")

    median_radius: float = float(np.median(radii_arr))
    mean_radius: float = float(np.mean(radii_arr))
    std_radius: float = float(np.std(radii_arr))
    min_radius: float = float(np.min(radii_arr))
    max_radius: float = float(np.max(radii_arr))

    mad: float = median_absolute_deviation(radii_arr)

    percentile_25: float = float(np.percentile(radii_arr, 25))
    percentile_75: float = float(np.percentile(radii_arr, 75))
    percentile_90: float = float(np.percentile(radii_arr, 90))
    percentile_95: float = float(np.percentile(radii_arr, 95))

    recommended_scale: float = math.ceil(
        float(
            np.clip(
                median_radius,
                args.min_scale,
                args.max_scale,
            )
        )
    )

    if (args.verbose):
        logger.info("========== Baysor Scale Estimation ==========")

        logger.info(f"Transcript file          : {args.transcripts}")
        logger.info(f"Estimation mode          : {'cell-based (prior)' if args.cell_column else 'density-based (no prior)'}")
        logger.info(f"Cells detected           : {total_cells}")
        logger.info(f"Cells used               : {used_cells}")
        logger.info(f"Minimum transcripts/cell : {args.min_transcripts}")
        logger.info(f"Radius percentile        : {args.percentile}")

        logger.info("Transcript-cloud radius statistics (µm)")
        logger.info("----------------------------------------")
        logger.info(f"Median : {median_radius:.2f}")
        logger.info(f"Mean   : {mean_radius:.2f}")
        logger.info(f"MAD    : {mad:.2f}")
        logger.info(f"Std    : {std_radius:.2f}")
        logger.info(f"Min    : {min_radius:.2f}")
        logger.info(f"25%    : {percentile_25:.2f}")
        logger.info(f"75%    : {percentile_75:.2f}")
        logger.info(f"90%    : {percentile_90:.2f}")
        logger.info(f"95%    : {percentile_95:.2f}")
        logger.info(f"Max    : {max_radius:.2f}")

        logger.info("=========================================")
        logger.info(f"Recommended Baysor --scale : {recommended_scale:.2f} µm")
        logger.info("=========================================")

    print(f"{recommended_scale:.2f}", file=open(f"{args.prefix}_scale_factor.txt", "w"))


if __name__ == "__main__":
    main()
