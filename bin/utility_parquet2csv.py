#!/usr/bin/env python3
"""
Convert a Parquet file to CSV format.

Reads a Parquet file and writes it as CSV, optionally gzip-compressed.
"""

import argparse
from pathlib import Path

import pandas as pd


def convert_parquet(
    transcripts: str,
    extension: str = ".csv",
    prefix: str = "",
) -> None:
    """
    Convert a Parquet file to CSV or CSV.GZ format.

    Args:
        transcripts: Filename of the input parquet file
        extension: Output extension ('.csv' or '.gz' for gzip)
        prefix: Output directory prefix
    """
    df = pd.read_parquet(transcripts, engine="pyarrow")

    Path(prefix).mkdir(parents=True, exist_ok=True)

    if extension == ".gz":
        output = transcripts.replace(".parquet", ".csv.gz")
        df.to_csv(f"{prefix}/{output}", compression="gzip", index=False)
    else:
        output = transcripts.replace(".parquet", ".csv")
        df.to_csv(f"{prefix}/{output}", index=False)

    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file to CSV format."
    )
    parser.add_argument(
        "--transcripts",
        required=True,
        help="Input parquet filename",
    )
    parser.add_argument(
        "--extension",
        default=".csv",
        help="Output extension: '.csv' or '.gz' (default: .csv)",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output directory prefix (sample ID)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_parquet(
        transcripts=args.transcripts,
        extension=args.extension,
        prefix=args.prefix,
    )
