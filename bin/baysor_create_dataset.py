#!/usr/bin/env python3
"""
Create a sampled dataset for Baysor preview mode.

Reads a CSV transcript file and randomly samples a fraction of rows,
writing the result to a new CSV file.
"""

import argparse
import csv
import os
import random
from pathlib import Path


class BaysorPreview():
    """
    Utility class to generate baysor preview dataset
    """
    @staticmethod
    def generate_dataset(
            transcripts: Path,
            sampled_transcripts: Path,
            sample_fraction: float = 0.3,
            random_state: int = 42,
            prefix: str = ""
        ) -> None:
        """
        Reads a csv file & randomly samples a fraction of rows,
        and writes the result to a .csv file.

        Args:
            transcripts: unziped transcripts.csv from xenium bundle
            sampled_transcripts: randomly subsampled transcripts.csv file
            sample_fraction: Fraction of rows to sample
            random_state: Seed for reproducibility
            prefix: Output directory prefix
        """

        random.seed(random_state)
        output_path = f"{prefix}/{sampled_transcripts}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(transcripts, mode='rt', newline='') as infile, \
            open(output_path, mode='wt', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            writer.writerow(header)

            for row in reader:
                if random.random() < float(sample_fraction):
                    writer.writerow(row)

        return None


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Create sampled dataset for Baysor preview"
    )
    parser.add_argument(
        "--transcripts", required=True,
        help="Path to transcripts CSV file"
    )
    parser.add_argument(
        "--sample-fraction", required=True, type=float,
        help="Fraction of rows to sample"
    )
    parser.add_argument(
        "--prefix", required=True,
        help="Output directory prefix"
    )
    args = parser.parse_args()

    sampled_transcripts = Path("sampled_transcripts.csv")

    # generate dataset
    BaysorPreview.generate_dataset(
        transcripts=Path(args.transcripts),
        sampled_transcripts=sampled_transcripts,
        sample_fraction=args.sample_fraction,
        prefix=args.prefix
    )

    return None


if __name__ == "__main__":
    main()
