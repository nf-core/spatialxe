"""Streaming aggregation over a Xenium ``transcripts.parquet``.

``molecule_qc_processing.main()`` used to open with

    df_spatial = pd.read_parquet(path, columns=required_columns)

and hold the whole table to the end of the function, then make ~13 passes over
it. On the 1.33 billion-row reference sample that frame costs ~127 GB of RSS
under *any* encoding — measured 117 B/row as ``object`` and 96 B/row
dictionary-encoded — which is what drove the 120 -> 240 -> 480 GB retry ladder.
Note that ``DataFrame.memory_usage(deep=True)`` is badly misleading here: it
over-reports ``object`` (it counts shared ``str`` objects once per reference) and
under-reports ``category`` (it ignores the Arrow dictionary buffers), so only RSS
measured in a subprocess means anything.

Every one of those passes is a reduction. The largest output is per-cell
(~530 k rows); the rest are per-FoV (~900), per-gene (~13.8 k), or scalars. The
two QV violin plots are the only row-level consumers and the previous code
*already* capped them at 10^6 rows each — it just obtained that cap by
materialising 1.33 billion rows and sampling down, i.e. reading 127 GB to keep
0.08 % of it.

Aggregation runs on **pyarrow Acero**, the streaming execution engine already
shipped in the container's pyarrow 21. Acero does the hash aggregation
out-of-core off the parquet scan, so there is no hand-rolled accumulation to get
wrong. Measured on a 20 M-row synthetic file with the production cardinalities
(700 k cells, 13.8 k genes, 900 FoVs): peak RSS 2.0 GB, per-cell aggregate in
1.9 s.

**One deliberate behaviour change.** The violin samples are no longer the same
rows as before. The previous code sampled the complete frame with
``random_state=42``, which is not reproducible without the complete frame.
Sampling here uses the A-Res scheme — draw an independent uniform key per row,
keep the ``k`` smallest — which is an exactly uniform sample of the whole stream,
deterministic for a given seed, and vectorised. The plotted distribution is
equivalent; individual rows differ. Every scalar and aggregate metric is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero

# Rows per record batch for the sampling pass. Large enough that per-batch
# overhead is negligible over ~660 batches, small enough never to be the peak.
DEFAULT_BATCH_ROWS = 2_000_000

# Matches the previous per-violin cap.
DEFAULT_SAMPLE_ROWS = 1_000_000

QV_THRESHOLD = 20.0

# Cap how far the scan may run ahead of the aggregate. Arrow's parallel scan will
# otherwise keep an unbounded number of batches in flight, and the pool grows with
# how far the scan outruns the consumer: measured on identical-schema files the
# default gave 2.24 GB at 20 M rows and 3.80 GB at 60 M (39 B/row and climbing),
# while capping the window gave 1.04 GB and 1.35 GB (7.8 B/row, so roughly 11 GB
# projected at 1.33 B rows against ~145 GB for the eager frame).
#
# Threads stay on: use_threads=False is marginally lower (1.02 GB) but discards
# scan parallelism on CPU-bound work. malloc_trim does not reclaim any of it,
# which confirms the memory is live Arrow buffers, not heap fragmentation.
_SCAN_READAHEAD = {"fragment_readahead": 1, "batch_readahead": 2}


def _aggregate(dataset, scan_columns, project_exprs, project_names, aggregates, keys):
    """Run one streaming hash aggregation off the parquet scan.

    ``scan_columns`` must list the *input* columns the projections reference, not
    the projection output names: a projection over a column the scan did not read
    yields nulls, which surface as NaN in the aggregate rather than as an error.
    """
    return acero.Declaration.from_sequence(
        [
            acero.Declaration(
                "scan",
                acero.ScanNodeOptions(
                    dataset, columns=list(scan_columns), **_SCAN_READAHEAD
                ),
            ),
            acero.Declaration(
                "project", acero.ProjectNodeOptions(project_exprs, project_names)
            ),
            acero.Declaration(
                "aggregate", acero.AggregateNodeOptions(aggregates, keys=keys)
            ),
        ]
    ).to_table(use_threads=True)


@dataclass
class TranscriptStats:
    """Small tables and scalars; nothing here scales with transcript count."""

    n_rows: int
    n_gene_rows: int
    n_unassigned: int
    qv_sum: float
    qv_above_threshold: int
    codeword_counts: pd.Series
    fov: pd.DataFrame  # index fov_name; n, mean_qv, pct_above_qv20, pct_unassigned
    gene: pd.DataFrame  # index feature_name; n_molecules, n_unassigned, is_gene
    cell_nucleus_fraction: pd.Series  # index cell_id (assigned cells only)

    @property
    def mean_qv(self) -> float:
        return self.qv_sum / self.n_rows if self.n_rows else float("nan")

    @property
    def pct_qv_above_threshold(self) -> float:
        if not self.n_rows:
            return float("nan")
        return 100.0 * self.qv_above_threshold / self.n_rows

    @property
    def pct_unassigned(self) -> float:
        if not self.n_rows:
            return float("nan")
        return 100.0 * self.n_unassigned / self.n_rows

    @property
    def n_features(self) -> int:
        return len(self.gene)


def aggregate_transcripts(path, background_cell_id: str) -> TranscriptStats:
    """Stream the transcript table, returning only small aggregates."""
    dataset = ds.dataset(str(path), format="parquet")

    unassigned = (pc.field("cell_id") == background_cell_id).cast(pa.int64())
    above = (pc.field("qv") > QV_THRESHOLD).cast(pa.int64())

    fov_tbl = _aggregate(
        dataset,
        ["fov_name", "qv", "cell_id"],
        # qv is float32 on disk; cast to float64 so the sum accumulates in double
        # precision, matching pandas .mean() which upcasts. In float32 the FoV
        # means differ from pandas in the 7th significant digit.
        [pc.field("fov_name"), pc.field("qv").cast(pa.float64()), above, unassigned],
        ["fov_name", "qv", "above", "unassigned"],
        [
            ("qv", "hash_count", None, "n"),
            ("qv", "hash_sum", None, "qv_sum"),
            ("above", "hash_sum", None, "n_above"),
            ("unassigned", "hash_sum", None, "n_unassigned"),
        ],
        ["fov_name"],
    )
    fov = fov_tbl.to_pandas().set_index("fov_name").sort_index()
    fov["mean_qv"] = fov["qv_sum"] / fov["n"]
    fov["pct_above_qv20"] = 100.0 * fov["n_above"] / fov["n"]
    fov["pct_unassigned"] = 100.0 * fov["n_unassigned"] / fov["n"]

    gene_tbl = _aggregate(
        dataset,
        ["feature_name", "is_gene", "cell_id"],
        [
            pc.field("feature_name"),
            pc.field("is_gene").cast(pa.int64()),
            unassigned,
        ],
        ["feature_name", "is_gene", "unassigned"],
        [
            ("is_gene", "hash_count", None, "n_molecules"),
            # is_gene is a property of the feature, so max == the value.
            ("is_gene", "hash_max", None, "is_gene"),
            ("unassigned", "hash_sum", None, "n_unassigned"),
        ],
        ["feature_name"],
    )
    gene = gene_tbl.to_pandas().set_index("feature_name").sort_index()
    gene["is_gene"] = gene["is_gene"].astype(bool)

    cw_tbl = _aggregate(
        dataset,
        ["codeword_category"],
        [pc.field("codeword_category")],
        ["codeword_category"],
        [("codeword_category", "hash_count", None, "n")],
        ["codeword_category"],
    )
    codeword = (
        cw_tbl.to_pandas()
        .set_index("codeword_category")["n"]
        .sort_values(ascending=False)
    )

    # Per-cell nucleus fraction, assigned cells only. Filter before aggregating so
    # the sentinel never becomes a group.
    cell_tbl = acero.Declaration.from_sequence(
        [
            acero.Declaration(
                "scan",
                acero.ScanNodeOptions(
                    dataset,
                    columns=["cell_id", "overlaps_nucleus"],
                    **_SCAN_READAHEAD,
                ),
            ),
            acero.Declaration(
                "filter",
                acero.FilterNodeOptions(pc.field("cell_id") != background_cell_id),
            ),
            acero.Declaration(
                "project",
                acero.ProjectNodeOptions(
                    [
                        pc.field("cell_id"),
                        pc.field("overlaps_nucleus").cast(pa.int64()),
                    ],
                    ["cell_id", "ov"],
                ),
            ),
            acero.Declaration(
                "aggregate",
                acero.AggregateNodeOptions(
                    [
                        ("ov", "hash_sum", None, "ov_sum"),
                        ("ov", "hash_count", None, "n"),
                    ],
                    keys=["cell_id"],
                ),
            ),
        ]
    ).to_table(use_threads=True)
    cell = cell_tbl.to_pandas().set_index("cell_id").sort_index()
    fraction = (cell["ov_sum"] / cell["n"]).astype(float)

    # Count gene rows directly. Deriving it from the per-gene table would assume
    # is_gene is a function of feature_name; true in Xenium data, but a scalar
    # aggregate is exact either way.
    gene_rows_tbl = acero.Declaration.from_sequence(
        [
            acero.Declaration(
                "scan",
                acero.ScanNodeOptions(dataset, columns=["is_gene"], **_SCAN_READAHEAD),
            ),
            acero.Declaration(
                "project",
                acero.ProjectNodeOptions(
                    [pc.field("is_gene").cast(pa.int64())], ["is_gene"]
                ),
            ),
            acero.Declaration(
                "aggregate",
                acero.AggregateNodeOptions([("is_gene", "sum", None, "n_gene")]),
            ),
        ]
    ).to_table()

    n_rows = int(fov["n"].sum())
    return TranscriptStats(
        n_rows=n_rows,
        n_gene_rows=int(gene_rows_tbl.column("n_gene")[0].as_py()),
        n_unassigned=int(fov["n_unassigned"].sum()),
        qv_sum=float(fov["qv_sum"].sum()),
        qv_above_threshold=int(fov["n_above"].sum()),
        codeword_counts=codeword,
        fov=fov[["n", "mean_qv", "pct_above_qv20", "pct_unassigned"]],
        gene=gene[["n_molecules", "n_unassigned", "is_gene"]],
        cell_nucleus_fraction=fraction,
    )


def sample_rows(
    path,
    columns: list[str],
    predicate_column: str = "is_gene",
    k: int = DEFAULT_SAMPLE_ROWS,
    seed: int = 42,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform samples of up to *k* rows where *predicate_column* is True / False.

    A-Res: give every row an independent uniform key and keep the ``k`` smallest.
    That is an exactly uniform sample of the stream, needs only ``k`` rows
    resident per group, and is deterministic for a given seed.
    """
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    keep = {True: (None, None), False: (None, None)}  # group -> (keys, frame)
    wanted = list(dict.fromkeys([*columns, predicate_column]))

    handle = pq.ParquetFile(str(path))
    for batch in handle.iter_batches(batch_size=batch_rows, columns=wanted):
        df = batch.to_pandas()
        del batch
        flag = df[predicate_column].to_numpy(dtype=bool, copy=False)
        for group in (True, False):
            sel = df.loc[flag if group else ~flag, columns]
            if sel.empty:
                continue
            new_keys = rng.random(len(sel))
            keys, frame = keep[group]
            if keys is None:
                keys, frame = new_keys, sel.reset_index(drop=True)
            else:
                keys = np.concatenate([keys, new_keys])
                frame = pd.concat([frame, sel], ignore_index=True)
            if len(keys) > k:
                take = np.argpartition(keys, k)[:k]
                keys = keys[take]
                frame = frame.iloc[take].reset_index(drop=True)
            keep[group] = (keys, frame)
        del df

    out = []
    for group in (True, False):
        _, frame = keep[group]
        out.append(frame if frame is not None else pd.DataFrame(columns=columns))
    return out[0], out[1]
