#!/usr/bin/env python3

"""
Transcript QC Processing Module
Performs all analysis from notebooks/1_qc_molecule.ipynb and generates figures and metrics.
Authors: Malwina Prater, mprater@altoslabs.com; Dongze He, dhe@altoslabs.com; Felix Krueger, fkrueger@altoslabs.com
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import transcript_stream
import scanpy as sc
import seaborn as sns

# ===========================================================================
# Vendored VERBATIM from upstream ``xenium_helpers.utils``
# (bin/xenium_helpers/src/xenium_helpers/utils.py) at nf-xenium-processing dev
# HEAD 5e35cae. This pipeline does not ship the xenium_helpers package, so the
# helpers this script needs are inlined here with their bodies unchanged.
# Re-sync: re-copy these definitions from that file; do NOT rename symbols
# inside this block (including the "mols"/"molecule" names) so it stays a
# mechanical copy.
# ===========================================================================


def calculate_noise_bound(
    n_molecules_non_gene_prefix: pd.Series, quant: float = 0.99
) -> tuple[float, float]:
    """Calculate the noise bounds based on the non-gene molecules."""
    from scipy.stats import median_abs_deviation, norm

    if n_molecules_non_gene_prefix.empty:
        return 0, 0

    quant_val = norm.ppf(quant)
    n_mols_log = np.log10(n_molecules_non_gene_prefix.values)
    std = median_abs_deviation(n_mols_log, scale="normal")
    noise_lb = np.mean(n_mols_log) - quant_val * std
    noise_ub = np.mean(n_mols_log) + quant_val * std
    return 10**noise_lb, 10**noise_ub


def estimate_min_mols_per_cell(n_mols_per_cell: List[int], min_value: int = 10):
    n_mols_per_cell = np.log10(np.asarray(n_mols_per_cell) + 1)
    nm_hist = np.histogram(n_mols_per_cell, bins=100)
    mode = nm_hist[1][nm_hist[0].argmax()]
    ci = np.quantile(n_mols_per_cell[n_mols_per_cell > mode], 0.99) - mode
    return max(min_value, int(round(10 ** (mode - ci))))


def format_yaml_like(data: dict, indent: int = 0) -> str:
    # Borrowed from nf-core
    """Formats a dictionary to a YAML-like string.
    Args:
        data (dict): The dictionary to format.
        indent (int): The current indentation level.
    Returns:
        str: A string formatted as YAML.
    """
    yaml_str = ""
    for key, value in data.items():
        spaces = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{spaces}{key}:\n{format_yaml_like(value, indent + 1)}"
        else:
            yaml_str += f"{spaces}{key}: {value}\n"
    return yaml_str


def dump_versions(
    file_path: str,
    packages: List[str],
    task_name: Optional[str] = None,
    show: bool = True,
    include_python: bool = True,
):
    # Inspired by nf-core
    from importlib.metadata import version
    import platform

    versions: Dict[str, str] = {k: version(k) for k in packages}

    if include_python:
        versions["python"] = platform.python_version()

    nested: Dict[str, Any] = (
        {task_name: versions} if task_name is not None else versions
    )
    versions_yaml: str = format_yaml_like(nested)

    if show:
        print(versions_yaml)

    with open(file_path, "w") as f:
        f.write(versions_yaml)


# ---------------------------------------------------------------------------
# Segmentation-software provenance for QC reports.
# ---------------------------------------------------------------------------

# Display names for pipeline resegmentation tools (raw params.segmentation
# value -> human-readable). Used as the name-only fallback when no parsed tool
# version is available.
SEGMENTATION_PRETTY = {
    "cellpose": "Cellpose",
    "cellpose_baysor": "Cellpose + Baysor",
    "proseg": "Proseg",
    "segger": "Segger",
}

# Per-method component tools as (display name, versions.yml key) pairs. The key
# is the tool name as it appears inside the segmentation modules' versions.yml
# (e.g. ``cellpose: 3.0.6``). Order defines how multi-tool labels read.
SEGMENTATION_TOOL_KEYS = {
    "cellpose": [("Cellpose", "cellpose")],
    "cellpose_baysor": [("Cellpose", "cellpose"), ("Baysor", "baysor")],
    "proseg": [("Proseg", "proseg")],
    "segger": [("Segger", "segger")],
}


def parse_tool_versions(version_files) -> Dict[str, str]:
    """Union a list of nf-core ``versions.yml`` files into a flat
    ``{tool: version}`` map. Each file maps ``process -> {tool: version}``; we
    flatten across processes (later entries win). Safe-fails per file. Versions
    are run-global (one tool version per pipeline run), so collecting across all
    samples/processes and flattening is correct."""
    import yaml  # local import — pyyaml is a runtime dep, not needed at import

    out: Dict[str, str] = {}
    for path in version_files or []:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        for tools in data.values():
            if isinstance(tools, dict):
                for tool, version in tools.items():
                    if version is not None:
                        out[str(tool)] = str(version).strip()
    return out


def _tool_label(display: str, key: str, tool_versions: Optional[Dict[str, str]]) -> str:
    """``"Cellpose"`` + version -> ``"Cellpose v3.0.6"`` (name only if absent)."""
    version = (tool_versions or {}).get(key)
    return f"{display} v{version}" if version else display


def read_xenium_analysis_sw_version(bundle_dir) -> Optional[str]:
    """Read ``analysis_sw_version`` from ``experiment.xenium`` (e.g.
    ``"xenium-4.0.1.0"``). Returns ``None`` on missing file, missing key, or
    malformed JSON. Mirrors ``read_xenium_pixel_size_um`` in bin/snr_metrics.py.
    """
    exp = Path(bundle_dir) / "experiment.xenium"
    if not exp.is_file():
        return None
    try:
        with open(exp, encoding="utf-8") as f:
            meta = json.load(f)
        version = meta.get("analysis_sw_version")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not version or not isinstance(version, str):
        return None
    return version


def _parse_xenium_version(analysis_sw_version: Optional[str]) -> Optional[str]:
    """``"xenium-4.0.1.0"`` -> ``"4.0.1"`` (major.minor.patch). Returns ``None``
    if no leading numeric components can be parsed."""
    if not analysis_sw_version:
        return None
    tail = analysis_sw_version.split("-", 1)[-1]  # drop a 'xenium-' style prefix
    nums = []
    for part in tail.split("."):
        if part.isdigit():
            nums.append(part)
        else:
            break
    if not nums:
        return None
    return ".".join(nums[:3])


def resolve_segmentation_software(
    bundle_dir,
    pipeline_segmentation: str = "skip",
    is_resegmented: bool = False,
    tool_versions: Optional[Dict[str, str]] = None,
) -> str:
    """Human-readable label for the segmentation software that produced the
    bundle a QC report describes.

    - Un-resegmented / pre-seg / ``skip``: the onboard analysis version from the
      bundle's ``experiment.xenium`` -> ``"Xenium Onboard Analysis v4.0.1"``.
    - Pipeline ``xr`` resegmentation: the reseg bundle's own
      ``analysis_sw_version`` -> ``"Xenium Ranger v4.0.1 (resegmentation)"``.
    - Other pipeline tools (cellpose / cellpose_baysor / proseg / segger): the
      tool name plus its version from ``tool_versions`` (parsed from the
      segmentation ``versions.yml``), e.g. ``"Cellpose v3.0.6"`` or
      ``"Cellpose v3.0.6 + Baysor v0.6.2"``. Falls back to name-only when the
      version is unavailable. Their reseg bundle is packaged via ``xeniumranger
      import-segmentation``, so its ``experiment.xenium`` would mislabel them as
      Xenium Ranger; the pipeline tool name is authoritative here.
    """
    seg = (pipeline_segmentation or "skip").strip()
    parsed = _parse_xenium_version(read_xenium_analysis_sw_version(bundle_dir))

    if not is_resegmented or seg == "skip":
        if parsed:
            return f"Xenium Onboard Analysis v{parsed}"
        return "Xenium Onboard Analysis (version unknown)"

    if seg == "xr":
        if parsed:
            return f"Xenium Ranger v{parsed} (resegmentation)"
        return "Xenium Ranger (resegmentation)"

    components = SEGMENTATION_TOOL_KEYS.get(seg)
    if components:
        return " + ".join(
            _tool_label(display, key, tool_versions) for display, key in components
        )
    return SEGMENTATION_PRETTY.get(seg, seg)


# ===========================================================================
# End vendored xenium_helpers.utils block.
# ===========================================================================

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

# Transcript-to-cell assignment sentinel: the cell_id value for transcripts
# that decoding placed but segmentation did not assign to any cell. Distinct
# from the "unassigned" codeword decoding category. Confirmed as the literal
# string "UNASSIGNED" on real Xenium bundles.
BACKGROUND_CELL_ID = "UNASSIGNED"

# Minimum observed transcript count for a gene to enter the per-gene
# unassigned-fraction breakdown (genes below this are too sparse for a stable
# fraction). Provisional — calibrate in the Phase B triage spike.
MIN_GENE_COUNT_FOR_UNASSIGNED = 10

# Cap on the per-FoV quality figure height, in inches.
_FOV_FIG_MAX_HEIGHT_IN = 40


def _min_count_threshold(counts) -> int | None:
    """Call-site guard for the vendored ``estimate_min_mols_per_cell``.

    That function is inside the vendored block and must stay byte-identical to
    upstream, so the degenerate-input check lives here instead. It does
    ``np.quantile(x[x > mode], 0.99)`` where ``mode`` is the left edge of the
    modal histogram bin; for a constant (or empty) input the modal bin's left
    edge equals the only value, the upper-tail slice is empty and numpy raises.
    Since the bin edges are strictly increasing, ``max > mode`` always holds for
    any non-constant input — so ``size == 0 or min == max`` is the exact
    degenerate condition, not a heuristic.

    Reachable with zero cells, one cell, or zero retained genes (all counts 0).
    Returns None in that case, meaning "threshold not computable": callers skip
    the plot cutoff line and emit JSON ``null``, which the report already
    renders as absent. Deliberately not a numeric fallback — any number here
    would read as a real noise floor that empty data had passed.
    """
    arr = np.asarray(counts, dtype=float).ravel()
    if arr.size == 0 or arr.min() == arr.max():
        return None
    return estimate_min_mols_per_cell(arr)


def _csv_float(value):
    """Parse a metrics_summary.csv cell to float, or None when blank/unparseable.
    Keeps "absent / blank" (None) distinct from a real 0.0 value. pandas reads a
    blank cell as float NaN, so guard both NaN and the literal string 'nan'."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _csv_str(value):
    """Parse a metrics_summary.csv cell to a non-empty string, or None. Treats a
    blank cell (pandas NaN → 'nan') as absent, so an empty stain_definition is
    correctly None rather than the string 'nan'."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def read_bundle_metrics(bundle_dir, is_resegmented):
    """Read the three 10x-defined QC metrics from the bundle's
    ``metrics_summary.csv`` (single data row). Returns a dict with float-or-None
    values plus a ``source`` tag.

    Fail-soft to None when the file is missing, the column is absent or blank.
    On resegmented bundles the CSV reflects the ONBOARD segmentation, not the
    pipeline's resegmentation, so the segmentation-dependent metrics are not
    meaningful — everything is set to None (N/A) in that case.

    ``stain_definition`` is carried through so the report can tell apart a run
    that used stain segmentation (gate stain_frac) from a nucleus-expansion run
    where stain_frac == 0 by design (informational, not a WARN).
    """
    out = {
        "nuclear_transcripts_per_100um2": None,
        "fraction_empty_cells": None,
        "segmented_cell_stain_frac": None,
        "stain_definition": None,
        "source": None,
    }
    if is_resegmented:
        out["source"] = (
            "n/a (resegmented bundle; metrics_summary.csv reflects onboard segmentation)"
        )
        return out
    csv_path = Path(bundle_dir) / "metrics_summary.csv"
    if not csv_path.is_file():
        return out
    try:
        row = pd.read_csv(csv_path, nrows=1).iloc[0].to_dict()
    except (OSError, ValueError, pd.errors.ParserError, IndexError):
        return out
    out["nuclear_transcripts_per_100um2"] = _csv_float(
        row.get("nuclear_transcripts_per_100um2")
    )
    out["fraction_empty_cells"] = _csv_float(row.get("fraction_empty_cells"))
    out["segmented_cell_stain_frac"] = _csv_float(row.get("segmented_cell_stain_frac"))
    out["stain_definition"] = _csv_str(row.get("stain_definition"))
    out["source"] = "metrics_summary.csv"
    return out


# ---------------------------------------------------------------------------
# Host memory instrumentation
#
# This module reports progress with print(), which Python block-buffers when
# stdout is not a tty. An OOM SIGKILL therefore discards every buffered line:
# a killed task's log contains the kill message and nothing else, which is why
# this OOM had been "addressed" by climbing the memory ladder rather than
# diagnosed. The module .nf now exports PYTHONUNBUFFERED=1, and the stage
# markers below report where memory actually goes.
#
# Duplicated from image_qc.py rather than shared via xenium_helpers on purpose:
# module binaries ship from the repo through moduleBinaries, but xenium_helpers
# is pip-installed into the container image, so a helper placed there would not
# take effect until the image is rebuilt. Consolidate when the image is bumped.
# ---------------------------------------------------------------------------

_GIB = float(1024**3)
_MEM_PEAK = {"working_set": 0.0, "rss": 0.0}

# (usage file, stat file, inactive key, active key) for cgroup v2 then v1.
_CGROUP_SOURCES = (
    (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.stat",
        "inactive_file",
        "active_file",
    ),
    (
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.stat",
        "total_inactive_file",
        "total_active_file",
    ),
)


def _cgroup_memory():
    """``(working_set_bytes, page_cache_bytes)`` for this cgroup, or None.

    The usage counter includes reclaimable page cache; ``usage - inactive_file``
    is what actually drives an OOM kill. Report both so they are never confused
    when sizing the `processing` label.
    """
    for usage_path, stat_path, inactive_key, active_key in _CGROUP_SOURCES:
        try:
            with open(usage_path) as fh:
                usage = float(fh.read().strip())
            inactive_file = 0.0
            active_file = 0.0
            with open(stat_path) as fh:
                for line in fh:
                    key, _, value = line.partition(" ")
                    if key == inactive_key:
                        inactive_file = float(value)
                    elif key == active_key:
                        active_file = float(value)
        except (OSError, ValueError):
            continue
        return max(usage - inactive_file, 0.0), inactive_file + active_file
    return None


def _process_rss():
    """Peak RSS of this process in bytes (VmHWM), excluding children."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _log_mem(stage, extra=""):
    """Print host memory at a stage boundary and track the high-water mark."""
    parts = []
    cgroup = _cgroup_memory()
    if cgroup is not None:
        working_set, page_cache = cgroup
        _MEM_PEAK["working_set"] = max(_MEM_PEAK["working_set"], working_set)
        parts.append(
            f"working_set={working_set / _GIB:.1f}GB "
            f"(+{page_cache / _GIB:.1f}GB reclaimable page cache)"
        )
    rss = _process_rss()
    if rss is not None:
        _MEM_PEAK["rss"] = max(_MEM_PEAK["rss"], rss)
        parts.append(f"peak_rss={rss / _GIB:.1f}GB")
    if extra:
        parts.append(extra)
    if parts:
        print(f"[MEM] {stage}: {' '.join(parts)}", flush=True)


def _log_mem_summary():
    """Final high-water summary — the number to size the memory request from."""
    print(
        f"[MEM] PEAK working_set={_MEM_PEAK['working_set'] / _GIB:.1f}GB "
        f"peak_rss={_MEM_PEAK['rss'] / _GIB:.1f}GB "
        "(working_set excludes reclaimable page cache)",
        flush=True,
    )


def _frame_gb(df):
    """Deep memory of a DataFrame in GB, for the [MEM] extra field."""
    try:
        return f"frame={df.memory_usage(deep=True).sum() / _GIB:.1f}GB rows={len(df):,}"
    except Exception:
        return ""


def read_random_parquet_row_groups(
    parquet_file, num_row_groups=4, random_seed=42, columns=None
):
    """
    Read a random subset of row groups from a parquet file. Each row group is a set of rows that are contiguous in the file. For 10x transcripts.parquet file, each row group has about 262,000 rows.
    """
    np.random.seed(random_seed)
    selected_row_groups = np.random.choice(
        parquet_file.metadata.num_row_groups, size=num_row_groups, replace=False
    )
    # Project columns here too: without `columns` this read pulled all ~20.
    return parquet_file.read_row_groups(
        selected_row_groups, columns=columns
    ).to_pandas()


def scaled_noise_threshold(
    nongene_feature_names,
    n_nongene_total: int,
    non_gene_prefixes: tuple[str, ...] = ("NegControl",),
) -> tuple[float, float, int, int]:
    """Noise threshold on the FULL-table scale, from a sampled non-gene column.

    Returns ``(threshold, scale, n_sampled, n_total)``.

    `calculate_noise_bound` is fed negative-control counts drawn from a *sample* of at
    most ``transcript_stream.DEFAULT_SAMPLE_ROWS`` non-gene rows, while the per-gene
    counts it is compared against are exact over the whole table. The threshold therefore
    has to be scaled up by how much of the non-gene population was sampled.

    The bug this exists to prevent: the scaling used to be ``num_transcripts /
    num_selected_transcripts``, and once the streaming aggregation made
    ``num_selected_transcripts`` the *full* row count, that ratio became exactly 1.0 and the
    threshold silently stayed sample-scale. Measured on the reference sample: threshold
    323 where the scaled value is ~1,400, with 5,005 of 5,006 genes retained. The factor
    must be over non-gene transcripts specifically, because that is the sampled population
    -- scaling by total rows would overshoot by the gene fraction.

    Extracted from ``main()`` so the scaling is testable; a test against
    ``transcript_stream`` alone cannot see it, and did not.
    """
    sampled = nongene_feature_names
    n_sampled = int(len(sampled))
    n_total = int(max(0, n_nongene_total))
    scale = (n_total / n_sampled) if n_sampled and n_total else 1.0
    # Only negative-control features feed the noise model — the report states this
    # explicitly. The prefix set is a parameter rather than a literal so
    # --non-gene-prefix actually takes effect; its default reproduces the
    # previously hard-coded "NegControl".
    _, threshold = calculate_noise_bound(
        sampled[sampled.str.startswith(non_gene_prefixes)].value_counts()
    )
    return float(threshold) * scale, scale, n_sampled, n_total


def main():
    parser = argparse.ArgumentParser(description="Transcript QC Processing")
    parser.add_argument(
        "--xenium-bundle-dir", required=True, help="Path to Xenium bundle directory"
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    # nargs="+" on both: modules/local/transcript_qc/main.nf splits the ";"-
    # separated config value (documented in nextflow_schema.json) into separate
    # bare tokens, e.g. `--non-gene-prefix 'A' 'B'`. Without nargs that form
    # makes argparse exit 2. Neither value is read anywhere in this script, so
    # the str-default/list-from-argv asymmetry is harmless.
    parser.add_argument(
        "--non-gene-prefix",
        nargs="+",
        # "NegControl" (not "NegControlProbe") so the default reproduces the
        # previously hard-coded selection and matches the pipeline default.
        default=["NegControl"],
        help="Prefix(es) identifying non-gene features for the noise model",
    )
    parser.add_argument(
        "--stain-names",
        nargs="+",
        help="Stain names (unused but kept for compatibility)",
    )
    parser.add_argument(
        "--task-process", default="TRANSCRIPT_QC", help="Task process name"
    )
    parser.add_argument(
        "--num-row-groups",
        type=int,
        default=None,
        help="Number of row groups to process",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads (for compatibility)"
    )
    parser.add_argument(
        "--pipeline-segmentation",
        default="skip",
        help="Pipeline segmentation method (params.segmentation); 'skip' for none",
    )
    parser.add_argument(
        "--is-resegmented",
        action="store_true",
        help="Set when this run analyses a pipeline-resegmented bundle (post-seg)",
    )
    parser.add_argument(
        "--seg-versions-file",
        nargs="*",
        default=None,
        help="Segmentation step versions.yml file(s); tool version parsed into "
        "the segmentation-software label (post-seg pipeline tools)",
    )

    args = parser.parse_args()

    # argparse yields a list with nargs="+", so normalise to the tuple
    # pandas' str.startswith expects for a multi-prefix test.
    non_gene_prefixes = tuple(
        args.non_gene_prefix
        if isinstance(args.non_gene_prefix, (list, tuple))
        else [args.non_gene_prefix]
    )

    # Validate parameters
    if args.xenium_bundle_dir is None or not os.path.exists(args.xenium_bundle_dir):
        raise FileNotFoundError(
            f'The given XENIUM_BUNDLE_DIR, "{args.xenium_bundle_dir}" doesn\'t exist'
        )

    XENIUM_BUNDLE_DIR = Path(args.xenium_bundle_dir)
    # Create output directories
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_fig_dir = outdir / "figures"
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    figures_source_dir = outdir / "figures_source"
    figures_source_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_path = outdir / "transcript_qc_metrics.json"

    NUM_ROW_GROUPS = args.num_row_groups
    TASK_PROCESS = args.task_process

    transcripts_parquet_path = XENIUM_BUNDLE_DIR / "transcripts.parquet"
    morphology_focus_dir = XENIUM_BUNDLE_DIR / "morphology_focus"
    cells_parquet_path = XENIUM_BUNDLE_DIR / "cells.parquet"
    cell_feature_matrix_h5_path = XENIUM_BUNDLE_DIR / "cell_feature_matrix.h5"

    # EXACT CODE FROM ORIGINAL NOTEBOOK - check if all required files are present
    required_files = [
        transcripts_parquet_path,
        morphology_focus_dir,
        cell_feature_matrix_h5_path,
        cells_parquet_path,
    ]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Required file not found: {file}")
            sys.exit(1)

    print("=== Transcript QC Processing ===")
    print(f"Input directory: {XENIUM_BUNDLE_DIR}")
    print(f"Output directory: {outdir}")

    # EXACT CODE FROM ORIGINAL NOTEBOOK - check if all required columns are present in the transcripts parquet file
    transcripts_parquet = pq.ParquetFile(transcripts_parquet_path)
    transcripts_parquet_columns = transcripts_parquet.schema.names
    num_transcripts = transcripts_parquet.metadata.num_rows
    print(f"Total number of transcripts: {num_transcripts:,}")

    # EXACT CODE FROM ORIGINAL NOTEBOOK - required columns
    required_columns = [
        "cell_id",
        "qv",
        "fov_name",
        "codeword_category",
        "is_gene",
        "feature_name",
        # Per-transcript nuclear-overlap flag — drives the §5.2 nucleus RNA
        # fraction (fraction of a cell's transcripts that overlap its nucleus).
        "overlaps_nucleus",
    ]
    missing_columns = [
        col for col in required_columns if col not in transcripts_parquet_columns
    ]
    if missing_columns:
        print(f"Missing required columns in transcripts.parquet: {missing_columns}")
        sys.exit(1)

    # Stream the table rather than hold it resident. The eager read cost ~145 GB
    # of RSS on the 1.33 B-row reference sample (117 B/row as object dtype) and
    # drove the 120 -> 240 -> 480 GB retry ladder, yet every consumer below is a
    # reduction: the largest output is per-cell at ~530 k rows. See
    # transcript_stream for the measurements.
    if NUM_ROW_GROUPS is not None:
        print(
            f"--num-row-groups ({NUM_ROW_GROUPS}) ignored: the streaming "
            "aggregation reads the whole file at bounded memory, so subsampling "
            "is no longer needed to avoid running out of it."
        )
        NUM_ROW_GROUPS = None

    _tx = transcript_stream.aggregate_transcripts(
        transcripts_parquet_path, BACKGROUND_CELL_ID
    )
    _log_mem("after streaming aggregation")

    # The two QV violin plots are the only row-level consumers, and the previous
    # code already capped them at 1e6 rows each -- it just got that cap by
    # materialising every row and sampling down.
    _SAMPLE_COLUMNS = ["qv", "codeword_category", "fov_name", "feature_name"]
    df_spatial_gene, df_spatial_nongene = transcript_stream.sample_rows(
        transcripts_parquet_path,
        _SAMPLE_COLUMNS,
        k=transcript_stream.DEFAULT_SAMPLE_ROWS,
    )
    _log_mem("after violin sampling")

    num_selected_transcripts = _tx.n_rows

    if num_selected_transcripts != num_transcripts:
        print(
            f"Number of random transcripts selected for analysis: {num_selected_transcripts:,} (out of {num_transcripts:,})"
        )

    codeword_category_counts = _tx.codeword_counts

    print(f"Features: {_tx.n_features:,}")
    print("\nFeature categories:")
    for cc in codeword_category_counts.sort_values(ascending=False).keys():
        count = codeword_category_counts[cc]
        percentage = count / num_selected_transcripts * 100
        print(f"  {cc:<26} - {count:>12,} transcripts ({percentage:>6.3f}%)")

    # EXACT CODE FROM ORIGINAL NOTEBOOK - quality distribution plots
    _log_mem("before quality plots")
    print("\nGenerating quality distribution plots...")

    # df_spatial_gene / df_spatial_nongene were sampled during the streaming pass.

    # define hue order: genes first, then non-genes
    codeword_categories_order = codeword_category_counts.index
    codeword_categories_order = (
        codeword_categories_order[codeword_categories_order.str.endswith("gene")]
        .sort_values(ascending=False)
        .tolist()
        + codeword_categories_order[~codeword_categories_order.str.endswith("gene")]
        .sort_values()
        .tolist()
    )

    df_spatial_quality = pd.concat([df_spatial_nongene, df_spatial_gene])

    print(
        f"Using {len(df_spatial_nongene):,} non-gene transcripts and {len(df_spatial_gene):,} gene transcripts for quality values (qv) distribution"
    )

    # Print total number of rows and rows per category in df_spatial_quality
    _log_mem("after quality sampling")
    print("\ntranscript per category:")
    print(df_spatial_quality["codeword_category"].value_counts())

    # Quality distribution density plot (Figure 1) — REMOVED 2026-05-22
    # per transcript QC v5 redesign feedback: redundant with the violin plot
    # below, which shows the same per-category QV distribution but is
    # easier to read (one violin per category, side-by-side, with median
    # and quartiles marked inside each violin).
    # The underlying df_spatial_quality data is still saved by the violin
    # plot block below — figures_source/quality_distributions_comprehensive.csv.

    # Figure 2: Quality distribution by codeword category (violin plot) —
    # axes swapped (was x=codeword_category/y=qv, now y=codeword_category/x=qv)
    # so QV reads left-to-right. The low-QV tail then sits on the LEFT (matches
    # the report prose) and the layout is consistent with the per-FoV QV violin
    # below, which is already horizontal. Category labels move to the y-axis and
    # read horizontally instead of rotated 45°.
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_spatial_quality,
        y="codeword_category",
        x="qv",
        hue="codeword_category",
        split=False,
        inner="box",
        palette="husl",
        density_norm="width",
        legend=False,
        order=codeword_categories_order,
    )
    plt.ylabel("Codeword Category", fontsize=12)
    plt.xlabel("Quality Value (qv)", fontsize=12)
    plt.title(
        "Distribution of Transcript Quality by Codeword Category", fontsize=14, pad=20
    )
    plt.tight_layout()
    # vertical reference line at qv=20 (matches horizontal-violin layout)
    plt.axvline(x=20, color="grey", linestyle="--")
    plt.savefig(
        output_fig_dir / "quality_distributions_comprehensive.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        output_fig_dir / "quality_distributions_comprehensive.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save data for quality distributions comprehensive
    df_spatial_quality.to_csv(
        figures_source_dir / "quality_distributions_comprehensive.csv", index=False
    )

    # Save df_spatial_quality to CSV
    output_file = outdir / "df_spatial_quality.csv"
    df_spatial_quality.to_csv(output_file, index=False)
    print(f"Saved df_spatial_quality to: {output_file}")

    # ===== Per-FoV QV-consistency stats (computed here so the §3.2 figure can
    # colour each FoV by its verdict; reused later for fov_summary — one
    # groupby over df_spatial, single source of truth). =====
    # Spike-validated design at plans/2026-05-22_SPIKE_fov_zscore_denominator.py
    # against 11 calibration samples (brain/lung/liver/skin/pancreas).
    #
    # Per-FoV stats — computed from df_spatial (full, not subsampled gene view).
    # Within-sample z-score on pct_above_qv20, one-sided (low outliers only).
    # Per-FoV median QV is deliberately NOT computed: Xenium QV caps at 40 so
    # the median saturates in every healthy FoV and carries no signal.
    _N_GATE_FOV = 100  # FoVs with fewer transcripts → N/A (sampling noise dominates)
    _Z_FAIL = 3.0
    _Z_WARN = 2.0

    # From the streaming pass. qv is accumulated in float64 there, so these means
    # are marginally more accurate than a pandas float32 column mean.
    _fov_n = _tx.fov["n"]
    _fov_mean = _tx.fov["mean_qv"]
    _fov_pct_above = _tx.fov["pct_above_qv20"]

    _eligible_mask = _fov_n >= _N_GATE_FOV
    _eligible_pct = _fov_pct_above[_eligible_mask]

    # _baseline_ok: a within-sample median/stdev can actually be computed
    # from the eligible FoVs (need ≥2 eligible FoVs with non-zero stdev).
    # When False, dev_z is meaningless and per-FoV status falls through
    # to N/A — see _per_fov_status below.
    _baseline_ok = len(_eligible_pct) >= 2 and _eligible_pct.std() > 0
    if _baseline_ok:
        _ref_median = float(_eligible_pct.median())
        _ref_stdev = float(_eligible_pct.std())
        _dev_z = (_fov_pct_above - _ref_median) / _ref_stdev
    else:
        _dev_z = pd.Series(np.zeros(len(_fov_pct_above)), index=_fov_pct_above.index)

    def _per_fov_status(n: int, dev: float) -> str:
        if n < _N_GATE_FOV:
            return "N/A"
        # If there's no within-sample baseline (single FoV, all-equal pct,
        # etc.), the dev_z above is 0.0 by fallback — meaningless. Don't
        # report PASS in that case; emit N/A so downstream consumers of
        # fov_summary[i].status see honest data.
        if not _baseline_ok:
            return "N/A"
        # One-sided: only low outliers (negative dev_z) flag.
        # Unusually high QV in a FoV is not a quality concern.
        if dev < -_Z_FAIL:
            return "FAIL"
        if dev < -_Z_WARN:
            return "WARN"
        return "PASS"

    _fov_qv_status = {
        str(_fov): _per_fov_status(int(_fov_n.loc[_fov]), float(_dev_z.loc[_fov]))
        for _fov in _fov_n.index
    }

    # Figure 3: Quality distribution by Field of View — axes swapped (was
    # x=fov_name/y=qv, now y=fov_name/x=qv). Xenium slides can carry 100+
    # FoVs; vertical violins crammed the labels and made the QV
    # distributions unreadable. Horizontal layout gives each FoV one row
    # and the figure height scales with FoV count.
    _n_fov = max(1, df_spatial_gene["fov_name"].nunique())
    # Bounded: this is the only artefact whose size grows without limit with the
    # sample. At dpi 300 an unbounded 0.22*n_fov height renders a 3000 x 59,400 px
    # canvas on a 900-FoV slide, and bbox_inches="tight" renders it twice; the PNG
    # is then base64-embedded by the report step.
    _fov_fig_height = min(_FOV_FIG_MAX_HEIGHT_IN, max(6, 0.22 * _n_fov))
    fig = plt.figure(figsize=(10, _fov_fig_height))
    # Single-colour violins. Per-FoV verdicts are reported in the §3.1 table
    # (sorted worst-first); colouring the violins by verdict was dropped — the
    # flagged FoVs are a handful among 100+ and were not visible on this tall,
    # thin-violin plot.
    sns.violinplot(
        data=df_spatial_gene,
        y="fov_name",
        x="qv",
        color="steelblue",
        inner="box",
        density_norm="width",
    )
    plt.ylabel("Field of View", fontsize=12)
    plt.xlabel("Quality Value (qv)", fontsize=12)
    plt.title(
        "Distribution of Transcript Quality by Field of View\n(gene transcripts only)",
        fontsize=14,
        pad=20,
    )
    plt.tight_layout()
    # vertical reference line at qv=20 (matches horizontal-violin layout)
    plt.axvline(x=20, color="grey", linestyle="--")
    plt.savefig(output_fig_dir / "quality_by_fov.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_fig_dir / "quality_by_fov.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save data for quality by fov
    df_spatial_gene.to_csv(figures_source_dir / "quality_by_fov.csv", index=False)

    del df_spatial_quality, df_spatial_gene

    # Noise threshold from the negative-control features.
    #
    # SCALE MISMATCH, fixed here. The threshold is derived from `df_spatial_nongene`,
    # which is a *sample* of at most transcript_stream.DEFAULT_SAMPLE_ROWS non-gene rows,
    # but it is compared against `_tx.gene["n_molecules"]`, which the streaming
    # aggregation computes exactly over the whole table. The two must be brought onto one
    # scale.
    #
    # The original line multiplied the threshold by `num_transcripts /
    # num_selected_transcripts`. That worked when both sides came from the same subsampled
    # frame, but `num_selected_transcripts` is now `_tx.n_rows` -- the full row count -- so
    # the ratio is exactly 1.0 and the threshold stayed sample-scale. Observed on the
    # reference sample: total_transcripts == selected_transcripts == 1,325,798,498, threshold
    # 323 where the correctly scaled value is ~1,400, and 5,005 of 5,006 genes retained.
    #
    # The right factor is over *non-gene* transcripts specifically, since that is the
    # population sampled.
    n_mols_threshold, nongene_scale, n_nongene_sampled, n_nongene_total = (
        scaled_noise_threshold(
            df_spatial_nongene["feature_name"],
            n_nongene_total=max(0, _tx.n_rows - _tx.n_gene_rows),
            non_gene_prefixes=non_gene_prefixes,
        )
    )
    print(
        f"Noise threshold for genes' transcript count: {n_mols_threshold:.0f} transcripts "
        f"(from {n_nongene_sampled:,} of {n_nongene_total:,} non-gene transcripts, "
        f"scale x{nongene_scale:.2f})"
    )

    # No rescaling: the streaming aggregation counts every row, so these are already
    # full-scale. Multiplying by num_transcripts / num_selected_transcripts here was a no-op
    # (that ratio is 1.0) and would double-count if the aggregation ever became partial.
    n_mols_per_gene_df = _tx.gene[["n_molecules", "is_gene"]].copy()

    # Figure 4: Distribution of transcripts per feature — smooth density (KDE)
    # by gene vs non-gene. The binned histogram was dropped; only the density
    # curve is shown, so the y-axis is density rather than a feature count.
    fig = plt.figure(figsize=(8, 4))
    sns.kdeplot(
        data=n_mols_per_gene_df,
        x="n_molecules",
        hue="is_gene",
        log_scale=True,
        fill=True,
        alpha=0.4,
        ax=plt.gca(),
    )
    plt.xlabel("Num. transcripts")
    plt.ylabel("Density")
    plt.axvline(x=n_mols_threshold, color="grey", linestyle="--")
    plt.title("Distribution of transcripts per feature", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(
        f"{output_fig_dir}/num_transcripts_per_feature.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_fig_dir}/num_transcripts_per_feature.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save data for transcripts per feature
    df_transcripts_per_feature = n_mols_per_gene_df.copy()
    df_transcripts_per_feature["n_mols_threshold"] = n_mols_threshold
    df_transcripts_per_feature.to_csv(
        figures_source_dir / "num_transcripts_per_feature.csv", index=True
    )

    retained_genes = n_mols_per_gene_df.query(
        "n_molecules > @n_mols_threshold and is_gene == True"
    )
    print(
        f"Number of genes with a total transcript count higher than the threshold : {len(retained_genes):,}"
    )
    retained_genes.to_csv(outdir / "retained_genes.csv", index=True)

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Cell size distribution
    _log_mem("before cell statistics")
    print("\nGenerating cell statistics plots...")

    # Check available columns and use appropriate cell area column
    available_columns = pq.ParquetFile(cells_parquet_path).schema.names
    cell_area_column = "cell_area" if "cell_area" in available_columns else "volume"
    cells_parquet = pd.read_parquet(
        cells_parquet_path, columns=["cell_id", cell_area_column]
    )
    # Rename the column for consistency
    cells_parquet.rename(columns={cell_area_column: "cell_size"}, inplace=True)

    # Figure 5: Cell size distribution. Rendered on both a linear and a log
    # x-axis so the §5.1 report tabset can flip between them: linear shows the
    # raw spread (a few very large cells stretch the axis and compress the
    # body), while log makes the log-normal body a readable bell and exposes
    # both tails (small = segmentation artefacts, large = merged cells).
    # Figure size + title style aligned with §5.4 / §5.5 (figsize=(8,4),
    # title fontsize=14, pad=20) for uniform appearance across cell-level plots.
    _mean_cs = cells_parquet["cell_size"].mean()
    _median_cs = cells_parquet["cell_size"].median()
    for _scale, _suffix in (("linear", ""), ("log", "_log")):
        fig = plt.figure(figsize=(8, 4))
        sns.kdeplot(
            data=cells_parquet,
            x="cell_size",
            fill=True,
            color="skyblue",
            alpha=0.5,
            log_scale=(_scale == "log"),
        )
        plt.xlabel("Cell size (pixels²)")
        plt.ylabel("Density")
        _scale_label = " (log scale)" if _scale == "log" else ""
        plt.title(f"Distribution of cell size{_scale_label}", fontsize=14, pad=20)

        # Add vertical lines for mean and median
        plt.axvline(_mean_cs, color="red", linestyle="--", label="Mean")
        plt.axvline(_median_cs, color="green", linestyle="--", label="Median")
        plt.legend()
        plt.tight_layout()

        # Save the plot (filename matches actual content — cell size; renamed
        # from genes_per_cell_distribution.* on 2026-05-22 per transcript QC v5
        # filename-content alignment pass). The "_log" suffix is the log-x twin.
        plt.savefig(
            os.path.join(outdir, "figures", f"cell_size_distribution{_suffix}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(outdir, "figures", f"cell_size_distribution{_suffix}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save data for cell size distribution
    df_cell_size = cells_parquet[["cell_id", "cell_size"]].copy()
    df_cell_size["mean_cell_size"] = cells_parquet["cell_size"].mean()
    df_cell_size["median_cell_size"] = cells_parquet["cell_size"].median()
    df_cell_size.to_csv(figures_source_dir / "cell_size_distribution.csv", index=False)
    # Capture per-sample aggregates for §5.1 status row (advisory PASS/WARN).
    median_cell_size = (
        float(cells_parquet["cell_size"].median()) if len(cells_parquet) > 0 else None
    )
    del cells_parquet

    # Cells read — nucleus_count drives the §5 "cells with no in-plane nucleus"
    # metric (a count of nuclei == 0). The total_counts > 0 filter sets that
    # denominator (cells carrying transcripts). Kept even though the nucleus RNA
    # fraction below no longer comes from this read.
    cells_parquet = pd.read_parquet(
        cells_parquet_path,
        columns=["nucleus_count", "total_counts"],
        filters=[("total_counts", ">", 0)],
    )

    # Metric 2 (SegTraQ Phase A): fraction of cells with no in-plane nucleus.
    # Uses nucleus_count == 0, NOT nucleus_area == 0 — no-nucleus cells still
    # carry nonzero nucleus_area, so the area test silently returns ~0% on
    # every sample. Denominator is the total_counts > 0 population this read is
    # filtered to (i.e. cells carrying transcripts).
    if "nucleus_count" in cells_parquet.columns:
        pct_cells_no_nucleus = float((cells_parquet["nucleus_count"] == 0).mean() * 100)
    else:
        pct_cells_no_nucleus = None
    del cells_parquet

    # 10x-defined gate: % of segmented cells with zero gene transcripts.
    # Derived in-house from cells.parquet transcript_counts over ALL cells
    # (single matched cell base — numerator and denominator from the same
    # module). Verified to reproduce 10x's metrics_summary.csv
    # fraction_empty_cells exactly on the XOA-3.2 and 4.0 example bundles, so it
    # sidesteps the cross-module derivation that yields impossible fractions
    # (qc_threshold_refinement report §3.3). Works on resegmented bundles too,
    # since it uses this run's actual cells.parquet rather than the onboard CSV.
    _cell_cols = pq.ParquetFile(cells_parquet_path).schema.names
    if "transcript_counts" in _cell_cols:
        _zt = pd.read_parquet(cells_parquet_path, columns=["transcript_counts"])
        pct_cells_zero_transcripts = (
            float((_zt["transcript_counts"] == 0).mean() * 100)
            if len(_zt) > 0
            else None
        )
        del _zt
    else:
        pct_cells_zero_transcripts = None

    # Figure 6 / §5.2: Nucleus RNA fraction per cell — the fraction of a cell's
    # assigned transcripts that overlap its nucleus. Derived from transcripts
    # (overlaps_nucleus), NOT from cells.parquet nucleus_count: nucleus_count is
    # a count of nuclei (0/1/2), so the old nucleus_count/(total_counts+1)
    # collapsed to ~1/total_counts — a latent bug, not a nuclear fraction.
    # Matches the MultiQC report definition: sum(overlaps_nucleus)/count per
    # cell_id over all assigned transcripts (cell_id != UNASSIGNED; no QV or
    # is_gene filter). Distribution is over cells with >= 1 assigned transcript.
    # Full-read only — a cell's transcripts span FoVs/row groups, so per-cell
    # fractions are unreliable under --num-row-groups; emit N/A there.
    # Streamed: the sentinel is filtered before aggregation, so it never becomes
    # a group, and the fraction is sum(overlaps)/count per cell.
    nucleus_count_fraction = _tx.cell_nucleus_fraction

    # Figure + source CSV + summary are full-read only — skip entirely under
    # subsampling (a misleading KDE next to an N/A table would be worse than
    # nothing). Figure 6 size/title style aligned with §5.1 / §5.4 / §5.5.
    _none_summary = {
        "mean": None,
        "median": None,
        "stdev": None,
        "pct_cells_above_60pct": None,
        "pct_cells_below_5pct": None,
    }
    if nucleus_count_fraction is not None and len(nucleus_count_fraction) > 0:
        fig = plt.figure(figsize=(8, 4))
        sns.kdeplot(x=nucleus_count_fraction, fill=True, color="skyblue", alpha=0.5)
        plt.xlabel("Nucleus RNA fraction per cell")
        plt.ylabel("Density")
        plt.title(
            "Empty plot — no nuclear transcripts detected"
            if not (nucleus_count_fraction > 0).any()
            else "Distribution of nucleus RNA fraction per cell",
            fontsize=14,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                outdir,
                "figures",
                "nucleus_transcript_fraction_per_cell_distribution.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(
                outdir,
                "figures",
                "nucleus_transcript_fraction_per_cell_distribution.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Save data for nucleus RNA fraction (one row per cell).
        pd.DataFrame(
            {
                "cell_id": nucleus_count_fraction.index,
                "nucleus_fraction": nucleus_count_fraction.values,
            }
        ).to_csv(
            figures_source_dir
            / "nucleus_transcript_fraction_per_cell_distribution.csv",
            index=False,
        )

        # Per-sample aggregates for the §5.2 status row (advisory PASS/WARN).
        _ncf = nucleus_count_fraction.dropna()
        if len(_ncf) > 0:
            nucleus_transcript_fraction_summary = {
                "mean": float(_ncf.mean()),
                "median": float(_ncf.median()),
                "stdev": float(_ncf.std()),
                "pct_cells_above_60pct": float((_ncf > 0.60).mean() * 100),
                "pct_cells_below_5pct": float((_ncf < 0.05).mean() * 100),
            }
        else:
            nucleus_transcript_fraction_summary = dict(_none_summary)
    else:
        # Subsampled run (or no assigned cells): full-read-only metric → N/A.
        nucleus_transcript_fraction_summary = dict(_none_summary)

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Figure 7: Nucleus-to-cell area fraction
    cells_parquet = pd.read_parquet(
        cells_parquet_path, columns=["nucleus_area", "cell_area"]
    )
    nucleus_size_fraction = cells_parquet["nucleus_area"] / (
        cells_parquet["cell_area"] + 1
    )

    # Figure 7 size + title style aligned with §5.1 / §5.2 / §5.4 / §5.5
    # for uniform appearance.
    fig = plt.figure(figsize=(8, 4))
    sns.kdeplot(x=nucleus_size_fraction, fill=True, color="skyblue", alpha=0.5)
    plt.xlabel("Nucleus to cell area ratio")
    plt.ylabel("Density")
    plt.title(
        # Same fix as §5.2: show the distribution unless NO cell has a
        # positive nucleus-to-cell ratio. `.all() != 0` was inverted — it read
        # False (→ "Empty") whenever any single cell had a zero ratio.
        "Distribution of nucleus-to-cell area ratio"
        if (nucleus_size_fraction.fillna(0) > 0).any()
        else "Empty plot — no nucleus transcripts detected",
        fontsize=14,
        pad=20,
    )
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(
            outdir, "figures", "nucleus_to_cell_size_fraction_per_cell_distribution.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(
            outdir, "figures", "nucleus_to_cell_size_fraction_per_cell_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save data for nucleus to cell size fraction
    df_nucleus_size_fraction = pd.DataFrame(
        {
            "nucleus_area": cells_parquet["nucleus_area"],
            "cell_area": cells_parquet["cell_area"],
            "nucleus_size_fraction": nucleus_size_fraction,
        }
    )
    df_nucleus_size_fraction.to_csv(
        figures_source_dir / "nucleus_to_cell_size_fraction_per_cell_distribution.csv",
        index=False,
    )
    del cells_parquet

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Load cell feature matrix
    ad = sc.read_10x_h5(cell_feature_matrix_h5_path)
    # filter for retained genes
    ad = ad[:, ad.var_names.isin(retained_genes.index)]
    print(f"AnnData object with n_obs × n_vars = {ad.shape[0]} × {ad.shape[1]}")

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Figure 8: Distribution of transcripts per cell
    n_mols_per_cell = ad.X.sum(axis=1).A1
    n_mols_threshold_cell = _min_count_threshold(n_mols_per_cell)

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(n_mols_per_cell, log_scale=True, bins=50, ax=plt.gca())
    plt.xlabel("Num. transcripts")
    plt.ylabel("Num. cells")
    if n_mols_threshold_cell is not None:
        plt.axvline(x=n_mols_threshold_cell, color="grey", linestyle="--")
    plt.title("Distribution of transcripts per Cell", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(
        f"{output_fig_dir}/num_transcripts_per_cell.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{output_fig_dir}/num_transcripts_per_cell.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Threshold for transcripts per cell: {n_mols_threshold_cell}")

    # Save data for transcripts per cell
    df_transcripts_per_cell = pd.DataFrame(
        {
            "n_transcripts_per_cell": n_mols_per_cell,
            "n_mols_threshold_cell": n_mols_threshold_cell,
        }
    )
    df_transcripts_per_cell.to_csv(
        figures_source_dir / "num_transcripts_per_cell.csv", index=False
    )

    # Convert numpy array to pandas DataFrame
    n_mols_per_cell_df = pd.DataFrame(n_mols_per_cell, columns=["num_of_transcripts"])
    # Filename matches actual content — transcripts; renamed from
    # num_transcripts_per_cell.csv on 2026-05-22 (CSV holds num_of_transcripts
    # column, not transcripts/genes).
    output_file = os.path.join(outdir, "num_transcripts_per_cell.csv")
    n_mols_per_cell_df.to_csv(output_file, index=False)
    print(f"Saved transcripts-per-cell distribution to: {output_file}")

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Figure 9: Distribution of genes per cell
    n_genes_per_cell = (ad.X != 0).sum(axis=1).A1
    n_genes_threshold = _min_count_threshold(n_genes_per_cell)

    fig = plt.figure(figsize=(8, 4))
    sns.histplot(n_genes_per_cell, log_scale=True, bins=50, ax=plt.gca())
    plt.xlabel("Num. genes")
    plt.ylabel("Num. cells")
    if n_genes_threshold is not None:
        plt.axvline(x=n_genes_threshold, color="grey", linestyle="--")
    plt.title("Distribution of number of detected genes per Cell", fontsize=14, pad=20)
    plt.tight_layout()
    # Filenames match actual content — genes per cell; renamed from
    # num_transcripts_per_cell.* on 2026-05-22 (xlabel reads "Num. genes",
    # title is "Distribution of number of detected genes per Cell").
    plt.savefig(
        f"{output_fig_dir}/num_genes_per_cell.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{output_fig_dir}/num_genes_per_cell.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data for genes per cell — figures_source/ version is the
    # richer pandas DataFrame (data + threshold). The bare numpy dump
    # at outdir-root (num_genes_per_cell.csv, line ~642 below) keeps
    # the legacy "single-column raw" name.
    df_genes_per_cell = pd.DataFrame(
        {"n_genes_per_cell": n_genes_per_cell, "n_genes_threshold": n_genes_threshold}
    )
    df_genes_per_cell.to_csv(figures_source_dir / "num_genes_per_cell.csv", index=False)

    # Convert numpy array to pandas DataFrame
    n_genes_per_cell_df = pd.DataFrame(n_genes_per_cell, columns=["num_of_genes"])
    # Save to CSV using os.path.join for path handling
    output_file = os.path.join(outdir, "num_genes_per_cell.csv")
    n_genes_per_cell_df.to_csv(output_file, index=False)
    print(f"Saved gene distribution to: {output_file}")

    # ===== transcript QC v5 redesign — Phase 0 schema extension (2026-05-22) =====
    # Spike-validated design at plans/2026-05-22_SPIKE_fov_zscore_denominator.py
    # against 11 calibration samples (brain/lung/liver/skin/pancreas).

    # Sample-wide QV summary, from the streaming pass.
    pct_transcripts_above_qv20 = float(_tx.pct_qv_above_threshold)
    mean_qv = float(_tx.mean_qv)

    # Per-FoV QV-consistency stats (_N_GATE_FOV, _Z_FAIL, _Z_WARN, _fov_n,
    # _fov_mean, _fov_pct_above, _eligible_mask, _baseline_ok, _dev_z,
    # _per_fov_status, _fov_qv_status) are computed earlier, just before the
    # §3.2 QV-by-FoV figure, so that plot can colour each FoV by its verdict.
    # They are reused here — single source of truth, one groupby over df_spatial.

    # Metric 3 (SegTraQ Phase A): per-FoV transcript density + unassigned%.
    # Full-read only — row groups are FoV-ordered, so under --num-row-groups
    # whole FoVs drop and per-FoV counts are no longer comparable; emit N/A.
    # _fov_n is the transcript count per FoV. Same one-sided z machinery as QV.
    # The _N_GATE_FOV gate marks FoVs below 100 transcripts N/A: those are
    # empty/edge tissue, not dropouts. An in-tissue dropout still carries
    # hundreds of transcripts, just fewer than its neighbours, so it stays
    # eligible and can flag.
    _fov_unassigned_pct = _tx.fov["pct_unassigned"]

    _mol_eligible = _fov_n[_eligible_mask]
    _mol_baseline_ok = (
        NUM_ROW_GROUPS is None and len(_mol_eligible) >= 2 and _mol_eligible.std() > 0
    )
    if _mol_baseline_ok:
        _mol_ref_median = float(_mol_eligible.median())
        _mol_ref_stdev = float(_mol_eligible.std())
        _mol_z = (_fov_n - _mol_ref_median) / _mol_ref_stdev
    else:
        _mol_z = pd.Series(np.zeros(len(_fov_n)), index=_fov_n.index)

    def _per_fov_transcript_status(n: int, dev: float) -> str:
        if NUM_ROW_GROUPS is not None:
            return "N/A"
        if n < _N_GATE_FOV:
            return "N/A"
        if not _mol_baseline_ok:
            return "N/A"
        # One-sided: only low-density FoVs (dropouts) flag.
        if dev < -_Z_FAIL:
            return "FAIL"
        if dev < -_Z_WARN:
            return "WARN"
        return "PASS"

    _subsampled = NUM_ROW_GROUPS is not None
    fov_summary = [
        {
            "fov_id": str(_fov),
            "n_transcripts": int(_fov_n.loc[_fov]),
            "mean_qv": float(_fov_mean.loc[_fov]),
            "pct_above_qv20": float(_fov_pct_above.loc[_fov]),
            "dev_z": float(_dev_z.loc[_fov]),
            "status": _fov_qv_status[str(_fov)],
            # SegTraQ Phase A per-FoV additions (N/A under subsampling)
            "transcript_dev_z": (None if _subsampled else float(_mol_z.loc[_fov])),
            "transcript_status": _per_fov_transcript_status(
                int(_fov_n.loc[_fov]), float(_mol_z.loc[_fov])
            ),
            "pct_unassigned": (
                None if _subsampled else float(_fov_unassigned_pct.loc[_fov])
            ),
        }
        for _fov in _fov_n.index
    ]
    fov_eligible_count = int(_eligible_mask.sum())
    if fov_eligible_count >= 2:
        fov_outlier_count = sum(1 for r in fov_summary if r["status"] == "FAIL")
        fov_warn_count = sum(1 for r in fov_summary if r["status"] == "WARN")
        fov_pct_fail = 100.0 * fov_outlier_count / fov_eligible_count
        fov_pct_warn = 100.0 * fov_warn_count / fov_eligible_count
    else:
        fov_outlier_count = None
        fov_warn_count = None
        fov_pct_fail = None
        fov_pct_warn = None

    # Per-FoV transcript-density dropout rollup (SegTraQ Phase A). Counts FoVs
    # flagged WARN or FAIL on low density as the sample-level dropout fraction
    # (advisory). N/A under subsampling or with too few eligible FoVs.
    if not _subsampled and fov_eligible_count >= 2:
        fov_density_dropout_count = sum(
            1 for r in fov_summary if r["transcript_status"] in ("FAIL", "WARN")
        )
        fov_density_pct_dropout = 100.0 * fov_density_dropout_count / fov_eligible_count
    else:
        fov_density_dropout_count = None
        fov_density_pct_dropout = None

    # Per-cell median aggregates (n_mols_per_cell + n_genes_per_cell are
    # numpy arrays defined earlier; reuse them).
    median_transcripts_per_cell = (
        int(np.median(n_mols_per_cell)) if len(n_mols_per_cell) > 0 else 0
    )
    median_genes_per_cell = (
        int(np.median(n_genes_per_cell)) if len(n_genes_per_cell) > 0 else 0
    )

    # Nucleus-to-cell area ratio summary (reuses nucleus_size_fraction Series
    # from the §5.3 figure block above).
    _nf = nucleus_size_fraction.dropna()
    if len(_nf) > 0:
        nucleus_to_cell_area_ratio = {
            "mean": float(_nf.mean()),
            "median": float(_nf.median()),
            "stdev": float(_nf.std()),
            "pct_cells_below_5pct": float((_nf < 0.05).mean() * 100),
            "pct_cells_above_60pct": float((_nf > 0.60).mean() * 100),
            "pct_cells_above_80pct": float((_nf > 0.80).mean() * 100),
        }
    else:
        nucleus_to_cell_area_ratio = {
            "mean": None,
            "median": None,
            "stdev": None,
            "pct_cells_below_5pct": None,
            "pct_cells_above_60pct": None,
            "pct_cells_above_80pct": None,
        }
    # ===== end Phase 0 schema extension =====

    # ===== SegTraQ-derived metrics — Phase A (2026-06-16) =====
    # Metric definitions mirrored from SegTraQ (github.com/LazDaria/SegTraQ,
    # MIT), reimplemented in pandas on the parquet transcript QC already loads.

    # Metric 1: transcripts not assigned to a cell. cell_id == "UNASSIGNED"
    # marks transcripts that decoding placed but segmentation left unassigned.
    # Distinct from the "unassigned" codeword decoding category. The headline
    # ratio is unbiased under row-group subsampling, so it is always computed.
    pct_transcripts_unassigned_to_cell = float(_tx.pct_unassigned)

    # Per-gene breakdown — full-read only (FoV-ordered row groups). CV across
    # genes is the Phase B triage signal (even vs gene-driven); surfaced here
    # only as a value + figure.
    # NUM_ROW_GROUPS is forced to None above (streaming reads the whole file), so
    # this guard is now always taken; kept to preserve the original intent.
    if NUM_ROW_GROUPS is None:
        # From the streaming pass: per-gene totals and unassigned counts, real
        # genes only. Previously this materialised _genes = df_spatial[is_gene],
        # a 7-column copy of ~80% of the rows, held for the rest of the block.
        _genes_only = _tx.gene[_tx.gene["is_gene"]]
        _per_gene = pd.DataFrame(
            {
                "total": _genes_only["n_molecules"],
                "unassigned": _genes_only["n_unassigned"],
            }
        )
        _per_gene["perc_unassigned"] = (
            _per_gene["unassigned"] / _per_gene["total"] * 100
        )
        _gated = _per_gene[_per_gene["total"] >= MIN_GENE_COUNT_FOR_UNASSIGNED]
        if len(_gated) > 1 and _gated["perc_unassigned"].mean() > 0:
            unassigned_per_gene_cv = float(
                _gated["perc_unassigned"].std() / _gated["perc_unassigned"].mean()
            )
        else:
            unassigned_per_gene_cv = None

        # Source data for the figure (genes that passed the count gate).
        _per_gene_out = _gated.sort_values(
            "perc_unassigned", ascending=False
        ).reset_index()
        _per_gene_out.to_csv(
            figures_source_dir / "unassigned_per_gene.csv", index=False
        )

        # Figure: per-gene distribution of unassigned fraction. The dashed line
        # marks the sample-wide rate, so genes leaking far above it stand out.
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(_gated["perc_unassigned"], bins=50, ax=plt.gca())
        plt.axvline(x=pct_transcripts_unassigned_to_cell, color="grey", linestyle="--")
        plt.xlabel("Transcripts not assigned to a cell, per gene (%)")
        plt.ylabel("Num. genes")
        plt.title("Per-gene transcript-to-cell assignment", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(
            output_fig_dir / "unassigned_per_gene.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            output_fig_dir / "unassigned_per_gene.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    else:
        unassigned_per_gene_cv = None
    # ===== end SegTraQ Phase A metrics =====

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Save metrics
    # Gene-only counts for §2.4 (the noise bound is applied to genes,
    # not to all features — non-gene categories like negative_control_*
    # don't go through this filter). Computed from the per-feature
    # rollup so we get the same denominator the histogram in §4.1 uses.
    _gene_feat = n_mols_per_gene_df.query("is_gene == True")
    _ctrl_feat = n_mols_per_gene_df.query("is_gene == False")
    total_genes_count = len(_gene_feat)
    filtered_genes_count = int((_gene_feat["n_molecules"] <= n_mols_threshold).sum())

    # §4 feature-level gene-vs-control transcript-count shift. A healthy
    # panel should have gene features carrying much higher per-feature
    # transcript counts than control features. log2(median_gene /
    # median_control) > 2 means genes carry at least 4× control medians.
    if len(_gene_feat) > 0 and len(_ctrl_feat) > 0:
        _median_gene = float(_gene_feat["n_molecules"].median())
        _median_ctrl = float(_ctrl_feat["n_molecules"].median())
        median_transcripts_per_gene_feature = _median_gene
        median_transcripts_per_control_feature = _median_ctrl
        log2_gene_vs_control_median_ratio = (
            float(np.log2(_median_gene / _median_ctrl)) if _median_ctrl > 0 else None
        )
    else:
        median_transcripts_per_gene_feature = None
        median_transcripts_per_control_feature = None
        log2_gene_vs_control_median_ratio = None

    metrics = {
        "total_transcripts": int(num_transcripts),
        "selected_transcripts": int(num_selected_transcripts),
        "total_features": int(_tx.n_features),
        "total_genes_count": total_genes_count,
        "filtered_genes_count": filtered_genes_count,
        "codeword_category_counts": {
            str(k): int(v) for k, v in codeword_category_counts.items()
        },
        "neg_control_quantile": int(n_mols_threshold),
        # None ⇒ not computable on degenerate counts (see _min_count_threshold);
        # emitted as JSON null, which the report already treats as absent.
        "min_transcripts_per_cell": (
            int(n_mols_threshold_cell) if n_mols_threshold_cell is not None else None
        ),
        "min_genes_per_cell": (
            int(n_genes_threshold) if n_genes_threshold is not None else None
        ),
        "retained_genes_count": len(retained_genes),
        "total_cells": int(ad.shape[0]),
        "analyzed_genes": int(ad.shape[1]),
        # ----- Phase 0 (v5 redesign) additions -----
        "pct_transcripts_above_qv20": pct_transcripts_above_qv20,
        "mean_qv": mean_qv,
        "fov_summary": fov_summary,
        "fov_eligible_count": fov_eligible_count,
        "fov_outlier_count": fov_outlier_count,
        "fov_warn_count": fov_warn_count,
        "fov_pct_fail": fov_pct_fail,
        "fov_pct_warn": fov_pct_warn,
        "median_transcripts_per_cell": median_transcripts_per_cell,
        "median_genes_per_cell": median_genes_per_cell,
        "nucleus_to_cell_area_ratio": nucleus_to_cell_area_ratio,
        # Section-summary additions (2026-05-22) for §5 cell-level summary
        "median_cell_size": median_cell_size,
        "nucleus_transcript_fraction_summary": nucleus_transcript_fraction_summary,
        # §4 feature-level shift (drives §1 row 3 in the v5 4-row scorecard)
        "median_transcripts_per_gene_feature": median_transcripts_per_gene_feature,
        "median_transcripts_per_control_feature": median_transcripts_per_control_feature,
        "log2_gene_vs_control_median_ratio": log2_gene_vs_control_median_ratio,
        # ----- SegTraQ-derived metrics (Phase A) -----
        "transcript_assignment": {
            "pct_transcripts_unassigned_to_cell": pct_transcripts_unassigned_to_cell,
            "unassigned_per_gene_cv": unassigned_per_gene_cv,
            "per_gene_min_count": MIN_GENE_COUNT_FOR_UNASSIGNED,
            "subsampled": NUM_ROW_GROUPS is not None,
        },
        "cells_without_nucleus": {
            "pct_cells_no_nucleus": pct_cells_no_nucleus,
            "denominator": "cells_with_total_counts_gt_0",
        },
        # % cells with zero gene transcripts (in-house, cells.parquet). Gated at
        # warn 5 / fail 10. None when transcript_counts is unavailable.
        "pct_cells_zero_transcripts": pct_cells_zero_transcripts,
        "fov_density_dropout_count": fov_density_dropout_count,
        "fov_density_pct_dropout": fov_density_pct_dropout,
        # 10x-defined bundle metrics read from metrics_summary.csv (None when
        # absent / blank / resegmented). See read_bundle_metrics().
        "bundle_metrics": read_bundle_metrics(XENIUM_BUNDLE_DIR, args.is_resegmented),
        # XOA (onboard analysis) version of the bundle, e.g. "xenium-4.0.1.0".
        "xoa_version": read_xenium_analysis_sw_version(XENIUM_BUNDLE_DIR),
        # Segmentation software that produced the bundle this report describes.
        "segmentation_software": resolve_segmentation_software(
            XENIUM_BUNDLE_DIR,
            args.pipeline_segmentation,
            args.is_resegmented,
            parse_tool_versions(args.seg_versions_file),
        ),
    }

    # Save metrics
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # EXACT CODE FROM ORIGINAL NOTEBOOK - Save versions
    dump_versions(
        str(outdir / "versions.yml"),
        ["matplotlib", "numpy", "pandas", "scipy", "seaborn", "scanpy", "pyarrow"],
        task_name=TASK_PROCESS,
    )

    _log_mem("processing complete")
    _log_mem_summary()
    print("\n=== Processing Complete ===")
    print(f"Generated {len(list(output_fig_dir.glob('*.pdf')))} figures")
    print(f"Generated {len(list(figures_source_dir.glob('*.csv')))} CSV data files")
    print(f"Saved metrics to: {output_metrics_path}")
    print(f"Output directory: {outdir}")
    print(f"Figures directory: {output_fig_dir}")
    print(f"Source data directory: {figures_source_dir}")


if __name__ == "__main__":
    main()
