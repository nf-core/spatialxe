#!/usr/bin/env python3
"""
Combined Image QC Script

This script combines three image QC scripts into one:
- image_qc_roi_processing.py: GPU-accelerated tile/pixel-level focus maps
- image_qc_processing.py: Cell-based QC with Quarto figures
- image_qc_mapping_to_cells.py: Maps tile results to cells

Author: Hanneke Okkenhaug, Malwina Prater
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import math
import os
import queue
import sys
import threading
import time
import traceback
import warnings
import click
import json
import numpy as np
import pandas as pd
import tifffile
import zarr
from pathlib import Path
import matplotlib.pyplot as plt
from napari_skimage_regionprops import regionprops_table
from skimage.segmentation import clear_border
from skimage import measure, color, morphology
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
import napari_simpleitk_image_processing as nsitk
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from tifffile import imread
import multiprocessing
import multiprocessing.connection
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from numpy.typing import NDArray
from scipy.ndimage import gaussian_laplace as scipy_gaussian_laplace
from scipy.ndimage import laplace as scipy_laplace
from scipy.ndimage import uniform_filter as scipy_uniform_filter
from scipy import ndimage
from sklearn.mixture import GaussianMixture

import snr_metrics

# Set matplotlib to use a non-interactive backend
import matplotlib

matplotlib.use("Agg")

# GPU backend detection (CuPy)
try:
    import cupy as cp  # type: ignore[import-untyped]
    import cupyx.scipy.ndimage  # type: ignore[import-untyped]  # noqa: F401  (binds `cupyx` for warmup)
    from cupyx.scipy.ndimage import laplace as cupy_laplace  # type: ignore[import-untyped]
    from cupyx.scipy.ndimage import uniform_filter as cupy_uniform_filter  # type: ignore[import-untyped]

    def cupy_gaussian_laplace(image, sigma):  # type: ignore[misc]
        """CuPy Laplacian of Gaussian: Gaussian smooth then Laplacian."""
        from cupyx.scipy.ndimage import gaussian_filter as _gf  # type: ignore[import-untyped]

        return cupy_laplace(_gf(image, sigma=sigma))

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ---------------------------------------------------------------------------
# Segmentation-software label helpers, vendored VERBATIM from upstream
# xenium_helpers.utils (bin/xenium_helpers/src/xenium_helpers/utils.py at
# nf-xenium-processing dev HEAD 5e35cae). This pipeline does not ship the
# xenium_helpers package, so the transitive closure needed by image_qc.py is
# inlined here to keep the script self-contained.
# Re-sync note: if upstream xenium_helpers.utils changes any of these symbols,
# re-copy this whole block verbatim rather than hand-patching it.
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


def read_xenium_major_version(bundle_dir) -> Optional[int]:
    """Major XOA version for a bundle, read from ``experiment.xenium``
    (``"xenium-4.0.1.0"`` -> ``4``). Returns ``None`` when the file is absent or
    the version cannot be parsed. Used to pick XOA-version-specific QC floors
    (e.g. intensity gates differ sharply between XOA 3.x and 4.0)."""
    parsed = _parse_xenium_version(read_xenium_analysis_sw_version(bundle_dir))
    if not parsed:
        return None
    first = parsed.split(".", 1)[0]
    return int(first) if first.isdigit() else None


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


# --------------------------- end vendored block ----------------------------

# Xenium pixel size in micrometers (used for coordinate conversions)
XENIUM_PIXEL_SIZE_UM = 0.2125

# Default CCFS threshold for classifying cells as low nuclear texture quality.
# CCFS measures per-cell nuclear contrast (local_var/local_mean), NOT optical blur.
# Calibrated on 5 samples (2026-04-02): cells below 0.02 lose >50% transcripts
# relative to top-50% CCFS cells in lung/liver.  Brain tissue is confounded by
# nucleus size (large neurons score lower) — interpret with caution.
DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD = 0.02

#       sensitivity for blurred cell detection (catches more background regions)
ROI_INTENSITY_THRESHOLD = 100.0
# 2026-06-26: tissue-tile gate lowered 0.5 -> 0.2. The background-from-nonzero mask is
# tighter (un-flooded), so real tissue tiles are only thinly covered; 0.5 discarded them.
# See plans/2026-06-26_PLAN_tissue-mask-recalibration.md.
ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC = 0.2
# Per-tile DAPI floor for the usable_tissue low-intensity term ONLY (decoupled from the
# is_low_intensity column, which stays at ROI_INTENSITY_THRESHOLD for the 1D-GMM tissue
# scope). Lowered to 50 to match the relaxed DAPI intensity QC (channels.DAPI.
# intensity_critical_v3/v4 = 50); a dim-but-real tissue tile should not count as unusable
# on absolute brightness alone.
DAPI_LOW_INTENSITY_FLOOR = 50.0


# Fallback intensity critical thresholds (used when YAML not provided)
_INTENSITY_CRITICAL_DEFAULTS = {"dapi": 500, "boundary": 100, "intrna": 300}

# Fixed per-channel colorbar caps for the §3.3 intensity spatial heatmaps.
# Calibrated against 9 tissues (lung / pancreas / liver / brain) — pooled p99
# was ~6300 (DAPI), ~7000 (Boundary), ~3600 (IntRNA). Caps were rounded down
# below pooled p99 so bright tissues (brain, pancreas) saturate to the
# extend="max" triangle and dim tissues (low-signal lung) genuinely render
# dim — cross-sample comparability over per-sample auto-scaling.
_INTENSITY_DISPLAY_CAP = {"dapi": 4000, "boundary": 4000, "intrna": 2000}


def _load_qc_thresholds(yaml_path: str | None) -> dict:
    """Load thresholds from roi_image_qc_thresholds.yaml.

    Returns the ``image_qc`` sub-dict, or an empty dict if the file
    is missing / unreadable.
    """
    if yaml_path is None:
        return {}
    try:
        import yaml

        p = Path(yaml_path)
        if not p.exists():
            logging.warning("Thresholds YAML not found: %s", yaml_path)
            return {}
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return d.get("image_qc", {}) if isinstance(d, dict) else {}
    except Exception as exc:
        logging.warning("Failed to load thresholds YAML: %s", exc)
        return {}


# Focus score percentile: Percentile of raw focus scores to use as threshold
# Applied to: tiles only after exclusion of low-intensity tiles (intensity >= ROI_INTENSITY_THRESHOLD)
# Purpose: Separate blurred vs in-focus tiles

ROI_FOCUS_SCORE_PERCENTILE = 5.0

# --- Per-cluster outlier detection (cluster_blur_outliers / cluster_ccfs_outliers) ---
# A cluster is flagged when its blur (or low-texture) rate is both a robust
# statistical outlier among the sample's clusters AND above an absolute floor.
CLUSTER_OUTLIER_MIN_CELLS = 50  # ignore tiny clusters
CLUSTER_OUTLIER_MIN_CLUSTERS = 5  # need enough clusters for robust stats
CLUSTER_OUTLIER_FLOOR_PCT = 15.0  # absolute minimum % to ever flag
CLUSTER_OUTLIER_MAD_Z = 3.5  # modified (MAD-based) z-score cutoff


def detect_cluster_outliers(cluster_stats, pct_key):
    """Flag clusters that stand apart from the rest of the sample.

    A cluster is flagged when its percentage (``pct_key``) is at least
    ``CLUSTER_OUTLIER_FLOOR_PCT`` AND it is a robust statistical outlier vs the
    other clusters, measured by a modified z-score ``(x - median) / (1.4826 *
    MAD) > CLUSTER_OUTLIER_MAD_Z``. When the MAD degenerates to 0 (common for
    low-texture, where most clusters sit near 0%), the floor alone decides, so a
    genuine spike is still caught. Needs at least ``CLUSTER_OUTLIER_MIN_CLUSTERS``
    clusters with ``>= CLUSTER_OUTLIER_MIN_CELLS`` cells; below that, robust
    stats are unreliable and nothing is flagged.

    Parameters
    ----------
    cluster_stats : dict[int, dict]
        Per-cluster stats, e.g. ``{cluster_id: {"n_cells": int, pct_key: float}}``.
    pct_key : str
        Key holding the percentage to test ("pct_blurred" or "pct_low_texture").

    Returns
    -------
    dict[int, dict]
        Subset of ``cluster_stats`` for flagged clusters, full stats retained
        (downstream report consumers read ``n_cells`` and the percentage).
    """
    eligible = {
        cl: s
        for cl, s in cluster_stats.items()
        if s.get("n_cells", 0) >= CLUSTER_OUTLIER_MIN_CELLS
    }
    if len(eligible) < CLUSTER_OUTLIER_MIN_CLUSTERS:
        return {}
    vals = np.array([eligible[cl][pct_key] for cl in eligible], dtype=float)
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    flagged = {}
    for cl, s in eligible.items():
        x = s[pct_key]
        if x < CLUSTER_OUTLIER_FLOOR_PCT:
            continue
        if mad <= 0 or (x - med) / mad > CLUSTER_OUTLIER_MAD_Z:
            flagged[cl] = s
    return flagged


def _run_figure_task(fn):
    """Wrapper for multiprocessing figure tasks that logs exceptions before exit.

    When using fork-based multiprocessing, child process tracebacks are lost
    if the child crashes. This wrapper catches and logs the full traceback
    in the child process before re-raising, so errors are visible in logs.
    """
    try:
        fn()
    except Exception:
        logging.error(f"Figure task {fn.__name__} failed:\n{traceback.format_exc()}")
        raise


def open_zarr(path: Path, zarr3: bool = False) -> zarr.Group:
    if zarr3:
        store = (
            zarr.storage.ZipStore(path)
            if path.suffix == ".zip"
            else zarr.storage.LocalStore(path)
        )
        return zarr.open_group(store=store, mode="r")
    else:
        """Open a Zarr file (compatible with zarr < 3)"""
        store = (
            zarr.ZipStore(path, mode="r")
            if path.suffix == ".zip"
            else zarr.DirectoryStore(path)
        )
        return zarr.group(store=store)


def load_and_prepare_data(xenium_bundle_dir, outdir):
    """Load all required data files and prepare paths"""

    xenium_bundle_dir = Path(xenium_bundle_dir)
    # Resolve outdir to absolute path to avoid nested directory issues
    # If outdir is already absolute, resolve() returns it as-is
    # If outdir is relative, resolve() makes it absolute relative to current working directory
    outdir = Path(outdir).resolve()

    # Required paths
    cells_parquet_path = xenium_bundle_dir / "cells.parquet"
    clusters_csv_path = (
        xenium_bundle_dir
        / "analysis"
        / "clustering"
        / "gene_expression_kmeans_10_clusters"
        / "clusters.csv"
    )
    umap_path = (
        xenium_bundle_dir
        / "analysis"
        / "umap"
        / "gene_expression_2_components"
        / "projection.csv"
    )
    cell_masks_path = xenium_bundle_dir / "cells.zarr.zip"
    morphology_focus_dir = xenium_bundle_dir / "morphology_focus"

    # Create output directories
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "xenium_bundle_dir": xenium_bundle_dir,
        "outdir": outdir,
        "figures_dir": figures_dir,
        "cells_parquet_path": cells_parquet_path,
        "clusters_csv_path": clusters_csv_path,
        "umap_path": umap_path,
        "cell_masks_path": cell_masks_path,
        "morphology_focus_dir": morphology_focus_dir,
    }


def _load_morphology_channels(
    xoa_morphology_files,
    *,
    level: int = 0,
):
    """
    Load DAPI/Boundary/IntRNA channels robustly from either:
    - a single multi-channel OME-TIFF, or
    - separate single-channel files (e.g. *_0000, *_0001, *_0002).
    """

    def _split_channels(arr):
        # Support both channel-first (C,Y,X) and channel-last (Y,X,C).
        # .copy() on each slice so the parent 3D array can be GC'd.
        if arr.ndim == 2:
            return arr, None, None
        if arr.ndim != 3:
            raise ValueError(f"Unexpected morphology array shape: {arr.shape}")
        if arr.shape[0] <= 4 and arr.shape[1] > 16 and arr.shape[2] > 16:
            d = arr[0].copy()
            b = arr[1].copy() if arr.shape[0] > 1 else None
            r = arr[2].copy() if arr.shape[0] > 2 else None
            return d, b, r
        if arr.shape[2] <= 4 and arr.shape[0] > 16 and arr.shape[1] > 16:
            d = arr[:, :, 0].copy()
            b = arr[:, :, 1].copy() if arr.shape[2] > 1 else None
            r = arr[:, :, 2].copy() if arr.shape[2] > 2 else None
            return d, b, r
        # Conservative fallback: prefer first axis as channels.
        d = arr[0].copy()
        b = arr[1].copy() if arr.shape[0] > 1 else None
        r = arr[2].copy() if arr.shape[0] > 2 else None
        return d, b, r

    primary = tifffile.imread(
        xoa_morphology_files[0], is_ome=False, level=level, aszarr=False
    )

    if primary.ndim == 3:
        dapi, boundary, intrna = _split_channels(primary)
        return dapi, boundary, intrna

    dapi = primary
    boundary = None
    intrna = None

    if len(xoa_morphology_files) > 1 and Path(xoa_morphology_files[1]).exists():
        try:
            b = tifffile.imread(
                xoa_morphology_files[1], is_ome=False, level=level, aszarr=False
            )
            boundary = _split_channels(b)[0] if getattr(b, "ndim", 0) == 3 else b
        except Exception as e:
            logging.warning(
                "Boundary channel load failed for %s (continuing with DAPI): %s",
                xoa_morphology_files[1],
                e,
            )
            boundary = None
    if len(xoa_morphology_files) > 2 and Path(xoa_morphology_files[2]).exists():
        try:
            r = tifffile.imread(
                xoa_morphology_files[2], is_ome=False, level=level, aszarr=False
            )
            intrna = _split_channels(r)[0] if getattr(r, "ndim", 0) == 3 else r
        except Exception as e:
            logging.warning(
                "IntRNA channel load failed for %s (continuing with DAPI): %s",
                xoa_morphology_files[2],
                e,
            )
            intrna = None

    return dapi, boundary, intrna


# Tissue-mask threshold guard (2026-06-23, fixes the generate_tissue_mask bug;
# see plans/2026-06-23_PLAN_fix-tissue-mask-bug.md and the Otsu spike). The old
# fixed 60th-percentile threshold assumed ~40% of the field is tissue and
# degenerated on sparse/dim slides. Otsu is background-aware and adapts to the
# real tissue fraction, but on a UNIMODAL field (all background or all tissue)
# Otsu still returns a split, fabricating a mask — so a guard rejects those.
# Calibrated on the spike's synthetic Gaussian fields: real bimodal class
# separation 8.9-24.4 sd vs unimodal 2.6-2.8 sd (clean margin around 3.0).
# CAVEAT: the 3.0 sd cutoff is synthetic-calibrated; confirm on real small0.
TISSUE_OTSU_GUARD_MIN_FG = 0.02  # foreground < 2% of field -> nothing detected
TISSUE_OTSU_GUARD_MAX_FG = 0.90  # foreground > 90% of field -> no background
TISSUE_OTSU_GUARD_MIN_SEP_SD = 3.0  # tissue mean must exceed bg mean by >= 3 bg-sd


def otsu_tissue_threshold_with_guard(small0):
    """Background-aware tissue threshold (Otsu) with a degeneracy guard.

    Returns ``(threshold, ok)``. ``ok=False`` means the field is unimodal /
    has no separable tissue, so no mask should be formed (the caller returns an
    empty mask, which downstream becomes ``tissue_mask_qc.status == FAIL``).

    The load-bearing test is class separation, not foreground fraction: on an
    all-background field Otsu splits the noise at ~40% foreground (inside any
    fraction band), so only the separation floor catches it.
    """
    arr = np.asarray(small0, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return None, False  # empty or uniform field
    t = float(threshold_otsu(finite))
    fg = arr >= t
    fg_frac = float(np.mean(fg))
    bg_vals = arr[(arr < t) & np.isfinite(arr)]
    fg_vals = arr[fg & np.isfinite(arr)]
    if bg_vals.size == 0 or fg_vals.size == 0:
        return t, False
    bg_std = float(bg_vals.std())
    sep_sd = (
        (float(fg_vals.mean()) - float(bg_vals.mean())) / bg_std
        if bg_std > 1e-9
        else np.inf
    )
    ok = (
        TISSUE_OTSU_GUARD_MIN_FG <= fg_frac <= TISSUE_OTSU_GUARD_MAX_FG
        and sep_sd >= TISSUE_OTSU_GUARD_MIN_SEP_SD
    )
    return t, ok


# Hysteresis tissue mask (2026-06-25): replaces the global-Otsu threshold, which
# under-captured dim/sparse tissue (validated non-circularly against decoded
# transcripts across 14 tissues — Otsu keeps ~20% of transcript tiles, hysteresis
# ~60%; see plans/2026-06-26_PLAN_tissue-mask-recalibration.md). Background mean + robust
# SD are estimated from NON-ZERO pixels (the level-3 DAPI overview is 57-80% exact zeros
# outside the imaged area; including them collapses the robust SD to 0 -> rsd falls back
# to 1 -> the hysteresis cutoffs become tiny and the mask floods). Estimating from
# non-zero pixels makes the spread genuinely per-sample. Tissue = pixels connected to a
# confident-bright seed (bg + SEED*robSD) grown down to a floor (bg + GROW*robSD), so the
# mask follows dim/uneven tissue without flooding background. The degeneracy guard is
# STRUCTURAL only (foreground fraction + class separation): verified empty-field-safe in
# plans/spikes/spike_empty_field_guard.py (an empty/noise field fails on fraction < 2% or
# separation < 3). No absolute brightness floor on the mask — absolute DAPI quality lives
# in usable_tissue via is_low_intensity (per-tile means, flood-immune).
TISSUE_HYST_SEED_SD = 3.0  # seed: confident tissue at bg + 3*robSD
TISSUE_HYST_GROW_SD = 1.0  # grow connected tissue down to bg + 1*robSD


def _robust_background_stats(small0):
    """``(bg_median, robust_sd)`` of the background, estimated from NON-ZERO pixels
    (Otsu split on the non-zero values, below-threshold = background). ``(None, None)``
    on a degenerate (empty / uniform) field. Estimating from non-zero pixels avoids the
    hard-zero collapse that flooded the mask (see module comment above)."""
    arr = np.asarray(small0, dtype=np.float64)
    nz = arr[np.isfinite(arr) & (arr > 0)]
    if nz.size == 0 or float(nz.max()) == float(nz.min()):
        return None, None
    t = float(threshold_otsu(nz))
    bg = nz[nz < t]
    if bg.size == 0:
        bg = nz
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    rsd = 1.4826 * mad if mad > 0 else 1.0
    return med, rsd


def hysteresis_tissue_mask_with_guard(small0):
    """Background-relative hysteresis tissue mask with a degeneracy guard.

    Returns ``(mask, low, ok)``:
      - mask: boolean tissue mask — pixels connected to a ``bg + 3*robSD`` seed,
        grown down to ``bg + 1*robSD``. Anchored to the field's own (non-zero)
        background, so it follows dim/uneven tissue instead of flooding.
      - low: the grow threshold (``bg + 1*robSD``); callers build the background
        components (edge/hole distance maps) from ``small0 < low``.
      - ok: ``False`` when the field is degenerate / has no separable tissue, so the
        caller returns an empty mask -> ``tissue_mask_qc.status == FAIL``.

    Guard (STRUCTURAL, both required): foreground fraction in ``[MIN_FG, MAX_FG]`` AND
    class separation ``>= MIN_SEP_SD`` background-SD. No absolute brightness floor:
    verified empty-field-safe in plans/spikes/spike_empty_field_guard.py (empty/noise
    fields fail on fraction < 2% or separation < 3; real tissue passes). Absolute DAPI
    quality is judged separately by usable_tissue (is_low_intensity), not by the mask.
    """
    bg_med, rsd = _robust_background_stats(small0)
    if bg_med is None:
        return np.zeros_like(small0, dtype=bool), None, False
    low = bg_med + TISSUE_HYST_GROW_SD * rsd
    high = bg_med + TISSUE_HYST_SEED_SD * rsd
    arr = np.asarray(small0, dtype=np.float64)
    mask = apply_hysteresis_threshold(arr, low, high)
    fg_vals = arr[mask & np.isfinite(arr)]
    bg_vals = arr[(~mask) & np.isfinite(arr)]
    if fg_vals.size == 0 or bg_vals.size == 0:
        return mask, low, False
    fg_frac = float(np.mean(mask))
    bg_std = float(bg_vals.std())
    sep_sd = (
        (float(fg_vals.mean()) - float(bg_vals.mean())) / bg_std
        if bg_std > 1e-9
        else np.inf
    )
    ok = (
        TISSUE_OTSU_GUARD_MIN_FG <= fg_frac <= TISSUE_OTSU_GUARD_MAX_FG
        and sep_sd >= TISSUE_OTSU_GUARD_MIN_SEP_SD
    )
    return mask, low, ok


def compute_tissue_mask(small0, min_size_hole=1500):
    """Shared tissue-mask logic — the single source of truth for the
    threshold + labelling, used by generate_tissue_mask AND the
    calculate_roi_focusscore* tissue filters so the three call sites cannot
    drift (2026-06-23 bug fix; previously the logic was copy-pasted three times,
    each with the `percentile 60` + `test_mask > 1` defects).

    2026-06-25: tissue threshold is now background-relative HYSTERESIS (see
    hysteresis_tissue_mask_with_guard) instead of global Otsu, which under-captured
    dim/sparse tissue. `test_mask > 0` keeps all foreground components.

    Returns ``(whole_sample, objects, holes)``:
      - whole_sample: labelled tissue mask. Empty when the guard rejects a
        degenerate / faint field (no separable tissue) -> downstream
        ``tissue_mask_qc.status == FAIL`` rather than a fabricated mask.
      - objects: labelled background components (``label(small0 < low)``, the
        hysteresis grow threshold) — callers that need the edge/distance map reuse
        this.
      - holes: labelled holes (border-cleared background components, small ones
        removed) — callers that need the hole distance map reuse this.
    """
    mask, low, ok = hysteresis_tissue_mask_with_guard(small0)
    if ok:
        thresh1 = small0 < low  # background (below the hysteresis grow threshold)
        thresh2 = mask  # tissue
    else:
        thresh1 = np.ones_like(small0, dtype=bool)
        thresh2 = np.zeros_like(small0, dtype=bool)
    objects = measure.label(thresh1)
    noborder = clear_border(objects)
    holes = morphology.remove_small_objects(noborder, min_size=min_size_hole)
    small_objects = noborder ^ holes
    test_mask = measure.label(thresh2) + small_objects
    whole_sample = measure.label(test_mask > 0)
    return whole_sample, objects, holes


def compute_multistain_tissue_mask(small0, small1, small2, min_size_hole=1500):
    """Tissue EXTENT mask combining all available morphology stains.

    DAPI nuclei are sparse in some tissues (muscle fibres, brain neuropil), so a
    DAPI-only mask under-captures them; the Boundary (membrane) and Interior
    (cytoplasm/rRNA) stains cover those regions. This builds a per-channel
    background-relative hysteresis mask (reusing hysteresis_tissue_mask_with_guard, so
    each channel is guarded against its OWN background) and ORs the channels that pass
    their guard. The union gets a foreground-fraction sanity check only (the SD-relative
    separation test is per-channel and undefined on a boolean union).

    Used for tissue EXTENT / coverage ONLY. The DAPI-only mask (compute_tissue_mask)
    still drives the focus/blur QC, because feeding nuclei-poor tiles into the DAPI
    focus GMM falsely reads as blurry. See plans/2026-06-26_PLAN_multistain-mask.md.

    DAPI-only bundles (small1 and small2 both None) return EXACTLY
    compute_tissue_mask(small0) — byte-identical to the DAPI-only behaviour.

    Returns ``(whole_sample, objects, holes)`` like compute_tissue_mask. The union's
    own background components (``objects``/``holes``) are rebuilt from ``~union`` (a
    union has no single grow threshold), so edge/hole geometry reflects the combined
    tissue extent.

    NOTE: dense-intensity-artefact subtraction is intentionally NOT applied — the
    dense-intensity mask is the brightest p97 of each channel, which includes real
    bright tissue, so subtracting it would remove real tissue. Artefact handling is a
    deferred follow-up (the spike measured ~1% false tissue without it).
    """
    if small1 is None and small2 is None:
        return compute_tissue_mask(small0, min_size_hole=min_size_hole)

    union = None
    for ch in (small0, small1, small2):
        if ch is None:
            continue
        mask, _low, ok = hysteresis_tissue_mask_with_guard(ch)
        if ok:
            union = mask if union is None else (union | mask)
    if union is None:
        union = np.zeros_like(small0, dtype=bool)

    fg_frac = float(union.mean())
    if TISSUE_OTSU_GUARD_MIN_FG <= fg_frac <= TISSUE_OTSU_GUARD_MAX_FG:
        thresh1 = ~union  # background = everything outside the combined tissue
        thresh2 = union
    else:
        thresh1 = np.ones_like(small0, dtype=bool)
        thresh2 = np.zeros_like(small0, dtype=bool)
    objects = measure.label(thresh1)
    noborder = clear_border(objects)
    holes = morphology.remove_small_objects(noborder, min_size=min_size_hole)
    small_objects = noborder ^ holes
    test_mask = measure.label(thresh2) + small_objects
    whole_sample = measure.label(test_mask > 0)
    return whole_sample, objects, holes


def _per_tile_coverage(whole_sample, x1, x2, y1, y2, downsample_factor=8):
    """Per-tile tissue-coverage fraction for ROI tiles given in full-resolution coords,
    computed from a labelled `whole_sample` mask at `downsample_factor` resolution via a
    summed-area table (same logic as the grid builders). Returns a float array aligned to
    the x1/x2/y1/y2 arrays."""
    mask = np.asarray(whole_sample) > 0
    h, w = mask.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    r1 = np.clip(np.asarray(y1) // downsample_factor, 0, h)
    r2 = np.clip((np.asarray(y2) - 1) // downsample_factor + 1, 0, h)
    c1 = np.clip(np.asarray(x1) // downsample_factor, 0, w)
    c2 = np.clip((np.asarray(x2) - 1) // downsample_factor + 1, 0, w)
    area = np.maximum((r2 - r1) * (c2 - c1), 1)
    s = ii[r2, c2] - ii[r1, c2] - ii[r2, c1] + ii[r1, c1]
    return s.astype(np.float64) / area


def generate_tissue_mask(
    xoa_morphology_files,
    small0,
    small1,
    small2,
    threshold_percentile=60,
    min_size_edge=500000,
    min_size_hole=1500,
    dense_intensity_region_percentile=97,
    min_size_dense_intensity_region=500,
    downsample_factor=8,
):
    """
    Generate tissue masks and distance maps from morphology images (cell/segmentation independent).

    Parameters:
    -----------
    xoa_morphology_files : list
        List of paths to morphology image files (for compatibility, not used if small0/1/2 provided)
    small0, small1, small2 : numpy.ndarray
        Downsampled morphology images (DAPI, Boundary, Interior)
    threshold_percentile : float, optional
        Percentile for thresholding (default: 60)
    min_size_edge : int, optional
        Minimum size for edge objects in downsampled space (default: 500000)
    min_size_hole : int, optional
        Minimum size for holes in downsampled space (default: 1500)
    dense_intensity_region_percentile : float, optional
        Percentile for dense intensity region detection (default: 97)
    min_size_dense_intensity_region : int, optional
        Minimum size for dense intensity regions (default: 500)
    downsample_factor : int, optional
        Downsampling factor (default: 8, for level 3)

    Returns:
    --------
    tuple
        (whole_sample, holes, dense_intensity_regions, distance_map, distance_map2,
         multistain_whole_sample, multistain_distance_map, multistain_distance_map2)
        - whole_sample: Labeled DAPI tissue mask (drives focus/blur QC)
        - holes: Labeled holes mask (DAPI)
        - dense_intensity_regions: Labeled dense intensity regions mask
        - distance_map: Distance to edge map (DAPI)
        - distance_map2: Distance to nearest hole map (DAPI)
        - multistain_whole_sample: Labeled multi-stain tissue-EXTENT mask (DAPI OR Boundary
          OR Interior), or None on DAPI-only bundles. Drives the reported coverage / extent.
        - multistain_distance_map / multistain_distance_map2: edge / hole distance maps for
          the multi-stain mask (None on DAPI-only bundles)
    """
    # Tissue mask via the shared helper (hysteresis + degeneracy guard + `> 0`).
    # 2026-06-23 bug fix: the old `np.percentile(small0, threshold_percentile)`
    # (60) assumed ~40% of the field is tissue and degenerated on sparse/dim
    # slides, and `test_mask > 1` emptied the mask on a single whole-field blob.
    # `threshold_percentile` is retained in the signature but no longer used.
    # See compute_tissue_mask and plans/2026-06-23_PLAN_fix-tissue-mask-bug.md.
    whole_sample, objects, holes = compute_tissue_mask(
        small0, min_size_hole=min_size_hole
    )

    # Edge + distance maps (generate_tissue_mask-specific; reuse `objects`/`holes`).
    mask = morphology.remove_small_objects(objects, min_size=min_size_edge)
    edge_sample = measure.label(mask)
    distance_map = nsitk.signed_maurer_distance_map(edge_sample)
    distance_map2 = nsitk.signed_maurer_distance_map(holes)

    # Detecting dense intensity regions — multi-channel co-thresholding when
    # Boundary / IntRNA channels are present, DAPI-only fallback otherwise.
    # `small1` / `small2` are None for DAPI-only bundles (returned from
    # `_load_morphology_channels`); blindly indexing them would crash.
    t0 = np.percentile(small0, dense_intensity_region_percentile)
    thresh_sum = (small0 >= t0).astype(np.int_)
    if small1 is not None:
        t1 = np.percentile(small1, dense_intensity_region_percentile)
        thresh_sum = thresh_sum + (small1 >= t1).astype(np.int_)
    if small2 is not None:
        t2 = np.percentile(small2, dense_intensity_region_percentile)
        thresh_sum = thresh_sum + (small2 >= t2).astype(np.int_)
    thresh_fill = nsitk.binary_fill_holes(thresh_sum)
    objects_art = measure.label(thresh_fill)
    dense_intensity_regions = morphology.remove_small_objects(
        objects_art, min_size=min_size_dense_intensity_region
    )

    # Multi-stain tissue-EXTENT mask (DAPI OR Boundary OR Interior) + its edge/hole distance
    # maps, for the extent metrics and the §2.3 figure. None on DAPI-only bundles so the
    # extent path falls back to the DAPI coverage and stays byte-identical.
    if small1 is None and small2 is None:
        ms_whole_sample = ms_distance_map = ms_distance_map2 = None
    else:
        ms_whole_sample, ms_objects, ms_holes = compute_multistain_tissue_mask(
            small0, small1, small2, min_size_hole=min_size_hole
        )
        ms_edge = morphology.remove_small_objects(ms_objects, min_size=min_size_edge)
        ms_distance_map = nsitk.signed_maurer_distance_map(measure.label(ms_edge))
        ms_distance_map2 = nsitk.signed_maurer_distance_map(ms_holes)

    return (
        whole_sample,
        holes,
        dense_intensity_regions,
        distance_map,
        distance_map2,
        ms_whole_sample,
        ms_distance_map,
        ms_distance_map2,
    )


# ---------------------------------------------------------------------------
# Internal helpers (from focus_score_maps)
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-8


def _get_backend(
    use_gpu: bool,
) -> tuple[Any, Any, Any]:
    """Return the array module, uniform_filter, and laplace for the chosen backend.

    Args:
        use_gpu: If ``True`` and CuPy is available, return GPU primitives.

    Returns:
        Tuple of ``(array_module, uniform_filter_fn, laplace_fn)``.

    Raises:
        RuntimeError: If ``use_gpu`` is ``True`` but CuPy is not installed.
    """
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError(
                "use_gpu=True but CuPy is not installed. "
                "Install CuPy or set use_gpu=False."
            )
        return cp, cupy_uniform_filter, cupy_laplace
    return np, scipy_uniform_filter, scipy_laplace


def _to_device(image: NDArray[np.generic], xp: Any) -> Any:
    """Move a numpy array to the target device (no-op for NumPy).

    Args:
        image: Input numpy array.
        xp: Array module (``numpy`` or ``cupy``).

    Returns:
        Array on the target device.
    """
    if xp is np:
        return image
    return cp.asarray(image)


def _to_numpy(arr: Any, xp: Any) -> NDArray[np.float32]:
    """Move an array back to host memory as float32 numpy (no-op for NumPy).

    Args:
        arr: Array on device.
        xp: Array module that created *arr*.

    Returns:
        numpy float32 array on the host.
    """
    if xp is np:
        return np.asarray(arr, dtype=np.float32)
    return cp.asnumpy(arr).astype(np.float32)


def _sanitize(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Replace NaN and Inf values with zero.

    Args:
        arr: Input array (modified in-place).

    Returns:
        The same array with non-finite values set to zero.
    """
    arr[~np.isfinite(arr)] = 0.0
    return arr


#: Longest-axis pixel budget for arrays sent to imshow/contour/label2rgb. A panel
#: is only ~1600 px at dpi 300, so imshow of an 86-megapixel map (12755x6738)
#: resamples all of it and discards ~97%: measured 15.9 s vs 0.5 s downsampled
#: first (32x). 2000 keeps every pixel a >=300-dpi panel can resolve.
_FIG_DISPLAY_MAX_PX = 2000


def _thumb(arr: Any, max_long: int = _FIG_DISPLAY_MAX_PX) -> Any:
    """Downsample a 2-D array to <= ``max_long`` on its long axis, for DISPLAY only.

    Float arrays are area-averaged (block mean, NaN-safe) to match matplotlib's own
    antialiased downscale, so the rendered panel is visually identical to
    ``imshow(arr)``. Integer/bool arrays (labels, masks) are block-MAX reduced (a mean
    of labels is meaningless): a block stays non-zero if any pixel in it is, so a
    feature thinner than ``step`` still renders instead of falling between strided
    rows. No-op on already-small / non-2-D input. Never use for exported data -- only
    at the draw call. Pair with ``extent`` of the *original* shape (see
    ``_imshow_thumb``) so axes are unchanged.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    # Ceil, not floor: floor overshoots the cap (12755 // 2000 = 6 leaves 2125 px;
    # 3000 // 2000 = 1 would not downsample a 3000 px axis at all).
    step = max(1, -(-max(a.shape) // max_long))
    if step == 1:
        return a
    ny, nx = (a.shape[0] // step) * step, (a.shape[1] // step) * step
    blocks = a[:ny, :nx].reshape(ny // step, step, nx // step, step)
    if np.issubdtype(a.dtype, np.floating):
        with warnings.catch_warnings():  # all-NaN blocks -> NaN, as intended
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(blocks, axis=(1, 3))
    return blocks.max(axis=(1, 3))


#: Target number of display bins along the long axis for Figure 5's binned focus
#: and classification heatmaps. ~180 keeps regional signal readable while dropping
#: the per-pixel speckle of an ~86-megapixel field, and it draws in ~1 s instead of
#: the ~126 s an imshow of the full field cost.
_FOCUS_HEATMAP_BINS_LONG = 180


def _bin_nanmean(arr: Any, step: int) -> Any:
    """Block-reduce a 2-D float array by ``step``x``step``, averaging over the finite
    (non-NaN) pixels of each block. Blocks with no finite pixel return NaN.

    NaN encodes "no tissue here" for the caller: pre-set non-tissue pixels to NaN and
    this returns the per-bin mean over tissue only. Uses the same NaN-safe block-mean as
    :func:`_thumb`; the all-NaN ``RuntimeWarning`` is suppressed because an empty
    (non-tissue) bin returning NaN is the intended result, not an error.
    """
    a = np.asarray(arr)
    ny = (a.shape[0] // step) * step
    nx = (a.shape[1] // step) * step
    blocks = a[:ny, :nx].reshape(ny // step, step, nx // step, step)
    with warnings.catch_warnings():  # all-NaN blocks -> NaN, as intended
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def _imshow_thumb(ax: Any, arr: Any, *, rgb: Any = None, **kwargs: Any) -> Any:
    """``ax.imshow`` of ``arr`` downsampled for speed but with the extent/axes of the
    FULL ``arr``, so the panel is visually identical to ``ax.imshow(arr)``. ``rgb`` is
    an optional callable (e.g. ``label2rgb``) applied to the downsampled array."""
    a = np.asarray(arr)
    disp = _thumb(a)
    if rgb is not None:
        disp = rgb(disp)
    if a.ndim == 2:
        kwargs.setdefault("extent", (-0.5, a.shape[1] - 0.5, a.shape[0] - 0.5, -0.5))
    return ax.imshow(disp, **kwargs)


def _contour_thumb(ax: Any, mask: Any, **kwargs: Any) -> Any:
    """``ax.contour`` of a boolean/label ``mask`` downsampled to display resolution,
    with X/Y mapped to the FULL-res pixel coordinate space so the boundary overlays an
    ``_imshow_thumb`` panel exactly. Runs marching-squares on the ~2000-px thumbnail
    instead of the full ~86-megapixel field (block-MAX keeps every thin boundary), so
    the traced contour is visually identical to ``ax.contour(mask)`` at >= 300 dpi but
    costs ~0.1 s instead of several seconds of contouring + tens of thousands of vector
    segments."""
    a = np.asarray(mask)
    m = _thumb(a)
    ny, nx = m.shape
    h, w = a.shape
    xs = np.linspace(0, w - 1, nx)
    ys = np.linspace(0, h - 1, ny)
    return ax.contour(xs, ys, m, **kwargs)


def _kde_density_grid(kde: Any, xs: Any, ys: Any, gridsize: int = 128) -> Any:
    """Evaluate a fitted ``gaussian_kde`` at every ``(xs, ys)`` via a coarse grid +
    bilinear interpolation, instead of ``kde(all points)`` which is O(N * n_fit) and
    dominates the large density-scatter figures (>100 s at N=531k, ~16 s here). Keeps
    ALL points -- no outlier dropped -- and the density coloring is visually identical
    (Spearman 1.000, max normalized-colour diff 6e-4 vs the exact eval)."""
    from scipy.interpolate import RegularGridInterpolator

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    gx = np.linspace(float(xs.min()), float(xs.max()), gridsize)
    gy = np.linspace(float(ys.min()), float(ys.max()), gridsize)
    grid_x, grid_y = np.meshgrid(gx, gy)
    gz = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(gridsize, gridsize)
    interp = RegularGridInterpolator((gy, gx), gz, bounds_error=False, fill_value=None)
    return interp(np.column_stack([ys, xs]))


def detect_gpu_ids() -> list[int]:
    """Detect available CUDA GPUs.

    Returns:
        List of GPU device IDs. Empty list if CuPy is not available or no GPUs
        are detected.
    """
    if not HAS_CUPY:
        logging.info("CuPy is not importable; GPU backend unavailable, using CPU.")
        return []
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        return list(range(n_devices))
    except Exception as exc:
        # CuPy is installed but the CUDA runtime could not be queried (driver
        # missing, GPU not attached to the container, init failure, ...). Surface
        # it: otherwise this is indistinguishable from "no GPU present" and a
        # silent CPU fallback on a GPU node looks like correct behaviour.
        logging.warning(
            "CuPy is installed but GPU detection failed (%s: %s); "
            "falling back to CPU backend.",
            type(exc).__name__,
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# Focus-map computation
# ---------------------------------------------------------------------------


def compute_ccfs_map(
    image: NDArray[np.generic],
    window_size: int = 35,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute a per-pixel CCFS (Coefficient of Contrast Focus Score) map.

    The CCFS focus score at each pixel is defined as::

        focus = local_var / local_mean

    which is equivalent to ``std**2 / mean`` computed over a square window
    centred on that pixel.

    Args:
        image: 2-D input image (any numeric dtype).
        window_size: Side length of the square averaging window.
        use_gpu: Use CuPy GPU backend when ``True``.
        gpu_id: CUDA device ID to use when ``use_gpu=True``.

    Returns:
        Tuple of ``(focus_map, mean_map)`` — both float32 arrays with the
        same shape as *image*.

    Raises:
        ValueError: If *image* is not 2-D or is smaller than the window in
            either dimension.
        RuntimeError: If *use_gpu* is ``True`` but CuPy is unavailable.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}.")
    if image.shape[0] < window_size or image.shape[1] < window_size:
        raise ValueError(
            f"Image shape {image.shape} is smaller than window_size "
            f"{window_size} in at least one dimension."
        )

    xp, uniform_filter, _ = _get_backend(use_gpu)

    if use_gpu:
        with cp.cuda.Device(gpu_id):
            image_f = cp.asarray(image.astype(np.float32, copy=False))
            local_mean = uniform_filter(image_f, size=window_size)
            local_sq_mean = uniform_filter(image_f**2, size=window_size)
            # Promote to float64 for subtraction to avoid catastrophic cancellation
            local_var = (
                local_sq_mean.astype(cp.float64) - local_mean.astype(cp.float64) ** 2
            )
            local_var = xp.maximum(local_var, 0.0)
            focus_map_dev = (
                local_var / (local_mean.astype(cp.float64) + _EPSILON)
            ).astype(cp.float32)
            focus_map = _to_numpy(focus_map_dev, xp)
            mean_map = _to_numpy(local_mean, xp)
    else:
        image_f = image.astype(np.float32, copy=False)
        local_mean = uniform_filter(image_f, size=window_size)
        sq = image_f**2
        del image_f
        local_sq_mean = uniform_filter(sq, size=window_size)
        del sq
        # Promote to float64 for the subtraction to avoid catastrophic
        # cancellation on high-intensity uint16 images.
        # Explicit steps to limit peak memory (avoid 3+ simultaneous f64 temps).
        local_var = local_sq_mean.astype(np.float64)
        del local_sq_mean
        temp_mean_f64 = local_mean.astype(np.float64)
        local_var -= temp_mean_f64**2
        del temp_mean_f64
        local_var = np.maximum(local_var, 0.0)
        # Compute focus_map = local_var / (local_mean + eps).
        # Save mean_map first, then free local_mean before creating f64 temp.
        mean_map = local_mean.astype(np.float32)
        temp_denom = local_mean.astype(np.float64)
        del local_mean
        temp_denom += _EPSILON
        local_var /= temp_denom
        del temp_denom
        focus_map = local_var.astype(np.float32)
        del local_var

    return _sanitize(focus_map), _sanitize(mean_map)


def compute_laplacian_variance_map(
    image: NDArray[np.generic],
    window_size: int = 35,
    use_gpu: bool = False,
    gpu_id: int = 0,
    lap_sigma: float = 1.0,
) -> NDArray[np.float32]:
    """Compute a per-pixel windowed Laplacian-variance focus map.

    This is a *local* variant of the standard Laplacian variance focus metric
    (Pech-Pacheco et al., 2000).  Instead of computing a single global
    variance, it produces a spatial map where each pixel holds the variance
    of the Laplacian of Gaussian (LoG) response inside the surrounding
    *window_size* × *window_size* neighbourhood::

        lap_var = uniform_filter(lap**2) - uniform_filter(lap)**2

    Using LoG (``gaussian_laplace`` with *lap_sigma*) rather than the bare
    3×3 Laplacian suppresses pixel-level noise and makes the metric more
    specific to genuine edge content (Sun et al., 2004; Pertuz et al., 2013).

    Variance is computed in float64 to avoid catastrophic cancellation in the
    ``E[X²] − E[X]²`` formula, then cast back to float32.

    Args:
        image: 2-D input image (any numeric dtype).
        window_size: Side length of the square averaging window.
        use_gpu: Use CuPy GPU backend when ``True``.
        gpu_id: CUDA device ID to use when ``use_gpu=True``.
        lap_sigma: Gaussian sigma for LoG pre-smoothing (pixels).
            Set to 0 to revert to the bare Laplacian.

    Returns:
        Float32 array with the same shape as *image*.

    Raises:
        ValueError: If *image* is not 2-D or is smaller than the window in
            either dimension.
        RuntimeError: If *use_gpu* is ``True`` but CuPy is unavailable.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}.")
    if image.shape[0] < window_size or image.shape[1] < window_size:
        raise ValueError(
            f"Image shape {image.shape} is smaller than window_size "
            f"{window_size} in at least one dimension."
        )

    xp, uniform_filter, laplace_fn = _get_backend(use_gpu)

    if use_gpu:
        with cp.cuda.Device(gpu_id):
            image_f = cp.asarray(image.astype(np.float32, copy=False))
            if lap_sigma > 0:
                lap = cupy_gaussian_laplace(image_f, sigma=lap_sigma)
            else:
                lap = laplace_fn(image_f)
            lap = lap.astype(cp.float64)
            lap_mean = uniform_filter(lap, size=window_size)
            lap_sq_mean = uniform_filter(lap * lap, size=window_size)
            lap_var = lap_sq_mean - lap_mean**2
            lap_var = xp.maximum(lap_var, 0.0)
            result = _to_numpy(lap_var.astype(cp.float32), xp)
    else:
        image_f = image.astype(np.float32, copy=False)
        if lap_sigma > 0:
            lap = scipy_gaussian_laplace(image_f, sigma=lap_sigma).astype(np.float64)
        else:
            lap = laplace_fn(image_f).astype(np.float64)
        del image_f
        lap_sq = lap * lap
        lap_mean = uniform_filter(lap, size=window_size)
        del lap
        lap_sq_mean = uniform_filter(lap_sq, size=window_size)
        del lap_sq
        np.square(lap_mean, out=lap_mean)  # in-place to avoid temporary
        lap_var = lap_sq_mean - lap_mean
        del lap_sq_mean, lap_mean
        lap_var = np.maximum(lap_var, 0.0)
        result = lap_var.astype(np.float32)
        del lap_var

    return _sanitize(result)


def _compute_channel_maps_on_gpu(
    channel: NDArray[np.generic],
    window_size: int,
    gpu_id: int,
    include_laplacian: bool = False,
    lap_sigma: float = 1.0,
    keep_mean_device: bool = False,
    keep_focus_device: bool = False,
    drop_mean_host: bool = False,
) -> dict[str, NDArray[np.float32]]:
    """Compute focus + mean maps for a single channel on a specific GPU.

    This fused implementation opens a single device context and transfers the
    image to the GPU once, computing CCFS (focus_map, mean_map) and optionally
    the Laplacian variance map within the same context. This avoids redundant
    CPU-to-GPU transfers of the full image.

    Args:
        channel: 2-D image array.
        window_size: Convolution window size.
        gpu_id: CUDA device ID.
        include_laplacian: Also compute Laplacian variance map.
        lap_sigma: Gaussian sigma for LoG pre-smoothing (0 = bare Laplacian).
        keep_mean_device: Also return the device (CuPy) mean map under
            ``mean_map_device`` without copying it to host, so a consumer can fold
            a reduction (the per-ROI Otsu SNR) on the GPU. Kept alive through the
            Laplacian phase and freed by the caller once folded.
        keep_focus_device: Same as *keep_mean_device* for the focus map, returned
            under ``focus_map_device``. The per-nucleus CCFS (``LabeledSumAccumulator``)
            and centre-pixel sampling fold on-device from it, so the full focus map
            never has to be read back for those reductions.
        drop_mean_host: Skip the ``mean_map`` host copy entirely. Valid only when no
            consumer reads the host mean map (all mean reductions run on-device from
            ``mean_map_device``); this is the transfer the device-resident fold
            eliminates. The host ``focus_map`` copy is always produced -- the Figure 5
            heatmap reduces it in float32 on the host, which a GPU reduction cannot
            reproduce to float rounding, so that copy stays.

    Returns:
        Dict with ``focus_map``, ``mean_map`` (unless *drop_mean_host*), optionally
        ``lap_var_map``, and -- when the respective flag is set --
        ``mean_map_device`` / ``focus_map_device`` (CuPy arrays on *gpu_id*).
    """
    if not HAS_CUPY:
        raise RuntimeError(
            "_compute_channel_maps_on_gpu requires CuPy but it is not installed."
        )

    pool = cp.get_default_memory_pool()

    with cp.cuda.Device(gpu_id):
        # Upload image to GPU
        image_f = cp.asarray(channel.astype(np.float32, copy=False))

        # --- CCFS (focus_map + mean_map) ---
        # Peak memory: max 3 arrays alive at once (~10.5GB for 34K×26K images)
        local_mean = cupy_uniform_filter(image_f, size=window_size)
        sq = image_f**2
        # Free image_f before allocating local_sq_mean to cap at 3 arrays
        del image_f
        pool.free_all_blocks()
        local_sq_mean = cupy_uniform_filter(sq, size=window_size)
        del sq
        # Promote to float64 for subtraction to avoid catastrophic cancellation
        local_var = (
            local_sq_mean.astype(cp.float64) - local_mean.astype(cp.float64) ** 2
        )
        del local_sq_mean
        cp.maximum(local_var, 0.0, out=local_var)
        focus_map_dev = (local_var / (local_mean.astype(cp.float64) + _EPSILON)).astype(
            cp.float32
        )
        del local_var

        # Transfer CCFS results to CPU. The host focus map is always produced --
        # the Figure 5 heatmap (BlockMeanAccumulator) reduces it in float32 on the
        # host, and a GPU reduction of float32 does not reproduce numpy's float32
        # summation order (measured ~2e-7 relative, enough to move Figure 5 pixels;
        # see docs/failures/2026-07-26_heatmap-16px-dtype-not-summation-order.md), so
        # that copy stays. The host mean map is skipped when every mean reduction
        # folds on-device (drop_mean_host) -- that is the D2H transfer this path drops.
        focus_map = cp.asnumpy(focus_map_dev).astype(np.float32)
        result: dict[str, NDArray[np.float32]] = {"focus_map": _sanitize(focus_map)}
        if not drop_mean_host:
            mean_map = cp.asnumpy(local_mean).astype(np.float32)
            result["mean_map"] = _sanitize(mean_map)

        # --- Device-resident maps handed to the on-GPU consumers ---
        # _sanitize each device array in place so it matches the host copy above
        # bit-for-bit: local_mean/focus are uniform_filter derivatives of a finite
        # image so this is a no-op in practice, but it keeps the on-device and host
        # paths structurally identical. Each kept array adds one float32 map of VRAM,
        # held past this block (through the Laplacian phase); the caller drops it
        # after folding.
        if keep_focus_device:
            focus_map_dev[~cp.isfinite(focus_map_dev)] = 0.0
            result["focus_map_device"] = focus_map_dev
        else:
            del focus_map_dev
        if keep_mean_device:
            # Hand RoiOtsuSnrAccumulator / LabeledSumAccumulator the device mean map
            # so their per-ROI / per-label reductions fold on the GPU (no full-map
            # D2H copy for those reductions).
            local_mean[~cp.isfinite(local_mean)] = 0.0
            result["mean_map_device"] = local_mean
        else:
            del local_mean
        pool.free_all_blocks()
        if include_laplacian:
            image_f = cp.asarray(channel.astype(np.float32, copy=False))
            if lap_sigma > 0:
                lap = cupy_gaussian_laplace(image_f, sigma=lap_sigma)
            else:
                lap = cupy_laplace(image_f)
            del image_f
            pool.free_all_blocks()
            lap = lap.astype(cp.float64)
            lap_mean = cupy_uniform_filter(lap, size=window_size)
            sq = lap * lap
            del lap
            pool.free_all_blocks()
            lap_sq_mean = cupy_uniform_filter(sq, size=window_size)
            del sq
            lap_var = lap_sq_mean - lap_mean**2
            del lap_sq_mean, lap_mean
            cp.maximum(lap_var, 0.0, out=lap_var)
            lap_var_np = cp.asnumpy(lap_var).astype(np.float32)
            del lap_var
            pool.free_all_blocks()
            result["lap_var_map"] = _sanitize(lap_var_np)

    return result


# Shared TIFF-tile decode pool. page.decode (imagecodecs) releases the GIL, so
# decoding the compressed blobs across a pool sized to the CPU count saturates the
# cores that would otherwise idle while the 4 GPU workers wait on a single-threaded
# decoder -- decode was the tile-pass wall (measured). Sized to the CPU count so the
# total decode concurrency is bounded by cores regardless of GPU count (no
# oversubscription). Lazily created; lives for the process (cleaned up at exit).
_DECODE_POOL: ThreadPoolExecutor | None = None
_DECODE_POOL_LOCK = threading.Lock()


def _get_decode_pool() -> ThreadPoolExecutor:
    global _DECODE_POOL
    if _DECODE_POOL is None:
        with _DECODE_POOL_LOCK:
            if _DECODE_POOL is None:
                _DECODE_POOL = ThreadPoolExecutor(
                    max_workers=max(2, os.cpu_count() or 4),
                    thread_name_prefix="tiff-decode",
                )
    return _DECODE_POOL


class _LazyTiffChannel:
    """Lazy 2-D view into one channel page of a TIFF file.

    Supports ``[y0:y1, x0:x1]`` slicing.  For tiled TIFFs (production
    Xenium images) only the overlapping tiles are decoded, keeping I/O
    minimal.  For non-tiled TIFFs (e.g. test images written by
    ``tifffile.imwrite``) the full page is cached on first access.

    Thread-safe: a lock serialises file-handle reads so that
    ``_process_tile_on_gpu`` can call ``[slice]`` from multiple threads.
    """

    def __init__(self, page, source: tuple[str, int] | None = None):
        """Accept a ``TiffPage`` or ``TiffFrame``.

        ``TiffFrame`` (non-first pages in a multi-page TIFF) lacks
        ``imagelength`` / ``is_tiled`` — we fall back to ``.shape`` and
        always use the cached-full-read path for frames.

        Args:
            page: The ``TiffPage`` / ``TiffFrame`` to wrap.
            source: ``(path, page_index)`` this channel can be reopened from in
                another process.  A live ``TiffPage`` holds an OS file handle
                and a ``threading.Lock``, so it cannot be pickled; the process
                pool re-opens the channel from this descriptor instead.
                ``None`` means the channel is not independently reopenable
                (e.g. it is backed by a cached full read), which disqualifies
                process-mode tiling.
        """
        import threading

        self._page = page
        # .shape works on both TiffPage and TiffFrame
        self.shape: tuple[int, int] = (page.shape[0], page.shape[1])
        self._lock = threading.Lock()
        self._cached_data: NDArray | None = None
        # TiffFrame has no is_tiled; treat as non-tiled (fallback path)
        self._is_tiled: bool = getattr(page, "is_tiled", False)
        # Profiling accumulators (guarded by _lock). Split the tile pass so we can
        # tell IO-bound from decode-bound from GPU-convolve-bound: convolve+upload is
        # then (compute_seconds - read_seconds - decode_seconds) at the caller.
        self._read_seconds: float = 0.0
        self._decode_seconds: float = 0.0
        self._read_bytes: int = 0
        # tifffile may lazily initialise its decoder on first use; warm it once under
        # the read lock (see _read_region_tiled) before any off-lock parallel decode.
        self._decoder_warmed: bool = False
        self._source: tuple[str, int] | None = source

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("_LazyTiffChannel supports only [y0:y1, x0:x1] slicing")
        yslice, xslice = key
        y0, y1, _ = yslice.indices(self.shape[0])
        x0, x1, _ = xslice.indices(self.shape[1])
        height = y1 - y0
        width = x1 - x0
        if height <= 0 or width <= 0:
            return np.empty((0, 0), dtype=self._page.dtype)
        if self._is_tiled:
            return self._read_region_tiled(y0, x0, height, width)
        return self._read_region_fallback(y0, x0, height, width)

    # ------------------------------------------------------------------
    # tiled path — decode only the tiles that overlap the request
    # ------------------------------------------------------------------

    def _read_region_tiled(self, y0: int, x0: int, height: int, width: int) -> NDArray:
        page = self._page
        tw, th = page.tilewidth, page.tilelength
        im_w, im_h = page.imagewidth, page.imagelength

        y1 = min(y0 + height, im_h)
        x1 = min(x0 + width, im_w)

        tile_y0 = y0 // th
        tile_x0 = x0 // tw
        tile_y1 = int(np.ceil(y1 / th))
        tile_x1 = int(np.ceil(x1 / tw))
        tiles_per_row = int(np.ceil(im_w / tw))

        buf_h = (tile_y1 - tile_y0) * th
        buf_w = (tile_x1 - tile_x0) * tw
        out = np.empty((buf_h, buf_w), dtype=page.dtype)

        fh = page.parent.filehandle
        jpegtables = page.jpegtables

        def _decode_blob(blob: tuple[int, int, int, bytes]) -> None:
            index, oi, oj, data = blob
            tile_arr, _indices, _shape = page.decode(data, index, jpegtables=jpegtables)
            out[oi : oi + th, oj : oj + tw] = tile_arr.squeeze()

        # Phase 1: read the compressed tile blobs under the lock. Only seek+read touch
        # the shared file handle, so the critical section is just the IO (measured at
        # ~1-3 s/channel -- negligible). Decoding is pulled OUT of the lock below.
        blobs: list[tuple[int, int, int, bytes]] = []  # (index, oi, oj, data)
        read_s = 0.0
        read_bytes = 0
        warm_decode_s = 0.0
        with self._lock:
            for ti in range(tile_y0, tile_y1):
                for tj in range(tile_x0, tile_x1):
                    index = ti * tiles_per_row + tj
                    offset = page.dataoffsets[index]
                    bytecount = page.databytecounts[index]
                    _t_io = time.perf_counter()
                    fh.seek(offset)
                    data = fh.read(bytecount)
                    read_s += time.perf_counter() - _t_io
                    read_bytes += bytecount
                    blobs.append(
                        (index, (ti - tile_y0) * th, (tj - tile_x0) * tw, data)
                    )
            self._read_seconds += read_s
            self._read_bytes += read_bytes
            # Warm tifffile's (possibly lazily-initialised) decoder ONCE, single-
            # threaded under the lock, so the concurrent off-lock decodes below cannot
            # race on a first-use init. Assembles the first tile of the first call.
            if not self._decoder_warmed and blobs:
                _t_warm = time.perf_counter()
                _decode_blob(blobs[0])
                warm_decode_s = time.perf_counter() - _t_warm
                blobs = blobs[1:]
                self._decoder_warmed = True

        # Phase 2: decode + assemble OFF the lock. page.decode is a pure function of the
        # read bytes and releases the GIL (imagecodecs), so across the GPU-worker
        # threads whole strips decode concurrently instead of single-file behind the
        # reader lock -- the serialized decode was ~91% of the tile-pass wall clock on a
        # 5.5 GP sample (run bVC3fObZxHK6p). Each tile writes a disjoint region of
        # `out`, so the assembly is race-free.
        _t_dec = time.perf_counter()
        if len(blobs) > 1:
            # Decode across the shared pool: page.decode releases the GIL, so the
            # strip's tiles decode on the otherwise-idle cores instead of single-file.
            # Exhaust the map generator so writes complete and errors propagate.
            for _ in _get_decode_pool().map(_decode_blob, blobs):
                pass
        else:
            for blob in blobs:
                _decode_blob(blob)
        decode_s = warm_decode_s + (time.perf_counter() - _t_dec)
        with self._lock:
            self._decode_seconds += decode_s

        ry0 = y0 - tile_y0 * th
        rx0 = x0 - tile_x0 * tw
        return out[ry0 : ry0 + (y1 - y0), rx0 : rx0 + (x1 - x0)]

    # ------------------------------------------------------------------
    # fallback path — cache full page (non-tiled / test images)
    # ------------------------------------------------------------------

    def _read_region_fallback(
        self, y0: int, x0: int, height: int, width: int
    ) -> NDArray:
        with self._lock:
            if self._cached_data is None:
                self._cached_data = self._page.asarray()
        return self._cached_data[y0 : y0 + height, x0 : x0 + width]


def _open_morphology_lazy(
    xoa_morphology_files,
    *,
    level: int = 0,
) -> tuple[list, tuple[int, int]]:
    """Open morphology channels as lazy TIFF page wrappers (no pixel data loaded).

    Uses ``tifffile.TiffFile`` directly instead of the zarr store interface,
    avoiding the zarr v3 dependency introduced by ``tifffile.imread(aszarr=True)``.

    Args:
        xoa_morphology_files: List of OME-TIFF file paths.
        level: Resolution level to open (0 = full resolution).

    Returns:
        (channels, image_shape) where channels is [dapi, boundary, intrna]
        (None for missing channels) and image_shape is (height, width).
    """
    tiff_handles: list[tifffile.TiffFile] = []  # prevent GC

    primary_tif = tifffile.TiffFile(str(xoa_morphology_files[0]))
    tiff_handles.append(primary_tif)

    # Use PHYSICAL pages in the file, not OME series pages.
    # Xenium multi-file OME-TIFFs report series.shape=(4,H,W) referencing
    # all files, but each file has only 1 physical page. series.pages would
    # include stubs for data in other files that decode to zeros.
    n_physical = len(primary_tif.pages)
    if level > 0:
        # For pyramid levels, use the series API to navigate levels
        series = primary_tif.series[0]
        if level < len(series.levels):
            pages = list(series.levels[level].pages)
        else:
            pages = list(primary_tif.pages)
    else:
        pages = list(primary_tif.pages)

    n_pages = len(pages)

    # Multi-channel single file: truly multi-page TIFF (each page = channel)
    # Only enter this path if the file physically contains multiple pages
    if n_pages >= 2 and n_physical >= 2:
        # Each page is one channel (C, H, W layout across pages).
        # At level 0 `pages` is the physical page list, so the list index is the
        # physical page index and each channel can be reopened elsewhere by
        # (path, page_index).  Pyramid levels come from the series API, where
        # that identity does not hold — leave those non-reopenable.
        primary_path = str(xoa_morphology_files[0])

        def _page_source(idx: int) -> tuple[str, int] | None:
            return (primary_path, idx) if level == 0 else None

        dapi = _LazyTiffChannel(pages[0], source=_page_source(0))
        shape = dapi.shape
        boundary = (
            _LazyTiffChannel(pages[1], source=_page_source(1)) if n_pages > 1 else None
        )
        intrna = (
            _LazyTiffChannel(pages[2], source=_page_source(2)) if n_pages > 2 else None
        )
        # Attach TiffFile handles to prevent garbage collection
        dapi._tiff_handles = tiff_handles  # type: ignore[attr-defined]
        return [dapi, boundary, intrna], shape

    # Single-channel (or single-page) primary
    # Check if the single page is actually multi-channel (C, H, W) in one page
    page0 = pages[0]
    arr_shape = page0.shape  # could be (H, W) or (C, H, W)
    if len(arr_shape) == 3:
        # Multi-channel packed into one page — fall back to cached full read
        full = page0.asarray()
        if arr_shape[0] <= 4 and arr_shape[1] > 16 and arr_shape[2] > 16:
            # Channel-first (C, H, W)
            shape = (arr_shape[1], arr_shape[2])
            ch_axis = 0
        elif arr_shape[2] <= 4 and arr_shape[0] > 16 and arr_shape[1] > 16:
            # Channel-last (H, W, C)
            shape = (arr_shape[0], arr_shape[1])
            ch_axis = 2
        else:
            shape = (arr_shape[1], arr_shape[2])
            ch_axis = 0
        n_ch = arr_shape[ch_axis]
        dapi = _LazyTiffChannel.__new__(_LazyTiffChannel)
        dapi._page = page0
        dapi.shape = shape
        import threading

        dapi._lock = threading.Lock()
        dapi._source = None  # cached full read: not reopenable in another process
        dapi._cached_data = np.take(full, 0, axis=ch_axis)
        boundary_ch = _LazyTiffChannel.__new__(_LazyTiffChannel) if n_ch > 1 else None
        if boundary_ch is not None:
            boundary_ch._page = page0
            boundary_ch.shape = shape
            boundary_ch._lock = threading.Lock()
            # cached full read: not reopenable in another process
            boundary_ch._source = None
            boundary_ch._cached_data = np.take(full, 1, axis=ch_axis)
        intrna_ch = _LazyTiffChannel.__new__(_LazyTiffChannel) if n_ch > 2 else None
        if intrna_ch is not None:
            intrna_ch._page = page0
            intrna_ch.shape = shape
            intrna_ch._lock = threading.Lock()
            # cached full read: not reopenable in another process
            intrna_ch._source = None
            intrna_ch._cached_data = np.take(full, 2, axis=ch_axis)
        dapi._tiff_handles = tiff_handles  # type: ignore[attr-defined]
        return [dapi, boundary_ch, intrna_ch], shape

    # Truly single-channel 2-D page
    dapi = _LazyTiffChannel(
        page0, source=(str(xoa_morphology_files[0]), 0) if level == 0 else None
    )
    shape = dapi.shape
    boundary = None
    intrna = None

    if len(xoa_morphology_files) > 1 and Path(xoa_morphology_files[1]).exists():
        try:
            t = tifffile.TiffFile(str(xoa_morphology_files[1]))
            tiff_handles.append(t)
            boundary = _LazyTiffChannel(
                t.pages[0],
                source=(str(xoa_morphology_files[1]), 0) if level == 0 else None,
            )
        except Exception as e:
            logging.warning(
                "Boundary TIFF open failed for %s: %s", xoa_morphology_files[1], e
            )

    if len(xoa_morphology_files) > 2 and Path(xoa_morphology_files[2]).exists():
        try:
            t = tifffile.TiffFile(str(xoa_morphology_files[2]))
            tiff_handles.append(t)
            intrna = _LazyTiffChannel(
                t.pages[0],
                source=(str(xoa_morphology_files[2]), 0) if level == 0 else None,
            )
        except Exception as e:
            logging.warning(
                "IntRNA TIFF open failed for %s: %s", xoa_morphology_files[2], e
            )

    # Attach TiffFile handles to prevent garbage collection
    dapi._tiff_handles = tiff_handles  # type: ignore[attr-defined]
    return [dapi, boundary, intrna], shape


def _compute_adaptive_tile_size(
    height: int,
    width: int,
    n_gpus: int,
    gpu_mem_bytes: int | None = None,
    target_utilization: float = 0.65,
    min_tiles_per_gpu: int = 2,
) -> int:
    """Compute tile size that maximizes GPU memory utilization.

    Balances two constraints:
    1. Each tile's peak GPU memory stays within target_utilization of VRAM.
    2. Total tiles >= n_gpus * min_tiles_per_gpu for load balancing.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        n_gpus: Number of available GPUs.
        gpu_mem_bytes: Total GPU VRAM in bytes. Auto-detected if None.
        target_utilization: Fraction of VRAM to target per tile (default 0.65).
        min_tiles_per_gpu: Minimum tiles per GPU for load balancing.

    Returns:
        Tile size in pixels (square tiles).
    """
    if gpu_mem_bytes is None:
        gpu_mem_bytes = cp.cuda.Device(0).mem_info[1]

    # Peak memory per pixel: ~6 concurrent float32 arrays
    BYTES_PER_PIXEL = 6 * 4  # 24 bytes

    # Max tile dim from GPU memory constraint
    max_pixels = int(gpu_mem_bytes * target_utilization / BYTES_PER_PIXEL)
    max_tile_dim = int(math.sqrt(max_pixels))

    # Min tiles needed for load balancing
    min_tiles = max(n_gpus * min_tiles_per_gpu, 1)

    # Minimum useful tile dimension (must fit the convolution window)
    min_dim = 256

    # Compute tile_size that produces at least min_tiles
    # Start from max_tile_dim and shrink until we have enough tiles
    tile_size = max_tile_dim
    while tile_size > min_dim:
        n_y = math.ceil(height / tile_size)
        n_x = math.ceil(width / tile_size)
        if n_y * n_x >= min_tiles:
            break
        tile_size = int(tile_size * 0.7)  # shrink by 30%

    # Clamp to minimum useful dimension
    tile_size = max(min_dim, tile_size)

    # If image is smaller than tile_size, just use image dims
    if height <= tile_size and width <= tile_size:
        tile_size = max(height, width)

    return tile_size


def _compute_adaptive_strip_height(
    height: int,
    width: int,
    n_gpus: int,
    gpu_mem_bytes: int | None = None,
    target_utilization: float = 0.65,
    min_strips_per_gpu: int = 2,
) -> int:
    """Rows per full-width strip that fit the VRAM budget.

    Full-width strips beat square tiles here for two measured reasons:

    * **Contiguous writes.** A square tile's write region is a rectangle, so a
      25,238-wide tile inside a 102,045-wide plane is 25,238 separate
      non-contiguous row segments. A strip's write region is one byte range. On
      the Fusion (FUSE/S3) work directory the scattered pattern was pathological
      -- 6 tiles took ~3 h against a ~46 min whole-run baseline, see
      docs/failures/2026-07-24_imageqc-mmap-over-fusion.md.
    * **Half the halo.** A strip needs overlap on top and bottom only, not all
      four edges, so less redundant convolution.

    At 48 GB VRAM and a 102,045 px width this gives ~12,700 rows, about 5 strips
    for the 5.5 gigapixel reference image.
    """
    if gpu_mem_bytes is None:
        gpu_mem_bytes = cp.cuda.Device(0).mem_info[1]

    # Same per-pixel budget as the square-tile sizing: ~6 concurrent float32
    # arrays live at once inside _compute_channel_maps_on_gpu.
    bytes_per_pixel = 6 * 4
    max_pixels = int(gpu_mem_bytes * target_utilization / bytes_per_pixel)
    rows = max(1, max_pixels // max(width, 1))

    # Enough strips to keep every GPU busy, when the image is tall enough.
    wanted = max(n_gpus * min_strips_per_gpu, 1)
    if rows * wanted > height:
        rows = max(1, height // wanted)

    # Round the strip *count* up to a multiple of the GPU count, then re-derive the
    # height from it. Two problems this fixes, both measured on run 3nkeHOEV1ONlbK
    # (102045 rows, 4 GPUs): height // wanted gave 12755 rows, so
    # ceil(102045/12755) = 9 strips -- eight full ones plus a 5-row sliver -- and 9
    # strips across 4 GPUs left occupancy at 74% / 88% / 100% / 100%. Deriving the
    # height from a count of 8 gives 12756 rows, no sliver, and two strips per GPU.
    # Increasing the count only ever shrinks strips, so the VRAM bound still holds.
    n_strips = max(1, -(-height // rows))
    n_strips = min(((n_strips + n_gpus - 1) // n_gpus) * n_gpus, height)
    rows = max(1, -(-height // n_strips))

    return int(min(rows, height))


def _compute_tile_grid(
    height: int,
    width: int,
    tile_size: int = 8192,
    overlap: int = 17,
    tile_width: int | None = None,
) -> list[dict[str, int]]:
    """Compute overlapping tile coordinates for tiled convolution.

    Each tile has a "write" region (non-overlapping, covers full image) and
    a "read" region (expanded by overlap, clipped to image bounds).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        tile_size: Core tile height (before overlap).
        overlap: Border pixels to add for convolution safety.
        tile_width: Core tile width; defaults to *tile_size* (square tiles). Pass
            the image width for full-width row strips, whose write region is one
            contiguous byte range.

    Returns:
        List of tile spec dicts with keys: read_y0, read_y1, read_x0, read_x1,
        write_y0, write_y1, write_x0, write_x1, trim_top, trim_bottom,
        trim_left, trim_right.
    """
    tiles = []
    step_x = int(tile_width) if tile_width else tile_size
    for y in range(0, height, tile_size):
        for x in range(0, width, step_x):
            wy0 = y
            wy1 = min(y + tile_size, height)
            wx0 = x
            wx1 = min(x + step_x, width)

            ry0 = max(0, wy0 - overlap)
            ry1 = min(height, wy1 + overlap)
            rx0 = max(0, wx0 - overlap)
            rx1 = min(width, wx1 + overlap)

            tiles.append(
                {
                    "read_y0": ry0,
                    "read_y1": ry1,
                    "read_x0": rx0,
                    "read_x1": rx1,
                    "write_y0": wy0,
                    "write_y1": wy1,
                    "write_x0": wx0,
                    "write_x1": wx1,
                    "trim_top": wy0 - ry0,
                    "trim_bottom": ry1 - wy1,
                    "trim_left": wx0 - rx0,
                    "trim_right": rx1 - wx1,
                }
            )
    return tiles


# ---------------------------------------------------------------------------
# Host memory instrumentation
# ---------------------------------------------------------------------------

_GIB = float(1024**3)

# High-water marks across the run, reported in the final summary.
_MEM_PEAK: dict[str, float] = {"working_set": 0.0, "rss": 0.0}


# (usage file, stat file, inactive_file key, active_file key) for cgroup v2 then
# v1.  AWS Batch nodes run either depending on the ECS AMI, so try both.
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


#: Kernel-maintained high-water usage for the whole cgroup, v2 then v1. Unlike the
#: sampled figures this is continuous and covers every process, so it cannot miss a
#: peak that falls between samples. It does include page cache, which makes it an
#: upper bound rather than an OOM-relevant working set.
_CGROUP_PEAK_PATHS = (
    "/sys/fs/cgroup/memory.peak",
    "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
)


def _cgroup_peak() -> float | None:
    """Peak cgroup usage in bytes, or None where the counter is absent."""
    for path in _CGROUP_PEAK_PATHS:
        try:
            with open(path) as fh:
                return float(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _cgroup_memory() -> tuple[float, float] | None:
    """``(working_set_bytes, page_cache_bytes)`` for this cgroup, or None.

    The usage counter includes reclaimable page cache, and the disk-backed focus
    planes deliberately generate a lot of it — reading usage alone would show a
    large number and wrongly suggest nothing improved.  The quantity that
    actually drives an OOM kill is the working set, ``usage - inactive_file``.
    Both are reported so the two are never confused when sizing the
    process_gpu_qc memory ladder.
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


# Cgroup memory hard limit, v2 then v1. Paired with `_cgroup_memory`'s working-set
# read to size the figure pool by available RAM headroom, not cores alone.
_CGROUP_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",  # v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
)


def _cgroup_memory_limit() -> float | None:
    """Cgroup memory hard limit in bytes, or None if unlimited/unreadable.

    v2 reports the literal string ``max`` when unlimited; v1 reports a sentinel
    close to ``2**63``, so an implausibly large limit is also treated as
    unlimited (a real Batch node is <=1 TB, well under ``2**60``).
    """
    for path in _CGROUP_LIMIT_PATHS:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = float(raw)
        except ValueError:
            continue
        if value >= float(1 << 60):
            return None
        return value
    return None


def _tree_rss() -> float | None:
    """RSS summed over every process in this PID namespace, in bytes.

    ``VmHWM`` from /proc/self/status is the main process only, and the figure phase
    forks up to ``_FIGURE_WORKERS_MAX`` children whose matplotlib buffers land on top
    of the parent's resident set. On run 3V53J4ewZt1vsU that made the difference
    between the 33.4 GB this module reported and the 100.6 GB Tower measured for the
    same task -- and the summary line told the reader to size the memory request from
    the smaller number, which would under-provision by 3x.

    Restricted to *our own* descendants rather than all of /proc. Summing everything
    is right in a container, whose PID namespace holds only our processes, but these
    modules also run directly on shared servers where it would silently add other
    users' processes to the total.

    Cgroup ``memory.peak`` would also cover children, but it counts reclaimable page
    cache — the confusion ``_cgroup_memory`` exists to avoid.
    """
    ppid_of: dict[int, int] = {}
    rss_of: dict[int, float] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/status") as fh:
                ppid = None
                rss = None
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("VmRSS:"):
                        rss = float(line.split()[1]) * 1024.0
                    if ppid is not None and rss is not None:
                        break
        except (OSError, ValueError, IndexError):
            continue  # exited between listdir and read, or not readable
        if ppid is None:
            continue
        ppid_of[pid] = ppid
        rss_of[pid] = rss or 0.0

    me = os.getpid()
    if me not in rss_of:
        return None
    # Walk parents to decide membership; the tree is shallow (pool workers are
    # direct children), so this stays cheap.
    total = 0.0
    for pid in rss_of:
        walker = pid
        for _ in range(64):  # bounded: never loop on a malformed parent chain
            if walker == me:
                total += rss_of[pid]
                break
            nxt = ppid_of.get(walker)
            if nxt is None or nxt == walker or nxt <= 1:
                break
            walker = nxt
    return total


def _process_rss() -> float | None:
    """Peak RSS of this process in bytes (VmHWM), excluding children."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _log_mem(stage: str) -> None:
    """Log host memory at a stage boundary and update the high-water marks.

    Instrumenting in-code rather than diagnosing after the fact: a Tower run
    only reports the peak for the whole task, which cannot say *which* stage
    drove it.
    """
    parts = []
    tree = _tree_rss()
    if tree is not None and tree > _MEM_PEAK.get("tree_rss", 0.0):
        _MEM_PEAK["tree_rss"] = tree
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
    if parts:
        logging.info(f"  [MEM] {stage}: {' '.join(parts)}")


def _log_mem_summary() -> None:
    """Final high-water summary — the number that sizes the memory request."""
    tree = _MEM_PEAK.get("tree_rss", 0.0)
    cgroup_peak = _cgroup_peak()
    parts = [
        f"cgroup_peak={cgroup_peak / _GIB:.1f}GB"
        if cgroup_peak is not None
        else "cgroup_peak=n/a",
        f"tree_rss={tree / _GIB:.1f}GB",
        f"working_set={_MEM_PEAK['working_set'] / _GIB:.1f}GB",
        f"main_process_rss={_MEM_PEAK['rss'] / _GIB:.1f}GB",
    ]
    logging.info("[MEM] PEAK " + " ".join(parts))
    # Each of these measures something different, and three of the four can understate
    # the task's real high-water mark. Spelling out which is which, because an earlier
    # version of this summary pointed at one number and was wrong twice over:
    #   cgroup_peak       continuous, all processes, but includes page cache
    #   tree_rss          all processes, but SAMPLED at stage boundaries -- it can miss
    #                     a peak between samples, and has come in *below*
    #                     main_process_rss for exactly that reason
    #   working_set       usage - inactive_file, the OOM-relevant quantity, also sampled
    #   main_process_rss  VmHWM: continuous, but this process only, so it excludes the
    #                     forked figure workers
    # The authoritative figure for sizing is the peakRss the Nextflow trace reports for
    # the task, which polls the whole process tree; these are for attributing cost to a
    # stage, which the trace cannot do.
    logging.info(
        "[MEM] For sizing use the Tower/trace peakRss for the task; the figures above "
        "attribute cost to stages and each understates the total in a different way "
        "(see the comment in _log_mem_summary)."
    )


# Absolute upper bound on concurrent forked figure workers. Each child inherits
# the parent's address space copy-on-write, so raising the count re-holds only
# each child's OWN matplotlib render buffers (~0.5-1 GB for a full-res 86 Mpx
# imshow), not the shared planes -- but N of those still land on the same cgroup,
# so `_figure_worker_limit` gates the pool on memory headroom, not cores alone.
_FIGURE_WORKERS_MAX = 8

#: Peak extra RAM a single figure child adds on top of the shared copy-on-write
#: planes, dominated by matplotlib's render buffers for a full-resolution imshow.
#: Deliberately generous (measured ~0.5-1 GB) so the memory gate errs toward
#: fewer workers rather than an OOM kill.
_FIGURE_PER_WORKER_GB = 2.0


def _cgroup_cpu_quota() -> int | None:
    """Cores the cgroup actually permits, or None if unlimited/unreadable.

    ``os.cpu_count()`` reports the host: a container given 30 of a 48-vCPU
    instance's cores still sees 48, so sizing a pool from it oversubscribes.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            raw_quota, raw_period = fh.read().split()
    except (OSError, ValueError):
        return None
    if raw_quota == "max":
        return None
    try:
        return max(1, int(int(raw_quota) / int(raw_period)))
    except (ValueError, ZeroDivisionError):
        return None


def _figure_worker_limit(n_tasks: int) -> int:
    """Concurrent figure-rendering processes to allow for ``n_tasks`` figures.

    Gated on BOTH the cgroup CPU quota and the cgroup memory headroom, because a
    high-core node can still OOM if every core forks a full-res-imshow child. The
    width is::

        min(n_tasks, cpu_quota, floor(available_gb / per_figure_gb), _FIGURE_WORKERS_MAX)

    where ``available_gb`` is the cgroup memory limit minus the working set
    (``usage - inactive_file``, the OOM-relevant quantity; see ``_cgroup_memory``).
    When the memory files cannot be read, falls back to the CPU-and-cap width and
    logs that memory gating was skipped. The cgroup CPU quota is preferred over
    ``os.cpu_count()`` (which reports the host, not the container's share).
    Override with IMAGE_QC_FIGURE_WORKERS.
    """
    if n_tasks <= 0:
        return 1

    override = os.environ.get("IMAGE_QC_FIGURE_WORKERS")
    if override:
        width = max(1, min(int(override), n_tasks))
        logging.info(f"[FIGPOOL] width={width} (IMAGE_QC_FIGURE_WORKERS override)")
        return width

    cpu = _cgroup_cpu_quota() or os.cpu_count() or 4
    width = min(n_tasks, cpu, _FIGURE_WORKERS_MAX)

    limit = _cgroup_memory_limit()
    mem = _cgroup_memory()
    if limit is not None and mem is not None:
        working_set, _page_cache = mem
        available_gb = max(0.0, limit - working_set) / _GIB
        mem_width = max(1, int(available_gb // _FIGURE_PER_WORKER_GB))
        width = max(1, min(width, mem_width))
        logging.info(
            f"[FIGPOOL] width={width} (tasks={n_tasks}, cpu_quota={cpu}, "
            f"mem_avail={available_gb:.1f}GB / {_FIGURE_PER_WORKER_GB:.0f}GB "
            f"-> {mem_width}, cap={_FIGURE_WORKERS_MAX})"
        )
    else:
        width = max(1, width)
        logging.info(
            f"[FIGPOOL] width={width} (tasks={n_tasks}, cpu_quota={cpu}, "
            f"cap={_FIGURE_WORKERS_MAX}, memory gate skipped: cgroup mem unreadable)"
        )
    return width


def _run_figure_pool(tasks, *, phase: str = "") -> None:
    """Render independent figure tasks concurrently in forked child processes.

    Rolling pool: keeps up to ``width`` children alive at once and starts the
    next task the instant a slot frees (submit + as-completed semantics), instead
    of the old batch barrier that started ``width`` children, joined ALL of them,
    then started the next batch -- so the slowest figure in a batch stalled every
    idle core until the whole batch drained (measured ~255 s for 11 ROI figures).

    NOT a ``concurrent.futures.ProcessPoolExecutor``: the tasks are closures over
    large numpy planes, and the executor's work queue pickles every submitted
    item even under a fork context (verified: "Can't pickle local object"). A raw
    fork ``Process`` never pickles its target, so children inherit the planes
    copy-on-write for free -- the whole reason this phase is fork-based. matplotlib
    is not thread-safe, so a thread pool is not an option either.

    Figure-failure semantics match the pre-pool code exactly: a task that raises
    logs its full traceback in the child (``_run_figure_task``) and the step
    CONTINUES, producing every other figure plus all metrics -- a single broken
    figure never nukes the QC step. The pool additionally logs any non-zero child
    exit (e.g. a hard crash / OOM-kill the old barrier ignored silently) naming the
    task, then carries on. This is a performance change only, not a change to what
    a figure failure does.
    """
    tasks = list(tasks)
    if not tasks:
        return

    width = _figure_worker_limit(len(tasks))
    mp_ctx = multiprocessing.get_context("fork")
    logging.info(f"[FIGPOOL] {phase or 'figures'}: dispatching {len(tasks)} task(s)")

    pending = list(reversed(tasks))  # pop() from the end preserves task order
    running: dict[Any, tuple[Any, str]] = {}  # sentinel fd -> (process, name)
    failures: list[tuple[str, int]] = []

    def _launch() -> None:
        fn = pending.pop()
        p = mp_ctx.Process(target=_run_figure_task, args=(fn,))
        p.start()
        running[p.sentinel] = (p, fn.__name__, time.perf_counter())

    while pending and len(running) < width:
        _launch()

    while running:
        # Block until at least one child exits; no busy-wait. A process sentinel
        # is a file descriptor that becomes ready when the process terminates.
        for sentinel in multiprocessing.connection.wait(list(running)):
            p, name, started = running.pop(sentinel)
            p.join()
            logging.info(
                f"  [TIMING] figure {name} ({phase or 'figures'}): "
                f"{time.perf_counter() - started:.1f}s"
            )
            if p.exitcode != 0:
                failures.append((name, p.exitcode))
            if pending:
                _launch()

    if failures:
        detail = ", ".join(f"{name} (exit {code})" for name, code in failures)
        # Match the pre-pool barrier: log loudly, do not abort the step. The child
        # already logged the Python traceback via _run_figure_task; this covers
        # non-zero exits (hard crash / OOM-kill) the old p.join() ignored silently.
        logging.error(
            f"[FIGPOOL] {phase or 'figures'}: figure task(s) failed: {detail}"
        )


# ---------------------------------------------------------------------------
# Tile consumers — fold a tile into small state, then drop it
#
# The disk-backed planes below bound host RAM, but they are not the right final
# design: they need ~154 GB of scratch on a 5.5 GP sample, mmap over a FUSE/S3
# work directory is pathological (see
# docs/failures/2026-07-24_imageqc-mmap-over-fusion.md), and these modules must
# run local / server / cloud on their way to nf-core/spatialaxe.
#
# No consumer of the full-resolution maps needs a whole map: ROI sampling reads
# one pixel per tile, SNR already loops per ROI window, the per-cell means are
# additive (sum, count) reductions, and the focus heatmap needs an 8x block mean.
# A consumer receives each finished tile and folds it into state that scales with
# the number of ROIs or cells, never with image size.
# ---------------------------------------------------------------------------


def _is_device_array(array: Any) -> bool:
    """True when *array* is a CuPy device array (so its reduction folds on-GPU)."""
    return HAS_CUPY and isinstance(array, cp.ndarray)


def _device_or_host(maps: dict[str, Any], key: str) -> Any:
    """The device variant (``key + "_device"``) of a map if the tile kept one on the
    GPU, else the host map under *key*. Lets a consumer fold on-device transparently:
    the streaming GPU path supplies ``focus_map_device`` / ``mean_map_device`` while
    the CPU / plane path supplies only the host ``focus_map`` / ``mean_map``.
    """
    device = maps.get(key + "_device")
    return maps.get(key) if device is None else device


class CentrePixelSampler:
    """Sample one pixel per ROI from tiles as they are produced.

    Replaces the centre-pixel sampling in ``downsample_maps_to_roi_dataframe``,
    which indexes ``map[cy, cx]`` on the assembled full-resolution planes. That
    pinned ~154 GB in order to read one pixel per ROI — 0.02 % of it.

    A tile's *trimmed* arrays cover exactly its write region, so an ROI whose
    centre lies in ``[write_y0, write_y1) x [write_x0, write_x1)`` is sampled at
    local offset ``(cy - write_y0, cx - write_x0)``. Write regions are disjoint
    and cover the image (``_compute_tile_grid``; asserted by
    ``test_no_write_overlap``), so every ROI is sampled exactly once.
    """

    wants_untrimmed = False
    #: This consumer does NOT read the host-side mean map, so the tile pass
    #: may drop that D2H copy. A consumer that needs it must set this True.
    reads_host_mean = False
    #: Row/col multiple this consumer needs its tile write origins to fall on.
    #: 1 means any split is fine. See BlockMeanAccumulator for why it needs more.
    write_alignment = 1
    #: Take the focus / mean maps from the GPU (``*_device``) when the tile pass
    #: kept them resident, so the one-pixel-per-ROI gather runs on-device and the
    #: full maps are not read back for it. The gather is a pure index, so the
    #: sampled float32 pixel is bit-identical to the host read.
    wants_device_focus = True
    wants_device_mean = True

    def __init__(self, cy: NDArray[np.integer], cx: NDArray[np.integer], keys):
        self.cy = np.asarray(cy)
        self.cx = np.asarray(cx)
        self.values: dict[str, NDArray[np.float64]] = {
            key: np.full(self.cy.size, np.nan, dtype=np.float64) for key in keys
        }
        # Sanity: every ROI must be claimed by exactly one tile.
        self._claimed = np.zeros(self.cy.size, dtype=np.int8)

    def consume(self, tile_spec: dict[str, int], maps: dict[str, Any]) -> None:
        """Fold one tile's trimmed maps into the per-ROI arrays.

        Each value map is taken device-first (``_device_or_host``): on the streaming
        GPU path the gather runs on the resident ``focus_map_device`` /
        ``mean_map_device`` and only the sampled pixels come back to host; on the CPU
        / plane path the host maps are gathered exactly as before. A gather is a pure
        index with no arithmetic, so the device and host reads return the identical
        float32 pixel.
        """
        wy0, wy1 = tile_spec["write_y0"], tile_spec["write_y1"]
        wx0, wx1 = tile_spec["write_x0"], tile_spec["write_x1"]
        inside = (self.cy >= wy0) & (self.cy < wy1) & (self.cx >= wx0) & (self.cx < wx1)
        if not inside.any():
            return
        local_y = self.cy[inside] - wy0
        local_x = self.cx[inside] - wx0
        self._claimed[inside] += 1
        for key in self.values:
            array = _device_or_host(maps, key)
            if array is None:
                continue
            if _is_device_array(array):
                # Gather on the array's own device (the fold runs after the GPU was
                # returned to the pool, so the thread's current device may differ).
                with array.device:
                    ly = cp.asarray(local_y)
                    lx = cp.asarray(local_x)
                    sampled = cp.asnumpy(array[ly, lx]).astype(np.float64)
            else:
                sampled = np.asarray(array)[local_y, local_x].astype(np.float64)
            self.values[key][inside] = sampled

    def finalize(self) -> dict[str, NDArray[np.float64]]:
        """Return the per-ROI arrays, checking every ROI was covered once."""
        unclaimed = int((self._claimed == 0).sum())
        duplicated = int((self._claimed > 1).sum())
        if unclaimed or duplicated:
            raise RuntimeError(
                f"ROI coverage is wrong: {unclaimed} ROI(s) claimed by no tile, "
                f"{duplicated} by more than one. Tile write regions must be "
                "disjoint and cover the image."
            )
        return dict(self.values)

    def spawn(self) -> "CentrePixelSampler":
        """A fresh, empty sampler sharing this one's ROI grid and keys.

        Used by the tile pass to give each worker thread its own accumulator so
        folds run without a lock; the partials are then reduced with ``merge``.
        """
        return CentrePixelSampler(self.cy, self.cx, list(self.values))

    def merge(self, other: "CentrePixelSampler") -> None:
        """Fold a per-worker partial into this one.

        Each ROI is claimed by exactly one tile, hence by exactly one worker, so the
        partials hold DISJOINT per-ROI entries: the merge is a pure overlay of the
        ROIs *other* claimed, with no arithmetic and so bit-identical to the serial
        fold regardless of order. The claim counts add so ``finalize`` still verifies
        global coverage.
        """
        overlap = (self._claimed > 0) & (other._claimed > 0)
        assert not overlap.any(), (
            "CentrePixelSampler partials claim overlapping ROIs; tile write regions "
            "must be disjoint (each ROI centre lands in exactly one write region)."
        )
        claimed = other._claimed > 0
        self._claimed += other._claimed
        for key in self.values:
            self.values[key][claimed] = other.values[key][claimed]


class LazyLabelPlane:
    """Lazily-sliced view over a full-resolution label plane in zarr.

    ``load_spatial_data`` does ``np.array(cell_masks_zarr["masks"]["1"])``, which
    materialises 22 GB of uint32 on a 5.5 gigapixel sample, and
    ``calculate_ccfs_from_focus_maps`` did the same for ``masks/0``. Neither needs
    the whole plane: the consumers of these arrays are row-block reductions and
    scattered point lookups, both of which this serves from zarr on demand.

    Supports the three access patterns the callers actually use:

    * ``plane[y0:y1]`` and ``plane[y0:y1, x0:x1]`` — row/rect slices, passed
      straight through to zarr.
    * ``plane[iy, ix]`` with integer arrays — coordinate lookup, served by reading
      only the row blocks the points fall in. ``masks[y, x]`` on a zarr array does
      not do numpy-style pair indexing, so this is why the wrapper exists.
    """

    def __init__(self, source, rows_per_chunk: int | None = None):
        self._source = source
        self.shape: tuple[int, int] = (int(source.shape[0]), int(source.shape[1]))
        self.dtype = getattr(source, "dtype", None)
        width = max(self.shape[1], 1)
        self.rows_per_chunk = rows_per_chunk or max(1, _LABEL_CHUNK_PIXELS // width)

    def __getitem__(self, key):
        # Coordinate lookup: two integer arrays.
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and all(
                isinstance(part, (np.ndarray, list)) or np.isscalar(part)
                for part in key
            )
            and not any(isinstance(part, slice) for part in key)
        ):
            return self._gather(np.asarray(key[0]), np.asarray(key[1]))
        return np.asarray(self._source[key])

    def _gather(
        self, rows: NDArray[np.integer], cols: NDArray[np.integer]
    ) -> NDArray[Any]:
        """Point lookup, reading only the row blocks the points land in."""
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        if rows.size != cols.size:
            raise ValueError("row and column index arrays must be the same length")
        out = None
        for start in range(0, self.shape[0], self.rows_per_chunk):
            stop = min(start + self.rows_per_chunk, self.shape[0])
            inside = (rows >= start) & (rows < stop)
            if not inside.any():
                continue
            block = np.asarray(self._source[start:stop])
            if out is None:
                out = np.zeros(rows.size, dtype=block.dtype)
            out[inside] = block[rows[inside] - start, cols[inside]]
            del block
        if out is None:
            out = np.zeros(rows.size, dtype=self.dtype or np.int64)
        return out

    def max(self) -> int:
        """Largest label value, read in row blocks."""
        highest = 0
        for start in range(0, self.shape[0], self.rows_per_chunk):
            stop = min(start + self.rows_per_chunk, self.shape[0])
            block = np.asarray(self._source[start:stop])
            if block.size:
                highest = max(highest, int(block.max()))
            del block
        return highest


class LabeledSumAccumulator:
    """Accumulate per-label ``(count, sum)`` from tiles as they are produced.

    The same additive reduction as ``_labeled_sums_chunked``, keyed on tile write
    regions instead of row blocks. Counts and sums are additive, so a cell split
    across tiles contributes partial sums to each and its final ``sum / count`` is
    exact — this is not an approximation.

    State is ``O(n_labels)``: two arrays per value plane, tens of MB for ~530 k
    cells, against the 22 GB per plane the whole-map path needed. The label plane
    is sliced per tile, so it is never materialised either — which also removes
    the ``cellseg_mask`` array the current path still holds.

    The per-label reduction (``xp.bincount``) is ``xp``-generic. On the streaming GPU
    path the value maps arrive resident on the device (``focus_map_device`` /
    ``mean_map_device``); this uploads the int label block — cheaper than reading the
    float value maps back — and runs every ``bincount`` on the GPU, returning only the
    per-label ``O(n_labels)`` vectors to host. That moves the largest term of the tile
    fold (~112 s/channel of host ``bincount``) onto the otherwise-idle GPU. The counts
    and float64 sums are the same additive reduction either way, and the GPU float64
    ``bincount`` matches the host result to float64 rounding (tests/test_tile_consumers).
    On the CPU / plane path the maps are host numpy and the fold is byte-identical to
    before.
    """

    wants_untrimmed = False
    #: This consumer does NOT read the host-side mean map, so the tile pass
    #: may drop that D2H copy. A consumer that needs it must set this True.
    reads_host_mean = False
    #: Row/col multiple this consumer needs its tile write origins to fall on.
    #: 1 means any split is fine. See BlockMeanAccumulator for why it needs more.
    write_alignment = 1
    #: Fold from the GPU-resident maps when the tile pass kept them there, so the
    #: per-label bincounts run on-device instead of a serial host fold.
    wants_device_focus = True
    wants_device_mean = True

    def __init__(
        self,
        label_plane,
        value_keys,
        include_coords: bool = False,
        skip_background: bool = False,
    ):
        self.label_plane = label_plane
        self.include_coords = include_coords
        # Drop label-0 pixels before reducing. Only valid where the caller never reads
        # index 0: the nuclear reduction does `labels[labels > 0]`, but the *cell*
        # reduction deliberately reads `cell_counts[0]` as the background pixel count
        # for CellID 0, so it must keep them.
        self.skip_background = skip_background
        self.counts = np.zeros(1, dtype=np.int64)
        self.sums: dict[str, NDArray[np.float64]] = {
            key: np.zeros(1, dtype=np.float64) for key in value_keys
        }
        if include_coords:
            self.sums["centroid_y_sum"] = np.zeros(1, dtype=np.float64)
            self.sums["centroid_x_sum"] = np.zeros(1, dtype=np.float64)

    def _add(self, key: str, block: NDArray[np.float64]) -> None:
        self.sums[key] = _grow_to(self.sums[key], block.size)
        self.sums[key][: block.size] += block

    def consume(self, tile_spec: dict[str, int], maps: dict[str, Any]) -> None:
        """Fold one tile's trimmed maps into the per-label accumulators.

        Runs on the GPU when the tile kept its maps device-resident (``xp=cp``),
        else on host (``xp=np``); ``_fold_blocks`` is written once against ``xp``.
        Blocked by rows within the tile, for the same reason
        ``_labeled_sums_chunked`` blocks: ``bincount`` needs ``intp`` labels and
        ``float64`` weights, so a whole-tile call casts both. A production tile is
        a full-width row strip, so that is 2.7 GB of labels plus 5.5 GB per
        coordinate array — larger than the tile's own maps. One block is ~32 M
        pixels regardless of tile size, which also keeps the on-device label upload
        and the ``K*256``-free bincount well under the tile's VRAM budget.
        """
        wx0, wx1 = tile_spec["write_x0"], tile_spec["write_x1"]
        tile_width = wx1 - wx0
        if tile_width <= 0 or tile_spec["write_y1"] <= tile_spec["write_y0"]:
            return

        # Take each value map device-first, then decide the fold backend from what the
        # tile actually handed over. A value-less counts-only accumulator (cells) has no
        # value map to check, so fall back to any resident device array in the tile.
        value_maps = {
            key: _device_or_host(maps, key)
            for key in self.sums
            if not key.startswith("centroid_")
        }
        device_arr = next((a for a in value_maps.values() if _is_device_array(a)), None)
        if device_arr is None:
            device_arr = next((a for a in maps.values() if _is_device_array(a)), None)

        if device_arr is not None:
            # Run every bincount on the map's own device (the fold happens after the
            # GPU is returned to the pool, so the thread's current device may differ).
            with device_arr.device:
                self._fold_blocks(tile_spec, value_maps, cp, on_gpu=True)
        else:
            self._fold_blocks(tile_spec, value_maps, np, on_gpu=False)

    def _fold_blocks(
        self,
        tile_spec: dict[str, int],
        value_maps: dict[str, Any],
        xp: Any,
        on_gpu: bool,
    ) -> None:
        """Row-blocked per-label reduction over one tile, on ``xp`` (numpy or cupy).

        Reduces on-device when ``xp`` is cupy, pulling only the ``O(n_labels)`` result
        vectors back to the host accumulators; the intermediate labels/values/coords
        never leave the device.
        """
        wy0, wy1 = tile_spec["write_y0"], tile_spec["write_y1"]
        wx0, wx1 = tile_spec["write_x0"], tile_spec["write_x1"]
        tile_width = wx1 - wx0
        rows_per_chunk = max(1, _LABEL_CHUNK_PIXELS // tile_width)

        def _host(arr: Any) -> Any:
            return cp.asnumpy(arr) if on_gpu else arr

        for y0 in range(wy0, wy1, rows_per_chunk):
            y1 = min(y0 + rows_per_chunk, wy1)
            labels_host = np.asarray(self.label_plane[y0:y1, wx0:wx1]).ravel()
            if labels_host.size == 0:
                continue
            # Upload the int label block to the device (cheaper than reading the float
            # value maps back), or keep it on host for the numpy path.
            labels = xp.asarray(labels_host) if on_gpu else labels_host
            del labels_host

            # Restrict to labelled pixels where index 0 is never read. On the reference
            # sample the nuclear mask is 9.8 % non-zero (505 k nuclei x ~1066 px of
            # 5.50 G), so this drops ~90 % of the work from every pass below --
            # measured at ~169 s for this accumulator, the largest term in the fold
            # after the Otsu SNR. `flatnonzero` also lets the coordinates be built at
            # the compressed size instead of materialising a full-chunk repeat/tile.
            selected = None
            if self.skip_background:
                selected = xp.flatnonzero(labels)
                labels = labels[selected]
                if labels.size == 0:
                    continue

            block_counts = xp.bincount(labels)
            n = int(block_counts.size)
            self.counts = _grow_to(self.counts, n)
            self.counts[:n] += _host(block_counts)

            # value_maps rows are trimmed to the write region, so map row `y0 - wy0`
            # is global row `y0`.
            for key, array in value_maps.items():
                if array is None:
                    continue
                values = xp.asarray(
                    array[y0 - wy0 : y1 - wy0], dtype=xp.float64
                ).ravel()
                if selected is not None:
                    values = values[selected]
                self._add(key, _host(xp.bincount(labels, weights=values, minlength=n)))
                del values

            if self.include_coords:
                # Global coordinates, so sum / count is regionprops' centroid in
                # image space rather than tile-local space.
                if selected is not None:
                    rows = (y0 + selected // tile_width).astype(xp.float64)
                    cols = (wx0 + selected % tile_width).astype(xp.float64)
                else:
                    rows = xp.repeat(xp.arange(y0, y1, dtype=xp.float64), tile_width)
                    cols = xp.tile(xp.arange(wx0, wx1, dtype=xp.float64), y1 - y0)
                self._add(
                    "centroid_y_sum",
                    _host(xp.bincount(labels, weights=rows, minlength=n)),
                )
                del rows
                self._add(
                    "centroid_x_sum",
                    _host(xp.bincount(labels, weights=cols, minlength=n)),
                )
                del cols
            del labels, block_counts, selected

    def finalize(self) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]:
        """Return ``(counts, sums)`` indexed by raw label value, 0 = background."""
        return self.counts, dict(self.sums)

    def spawn(self) -> "LabeledSumAccumulator":
        """A fresh, empty accumulator sharing this one's label plane and keys.

        Used by the tile pass to give each worker thread its own accumulator so
        folds run without a lock; the partials are then reduced with ``merge``. The
        label plane is shared by reference (read-only, per-tile slices), not copied.
        """
        value_keys = [k for k in self.sums if not k.startswith("centroid_")]
        return LabeledSumAccumulator(
            self.label_plane,
            value_keys,
            include_coords=self.include_coords,
            skip_background=self.skip_background,
        )

    def merge(self, other: "LabeledSumAccumulator") -> None:
        """Fold a per-worker partial into this one by element-wise addition.

        Counts and per-label float64 sums are additive, so adding a worker's
        partial totals reduces the same values as the serial per-tile fold -- but
        it regroups the float64 additions (per-worker subtotals then combined,
        rather than one running sum in tile order). Reassociation of float64 sums
        can differ by ULPs; that this reproduces the serial fold byte-for-byte is
        empirical and is pinned by tests/test_fold_parallel_equivalence, not
        assumed. Arrays are grown to the longer length before adding.
        """
        n = max(self.counts.size, other.counts.size)
        self.counts = _grow_to(self.counts, n)
        self.counts[: other.counts.size] += other.counts
        for key, total in other.sums.items():
            self.sums[key] = _grow_to(self.sums[key], total.size)
            self.sums[key][: total.size] += total

    def means(self, labels: NDArray[np.integer]) -> dict[str, NDArray[np.float64]]:
        """Per-label means for *labels*, NaN where a label has no pixels."""
        counts = self.counts[labels].astype(np.float64)
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for key, total in self.sums.items():
                out[key] = total[labels] / counts
        return out


def _block_sum(
    array: NDArray[np.generic], y_offset: int, x_offset: int, factor: int
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    """Sum *array* into the global factor-grid blocks it overlaps.

    ``array[0, 0]`` sits at global ``(y_offset, x_offset)``. Returns the block
    sums plus the global block row/column indices they belong to.

    Two ``np.add.reduceat`` passes rather than a flat block-index array: for a
    36k x 36k tile the index array alone would be 1.3e9 int64 = 10 GB, whereas
    the row-reduced intermediate is (n_block_rows, width). ``reduceat`` also
    handles the ragged first/last groups natively when the tile does not start on
    a block boundary.
    """
    values = np.asarray(array, dtype=np.float64)
    rows = np.arange(values.shape[0]) + y_offset
    row_starts = np.flatnonzero(
        np.r_[True, (rows[1:] // factor) != (rows[:-1] // factor)]
    )
    partial = np.add.reduceat(values, row_starts, axis=0)

    cols = np.arange(values.shape[1]) + x_offset
    col_starts = np.flatnonzero(
        np.r_[True, (cols[1:] // factor) != (cols[:-1] // factor)]
    )
    block = np.add.reduceat(partial, col_starts, axis=1)
    return block, rows[row_starts] // factor, cols[col_starts] // factor


class BlockMeanAccumulator:
    """Build a factor-``f`` down-sampled canvas from tiles, matching skimage.

    ``_fig5_focus_heatmap`` calls ``downscale_local_mean(dapi_focus, (8, 8))`` on
    the assembled plane — the only use of that plane, and it pads the array before
    reducing, so on a 5.5 GP sample it costs another ~22 GB on top of the 22 GB
    plane it reads.

    ``downscale_local_mean`` zero-pads incomplete blocks and divides by the
    *full* block size, so accumulating per-block sums and dividing by
    ``factor**2`` reproduces it **bit-exactly**, including the partial final block, and
    that is verified against skimage rather than argued (``TestBlockMeanEpsilon``).

    Exactness needs two things, and the *dtype* one is by far the larger:

    First, the same precision. Production focus maps are float32, so the plane-based path
    reduces in float32; this class reduces in whatever dtype it is handed and must not
    widen it. An earlier version upcast to float64 and disagreed with skimage in 200 of
    200 measured random planes, by up to 1.86e-07 relative.

    Second, *grouping*: every block reduced by one addition chain, never
    as two partials summed across a tile seam. The dispatcher arranges that by aligning
    write boundaries to ``factor`` -- it reads ``write_alignment`` off each consumer, so
    a caller cannot forget -- and ``consume`` refuses an unaligned write rather than
    silently accumulating one.

    The axis order within a block turns out not to matter -- ``sum(axis=(1, 3))`` and
    ``sum(axis=3).sum(axis=1)`` agree with skimage bit-for-bit over 300 random
    shapes/magnitudes, and mutating one to the other is an equivalent mutant. Only the
    grouping is load-bearing.

    An earlier version summed with two ``np.add.reduceat`` passes over an unaligned grid,
    so a block straddling a seam *was* accumulated as two partials. That is real, but it
    contributes only ~3.5e-16 and was **not** the cause of the 16 pixels of Figure 5 that
    differed from the plane-based path: the float64 upcast above was, at ~1e-7. Fixing
    the ordering alone left the output byte-for-byte unchanged (16 px, delta 1) --
    see ``docs/failures/2026-07-26_heatmap-16px-dtype-not-summation-order.md``.

    State is the ``(H/f, W/f)`` canvas: 0.34 GB at f=8 on a 5.5 GP sample,
    against 44 GB for the plane plus skimage's pad.
    """

    #: Every consumer takes ``consume(tile_spec, maps)`` where *maps* is the dict
    #: of this tile's channel maps. A uniform signature lets the tile dispatcher
    #: feed them all without knowing which is which.
    wants_untrimmed = False
    #: This consumer does NOT read the host-side mean map, so the tile pass
    #: may drop that D2H copy. A consumer that needs it must set this True.
    reads_host_mean = False
    #: Row/col multiple this consumer needs its tile write origins to fall on.
    #: 1 means any split is fine. See BlockMeanAccumulator for why it needs more.
    write_alignment = 1

    def __init__(
        self,
        shape: tuple[int, int],
        factor: int = 8,
        map_key: str = "focus_map",
    ):
        self.map_key = map_key
        self.factor = int(factor)
        # Blocks must not straddle tiles, so the dispatcher must land write origins on
        # multiples of the factor. Declared here rather than passed in by the caller:
        # a caller that forgot would only find out via consume()'s RuntimeError.
        self.write_alignment = self.factor
        #: dtype of the first array consumed; the canvas is returned in it.
        self._out_dtype: np.dtype | None = None
        self.shape = (int(shape[0]), int(shape[1]))
        rows = -(-self.shape[0] // self.factor)  # ceil
        cols = -(-self.shape[1] // self.factor)
        self.sums = np.zeros((rows, cols), dtype=np.float64)

    def consume(self, tile_spec: dict[str, int], maps: dict[str, Any]) -> None:
        """Fold this tile's ``map_key`` plane into the block canvas.

        Reduces each block with a single reshape, matching ``downscale_local_mean``'s own
        grouping and summation order, which makes the canvas **bit-exact** rather than
        equal to within a float64 epsilon. That requires every block to lie wholly inside
        one tile, which the dispatcher guarantees by aligning write boundaries to
        ``factor``; the check below is that contract, and it raises rather than silently
        degrading.

        Previously this used two ``np.add.reduceat`` passes, so a block straddling a seam
        was summed as two partials in a different order from skimage. The resulting
        ~3.5e-16 discrepancy reached the rendered figure: 16 of Figure 5's 5,553,835
        pixels landed 1/255 apart from the plane-based path.

        The image's own trailing rows and columns may still form a partial block; those
        are zero-padded exactly as skimage pads, which is bit-exact for dimensions that
        are not multiples of the factor.
        """
        array = maps.get(self.map_key)
        if array is None:
            return
        # Reduce in the INPUT's dtype, never a wider one. Production focus maps are
        # float32 and the plane-based path calls downscale_local_mean on them, so it
        # reduces in float32; accumulating in float64 here disagreed with it by up to
        # 1.86e-07 relative in 200 of 200 measured cases -- enough to move 16 of Figure
        # 5's 5.5M pixels across a uint8 boundary. That dtype term is ~5e8 times larger
        # than the summation-order term this class used to blame.
        # No integer special-case: numpy already promotes e.g. uint16 sums to uint64,
        # which lands on skimage's float64 mean exactly. Verified, not assumed --
        # test_integer_input_follows_skimage_to_float64, and a mutation removing an
        # explicit promotion survived because it was doing nothing.
        values = np.asarray(array)
        if self._out_dtype is None:
            # The canvas must come out in the dtype downscale_local_mean would have
            # produced, not merely with the same values. Figure 5 takes its colour scale
            # from np.percentile of this array, and percentile on a float64 container
            # returns 2082.678369140625 where float32 returns 2082.6785 -- a different
            # vmin/vmax, which flips pixels sitting on a colormap boundary. That was the
            # third and last cause of the differing heatmap pixels.
            self._out_dtype = (
                values.dtype
                if np.issubdtype(values.dtype, np.floating)
                else np.dtype(np.float64)  # skimage's np.mean promotes integers
            )
        wy0, wx0 = tile_spec["write_y0"], tile_spec["write_x0"]
        f = self.factor
        if wy0 % f or wx0 % f:
            raise RuntimeError(
                f"tile write origin ({wy0}, {wx0}) is not aligned to the heatmap block "
                f"size {f}: blocks would straddle tiles and the canvas would stop "
                "matching downscale_local_mean exactly. _compute_channel_maps_tiled "
                "derives this from each consumer's write_alignment, so reaching here "
                "means the accumulator was driven by something else."
            )
        pad_y = (-values.shape[0]) % f
        pad_x = (-values.shape[1]) % f
        if pad_y or pad_x:
            values = np.pad(values, ((0, pad_y), (0, pad_x)), mode="constant")
        blocks = values.reshape(values.shape[0] // f, f, values.shape[1] // f, f).sum(
            axis=(1, 3)
        )
        r0, c0 = wy0 // f, wx0 // f
        self.sums[r0 : r0 + blocks.shape[0], c0 : c0 + blocks.shape[1]] += blocks

    def finalize(self) -> NDArray[np.floating]:
        """Return the block means, identical to ``downscale_local_mean`` in value *and*
        dtype. The dtype is part of the contract: Figure 5 derives its colour scale from
        ``np.percentile`` of this array, which answers differently in float32 and float64.
        """
        means = self.sums / float(self.factor * self.factor)
        if self._out_dtype is not None and means.dtype != self._out_dtype:
            means = means.astype(self._out_dtype)
        return means

    def spawn(self) -> "BlockMeanAccumulator":
        """A fresh, empty canvas of the same shape / factor / map key.

        Used by the tile pass to give each worker thread its own accumulator so
        folds run without a lock; the partials are then reduced with ``merge``.
        """
        return BlockMeanAccumulator(
            self.shape, factor=self.factor, map_key=self.map_key
        )

    def merge(self, other: "BlockMeanAccumulator") -> None:
        """Fold a per-worker partial canvas into this one.

        Write origins are aligned to ``factor`` (the dispatcher enforces it via
        ``write_alignment``), so every block lies wholly inside one tile and hence
        one worker. The per-worker canvases are therefore DISJOINT -- each block is
        non-zero in exactly one partial -- so the element-wise add is a pure overlay
        (``0.0 + x == x``) with no reassociation, bit-identical to the serial fold.
        """
        self.sums += other.sums
        if other._out_dtype is not None:
            if self._out_dtype is not None and self._out_dtype != other._out_dtype:
                raise RuntimeError(
                    f"BlockMeanAccumulator partials disagree on output dtype "
                    f"({self._out_dtype} vs {other._out_dtype}); every tile's map "
                    "must have the same dtype."
                )
            self._out_dtype = other._out_dtype


#: ROIs folded per batched-Otsu call. Two constraints set it:
#:
#: 1. **cupy bincount ceiling (hard).** ``roi_snr_db_batch`` builds a per-row
#:    histogram as one flattened ``cp.bincount`` over ``K*1225`` indices into
#:    ``K*256`` bins. In cupy 14.0.1 that bincount raises
#:    ``cudaErrorIllegalAddress`` once K is large: measured OK at K=40_000
#:    (10.2M bins / 49M elems), FAIL at K=50_000 (12.8M / 61.25M). This is what
#:    crashed Tower run 4bBYe2TAP5QEqA on the first fold. Staying an order of
#:    magnitude under that boundary makes the fault structurally impossible and
#:    leaves headroom for other GPUs/cupy builds.
#: 2. VRAM: bounds the ``(K, N)`` / ``(K, 256)`` float64 temporaries so a tall
#:    strip's owned ROIs stay well under the tile's ~10.5 GB budget.
#:
#: 8192 gives ~5x margin on (1) and a small device peak; the extra kernel launches
#: over a larger chunk are negligible against the fold. See
#: docs/failures/2026-07-27_gpu-otsu-cuda-illegal-address.md.
_OTSU_ROI_CHUNK = 8_192


class RoiOtsuSnrAccumulator:
    """Per-ROI Otsu SNR in dB, computed from tiles as they are produced.

    Replaces ``snr_metrics.compute_image_snr_from_pixel_maps``'s loop over
    ``dapi_mean_map[y1:y2, x1:x2]``, which needed the assembled 22 GB plane.

    The reduction is ``xp``-generic. In the streaming GPU path the tile pass hands
    over the tile's **device** ``mean_map`` (a CuPy array still resident in VRAM,
    under key ``"mean_map_device"``) and the batched Otsu SNR
    (``snr_metrics.roi_snr_db_batch``, ``xp=cp``) folds every owned interior ROI in
    one vectorized GPU pass, returning only the small per-ROI dB vector to host --
    instead of the 4.49 M-iteration Python loop over host windows this class used
    to run (~439 s of CPU). On the CPU / no-GPU path the map is a numpy array and
    the identical batched kernel runs with ``xp=np``. Either way the result matches
    the per-ROI scalar ``snr_metrics.roi_snr_db`` to float64 rounding (~1e-14 dB;
    see ``tests/test_roi_snr_db_batch.py``).

    Windows clipped at the image edge (variable size) or holding a non-finite
    pixel fall back to the scalar ``roi_snr_db``, which the batched kernel -- it
    assumes finite, full-size windows -- does not model. Only the full-size
    interior windows, which share one shape, are batched.

    Unlike the other consumers this one needs the **untrimmed** tile: an ROI is
    owned by the tile whose write region contains its top-left corner, and its
    window then extends up to ``roi_size - 1`` px beyond that write region. The
    read region extends by ``overlap``, so the window is covered exactly when
    ``overlap >= roi_size``. That is currently true (both are ``window_size``,
    35) but it is a real constraint, so a violation raises rather than silently
    truncating a window and reporting a wrong dB.
    """

    #: Tells the tile dispatcher to hand this consumer the haloed tile.
    wants_untrimmed = True
    #: This consumer does NOT read the host-side mean map, so the tile pass
    #: may drop that D2H copy. A consumer that needs it must set this True.
    reads_host_mean = False
    #: Tells the tile dispatcher to keep the tile's mean map on the GPU and pass
    #: the device array (``"mean_map_device"``) so the Otsu SNR folds on-device.
    wants_device_mean = True
    write_alignment = 1

    def __init__(
        self,
        y1: NDArray[np.integer],
        y2: NDArray[np.integer],
        x1: NDArray[np.integer],
        x2: NDArray[np.integer],
        image_shape: tuple[int, int],
        map_key: str = "mean_map",
    ):
        self.map_key = map_key
        self.y1 = np.asarray(y1, dtype=np.int64)
        self.y2 = np.asarray(y2, dtype=np.int64)
        self.x1 = np.asarray(x1, dtype=np.int64)
        self.x2 = np.asarray(x2, dtype=np.int64)
        self.height, self.width = int(image_shape[0]), int(image_shape[1])
        self.db = np.full(self.y1.size, np.nan, dtype=np.float64)
        self._claimed = np.zeros(self.y1.size, dtype=np.int8)
        # The full ROI side; interior windows equal it, edge-clipped windows are
        # smaller. Windows of exactly this shape are the ones the batched kernel
        # folds together (it needs one uniform shape); everything else is scalar.
        self._full_h = int((self.y2 - self.y1).max()) if self.y1.size else 0
        self._full_w = int((self.x2 - self.x1).max()) if self.x1.size else 0

    def consume(self, tile_spec: dict[str, int], maps: dict[str, Any]) -> None:
        """Fold this tile's *untrimmed* (haloed) mean map into the per-ROI dB array.

        Reads the device mean map (``"mean_map_device"``) when the tile pass kept
        one resident, else the host ``map_key``; ``xp`` follows the array type so
        the same code batches on GPU (``cp``) or CPU (``np``). Full-size interior
        windows go through ``snr_metrics.roi_snr_db_batch`` in bounded chunks;
        edge-clipped or non-finite windows fall back to the scalar
        ``roi_snr_db``.
        """
        # The device map the tile pass keeps resident is the mean map specifically
        # (``mean_map_device``). Only reach for it when this accumulator is actually
        # configured for the mean map; otherwise fall through to the configured
        # ``map_key``. Without this guard a focus-configured instance would silently
        # reduce the mean map on a GPU tile and return plausible but wrong numbers.
        array = maps.get("mean_map_device") if self.map_key == "mean_map" else None
        if array is None:
            array = maps.get(self.map_key)
        if array is None:
            return

        on_gpu = HAS_CUPY and isinstance(array, cp.ndarray)
        xp = cp if on_gpu else np

        ry0, ry1 = tile_spec["read_y0"], tile_spec["read_y1"]
        rx0, rx1 = tile_spec["read_x0"], tile_spec["read_x1"]
        wy0, wy1 = tile_spec["write_y0"], tile_spec["write_y1"]
        wx0, wx1 = tile_spec["write_x0"], tile_spec["write_x1"]

        owned = (self.y1 >= wy0) & (self.y1 < wy1) & (self.x1 >= wx0) & (self.x1 < wx1)
        owned_idx = np.flatnonzero(owned)
        if owned_idx.size == 0:
            return

        # Clip each owned window to the image exactly as the whole-map path does.
        y_start = np.maximum(0, self.y1[owned_idx])
        x_start = np.maximum(0, self.x1[owned_idx])
        y_stop = np.minimum(self.height, self.y2[owned_idx])
        x_stop = np.minimum(self.width, self.x2[owned_idx])
        self._claimed[owned_idx] += 1

        valid = (y_stop > y_start) & (x_stop > x_start)
        # The read region must cover every non-empty window, or a window would be
        # silently truncated and its dB wrong. Raise, as the per-ROI path did.
        covered = (
            (ry0 <= y_start) & (y_stop <= ry1) & (rx0 <= x_start) & (x_stop <= rx1)
        )
        bad = np.flatnonzero(valid & ~covered)
        if bad.size:
            b = int(bad[0])
            raise RuntimeError(
                f"ROI window [{int(y_start[b])}:{int(y_stop[b])}, "
                f"{int(x_start[b])}:{int(x_stop[b])}] is not covered by its tile's "
                f"read region [{ry0}:{ry1}, {rx0}:{rx1}]. The tile overlap must be "
                "at least the ROI size."
            )

        # Window bounds relative to the read region.
        y0r = (y_start - ry0).astype(np.int64)
        x0r = (x_start - rx0).astype(np.int64)
        y1r = (y_stop - ry0).astype(np.int64)
        x1r = (x_stop - rx0).astype(np.int64)
        # Full-size interior windows share one shape and are batched together;
        # partial (edge-clipped) windows keep the scalar path.
        full = (
            valid
            & (y_stop - y_start == self._full_h)
            & (x_stop - x_start == self._full_w)
        )

        def _scalar(positions: NDArray[np.integer]) -> None:
            for pos in positions:
                win = array[y0r[pos] : y1r[pos], x0r[pos] : x1r[pos]]
                if on_gpu:
                    win = cp.asnumpy(win)
                self.db[owned_idx[pos]] = snr_metrics.roi_snr_db(win)

        def _batch_reduce(st: Any) -> NDArray[np.float64]:
            # GPU: batched Otsu on-device (CuPy), no map transfer. CPU: numba prange
            # kernel (~50x the Python loop). The pure-numpy batch is memory-bound and
            # no faster than the loop, so it is only the fallback when numba is absent.
            # All three match the scalar roi_snr_db to float64 rounding.
            if on_gpu:
                return cp.asnumpy(snr_metrics.roi_snr_db_batch(st, xp=cp))
            if snr_metrics._HAS_NUMBA:
                return snr_metrics.roi_snr_db_numba(st)
            return snr_metrics.roi_snr_db_batch(st, xp=np)

        def _reduce() -> None:
            full_pos = np.flatnonzero(full)
            for start in range(0, full_pos.size, _OTSU_ROI_CHUNK):
                sel = full_pos[start : start + _OTSU_ROI_CHUNK]
                # xp.stack, not xp.asarray(list): the windows are on-device CuPy
                # slices, and cp.asarray of a Python list of CuPy arrays is
                # version-fragile, whereas stack of same-shape arrays is not.
                stack = xp.stack([array[y0r[p] : y1r[p], x0r[p] : x1r[p]] for p in sel])
                # The batched/numba kernels assume finite windows; route any
                # non-finite one to the scalar path (which filters them itself).
                finite = xp.isfinite(stack.reshape(stack.shape[0], -1)).all(axis=1)
                finite_host = cp.asnumpy(finite) if on_gpu else finite
                if bool(finite_host.all()):
                    self.db[owned_idx[sel]] = _batch_reduce(stack)
                else:
                    self.db[owned_idx[sel[finite_host]]] = _batch_reduce(stack[finite])
                    _scalar(sel[~finite_host])
            _scalar(np.flatnonzero(valid & ~full))

        if on_gpu:
            # Run every CuPy op under the array's own device -- the fold happens
            # after the GPU is returned to the pool, so the thread's current
            # device may be another one.
            with array.device:
                _reduce()
        else:
            _reduce()

    def finalize(self) -> NDArray[np.float64]:
        """Return the per-ROI dB array, checking each ROI was claimed once."""
        unclaimed = int((self._claimed == 0).sum())
        duplicated = int((self._claimed > 1).sum())
        if unclaimed or duplicated:
            raise RuntimeError(
                f"ROI coverage is wrong: {unclaimed} claimed by no tile, "
                f"{duplicated} by more than one."
            )
        return self.db

    def spawn(self) -> "RoiOtsuSnrAccumulator":
        """A fresh, empty accumulator sharing this one's ROI geometry.

        Used by the tile pass to give each worker thread its own accumulator so
        folds run without a lock; the partials are then reduced with ``merge``.
        """
        return RoiOtsuSnrAccumulator(
            self.y1,
            self.y2,
            self.x1,
            self.x2,
            (self.height, self.width),
            map_key=self.map_key,
        )

    def merge(self, other: "RoiOtsuSnrAccumulator") -> None:
        """Fold a per-worker partial into this one.

        An ROI is owned by the single tile whose write region holds its top-left
        corner, so partials hold DISJOINT per-ROI dB entries: the merge overlays the
        ROIs *other* computed, with no arithmetic and so bit-identical to the serial
        fold. The claim counts add so ``finalize`` still verifies global coverage.
        """
        overlap = (self._claimed > 0) & (other._claimed > 0)
        assert not overlap.any(), (
            "RoiOtsuSnrAccumulator partials claim overlapping ROIs; each ROI must be "
            "owned by exactly one tile (its top-left corner lands in one write region)."
        )
        claimed = other._claimed > 0
        self._claimed += other._claimed
        self.db[claimed] = other.db[claimed]


# ---------------------------------------------------------------------------
# Full-resolution plane storage (RAM or disk-backed)
# ---------------------------------------------------------------------------

# Planes at or above this size are backed by a file rather than anonymous RAM.
_PLANE_SPILL_BYTES = 2 * 1024**3


class _PlaneStore:
    """Owns the full-resolution output planes for one channel.

    A plane that reaches ``_PLANE_SPILL_BYTES`` is backed by an ``np.memmap``
    file in *plane_dir* (the task working directory by default) instead of
    anonymous RAM.  Nothing declares these files as process outputs, so Nextflow
    discards them with the work directory; use the ``scratch`` directive to put
    that directory on node-local NVMe.  Two
    things follow from that, and together they are why this class exists:

    * **Bounded resident set.** Every consumer of these planes reads row blocks
      or ROI windows, never the whole array (see ``_labeled_sums_chunked`` and
      the SNR per-tile loop).  File backing turns 22 GB of unreclaimable
      anonymous pages per plane into page cache the kernel can evict under
      pressure.  On a 5.5 gigapixel sample the seven planes come to ~154 GB
      resident, which is what forced the 180 -> 720 GB retry ladder.
    * **Real multi-GPU parallelism.** A memmap is shared between processes by
      *filename*, so tile workers can be separate processes — one per GPU, each
      with its own interpreter, its own GIL and its own CUDA context — all
      writing into the same planes with no pixel data ever pickled.  Thread
      workers cannot scale much past a single GPU's throughput no matter how
      many devices are present, because the host-side numpy in every tile (the
      input dtype cast, ``_sanitize``, the post-D2H cast) holds the GIL.  NVLink
      is irrelevant to this workload: tiles are independent and nothing is
      exchanged between devices.

    Planes are always handed out as ordinary ndarrays, so consumers never need
    to know which backing is in use.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        keys: list[str],
        plane_dir: Path | None = None,
        prefix: str = "plane",
        dtype: Any = np.float32,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(dtype)
        self.keys = list(keys)
        plane_bytes = self.shape[0] * self.shape[1] * self.dtype.itemsize
        # Small planes are not worth a file; large ones always get one.
        self.on_disk = plane_bytes >= _PLANE_SPILL_BYTES
        self._dir = Path(plane_dir or Path.cwd()) if self.on_disk else None
        self.paths: dict[str, str] = {}
        self._arrays: dict[str, NDArray[Any]] = {}

        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            required = plane_bytes * len(self.keys)
            free = shutil.disk_usage(self._dir).free
            if free < required:
                raise RuntimeError(
                    f"scratch dir {self._dir} has {free / 1024**3:.1f} GB free, but "
                    f"{len(self.keys)} plane(s) of {plane_bytes / 1024**3:.1f} GB "
                    f"need {required / 1024**3:.1f} GB. Point --scratch-dir at a "
                    "larger filesystem, or drop the flag to keep planes in RAM."
                )
            logging.info(
                f"    Plane store: {len(self.keys)} x "
                f"{plane_bytes / 1024**3:.1f} GB on disk at {self._dir}"
            )

        for key in self.keys:
            if self._dir is not None:
                path = self._dir / f"{prefix}_{key}.dat"
                self._arrays[key] = np.memmap(
                    path, dtype=self.dtype, mode="w+", shape=self.shape
                )
                self.paths[key] = str(path)
            else:
                self._arrays[key] = np.empty(self.shape, dtype=self.dtype)

    def arrays(self) -> dict[str, Any]:
        """Plane arrays keyed by name, for in-process use."""
        return dict(self._arrays)

    def descriptors(self) -> dict[str, str] | None:
        """``{key: path}`` for reopening in another process, None when in RAM."""
        return dict(self.paths) if self.on_disk else None

    def flush(self) -> None:
        """Push dirty memmap pages to the backing files."""
        for arr in self._arrays.values():
            if isinstance(arr, np.memmap):
                arr.flush()

    def release(self) -> None:
        """Drop references and delete the backing files."""
        self.flush()
        self._arrays.clear()
        for path in self.paths.values():
            Path(path).unlink(missing_ok=True)
        self.paths.clear()


# ---------------------------------------------------------------------------
# Process-per-GPU tile workers
# ---------------------------------------------------------------------------

# Per-process state, populated by _tile_worker_init in each pool worker.
_WORKER: dict[str, Any] = {}


def _tile_worker_init(
    slot_counter,
    gpu_ids: tuple[int, ...],
    channel_source: tuple[str, int],
    plane_paths: dict[str, str],
    shape: tuple[int, int],
    dtype_str: str,
) -> None:
    """Pool initializer: claim one GPU, open our own TIFF handle and plane views.

    Each worker claims a distinct device by taking the next slot from a shared
    counter, so with ``processes == len(gpu_ids)`` every GPU gets exactly one
    process.  Tiles are then pulled from the pool's task queue, which
    load-balances naturally — unlike static ``gpu_ids[i % n_gpus]`` round-robin,
    which leaves fast devices idle whenever tile costs differ (the DAPI channel
    also computes the Laplacian, so its tiles cost roughly 2.5x the others').
    """
    # A spawned child never runs main(), so it never runs logging.basicConfig and
    # the root logger would sit at WARNING -- silencing every worker-side message
    # exactly where multi-GPU problems would show up.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    with slot_counter.get_lock():
        slot = slot_counter.value
        slot_counter.value += 1
    gpu_id = gpu_ids[slot % len(gpu_ids)]

    path, page_index = channel_source
    tif = tifffile.TiffFile(path)
    channel = _LazyTiffChannel(tif.pages[page_index], source=channel_source)
    channel._tiff_handles = [tif]  # type: ignore[attr-defined]

    dtype = np.dtype(dtype_str)
    _WORKER.clear()
    _WORKER.update(
        gpu_id=gpu_id,
        channel=channel,
        planes={
            key: np.memmap(p, dtype=dtype, mode="r+", shape=tuple(shape))
            for key, p in plane_paths.items()
        },
    )
    logging.info(f"    Tile worker pid={os.getpid()} bound to GPU {gpu_id}")


def _tile_worker_run(
    task: tuple[dict[str, int], int, bool, float],
) -> tuple[int, float]:
    """Pool task: compute one tile on this process's GPU. Returns (gpu_id, seconds)."""
    tile_spec, window_size, include_laplacian, lap_sigma = task
    t0 = time.perf_counter()
    _process_tile_on_gpu(
        _WORKER["channel"],
        tile_spec,
        window_size,
        _WORKER["gpu_id"],
        _WORKER["planes"],
        include_laplacian,
        lap_sigma,
    )
    for plane in _WORKER["planes"].values():
        if isinstance(plane, np.memmap):
            plane.flush()
    return int(_WORKER["gpu_id"]), time.perf_counter() - t0


def _log_gpu_balance(timings: list[tuple[int, float]], label: str) -> None:
    """Log per-GPU occupancy so multi-GPU scaling is measurable, not assumed.

    An aggregate wall-clock number cannot reveal load imbalance: it reports
    max(per-device time), so an idle device looks identical to a saturated one.
    """
    if not timings:
        return
    per_gpu: dict[int, list[float]] = {}
    for gpu_id, seconds in timings:
        per_gpu.setdefault(gpu_id, []).append(seconds)
    busiest = max(sum(v) for v in per_gpu.values()) or 1.0
    for gpu_id in sorted(per_gpu):
        secs = per_gpu[gpu_id]
        total = sum(secs)
        logging.info(
            f"  [TIMING] {label} GPU {gpu_id}: {len(secs)} tiles, "
            f"{total:.1f}s busy ({100.0 * total / busiest:.0f}% of busiest)"
        )


# ---------------------------------------------------------------------------
# Focus-map computation
# ---------------------------------------------------------------------------


def _process_tile_on_gpu(
    channel_data,
    tile_spec: dict[str, int],
    window_size: int,
    gpu_id: int,
    out: dict[str, NDArray[np.float32] | None],
    include_laplacian: bool = False,
    lap_sigma: float = 1.0,
) -> None:
    """Read a tile, compute focus maps on GPU, trim overlap, write into *out*.

    Writes its trimmed results straight into the caller's preallocated
    full-resolution planes rather than returning them.  This is what bounds
    host RAM: a returned dict would be pinned by ``Future._result`` until the
    assembly loop finished, and ``as_completed()`` holds every ``Future`` for
    the duration of iteration — so returning arrays retains *every* tile's
    output (roughly one extra full copy of each plane, plus halo).  Writing in
    place means the ``Future`` carries only ``None``.

    Each tile's write region is disjoint from every other tile's (guaranteed by
    ``_compute_tile_grid``; asserted by ``test_no_write_overlap``), so
    concurrent in-place writes from the worker threads are safe.

    Args:
        channel_data: Array-like supporting slicing (_LazyTiffChannel or numpy array).
        tile_spec: Dict from _compute_tile_grid() with read/write/trim keys.
        window_size: Convolution window size.
        gpu_id: CUDA device ID.
        out: Dict of preallocated full-res planes keyed ``focus_map`` /
            ``mean_map`` / ``lap_var_map``.  A ``None`` value skips that plane.
        include_laplacian: Also compute Laplacian variance map.
        lap_sigma: Gaussian sigma for LoG.
    """
    # Read tile (triggers actual I/O for lazy TIFF-backed arrays)
    tile = np.asarray(
        channel_data[
            tile_spec["read_y0"] : tile_spec["read_y1"],
            tile_spec["read_x0"] : tile_spec["read_x1"],
        ]
    )

    # Compute on GPU using existing function
    result = _compute_channel_maps_on_gpu(
        tile, window_size, gpu_id, include_laplacian, lap_sigma
    )
    del tile

    # Trim overlap and write each output straight into the caller's plane
    tt = tile_spec["trim_top"]
    tb = tile_spec["trim_bottom"]
    tl = tile_spec["trim_left"]
    tr = tile_spec["trim_right"]
    wy0, wy1 = tile_spec["write_y0"], tile_spec["write_y1"]
    wx0, wx1 = tile_spec["write_x0"], tile_spec["write_x1"]

    for key in list(result):
        plane = out.get(key)
        arr = result.pop(key)
        if plane is None:
            continue
        h, w = arr.shape
        y1 = h - tb if tb > 0 else h
        x1 = w - tr if tr > 0 else w
        plane[wy0:wy1, wx0:wx1] = arr[tt:y1, tl:x1]


def _process_tile_for_consumers(
    channel_data,
    tile_spec: dict[str, int],
    window_size: int,
    gpu_id: int,
    include_laplacian: bool = False,
    lap_sigma: float = 1.0,
    keep_mean_device: bool = False,
    keep_focus_device: bool = False,
    drop_mean_host: bool = False,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]:
    """Compute one tile and return ``(trimmed, untrimmed)`` maps for consumers.

    Returns rather than writes: with no plane to write into, the parent folds each
    tile into the consumers and drops it. Both forms are needed —
    ``RoiOtsuSnrAccumulator`` requires the haloed (untrimmed) tile because an ROI
    it owns can extend up to ``roi_size - 1`` px past the write region, while the
    other consumers want the trimmed tile that maps exactly onto the write region.

    Device maps (``focus_map_device`` / ``mean_map_device``) are trimmed too — the
    write-region view for the trimmed consumers (``CentrePixelSampler``,
    ``LabeledSumAccumulator``) while the untrimmed dict keeps the full device array
    for ``RoiOtsuSnrAccumulator``. Slicing a CuPy array is a view, so this adds no
    device allocation.

    The untrimmed arrays are *views* into the same buffers, so returning both adds
    no allocation.
    """
    tile = np.asarray(
        channel_data[
            tile_spec["read_y0"] : tile_spec["read_y1"],
            tile_spec["read_x0"] : tile_spec["read_x1"],
        ]
    )
    untrimmed = _compute_channel_maps_on_gpu(
        tile,
        window_size,
        gpu_id,
        include_laplacian,
        lap_sigma,
        keep_mean_device=keep_mean_device,
        keep_focus_device=keep_focus_device,
        drop_mean_host=drop_mean_host,
    )
    del tile

    top = tile_spec["trim_top"]
    bottom = tile_spec["trim_bottom"]
    left = tile_spec["trim_left"]
    right = tile_spec["trim_right"]
    trimmed = {}
    for key, arr in untrimmed.items():
        # Trim every map, host and device alike, to the write region. The trimmed
        # device slices are what CentrePixelSampler / LabeledSumAccumulator fold;
        # the untrimmed dict keeps the full device arrays for RoiOtsuSnrAccumulator,
        # whose owned ROI windows reach past the write region into the halo.
        height, width = arr.shape
        y_stop = height - bottom if bottom > 0 else height
        x_stop = width - right if right > 0 else width
        trimmed[key] = arr[top:y_stop, left:x_stop]
    return trimmed, untrimmed


def _compute_channel_maps_tiled(
    channel_data,
    image_shape: tuple[int, int],
    window_size: int,
    gpu_ids: list[int],
    include_laplacian: bool = False,
    lap_sigma: float = 1.0,
    gpu_mem_bytes: int | None = None,
    plane_dir: Path | None = None,
    plane_prefix: str = "plane",
    consumers: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Compute focus maps for one channel using tiled multi-GPU processing.

    Tiles the image with convolution-safe overlap, distributes tiles across the
    available GPUs, and assembles the results into full-resolution planes.  Tile
    size is chosen adaptively to fit GPU memory.

    With more than one GPU and disk-backed planes, tiles are dispatched to one
    worker **process** per GPU (see ``_tile_worker_init``); otherwise a thread
    pool is used, which keeps single-GPU behaviour unchanged.

    Args:
        channel_data: Array-like (_LazyTiffChannel or numpy) with shape matching image_shape.
        image_shape: (height, width).
        window_size: Convolution window size.
        gpu_ids: List of CUDA device IDs.
        include_laplacian: Also compute Laplacian variance.
        lap_sigma: Gaussian sigma for LoG.
        gpu_mem_bytes: Total GPU VRAM in bytes. Auto-detected if None.
        plane_dir: Directory for the disk-backed output planes.  Defaults to
            the task working directory, which is what Nextflow's ``scratch``
            directive relocates onto node-local storage.
        plane_prefix: Filename prefix / log label for this channel's planes.

    Returns:
        Dict with 'focus_map', 'mean_map', and optionally 'lap_var_map'
        (full-resolution float32 arrays; memmaps when disk-backed).
    """
    H, W = image_shape
    n_gpus = len(gpu_ids)
    # Overlap must cover both uniform_filter radius (window_size // 2) and the
    # Gaussian pre-smoothing in the Laplacian path (~3 * lap_sigma).  Using the
    # full window_size is safe for all kernels and adds negligible I/O overhead.
    overlap = window_size
    # Streaming mode needs MORE than the convolution radius. RoiOtsuSnrAccumulator
    # reads ROI windows out of the *untrimmed* tile, and it claims an ROI by its top
    # edge y1, so the window reaches `roi_size - 1` px past write_y1 -- while
    # uniform_filter's reflect padding corrupts the last `window_size // 2` rows of the
    # read region. With overlap == window_size == roi_size == 35 the padded band starts
    # at write_y1 + 18 and the ROI reaches write_y1 + 33, so up to 16 of its 35 rows
    # came from padding.
    #
    # Measured before this fix: 2,855 of 335,241 ROIs (0.85 %) had a different
    # snr_image_otsu_db from the plane-based path, by up to 19.9 % relative, and that
    # single column was the ONLY difference across 66 output files
    # (runs 5ZL9TWmJ701ham vs 1AjX0aBBfdbAFQ).
    #
    # roi_size is the ROI side, which equals window_size on the production path but is
    # kept separate here because the requirement is genuinely about the ROI, not the
    # kernel.
    if consumers:
        roi_reach = window_size  # ROI side length; grid stride defaults to roi_size
        needed = roi_reach - 1 + window_size // 2
        if needed > overlap:
            overlap = needed
    # VRAM cost of the larger halo, checked rather than assumed: the strip height from
    # _compute_adaptive_strip_height does NOT include the overlap, so the read region
    # always exceeds that budget by 2*overlap rows. At the production geometry (20409-row
    # strips, 53908 px wide, ~24 B/px of concurrent buffers) going 35 -> 51 adds 32 rows,
    # i.e. 0.039 GB, taking a tile from 24.68 to 24.71 GB of the 44.5 GB device. The
    # budget was already optimistic by 70 rows; it is now optimistic by 102.

    # Fail loud if lap_sigma is raised past what the halo can cover: the LoG
    # needs window_size//2 (uniform_filter) + ~4*lap_sigma+1 (Gaussian+Laplace)
    # of halo; beyond `overlap` the DAPI lap_var map would differ from the
    # whole-image result at tile seams (silently). Default lap_sigma=1.0 is safe.
    if include_laplacian:
        lap_halo = window_size // 2 + int(math.ceil(4 * lap_sigma)) + 1
        if lap_halo > overlap:
            logging.warning(
                f"  lap_sigma={lap_sigma} needs {lap_halo}px halo > overlap "
                f"{overlap}px; Laplacian tile seams may differ from whole-image. "
                f"Raise the tile overlap or lower lap_sigma."
            )

    # Full-width row strips rather than square tiles: a strip's write region is
    # one contiguous byte range, and it needs halo on top/bottom only. See
    # _compute_adaptive_strip_height for the measurements that motivated this.
    strip_height = _compute_adaptive_strip_height(H, W, n_gpus, gpu_mem_bytes)
    # Derived from the consumers, not supplied by the caller: BlockMeanAccumulator needs
    # write origins on multiples of its block size, and a caller who had to remember to
    # say so would discover the omission only as a RuntimeError mid-fold.
    align_writes_to = 1
    for consumer in consumers or ():
        align_writes_to = math.lcm(
            align_writes_to, int(getattr(consumer, "write_alignment", 1) or 1)
        )
    if align_writes_to > 1:
        # Align write boundaries to the heatmap block size so every block lies wholly
        # inside one tile; BlockMeanAccumulator then reduces it with one reshape and the
        # canvas is bit-exact. See that class for the 16-pixel figure difference.
        # max(), not a conditional skip: a strip shorter than one block would otherwise
        # be left unaligned and trip the accumulator's contract check at runtime. Rounding
        # up to one block costs at most align_writes_to rows of halo.
        strip_height = max(
            align_writes_to, (strip_height // align_writes_to) * align_writes_to
        )
    tiles = _compute_tile_grid(H, W, strip_height, overlap, tile_width=W)

    logging.info(
        f"    Tiled processing: {len(tiles)} row strip(s) "
        f"({strip_height}x{W}px), {n_gpus} GPU(s), overlap={overlap}px"
    )

    # Ensure CUDA_PATH is set for CuPy NVRTC kernel compilation in worker
    # threads.  CuPy auto-detects CUDA in the main thread but worker threads
    # can fail with "Failed to auto-detect CUDA root directory".  The pip
    # package nvidia-cuda-runtime-cu12 installs headers under
    # site-packages/nvidia/cuda_runtime/include/.
    if "CUDA_PATH" not in os.environ:
        try:
            import nvidia.cuda_runtime as _cr

            _cr_dir = _cr.__path__[0]  # .../site-packages/nvidia/cuda_runtime
            if os.path.isdir(os.path.join(_cr_dir, "include")):
                os.environ["CUDA_PATH"] = _cr_dir
                logging.info(f"    Set CUDA_PATH={_cr_dir}")
        except (ImportError, IndexError, AttributeError):
            pass  # CUDA_PATH remains unset; CuPy will try its own detection

    # Warm up CuPy kernel cache in the main thread.  NVRTC compiles kernels on
    # first use; running a tiny operation here populates the disk cache so that
    # worker threads hit the cache instead of compiling in parallel.
    try:
        _warmup = cp.array([1.0], dtype=cp.float32)
        cupyx.scipy.ndimage.uniform_filter(_warmup.reshape(1, 1), size=1)
        del _warmup
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Consumer mode: no planes at all. Each tile is folded into the consumers and
    # dropped, so peak host memory is O(max_inflight * tile) + O(n_rois, n_cells) --
    # NOT O(tile). Results sit in their futures until the parent pops and folds them,
    # and folding is serialised here, so up to max_inflight tiles can be complete and
    # resident at once. Measured on a 102045x53908 sample: 12 strips of 8504 rows is
    # 1.71 GB per map, so DAPI's 3 maps x 8 in flight is ~41 GB worst case against
    # ~154 GB of planes for the whole image.
    # ------------------------------------------------------------------
    if consumers:
        # Each worker THREAD folds its tiles into its OWN set of consumer
        # accumulators, then the per-thread partials are reduced into the caller's
        # consumers after the executor drains. There is no fold lock, so the n_gpus
        # workers fold CONCURRENTLY -- the fold is the wall clock here (dominated by
        # RoiOtsuSnrAccumulator's per-ROI Otsu, LabeledSumAccumulator's bincounts),
        # so a single fold_lock serialised the folds and threw away the multi-GPU
        # parallelism (the DAPI fold measured ~240 s of the run behind that lock).
        #
        # Correctness of the reduce, per consumer:
        #   * RoiOtsuSnrAccumulator / CentrePixelSampler: every ROI is owned by
        #     exactly one tile (the halo guarantees it), so partials hold DISJOINT
        #     per-ROI entries -- merge is an overlay, bit-identical to the serial
        #     fold regardless of order.
        #   * BlockMeanAccumulator: write origins are aligned to the block size, so
        #     every block lies wholly in one tile -> one partial; canvases are
        #     disjoint and add exactly.
        #   * LabeledSumAccumulator: per-label float64 count/sum arrays are added
        #     element-wise. This regroups the additions relative to the serial
        #     tile-order fold; that it still reproduces them byte-for-byte is
        #     empirical, pinned by tests/test_fold_parallel_equivalence.
        #
        # Folding in the worker (not the dispatch loop) is also what bounds memory: a
        # tile is released as soon as it is folded, instead of staying alive in its
        # Future until the parent catches up.
        gpu_pool: queue.Queue[int] = queue.Queue()
        for _gpu in gpu_ids:
            gpu_pool.put(_gpu)

        # Per-worker-thread consumer sets, created lazily on the thread's first tile
        # and registered under a lock with a stable creation index so the reduce
        # order is deterministic (run-to-run reproducible). threading.local() keys
        # the set to the OS thread the ThreadPoolExecutor reuses, so a thread folds
        # all its tiles into the one set it created.
        _thread_state = threading.local()
        partials: list[tuple[int, list[Any]]] = []
        partials_lock = threading.Lock()
        per_consumer_lock = threading.Lock()

        def _thread_consumers() -> list[Any]:
            local = getattr(_thread_state, "consumers", None)
            if local is not None:
                return local
            local = [c.spawn() for c in consumers]
            with partials_lock:
                index = len(partials)
                partials.append((index, local))
            _thread_state.consumers = local
            return local

        # Keep the maps resident on the GPU when a consumer folds a reduction there
        # (RoiOtsuSnrAccumulator, LabeledSumAccumulator, CentrePixelSampler do their
        # per-ROI / per-label / centre-pixel reductions on-device). The host mean map
        # is dropped when every consumer that reads the mean map takes the device one
        # -- then nothing reads the host copy and the D2H transfer is eliminated. The
        # host focus map is always produced: BlockMeanAccumulator (Figure 5) reduces
        # it in float32 on the host, which a GPU reduction cannot match to float
        # rounding, so it never sets wants_device_focus.
        keep_mean_device = any(
            getattr(c, "wants_device_mean", False) for c in consumers
        )
        keep_focus_device = any(
            getattr(c, "wants_device_focus", False) for c in consumers
        )
        drop_mean_host = keep_mean_device and not any(
            getattr(c, "reads_host_mean", False) for c in consumers
        )

        timings: list[tuple[int, float]] = []
        fold_seconds = 0.0
        per_consumer: dict[str, float] = {}

        def _run_tile(spec) -> tuple[int, float, float]:
            gpu_id = gpu_pool.get()
            try:
                t0 = time.perf_counter()
                trimmed, untrimmed = _process_tile_for_consumers(
                    channel_data,
                    spec,
                    window_size,
                    gpu_id,
                    include_laplacian,
                    lap_sigma,
                    keep_mean_device=keep_mean_device,
                    keep_focus_device=keep_focus_device,
                    drop_mean_host=drop_mean_host,
                )
                compute_seconds = time.perf_counter() - t0

                # Fold into THIS thread's own consumer set -- no lock, so the n_gpus
                # workers fold concurrently.
                local_consumers = _thread_consumers()
                t1 = time.perf_counter()
                local_per: dict[str, float] = {}
                for consumer in local_consumers:
                    t_c = time.perf_counter()
                    if getattr(consumer, "wants_untrimmed", False):
                        consumer.consume(spec, untrimmed)
                    else:
                        consumer.consume(spec, trimmed)
                    name = type(consumer).__name__
                    local_per[name] = local_per.get(name, 0.0) + (
                        time.perf_counter() - t_c
                    )
                fold = time.perf_counter() - t1
                # Aggregate the per-consumer fold time across threads (these overlap
                # in wall clock now, so this is summed CPU, not wall time).
                with per_consumer_lock:
                    for name, seconds in local_per.items():
                        per_consumer[name] = per_consumer.get(name, 0.0) + seconds
                del trimmed, untrimmed
                return gpu_id, compute_seconds, fold
            finally:
                # Release the GPU slot AFTER the fold, not before it. With
                # keep_mean_device/keep_focus_device this tile's mean/focus maps are
                # device arrays that the fold reduces on THIS card. Releasing the slot
                # before the fold (the previous behaviour) let another worker grab this
                # card and start a second tile's compute while these maps were still
                # resident -- an unbounded cross-worker pile-up (up to one compute +
                # n_gpus-1 held maps on one card) that breaks the per-device VRAM bound
                # #55 established, since _compute_adaptive_strip_height sizes for one
                # tile's compute arrays only. Holding the slot through the fold bounds
                # each card to a single tile at a time (its compute arrays, then its
                # held maps), so the strip-height budget is the real per-card ceiling.
                # The trade is small: the fold runs on this card regardless (on-device
                # reduction) and is short since the GPU Otsu change, so the only lost
                # overlap is a peer tile's compute starting on this card mid-fold.
                gpu_pool.put(gpu_id)

        t_dispatch = time.perf_counter()
        logging.info(
            f"    Consumer mode: {len(tiles)} tiles, {n_gpus} GPU(s), "
            f"per-worker parallel fold, no planes materialised"
        )
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            for gpu_id, compute_seconds, fold in executor.map(_run_tile, tiles):
                timings.append((gpu_id, compute_seconds))
                fold_seconds += fold

        # Reduce the per-thread partials into the caller's consumers, in a
        # deterministic order (stable creation index). The caller's consumers never
        # consumed a tile, so they start empty and this fills them for the
        # downstream finalize().
        for _index, local_consumers in sorted(partials, key=lambda item: item[0]):
            for target, part in zip(consumers, local_consumers):
                target.merge(part)

        elapsed = time.perf_counter() - t_dispatch
        logging.info(
            f"  [TIMING] {plane_prefix} tiled compute (consumers, {len(tiles)} "
            f"tiles, {n_gpus} GPU): {elapsed:.1f}s"
        )
        logging.info(
            f"  [TIMING] {plane_prefix} consumer folding: {fold_seconds:.1f}s "
            f"({100.0 * fold_seconds / max(elapsed, 1e-9):.0f}% of wall clock)"
        )
        for name, seconds in sorted(per_consumer.items(), key=lambda kv: -kv[1]):
            logging.info(
                f"  [TIMING] {plane_prefix} fold {name}: {seconds:.1f}s "
                f"({100.0 * seconds / max(fold_seconds, 1e-9):.0f}% of folding)"
            )
        # Read/decode/convolve split. read+decode are serialized under the reader's
        # lock, so compare their sum to the wall clock (elapsed): sum ~ elapsed means
        # the reader is the wall (IO/decode-bound, GPUs starve); sum << elapsed means
        # the GPU convolution dominates. total_compute is the summed per-tile
        # read+upload+convolve across all workers, so convolve+upload ~ total_compute
        # - read - decode.
        read_s = float(getattr(channel_data, "_read_seconds", 0.0))
        decode_s = float(getattr(channel_data, "_decode_seconds", 0.0))
        read_gb = float(getattr(channel_data, "_read_bytes", 0)) / 1024**3
        total_compute = sum(secs for _gid, secs in timings)
        logging.info(
            f"  [TIMING] {plane_prefix} read (S3/fh): {read_s:.1f}s "
            f"({read_gb:.2f} GB, {read_gb / max(read_s, 1e-9):.2f} GB/s), "
            f"decode: {decode_s:.1f}s, read+decode = "
            f"{100.0 * (read_s + decode_s) / max(elapsed, 1e-9):.0f}% of wall clock"
        )
        logging.info(
            f"  [TIMING] {plane_prefix} convolve+upload ~= "
            f"{max(total_compute - read_s - decode_s, 0.0):.1f}s "
            f"(sum tile-compute {total_compute:.1f}s - read {read_s:.1f}s - "
            f"decode {decode_s:.1f}s)"
        )
        _log_gpu_balance(timings, f"{plane_prefix} streamed")
        return None

    # Allocate the output planes.  With a scratch dir configured these are
    # disk-backed memmaps rather than anonymous RAM, which both bounds the
    # resident set and lets the tile workers be separate processes — see
    # _PlaneStore.  Workers write their trimmed tiles straight into these, so no
    # per-tile result is ever retained (`as_completed()` holds every Future for
    # the duration of iteration, so a worker that *returned* its arrays would
    # keep all of them alive until the executor block exited).
    keys = ["focus_map", "mean_map"] + (["lap_var_map"] if include_laplacian else [])
    store = _PlaneStore(
        (H, W), keys, plane_dir=plane_dir, prefix=plane_prefix, dtype=np.float32
    )
    out_planes: dict[str, Any] = store.arrays()
    out_planes.setdefault("lap_var_map", None)

    channel_source = getattr(channel_data, "_source", None)
    plane_paths = store.descriptors()
    # Process mode needs: more than one GPU to be worth it, a reopenable channel
    # (a live TiffPage holds an OS handle and a lock, so it cannot be pickled),
    # and file-backed planes to write into.  Single-GPU runs keep the thread
    # path, so their behaviour is unchanged.
    use_processes = (
        n_gpus > 1 and channel_source is not None and plane_paths is not None
    )
    tasks = [(spec, window_size, include_laplacian, lap_sigma) for spec in tiles]
    t_dispatch = time.perf_counter()

    if use_processes:
        assert channel_source is not None and plane_paths is not None
        # One process per GPU.  Separate interpreters mean the host-side numpy
        # in each tile (input cast, _sanitize, post-D2H cast) no longer
        # serialises on a shared GIL, and each process builds its own CUDA
        # context.  Must be "spawn": CUDA does not survive fork().
        logging.info(
            f"    Dispatching {len(tasks)} tiles to {n_gpus} worker process(es), "
            f"one per GPU {gpu_ids} (spawn)"
        )
        ctx = multiprocessing.get_context("spawn")
        slot_counter = ctx.Value("i", 0)
        with ctx.Pool(
            processes=n_gpus,
            initializer=_tile_worker_init,
            initargs=(
                slot_counter,
                tuple(gpu_ids),
                channel_source,
                plane_paths,
                (H, W),
                np.dtype(np.float32).str,
            ),
        ) as pool:
            timings = list(pool.imap_unordered(_tile_worker_run, tasks))
    else:
        # Pipelined I/O + GPU: each worker reads its tile from TIFF then
        # computes on its assigned GPU.  The TIFF file handle lock serializes
        # reads, but I/O for tile N+1 overlaps with GPU compute for tile N.
        def _run_tile(index: int, spec: dict[str, int]) -> tuple[int, float]:
            gpu_id = gpu_ids[index % n_gpus]
            t0 = time.perf_counter()
            _process_tile_on_gpu(
                channel_data,
                spec,
                window_size,
                gpu_id,
                out_planes,
                include_laplacian,
                lap_sigma,
            )
            return gpu_id, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [
                executor.submit(_run_tile, i, spec) for i, spec in enumerate(tiles)
            ]
            timings = [f.result() for f in as_completed(futures)]

    store.flush()
    mode = "process" if use_processes else "thread"
    logging.info(
        f"  [TIMING] {plane_prefix} tiled compute ({mode}, {len(tiles)} tiles, "
        f"{n_gpus} GPU): {time.perf_counter() - t_dispatch:.1f}s"
    )
    _log_gpu_balance(timings, f"{plane_prefix} tiled")

    # Per-tile results are already sanitized in _compute_channel_maps_on_gpu,
    # so no redundant final full-array _sanitize here.
    return {
        "focus_map": out_planes["focus_map"],
        "mean_map": out_planes["mean_map"],
        "lap_var_map": out_planes.get("lap_var_map"),
    }


def compute_all_focus_maps(
    channels: NDArray[np.generic],
    window_size: int = 35,
    use_gpu: bool = False,
    gpu_ids: list[int] | None = None,
    lap_sigma: float = 1.0,
) -> dict[str, NDArray[np.float32] | None]:
    """Compute focus maps for all available image channels.

    When ``gpu_ids`` contains multiple IDs, channel computations are
    distributed across GPUs in parallel using a thread pool. Each channel's
    convolution work runs entirely on one GPU; with 3 channels and 3+ GPUs
    all channels are processed concurrently.

    Args:
        channels: Either a 2-D array (single DAPI channel, shape ``(H, W)``)
            or a 3-D array with channels first (shape ``(C, H, W)``).
        window_size: Side length of the square averaging window.
        use_gpu: Use CuPy GPU backend when ``True``. Ignored when *gpu_ids*
            is provided (GPU is assumed).
        gpu_ids: List of CUDA device IDs for multi-GPU parallelism. If
            ``None`` and ``use_gpu=True``, uses device 0 only. If ``None``
            and ``use_gpu=False``, runs on CPU.

    Returns:
        Dictionary with the following keys (values are ``None`` when the
        corresponding channel is not present in *channels*):

        - ``dapi_focus_map`` — CCFS focus map for DAPI (channel 0).
        - ``dapi_mean_map`` — Local mean map for DAPI.
        - ``dapi_lap_var_map`` — Laplacian variance map for DAPI.
        - ``boundary_focus_map`` — CCFS focus map for Boundary (channel 1).
        - ``boundary_mean_map`` — Local mean map for Boundary.
        - ``intrna_focus_map`` — CCFS focus map for IntRNA (channel 2).
        - ``intrna_mean_map`` — Local mean map for IntRNA.

    Raises:
        ValueError: If *channels* has fewer than 2 or more than 3 dimensions.
    """
    # Resolve GPU configuration
    if gpu_ids is not None and len(gpu_ids) > 0:
        use_gpu = True
    elif use_gpu and gpu_ids is None:
        gpu_ids = [0]

    # Split channels — .copy() so the parent 3-D array can be freed.
    if channels.ndim == 2:
        dapi = channels.copy()
        boundary = None
        intrna = None
    elif channels.ndim == 3:
        n_ch = channels.shape[0]
        dapi = channels[0].copy()
        boundary = channels[1].copy() if n_ch > 1 else None
        intrna = channels[2].copy() if n_ch > 2 else None
    else:
        raise ValueError(
            f"Expected 2-D or 3-D array, got {channels.ndim}-D "
            f"(shape {channels.shape})."
        )
    del channels

    # ---- Multi-GPU path: distribute channels across GPUs ----
    if use_gpu and gpu_ids is not None and len(gpu_ids) > 0:
        # Build work items: (channel_name, channel_data, include_laplacian)
        work_items: list[tuple[str, NDArray[np.generic], bool]] = [
            ("dapi", dapi, True),  # DAPI always gets Laplacian
        ]
        if boundary is not None:
            work_items.append(("boundary", boundary, False))
        if intrna is not None:
            work_items.append(("intrna", intrna, False))

        # Assign GPUs round-robin
        results: dict[str, dict[str, NDArray[np.float32]]] = {}
        n_gpus = len(gpu_ids)
        logging.info(
            f"    Distributing {len(work_items)} channel(s) across {n_gpus} GPU(s): {gpu_ids}"
        )

        t_gpu = time.perf_counter()
        gpu_failed = False
        try:
            with ThreadPoolExecutor(
                max_workers=min(len(work_items), n_gpus)
            ) as executor:
                futures = {}
                for idx, (name, ch_data, inc_lap) in enumerate(work_items):
                    assigned_gpu = gpu_ids[idx % n_gpus]
                    future = executor.submit(
                        _compute_channel_maps_on_gpu,
                        ch_data,
                        window_size,
                        assigned_gpu,
                        inc_lap,
                        lap_sigma,
                    )
                    futures[future] = name

                for future in as_completed(futures):
                    ch_name = futures[future]
                    results[ch_name] = future.result()
            logging.info(
                f"  [TIMING] GPU compute (multi-GPU, {len(work_items)} channels): {time.perf_counter() - t_gpu:.1f}s"
            )
        except Exception as gpu_err:
            logging.warning(
                f"  WARNING: GPU computation failed ({gpu_err}), falling back to CPU..."
            )
            gpu_failed = True

        if not gpu_failed:
            # Assemble output dict from GPU results
            dapi_res = results["dapi"]
            return {
                "dapi_focus_map": dapi_res["focus_map"],
                "dapi_mean_map": dapi_res["mean_map"],
                "dapi_lap_var_map": dapi_res.get("lap_var_map"),
                "boundary_focus_map": results["boundary"]["focus_map"]
                if "boundary" in results
                else None,
                "boundary_mean_map": results["boundary"]["mean_map"]
                if "boundary" in results
                else None,
                "intrna_focus_map": results["intrna"]["focus_map"]
                if "intrna" in results
                else None,
                "intrna_mean_map": results["intrna"]["mean_map"]
                if "intrna" in results
                else None,
            }
        # else: fall through to CPU path below

    # ---- Single-GPU or CPU fallback path ----
    # Reached when: no multi-GPU available, use_gpu=False, or GPU OOM fallback
    use_gpu = (
        False  # Force CPU to avoid repeated OOM if we fell through from GPU failure
    )
    gpu_id = gpu_ids[0] if gpu_ids else 0

    t_gpu = time.perf_counter()
    # Process channels sequentially, freeing each input before the next
    # to keep peak memory at ~1 channel + its output maps.

    # DAPI — always present
    dapi_focus_map, dapi_mean_map = compute_ccfs_map(
        dapi, window_size=window_size, use_gpu=use_gpu, gpu_id=gpu_id
    )
    dapi_lap_var_map = compute_laplacian_variance_map(
        dapi,
        window_size=window_size,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        lap_sigma=lap_sigma,
    )
    del dapi

    # Boundary (channel 1)
    boundary_focus_map: NDArray[np.float32] | None = None
    boundary_mean_map: NDArray[np.float32] | None = None
    if boundary is not None:
        boundary_focus_map, boundary_mean_map = compute_ccfs_map(
            boundary, window_size=window_size, use_gpu=use_gpu, gpu_id=gpu_id
        )
        del boundary

    # IntRNA (channel 2)
    intrna_focus_map: NDArray[np.float32] | None = None
    intrna_mean_map: NDArray[np.float32] | None = None
    if intrna is not None:
        intrna_focus_map, intrna_mean_map = compute_ccfs_map(
            intrna, window_size=window_size, use_gpu=use_gpu, gpu_id=gpu_id
        )
        del intrna

    backend_label = "GPU" if use_gpu else "CPU"
    logging.info(
        f"  [TIMING] {backend_label} compute (single device, all channels): {time.perf_counter() - t_gpu:.1f}s"
    )

    return {
        "dapi_focus_map": dapi_focus_map,
        "dapi_mean_map": dapi_mean_map,
        "dapi_lap_var_map": dapi_lap_var_map,
        "boundary_focus_map": boundary_focus_map,
        "boundary_mean_map": boundary_mean_map,
        "intrna_focus_map": intrna_focus_map,
        "intrna_mean_map": intrna_mean_map,
    }


# ---------------------------------------------------------------------------
# Down-sampling to per-tile DataFrame
# ---------------------------------------------------------------------------


def _build_roi_grid(
    image_shape: tuple[int, int],
    roi_size: int,
    stride: int,
    tissue_mask: NDArray[np.generic] | None,
    downsample_factor: int = 8,
) -> dict[str, Any]:
    """Build the tile grid and compute tissue coverages (one-time setup).

    Returns a dict with keys: ``x1_arr``, ``x2_arr``, ``y1_arr``, ``y2_arr``,
    ``cx``, ``cy``, ``n_rois``, ``height``, ``width``, ``tissue_coverages``,
    ``is_boundary_roi``, ``roi_coords_arr``.
    """
    height, width = image_shape

    # Build ROI grid — vectorized
    x1_arr, x2_arr, y1_arr, y2_arr, cx, cy = compute_roi_grid(
        height, width, roi_size, stride
    )
    n_rois = len(x1_arr)
    roi_coords_arr = np.stack([x1_arr, x2_arr, y1_arr, y2_arr], axis=1)

    # Tissue coverage
    if tissue_mask is not None:
        binary_mask = (tissue_mask > 0).astype(np.float64)
        mask_h, mask_w = binary_mask.shape
        x1_ds = np.clip(x1_arr // downsample_factor, 0, mask_w)
        x2_ds = np.clip((x2_arr - 1) // downsample_factor + 1, 0, mask_w)
        y1_ds = np.clip(y1_arr // downsample_factor, 0, mask_h)
        y2_ds = np.clip((y2_arr - 1) // downsample_factor + 1, 0, mask_h)

        integral = np.zeros((mask_h + 1, mask_w + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(binary_mask, axis=0), axis=1)
        block_sums = (
            integral[y2_ds, x2_ds]
            - integral[y1_ds, x2_ds]
            - integral[y2_ds, x1_ds]
            + integral[y1_ds, x1_ds]
        )
        block_w = x2_ds - x1_ds
        block_h = y2_ds - y1_ds
        total_pixels_arr = (block_w * block_h).astype(np.float64)
        total_pixels_arr[total_pixels_arr == 0] = 1.0
        tissue_coverages = block_sums / total_pixels_arr
    else:
        tissue_coverages = np.ones(n_rois, dtype=np.float64)

    half_win = roi_size // 2
    is_boundary_roi = (
        (cx < half_win)
        | (cy < half_win)
        | (cx >= width - half_win)
        | (cy >= height - half_win)
    )

    return {
        "x1_arr": x1_arr,
        "x2_arr": x2_arr,
        "y1_arr": y1_arr,
        "y2_arr": y2_arr,
        "cx": cx,
        "cy": cy,
        "n_rois": n_rois,
        "height": height,
        "width": width,
        "tissue_coverages": tissue_coverages,
        "is_boundary_roi": is_boundary_roi,
        "roi_coords_arr": roi_coords_arr,
    }


def compute_roi_grid(
    height: int,
    width: int,
    roi_size: int = 35,
    stride: int | None = None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """The ROI grid and its centre pixels: ``(x1, x2, y1, y2, cx, cy)``.

    Shared by :func:`downsample_maps_to_roi_dataframe` and the streaming path,
    which needs the centres up front to build :class:`CentrePixelSampler`. The two
    must agree exactly — the sampler fills a positional array that the DataFrame
    then labels with these coordinates, so any drift mislabels every ROI silently
    rather than raising.

    Partial tiles at the right/bottom edge are kept when at least half a tile
    remains, and their centre is the midpoint of the *clipped* extent, not
    ``x1 + roi_size // 2``.
    """
    if stride is None:
        stride = roi_size
    yy, xx = np.meshgrid(
        np.arange(0, height, stride), np.arange(0, width, stride), indexing="ij"
    )
    x1_all, y1_all = xx.ravel(), yy.ravel()
    x2_all = np.minimum(x1_all + roi_size, width)
    y2_all = np.minimum(y1_all + roi_size, height)

    valid = (x2_all - x1_all >= roi_size // 2) & (y2_all - y1_all >= roi_size // 2)
    x1_arr, x2_arr = x1_all[valid], x2_all[valid]
    y1_arr, y2_arr = y1_all[valid], y2_all[valid]
    if len(x1_arr) == 0:
        raise ValueError(
            f"No tiles generated. Image: {height}x{width}, "
            f"roi_size={roi_size}, stride={stride}."
        )
    return (
        x1_arr,
        x2_arr,
        y1_arr,
        y2_arr,
        (x1_arr + x2_arr) // 2,
        (y1_arr + y2_arr) // 2,
    )


def downsample_maps_to_roi_dataframe(
    focus_maps: dict[str, NDArray[np.float32] | None],
    tissue_mask: NDArray[np.generic] | None,
    roi_size: int = 35,
    stride: int | None = None,
    downsample_factor: int = 8,
    min_tissue_coverage: float = 0.0,
    image_shape: tuple[int, int] | None = None,
    presampled: dict[str, NDArray[np.float64]] | None = None,
) -> pd.DataFrame:
    """Down-sample pixel-level focus maps to per-tile scalars.

    When *presampled* is given, the centre-pixel sampling step is skipped and those
    per-ROI arrays are used instead. That is how the streaming path reuses every
    column, threshold and normalisation below without ever assembling a
    full-resolution plane: ``CentrePixelSampler`` produces exactly the arrays this
    function would have read out of ``map[cy, cx]``. Keys are the map names
    (``dapi_focus_map``, ``boundary_mean_map``, ...); *focus_maps* may then be an
    empty dict.

    The output DataFrame has **exactly** the same columns as the legacy
    ``calculate_roi_focusscore()`` function, ensuring full backward
    compatibility.

    For each tile the focus score is obtained by sampling the **centre pixel**
    of the corresponding pixel-level map. Because ``uniform_filter`` at pixel
    ``(cy, cx)`` computes statistics over the surrounding ``window_size x
    window_size`` window, the centre pixel of a ``roi_size x roi_size`` tile
    yields the exact block-level statistic (for non-edge tiles). Edge tiles may
    differ slightly because ``uniform_filter`` uses ``mode='reflect'`` padding
    while the legacy code clips to the image boundary.

    Args:
        focus_maps: Dictionary returned by :func:`compute_all_focus_maps`.
            Required key: ``dapi_focus_map``. All other keys may be ``None``.
        tissue_mask: Labelled tissue mask at down-sampled resolution (e.g.
            level 3). Pass ``None`` to skip tissue filtering (all tiles get
            ``tissue_coverage=1.0``).
        roi_size: Side length of each square tile in pixels.
        stride: Grid spacing in pixels. Defaults to *roi_size* (non-
            overlapping grid).
        downsample_factor: Scale factor between full-resolution coordinates
            and *tissue_mask* coordinates (default 8 for level 3).
        min_tissue_coverage: Minimum tissue fraction for a tile to be
            included. ``0.0`` means any tissue overlap is sufficient. This
            parameter is recorded but **not** used for filtering — all tiles
            are returned so the caller can filter as needed.
        image_shape: ``(height, width)`` of the full-resolution image. If
            ``None`` the shape is inferred from ``dapi_focus_map``.

    Returns:
        :class:`~pandas.DataFrame` with columns identical to the legacy
        ``calculate_roi_focusscore()`` output.

    Raises:
        ValueError: If ``dapi_focus_map`` is missing from *focus_maps* or the
            Tile grid is empty.
    """
    # ------------------------------------------------------------------
    # Unpack maps
    # ------------------------------------------------------------------
    # In streaming mode there are no planes to inspect, so channel presence — which
    # decides both the sampling below and the column set — comes from the same dict
    # the samples do.
    source: dict[str, object] = presampled if presampled is not None else focus_maps  # type: ignore[assignment]
    if source.get("dapi_focus_map") is None:
        which = "presampled" if presampled is not None else "focus_maps"
        raise ValueError(f"{which} must contain 'dapi_focus_map'.")
    has_boundary = source.get("boundary_focus_map") is not None
    has_intrna = source.get("intrna_focus_map") is not None

    dapi_focus_map = focus_maps.get("dapi_focus_map")
    dapi_mean_map = focus_maps.get("dapi_mean_map")
    dapi_lap_var_map = focus_maps.get("dapi_lap_var_map")
    boundary_focus_map = focus_maps.get("boundary_focus_map")
    boundary_mean_map = focus_maps.get("boundary_mean_map")
    intrna_focus_map = focus_maps.get("intrna_focus_map")
    intrna_mean_map = focus_maps.get("intrna_mean_map")

    # ------------------------------------------------------------------
    # Image shape
    # ------------------------------------------------------------------
    if image_shape is not None:
        height, width = image_shape
    elif dapi_focus_map is not None:
        height, width = dapi_focus_map.shape
    else:
        raise ValueError("image_shape is required when passing presampled arrays.")

    if stride is None:
        stride = roi_size

    # ------------------------------------------------------------------
    # Build ROI grid — vectorized (identical logic to legacy code)
    # ------------------------------------------------------------------
    t_grid = time.perf_counter()

    x_starts = np.arange(0, width, stride)
    y_starts = np.arange(0, height, stride)
    yy, xx = np.meshgrid(y_starts, x_starts, indexing="ij")
    x1_all = xx.ravel()
    y1_all = yy.ravel()
    x2_all = np.minimum(x1_all + roi_size, width)
    y2_all = np.minimum(y1_all + roi_size, height)

    # Filter out small edge ROIs (same threshold as legacy code)
    valid = (x2_all - x1_all >= roi_size // 2) & (y2_all - y1_all >= roi_size // 2)
    x1_arr = x1_all[valid]
    x2_arr = x2_all[valid]
    y1_arr = y1_all[valid]
    y2_arr = y2_all[valid]

    n_rois = len(x1_arr)
    if n_rois == 0:
        raise ValueError(
            f"No tiles generated from grid. "
            f"Image: {height}x{width}, roi_size={roi_size}, stride={stride}."
        )

    # Keep a structured array for backward-compatible DataFrame columns
    # Columns: x1, x2, y1, y2
    roi_coords_arr = np.stack([x1_arr, x2_arr, y1_arr, y2_arr], axis=1)

    logging.info(
        f"  [TIMING] Tile grid generation ({n_rois} tiles): {time.perf_counter() - t_grid:.1f}s"
    )

    # ------------------------------------------------------------------
    # Tissue coverage — vectorized via block sums
    # ------------------------------------------------------------------
    t_tissue = time.perf_counter()

    tissue_coverages_arr: NDArray[np.float64]
    if tissue_mask is not None:
        # Convert tissue mask to binary
        binary_mask = (tissue_mask > 0).astype(np.float64)
        mask_h, mask_w = binary_mask.shape

        # Compute downsampled ROI coordinates (vectorized)
        x1_ds = x1_arr // downsample_factor
        x2_ds = (x2_arr - 1) // downsample_factor + 1
        y1_ds = y1_arr // downsample_factor
        y2_ds = (y2_arr - 1) // downsample_factor + 1

        # Clip to mask boundaries
        x1_ds = np.clip(x1_ds, 0, mask_w)
        x2_ds = np.clip(x2_ds, 0, mask_w)
        y1_ds = np.clip(y1_ds, 0, mask_h)
        y2_ds = np.clip(y2_ds, 0, mask_h)

        # Check if all ROI blocks have uniform downsampled size
        block_w = x2_ds - x1_ds
        block_h = y2_ds - y1_ds
        uniform_w = int(block_w[0]) if len(block_w) > 0 else 0
        uniform_h = int(block_h[0]) if len(block_h) > 0 else 0
        all_uniform = bool(
            np.all(block_w == uniform_w) and np.all(block_h == uniform_h)
        )

        if all_uniform and uniform_w > 0 and uniform_h > 0:
            # Fast path: use a 2D integral image (summed-area table)
            # to compute block sums in O(1) per tile
            integral = np.zeros((mask_h + 1, mask_w + 1), dtype=np.float64)
            integral[1:, 1:] = np.cumsum(np.cumsum(binary_mask, axis=0), axis=1)
            # Block sum = integral[y2,x2] - integral[y1,x2] - integral[y2,x1] + integral[y1,x1]
            block_sums = (
                integral[y2_ds, x2_ds]
                - integral[y1_ds, x2_ds]
                - integral[y2_ds, x1_ds]
                + integral[y1_ds, x1_ds]
            )
            total_pixels = uniform_w * uniform_h
            tissue_coverages_arr = block_sums / total_pixels
        else:
            # Fallback: still use integral image but handle variable block sizes
            integral = np.zeros((mask_h + 1, mask_w + 1), dtype=np.float64)
            integral[1:, 1:] = np.cumsum(np.cumsum(binary_mask, axis=0), axis=1)
            block_sums = (
                integral[y2_ds, x2_ds]
                - integral[y1_ds, x2_ds]
                - integral[y2_ds, x1_ds]
                + integral[y1_ds, x1_ds]
            )
            total_pixels_arr = (block_w * block_h).astype(np.float64)
            # Avoid division by zero
            total_pixels_arr[total_pixels_arr == 0] = 1.0
            tissue_coverages_arr = block_sums / total_pixels_arr
    else:
        tissue_coverages_arr = np.ones(n_rois, dtype=np.float64)

    logging.info(
        f"  [TIMING] Tissue coverage computation: {time.perf_counter() - t_tissue:.1f}s"
    )

    # ------------------------------------------------------------------
    # Sample centre pixel for each ROI — vectorized fancy indexing
    # ------------------------------------------------------------------
    t_sample = time.perf_counter()

    # Compute centre coordinates for all ROIs at once
    cx = (x1_arr + x2_arr) // 2
    cy = (y1_arr + y2_arr) // 2

    if presampled is not None:
        # Streaming path: the tile pass already read one pixel per ROI.
        def _sampled(name: str) -> NDArray[np.float64] | None:
            arr = presampled.get(name)
            return None if arr is None else np.asarray(arr, dtype=np.float64)

        dapi_focus_scores = _sampled("dapi_focus_map")
        if dapi_focus_scores is None:
            raise ValueError("presampled must contain 'dapi_focus_map'")
        _zeros = np.zeros(n_rois, dtype=np.float64)
        dapi_intensities = _sampled("dapi_mean_map")
        if dapi_intensities is None:
            dapi_intensities = _zeros
        dapi_lap_vars = _sampled("dapi_lap_var_map")
        if dapi_lap_vars is None:
            dapi_lap_vars = _zeros
        boundary_focus_scores = _sampled("boundary_focus_map") if has_boundary else None
        boundary_intensities = _sampled("boundary_mean_map") if has_boundary else None
        intrna_focus_scores = _sampled("intrna_focus_map") if has_intrna else None
        intrna_intensities = _sampled("intrna_mean_map") if has_intrna else None
        logging.info(
            f"  [TIMING] Centre-pixel sampling ({n_rois} tiles): from the tile pass"
        )
    else:
        # DAPI channels (always present)
        dapi_focus_scores = dapi_focus_map[cy, cx].astype(np.float64)
        dapi_intensities = (
            dapi_mean_map[cy, cx].astype(np.float64)
            if dapi_mean_map is not None
            else np.zeros(n_rois, dtype=np.float64)
        )
        dapi_lap_vars = (
            dapi_lap_var_map[cy, cx].astype(np.float64)
            if dapi_lap_var_map is not None
            else np.zeros(n_rois, dtype=np.float64)
        )

        # Boundary channel
        if has_boundary:
            boundary_focus_scores = boundary_focus_map[cy, cx].astype(np.float64)  # type: ignore[index]
            boundary_intensities = boundary_mean_map[cy, cx].astype(np.float64)  # type: ignore[index]
        else:
            boundary_focus_scores = None
            boundary_intensities = None

        # IntRNA channel
        if has_intrna:
            intrna_focus_scores = intrna_focus_map[cy, cx].astype(np.float64)  # type: ignore[index]
            intrna_intensities = intrna_mean_map[cy, cx].astype(np.float64)  # type: ignore[index]
        else:
            intrna_focus_scores = None
            intrna_intensities = None

        logging.info(
            f"  [TIMING] Centre-pixel sampling ({n_rois} tiles): "
            f"{time.perf_counter() - t_sample:.1f}s"
        )

    # ------------------------------------------------------------------
    # Normalize focus scores with RobustScaler
    # ------------------------------------------------------------------
    t_scaler = time.perf_counter()
    scaler_dapi = RobustScaler()
    dapi_focus_scores_norm = scaler_dapi.fit_transform(
        dapi_focus_scores.reshape(-1, 1)
    ).flatten()

    if has_boundary:
        scaler_boundary = RobustScaler()
        boundary_focus_scores_norm: NDArray[np.float64] | None = (
            scaler_boundary.fit_transform(
                boundary_focus_scores.reshape(-1, 1)  # type: ignore[union-attr]
            ).flatten()
        )
    else:
        boundary_focus_scores_norm = None

    if has_intrna:
        scaler_intrna = RobustScaler()
        intrna_focus_scores_norm: NDArray[np.float64] | None = (
            scaler_intrna.fit_transform(
                intrna_focus_scores.reshape(-1, 1)  # type: ignore[union-attr]
            ).flatten()
        )
    else:
        intrna_focus_scores_norm = None

    logging.info(
        f"  [TIMING] RobustScaler normalization: {time.perf_counter() - t_scaler:.1f}s"
    )

    # ------------------------------------------------------------------
    # Mark boundary tiles (centre within roi_size//2 of image edge)
    # ------------------------------------------------------------------
    half_win = roi_size // 2
    is_boundary_roi = (
        (cx < half_win)
        | (cy < half_win)
        | (cx >= width - half_win)
        | (cy >= height - half_win)
    )

    # ------------------------------------------------------------------
    # Build output DataFrame (column order matches legacy code)
    # ------------------------------------------------------------------
    df_data: dict[str, Any] = {
        "roi_id": np.arange(n_rois),
        "x1": roi_coords_arr[:, 0],
        "x2": roi_coords_arr[:, 1],
        "y1": roi_coords_arr[:, 2],
        "y2": roi_coords_arr[:, 3],
        # DAPI focus scores (duplicated for backward compatibility)
        "focus_score": dapi_focus_scores,
        "focus_score_norm": dapi_focus_scores_norm,
        "dapi_focus_score": dapi_focus_scores,
        "dapi_focus_score_norm": dapi_focus_scores_norm,
        # Laplacian variance (DAPI)
        "dapi_lap_var": dapi_lap_vars,
        # Intensities
        "dapi_intensity": dapi_intensities,
        "raw_intensity": dapi_intensities.copy(),  # backward compatibility
        # Boundary channel
        "boundary_focus_score": (boundary_focus_scores if has_boundary else np.nan),
        "boundary_focus_score_norm": (
            boundary_focus_scores_norm if has_boundary else np.nan
        ),
        "boundary_intensity": (boundary_intensities if has_boundary else np.nan),
        # IntRNA channel
        "intrna_focus_score": (intrna_focus_scores if has_intrna else np.nan),
        "intrna_focus_score_norm": (intrna_focus_scores_norm if has_intrna else np.nan),
        "intrna_intensity": (intrna_intensities if has_intrna else np.nan),
        # Tissue coverage
        "tissue_coverage": tissue_coverages_arr,
        "overlaps_tissue": tissue_coverages_arr > 0.0,
        # Boundary flag: True if ROI centre is near image edge
        "is_boundary_roi": is_boundary_roi,
    }

    return pd.DataFrame(df_data)


def calculate_roi_focusscore_without_laplace(
    xoa_morphology_files,
    roi_size=35,
    stride=None,
    tissue_filter=True,
    min_tissue_coverage=0.0,
    downsample_factor=8,
):
    """
    Calculate cell-independent tile-based focus scores using a regular grid.

    Creates a regular lattice/grid of tiles across the entire slide and calculates
    focus scores for each tile independently of cell locations. Calculates focus scores
    for all available channels (DAPI, Boundary, IntRNA).

    Parameters:
    -----------
    xoa_morphology_files : list
        List of paths to morphology image files
    roi_size : int, optional
        Size of square tile in pixels (default: 35)
    stride : int, optional
        Grid spacing in pixels. If None, uses non-overlapping grid (stride = roi_size)
    tissue_filter : bool, optional
        Enable tissue region filtering (default: True)
    min_tissue_coverage : float, optional
        Minimum fraction of tile that must be tissue (default: 0.0, i.e., any tissue overlap)
    downsample_factor : int, optional
        Downsampling factor for tissue mask (default: 8, for level 3)

    Returns:
    --------
    pandas DataFrame
        DataFrame with columns:
        - roi_id: Unique identifier
        - x1, x2, y1, y2: Tile boundaries (full resolution)
        - focus_score: Raw DAPI focus score (std² / mean) [backward compatibility]
        - focus_score_norm: Normalized DAPI focus score [backward compatibility]
        - dapi_focus_score, dapi_focus_score_norm: DAPI focus scores
        - boundary_focus_score, boundary_focus_score_norm: Boundary focus scores (if available)
        - intrna_focus_score, intrna_focus_score_norm: IntRNA focus scores (if available)
        - dapi_intensity: Mean DAPI intensity per tile
        - boundary_intensity: Mean Boundary intensity per tile (if available)
        - intrna_intensity: Mean IntRNA intensity per tile (if available)
        - raw_intensity: Mean DAPI intensity per tile [backward compatibility]
        - tissue_coverage: Fraction of tile that is tissue (0.0-1.0)
    """

    # Set stride (non-overlapping if not specified)
    if stride is None:
        stride = roi_size

    # Load channels from either multi-channel stack or split single-channel files.
    dapi_image, boundary_image, intrna_image = _load_morphology_channels(
        xoa_morphology_files, level=0
    )
    n_channels = 1 + int(boundary_image is not None) + int(intrna_image is not None)

    height, width = dapi_image.shape

    # Check which channels are available
    has_boundary = n_channels > 1
    has_intrna = n_channels > 2

    if has_boundary:
        logging.info(
            f"  Found {n_channels} channels: DAPI, Boundary"
            + (", IntRNA" if has_intrna else "")
        )
    else:
        logging.info(f"  Found {n_channels} channel(s): DAPI only")

    # Generate tissue mask if filtering enabled
    tissue_mask = None
    if tissue_filter:
        # Load downsampled DAPI for tissue mask generation
        small0_ds, _, _ = _load_morphology_channels(xoa_morphology_files, level=3)
        # Generate tissue mask (simplified version - just need whole_sample)
        # Tissue mask via the shared helper (hysteresis + degeneracy guard + `> 0`) —
        # single source of truth, see compute_tissue_mask. Fixed 2026-06-23 (was
        # the percentile-60 + `test_mask > 1` defect). This path builds
        # tissue_coverage, so the fix here is what corrects the §5.5 mask gate
        # and the GMM tissue selection on sparse/dim slides.
        tissue_mask = compute_tissue_mask(small0_ds)[0]
        del small0_ds

    # Create grid of ROI coordinates
    n_x = (width // stride) + (1 if width % stride > 0 else 0)
    n_y = (height // stride) + (1 if height % stride > 0 else 0)

    roi_coords = []
    for y_idx in range(n_y):
        for x_idx in range(n_x):
            x1 = x_idx * stride
            x2 = min(x1 + roi_size, width)
            y1 = y_idx * stride
            y2 = min(y1 + roi_size, height)

            # Skip if ROI is too small (at edges)
            if (x2 - x1) < roi_size // 2 or (y2 - y1) < roi_size // 2:
                continue

            roi_coords.append((x1, x2, y1, y2))

    # Calculate tissue coverage for ALL tiles (no filtering)
    # This ensures we always have tiles to process, even if tissue detection fails
    tissue_coverages = []
    if tissue_filter and tissue_mask is not None:
        for x1, x2, y1, y2 in roi_coords:
            # Scale coordinates to downsampled space
            x1_ds = x1 // downsample_factor
            x2_ds = (x2 - 1) // downsample_factor + 1
            y1_ds = y1 // downsample_factor
            y2_ds = (y2 - 1) // downsample_factor + 1

            # Clip to mask boundaries
            x1_ds = max(0, x1_ds)
            x2_ds = min(tissue_mask.shape[1], x2_ds)
            y1_ds = max(0, y1_ds)
            y2_ds = min(tissue_mask.shape[0], y2_ds)

            # Calculate tissue coverage
            roi_mask = tissue_mask[y1_ds:y2_ds, x1_ds:x2_ds]
            tissue_pixels = np.sum(roi_mask > 0)
            total_pixels = roi_mask.size
            coverage = tissue_pixels / total_pixels if total_pixels > 0 else 0.0
            tissue_coverages.append(coverage)
    else:
        # If tissue filtering is disabled, assume all tiles have full tissue coverage
        tissue_coverages = [1.0] * len(roi_coords)

    # Calculate focus scores for all ROIs and all available channels
    n_rois = len(roi_coords)

    # Basic safety check (should never trigger now, but defensive programming)
    if n_rois == 0:
        raise ValueError(
            f"ERROR: No tiles generated from grid!\n"
            f"  - Image dimensions: {height}x{width} (full resolution)\n"
            f"  - Tile size: {roi_size}px, stride: {stride}px\n"
            f"  - This should not happen. Please check image dimensions and tile parameters.\n"
        )

    # Initialize arrays for all channels
    dapi_focus_scores = np.empty(n_rois, dtype=np.float64)
    dapi_intensities = np.empty(n_rois, dtype=np.float64)

    boundary_focus_scores = np.empty(n_rois, dtype=np.float64) if has_boundary else None
    boundary_intensities = np.empty(n_rois, dtype=np.float64) if has_boundary else None

    intrna_focus_scores = np.empty(n_rois, dtype=np.float64) if has_intrna else None
    intrna_intensities = np.empty(n_rois, dtype=np.float64) if has_intrna else None

    # Calculate focus scores for each ROI
    for i, (x1, x2, y1, y2) in enumerate(roi_coords):
        # DAPI channel
        roi_dapi = dapi_image[y1:y2, x1:x2]
        mean_dapi = np.mean(roi_dapi)
        std_dapi = np.std(roi_dapi)
        dapi_focus_scores[i] = (
            (std_dapi * std_dapi) / mean_dapi if mean_dapi > 0 else 0.0
        )
        dapi_intensities[i] = mean_dapi

        # Boundary channel (if available)
        if has_boundary:
            roi_boundary = boundary_image[y1:y2, x1:x2]
            mean_boundary = np.mean(roi_boundary)
            std_boundary = np.std(roi_boundary)
            boundary_focus_scores[i] = (
                (std_boundary * std_boundary) / mean_boundary
                if mean_boundary > 0
                else 0.0
            )
            boundary_intensities[i] = mean_boundary

        # IntRNA channel (if available)
        if has_intrna:
            roi_intrna = intrna_image[y1:y2, x1:x2]
            mean_intrna = np.mean(roi_intrna)
            std_intrna = np.std(roi_intrna)
            intrna_focus_scores[i] = (
                (std_intrna * std_intrna) / mean_intrna if mean_intrna > 0 else 0.0
            )
            intrna_intensities[i] = mean_intrna

    # Normalize focus scores for each channel separately
    # Safety check: Ensure we have data before normalizing
    if len(dapi_focus_scores) == 0:
        raise ValueError(
            f"ERROR: Cannot normalize focus scores - empty array detected!\n"
            f"  - Number of tiles: {n_rois}\n"
            f"  - This should have been caught earlier. Please report this issue.\n"
        )

    scaler_dapi = RobustScaler()
    dapi_focus_scores_norm = scaler_dapi.fit_transform(
        dapi_focus_scores.reshape(-1, 1)
    ).flatten()

    if has_boundary:
        if len(boundary_focus_scores) == 0:
            raise ValueError(
                "ERROR: Cannot normalize boundary focus scores - empty array detected!"
            )
        scaler_boundary = RobustScaler()
        boundary_focus_scores_norm = scaler_boundary.fit_transform(
            boundary_focus_scores.reshape(-1, 1)
        ).flatten()
    else:
        boundary_focus_scores_norm = None

    if has_intrna:
        if len(intrna_focus_scores) == 0:
            raise ValueError(
                "ERROR: Cannot normalize IntRNA focus scores - empty array detected!"
            )
        scaler_intrna = RobustScaler()
        intrna_focus_scores_norm = scaler_intrna.fit_transform(
            intrna_focus_scores.reshape(-1, 1)
        ).flatten()
    else:
        intrna_focus_scores_norm = None

    # Create output DataFrame
    # roi_id: Simple integer IDs (0, 1, 2, ...) for easy indexing
    df_data = {
        "roi_id": range(n_rois),
        "x1": [coords[0] for coords in roi_coords],
        "x2": [coords[1] for coords in roi_coords],
        "y1": [coords[2] for coords in roi_coords],
        "y2": [coords[3] for coords in roi_coords],
        # DAPI focus scores (also kept as focus_score/focus_score_norm for backward compatibility)
        "focus_score": dapi_focus_scores,
        "focus_score_norm": dapi_focus_scores_norm,
        "dapi_focus_score": dapi_focus_scores,
        "dapi_focus_score_norm": dapi_focus_scores_norm,
        "dapi_intensity": dapi_intensities,
        "raw_intensity": dapi_intensities,  # Backward compatibility
        "tissue_coverage": tissue_coverages if tissue_filter else [1.0] * n_rois,
        "overlaps_tissue": [
            coverage > 0.0
            for coverage in (tissue_coverages if tissue_filter else [1.0] * n_rois)
        ],
    }

    # Add Boundary channel data if available
    if has_boundary:
        df_data["boundary_focus_score"] = boundary_focus_scores
        df_data["boundary_focus_score_norm"] = boundary_focus_scores_norm
        df_data["boundary_intensity"] = boundary_intensities
    else:
        df_data["boundary_focus_score"] = np.nan
        df_data["boundary_focus_score_norm"] = np.nan
        df_data["boundary_intensity"] = np.nan

    # Add IntRNA channel data if available
    if has_intrna:
        df_data["intrna_focus_score"] = intrna_focus_scores
        df_data["intrna_focus_score_norm"] = intrna_focus_scores_norm
        df_data["intrna_intensity"] = intrna_intensities
    else:
        df_data["intrna_focus_score"] = np.nan
        df_data["intrna_focus_score_norm"] = np.nan
        df_data["intrna_intensity"] = np.nan

    df_grid_roi = pd.DataFrame(df_data)

    # Clean up
    del dapi_image
    if has_boundary:
        del boundary_image
    if has_intrna:
        del intrna_image

    return df_grid_roi


@dataclass
class StreamedTileResults:
    """The small reductions that replace the full-resolution pixel planes.

    Assembling the planes cost ~154 GB of scratch on a 5.5 gigapixel sample, and
    `mmap` over a FUSE/S3 work directory is pathological (see
    `docs/failures/2026-07-24_imageqc-mmap-over-fusion.md`). No downstream consumer
    needs a whole plane, so in streaming mode each tile is folded into these arrays
    and dropped. Everything here is O(n_ROIs), O(n_cells) or O(pixels / factor^2) —
    tens of MB, not tens of GB.

    Attributes:
        presampled: One centre pixel per ROI, keyed by map name
            (``dapi_focus_map``, ``boundary_mean_map``, ...). Feeds
            :func:`downsample_maps_to_roi_dataframe`'s ``presampled`` argument.
        roi_snr_db: Per-ROI Otsu SNR in dB, replacing
            ``snr_metrics.compute_image_snr_from_pixel_maps``' loop over the DAPI
            mean plane. Aligned with the ROI grid.
        focus_heatmap: DAPI focus map down-sampled by ``heatmap_factor``, replacing
            ``downscale_local_mean`` on the assembled plane in Figure 5.
        nuclear_counts / nuclear_sums: Per-nucleus pixel counts and value sums over
            ``masks/0``, including the ``centroid_*_sum`` keys. Replaces the
            ``_labeled_sums_chunked`` pass in :func:`calculate_ccfs_from_focus_maps`.
        cell_counts / cell_sums: The same over ``masks/1``, giving cell areas and
            per-cell boundary/IntRNA means.
    """

    presampled: dict[str, NDArray[np.float64]]
    roi_snr_db: NDArray[np.float64] | None = None
    focus_heatmap: NDArray[np.float64] | None = None
    heatmap_factor: int = 8
    nuclear_counts: NDArray[np.int64] | None = None
    nuclear_sums: dict[str, NDArray[np.float64]] | None = None
    cell_counts: NDArray[np.int64] | None = None
    cell_sums: dict[str, NDArray[np.float64]] | None = None

    @property
    def has_per_cell(self) -> bool:
        return self.nuclear_counts is not None


def _stream_channels(
    lazy_channels,
    image_shape: tuple[int, int],
    roi_size: int,
    stride: int | None,
    gpu_ids: list[int],
    *,
    lap_sigma: float,
    gpu_mem_bytes: int | None,
    cell_masks_path: Path | None = None,
    heatmap_factor: int = 8,
) -> StreamedTileResults:
    """Run the tiled compute per channel, folding each tile into small reductions.

    The plane-based path assembles a full-resolution map per channel and then makes
    one pass over it per consumer. Here the consumers are attached to the tile
    dispatch instead, so a tile is reduced and dropped as soon as it is computed
    and no plane is ever allocated.

    Which consumers attach depends on the channel, because the downstream metrics
    do not use every channel the same way:

    * every channel needs one centre pixel per ROI, for the ROI DataFrame;
    * only DAPI feeds the Figure 5 heatmap, the Otsu SNR and the per-nucleus CCFS;
    * cell areas come from any single channel's pass over ``masks/1``, so they are
      taken from DAPI, which is always present, while boundary and IntRNA each
      contribute their own per-cell mean.

    The label planes stay lazy: ``LabeledSumAccumulator`` slices row blocks out of
    zarr per tile, so neither ``masks/0`` nor ``masks/1`` (22 GB each on a 5.5
    gigapixel sample) is materialised.
    """
    x1, x2, y1, y2, cx, cy = compute_roi_grid(
        image_shape[0], image_shape[1], roi_size, stride
    )
    logging.info(f"  Streaming mode: {len(cx):,} ROIs, no pixel planes materialised")

    nuclear_plane = cell_plane = None
    if cell_masks_path is not None:
        masks = open_zarr(cell_masks_path).get("masks")
        nuclear_plane = LazyLabelPlane(masks.get("0"))
        cell_plane = LazyLabelPlane(masks.get("1"))
        logging.info("  Per-cell CCFS will be reduced during the tile pass")

    presampled: dict[str, NDArray[np.float64]] = {}
    nuclear_counts: NDArray[np.int64] | None = None
    nuclear_sums: dict[str, NDArray[np.float64]] | None = None
    cell_counts: NDArray[np.int64] | None = None
    cell_sums: dict[str, NDArray[np.float64]] = {}
    roi_snr_db: NDArray[np.float64] | None = None
    focus_heatmap: NDArray[np.float64] | None = None

    for name, channel_data in (
        ("dapi", lazy_channels[0]),
        ("boundary", lazy_channels[1]),
        ("intrna", lazy_channels[2]),
    ):
        if channel_data is None:
            continue
        is_dapi = name == "dapi"

        # Built by a factory, because the fold runs concurrently on one set of
        # accumulators per slot and the parent merges them. Order must be identical
        # across sets -- merging pairs them positionally.
        def _make_consumers() -> tuple[list[Any], dict[str, Any]]:
            made: dict[str, Any] = {
                "sampler": CentrePixelSampler(
                    cy,
                    cx,
                    ["focus_map", "mean_map"] + (["lap_var_map"] if is_dapi else []),
                )
            }
            ordered: list[Any] = [made["sampler"]]
            if is_dapi:
                made["heat"] = BlockMeanAccumulator(
                    image_shape, factor=heatmap_factor, map_key="focus_map"
                )
                made["snr"] = RoiOtsuSnrAccumulator(
                    y1, y2, x1, x2, image_shape, map_key="mean_map"
                )
                ordered += [made["heat"], made["snr"]]
                if nuclear_plane is not None:
                    made["nuc"] = LabeledSumAccumulator(
                        nuclear_plane,
                        ["focus_map", "mean_map"],
                        include_coords=True,
                        # CCFS reads only labels > 0 from this one.
                        skip_background=True,
                    )
                    made["cells"] = LabeledSumAccumulator(cell_plane, [])
                    ordered += [made["nuc"], made["cells"]]
            elif cell_plane is not None:
                made["cells"] = LabeledSumAccumulator(cell_plane, ["mean_map"])
                ordered.append(made["cells"])
            return ordered, made

        consumers, named = _make_consumers()
        sampler = named["sampler"]
        heat = named.get("heat")
        snr = named.get("snr")
        nuc = named.get("nuc")
        cells = named.get("cells")

        t_ch = time.perf_counter()
        _compute_channel_maps_tiled(
            channel_data,
            image_shape,
            roi_size,
            gpu_ids,
            include_laplacian=is_dapi,
            lap_sigma=lap_sigma,
            gpu_mem_bytes=gpu_mem_bytes,
            plane_prefix=name,
            consumers=consumers,
        )
        logging.info(
            f"  [TIMING] {name} channel streamed: {time.perf_counter() - t_ch:.1f}s"
        )
        _log_mem(f"{name} channel streamed")

        # CentrePixelSampler keys are per-channel ("focus_map"); the ROI DataFrame
        # wants them qualified ("dapi_focus_map").
        for key, values in sampler.finalize().items():
            presampled[f"{name}_{key}"] = values

        if is_dapi:
            focus_heatmap = heat.finalize()
            roi_snr_db = snr.finalize()
            if nuc is not None:
                nuclear_counts, nuclear_sums = nuc.finalize()
                cell_counts, _ = cells.finalize()
        elif cells is not None:
            cell_sums[name] = cells.finalize()[1]["mean_map"]

    # No index alignment needed: every channel's pass covers the whole image, so each
    # per-label array grows to the same largest label. Asserted in
    # test_cell_arrays_share_one_label_index.

    return StreamedTileResults(
        presampled=presampled,
        roi_snr_db=roi_snr_db,
        focus_heatmap=focus_heatmap,
        heatmap_factor=heatmap_factor,
        nuclear_counts=nuclear_counts,
        nuclear_sums=nuclear_sums,
        cell_counts=cell_counts,
        cell_sums=cell_sums or None,
    )


def calculate_roi_focusscore(
    xoa_morphology_files,
    roi_size=35,
    stride=None,
    tissue_filter=True,
    min_tissue_coverage=0.0,
    downsample_factor=8,
    use_gpu=False,
    gpu_ids=None,
    return_pixel_maps=False,
    stream_tiles=False,
    cell_masks_path=None,
    heatmap_factor=8,
    lap_sigma: float = 1.0,
    small0_ds=None,
    tissue_mask=None,
):
    """
    Calculate cell-independent tile-based focus scores using a regular grid.

    Uses efficient whole-image convolution operations (uniform_filter + laplace)
    to produce per-pixel focus maps, then samples the centre pixel of each tile
    for backward-compatible per-tile scalars. GPU acceleration is supported via
    CuPy when ``use_gpu=True``. Multi-GPU parallelism is available via
    ``gpu_ids``.

    Parameters:
    -----------
    xoa_morphology_files : list
        List of paths to morphology image files
    roi_size : int, optional
        Size of square tile in pixels (default: 35)
    stride : int, optional
        Grid spacing in pixels. If None, uses non-overlapping grid (stride = roi_size)
    tissue_filter : bool, optional
        Enable tissue region filtering (default: True)
    min_tissue_coverage : float, optional
        Minimum fraction of tile that must be tissue (default: 0.0)
    downsample_factor : int, optional
        Downsampling factor for tissue mask (default: 8, for level 3)
    use_gpu : bool, optional
        Use CuPy GPU backend for convolutions (default: False)
    gpu_ids : list[int] or None, optional
        List of CUDA device IDs for multi-GPU parallelism. Channels are
        distributed across GPUs. If None and use_gpu=True, GPUs are
        auto-detected (falling back to device 0).
    return_pixel_maps : bool, optional
        If True, also return the pixel-level focus map dict (default: False)
    small0_ds : numpy.ndarray or None, optional
        Pre-loaded level-3 DAPI plane. When provided (with ``tissue_mask``), the
        redundant level-3 decode is skipped. ``main`` already holds this array.
    tissue_mask : numpy.ndarray or None, optional
        Pre-computed labelled tissue mask (``compute_tissue_mask(small0_ds)[0]``).
        When provided, the mask recompute is skipped -- ``main`` already produced
        the identical array via ``generate_tissue_mask`` (same ``small0``, same
        ``compute_tissue_mask`` default ``min_size_hole=1500``), so the result is
        bit-identical. Ignored when ``tissue_filter`` is False.

    Returns:
    --------
    pandas DataFrame, or ``(DataFrame, focus_maps, streamed)`` when
    *return_pixel_maps* is set. Exactly one of the last two is not ``None``:
    ``focus_maps`` on the plane-based path, a :class:`StreamedTileResults` when
    *stream_tiles* folded each tile into small reductions instead.
        DataFrame with columns:
        - roi_id: Unique identifier
        - x1, x2, y1, y2: Tile boundaries (full resolution)
        - focus_score: Raw DAPI focus score (std^2 / mean) [backward compat]
        - focus_score_norm: Normalized DAPI focus score [backward compat]
        - dapi_focus_score, dapi_focus_score_norm: DAPI focus scores
        - dapi_lap_var: Laplacian variance (DAPI) per tile
        - boundary_focus_score, boundary_focus_score_norm: Boundary (if avail)
        - intrna_focus_score, intrna_focus_score_norm: IntRNA (if available)
        - dapi_intensity, boundary_intensity, intrna_intensity: Mean per tile
        - raw_intensity: Mean DAPI intensity per tile [backward compat]
        - tissue_coverage: Fraction of tile that is tissue (0.0-1.0)
        - overlaps_tissue: Boolean, tile overlaps any tissue (>0 coverage)
    """
    if stride is None:
        stride = roi_size

    # Channel detection + full-resolution loading happen per-path below: the
    # GPU path opens lazy TIFF wrappers (no full-res pixels in host RAM) and
    # computes in memory-bounded tiles; the CPU path eager-loads numpy arrays.
    # This avoids materialising the whole image (tens of GB on large samples)
    # before the GPU branch.

    # Generate tissue mask if filtering enabled. `main` already loads the level-3
    # DAPI plane and computes this exact mask (compute_tissue_mask(small0)[0], the
    # `whole_sample` output of generate_tissue_mask) — when it threads `small0_ds`
    # and `tissue_mask` in we skip a redundant level-3 decode + mask recompute
    # (~20-35s on a 5.5 GP sample). Bit-identical: same small0, same
    # compute_tissue_mask default min_size_hole=1500.
    if tissue_filter and tissue_mask is None:
        if small0_ds is None:
            small0_ds, _, _ = _load_morphology_channels(xoa_morphology_files, level=3)
        # Tissue mask via the shared helper (hysteresis + degeneracy guard + `> 0`) —
        # single source of truth, see compute_tissue_mask. Fixed 2026-06-23 (was
        # the percentile-60 + `test_mask > 1` defect). This path builds
        # tissue_coverage, so the fix here is what corrects the §5.5 mask gate
        # and the GMM tissue selection on sparse/dim slides.
        tissue_mask = compute_tissue_mask(small0_ds)[0]
        del small0_ds
    elif not tissue_filter:
        tissue_mask = None

    # Resolve GPU configuration
    _use_gpu = use_gpu
    if gpu_ids is not None and len(gpu_ids) > 0:
        _use_gpu = True

    # ------------------------------------------------------------------
    # GPU path: memory-bounded tiled compute with lazy tile reads. Each
    # channel is read from disk and computed in tiles so neither a full-res
    # channel (tens of GB) nor its float64 intermediates land on the GPU or
    # in host RAM at once. Ports the tiled+lazy path from commit 5039b6a.
    # ------------------------------------------------------------------
    if _use_gpu:
        # Auto-detect GPUs if the caller enabled use_gpu without naming devices
        # (mem_info + tiled dispatch below index gpu_ids[0]).
        if not gpu_ids:
            gpu_ids = detect_gpu_ids() or [0]

        # Lazy TIFF page wrappers — no full-res pixel data loaded into RAM.
        lazy_channels, img_shape = _open_morphology_lazy(xoa_morphology_files, level=0)
        has_boundary = lazy_channels[1] is not None
        has_intrna = lazy_channels[2] is not None
        n_channels = 1 + int(has_boundary) + int(has_intrna)

        gpu_label = f"gpu_ids={gpu_ids}" if gpu_ids else f"gpu={_use_gpu}"
        logging.info(
            f"  Found {n_channels} channel(s); computing per-pixel focus maps "
            f"(window={roi_size}, {gpu_label}, tiled)..."
        )
        logging.info(
            f"  Image shape: {img_shape[0]}x{img_shape[1]}, {n_channels} channel(s)"
        )

        t_gpu = time.perf_counter()
        gpu_mem = cp.cuda.Device(gpu_ids[0]).mem_info[1]  # total VRAM per device
        logging.info(f"  GPU VRAM: {gpu_mem / 1024**3:.1f} GB per device")

        if stream_tiles:
            streamed = _stream_channels(
                lazy_channels,
                img_shape,
                roi_size,
                stride,
                gpu_ids,
                lap_sigma=lap_sigma,
                gpu_mem_bytes=gpu_mem,
                cell_masks_path=cell_masks_path,
                heatmap_factor=heatmap_factor,
            )
            logging.info(
                f"  [TIMING] GPU compute (streamed, {n_channels} ch): "
                f"{time.perf_counter() - t_gpu:.1f}s"
            )
            df_grid_roi = downsample_maps_to_roi_dataframe(
                {},
                tissue_mask=tissue_mask,
                roi_size=roi_size,
                stride=stride,
                downsample_factor=downsample_factor,
                min_tissue_coverage=min_tissue_coverage,
                image_shape=img_shape,
                presampled=streamed.presampled,
            )
            if return_pixel_maps:
                return df_grid_roi, None, streamed
            return df_grid_roi

        dapi_result = _compute_channel_maps_tiled(
            lazy_channels[0],
            img_shape,
            roi_size,
            gpu_ids,
            include_laplacian=True,
            lap_sigma=lap_sigma,
            gpu_mem_bytes=gpu_mem,
            plane_prefix="dapi",
        )
        boundary_result = None
        if has_boundary and lazy_channels[1] is not None:
            boundary_result = _compute_channel_maps_tiled(
                lazy_channels[1],
                img_shape,
                roi_size,
                gpu_ids,
                include_laplacian=False,
                lap_sigma=lap_sigma,
                gpu_mem_bytes=gpu_mem,
                plane_prefix="boundary",
            )
        intrna_result = None
        if has_intrna and lazy_channels[2] is not None:
            intrna_result = _compute_channel_maps_tiled(
                lazy_channels[2],
                img_shape,
                roi_size,
                gpu_ids,
                include_laplacian=False,
                lap_sigma=lap_sigma,
                gpu_mem_bytes=gpu_mem,
                plane_prefix="intrna",
            )
        logging.info(
            f"  [TIMING] GPU compute (tiled, {n_channels} ch): "
            f"{time.perf_counter() - t_gpu:.1f}s"
        )

        focus_maps = {
            "dapi_focus_map": dapi_result["focus_map"],
            "dapi_mean_map": dapi_result["mean_map"],
            "dapi_lap_var_map": dapi_result.get("lap_var_map"),
            "boundary_focus_map": (
                boundary_result["focus_map"] if boundary_result else None
            ),
            "boundary_mean_map": (
                boundary_result["mean_map"] if boundary_result else None
            ),
            "intrna_focus_map": intrna_result["focus_map"] if intrna_result else None,
            "intrna_mean_map": intrna_result["mean_map"] if intrna_result else None,
        }
        logging.info("  Per-pixel focus maps computed (tiled).")

        df_grid_roi = downsample_maps_to_roi_dataframe(
            focus_maps,
            tissue_mask=tissue_mask,
            roi_size=roi_size,
            stride=stride,
            downsample_factor=downsample_factor,
            min_tissue_coverage=min_tissue_coverage,
        )

        if return_pixel_maps:
            return df_grid_roi, focus_maps, None
        return df_grid_roi

    # ------------------------------------------------------------------
    # CPU path: incremental per-channel compute→downsample→free to limit
    # peak memory.  At most ~3 maps + convolution intermediates at once.
    # ------------------------------------------------------------------
    # Eager-load full-resolution channels (CPU path needs numpy arrays).
    dapi_image, boundary_image, intrna_image = _load_morphology_channels(
        xoa_morphology_files, level=0
    )
    has_boundary = boundary_image is not None
    has_intrna = intrna_image is not None

    logging.info(f"  Computing per-pixel focus maps (window={roi_size}, gpu=False)...")

    # Build ROI grid once (cheap — just coordinate arrays)
    image_shape = dapi_image.shape[:2]
    grid = _build_roi_grid(
        image_shape, roi_size, stride, tissue_mask, downsample_factor
    )
    cx, cy = grid["cx"], grid["cy"]
    n_rois = grid["n_rois"]
    logging.info(f"  Tile grid: {n_rois:,} tiles")

    # -- DAPI (always present) --
    dapi_focus_map, dapi_mean_map = compute_ccfs_map(
        dapi_image, window_size=roi_size, use_gpu=False, gpu_id=0
    )
    dapi_lap_var_map = compute_laplacian_variance_map(
        dapi_image,
        window_size=roi_size,
        use_gpu=False,
        gpu_id=0,
        lap_sigma=lap_sigma,
    )
    del dapi_image
    # Sample per-tile scalars
    dapi_focus_scores = dapi_focus_map[cy, cx].astype(np.float64)
    dapi_intensities = dapi_mean_map[cy, cx].astype(np.float64)
    dapi_lap_vars = dapi_lap_var_map[cy, cx].astype(np.float64)
    del dapi_lap_var_map  # not needed downstream
    logging.info("  DAPI maps computed and sampled.")

    # -- Boundary --
    boundary_focus_scores = None
    boundary_intensities = None
    boundary_mean_map = None
    if has_boundary:
        b_focus, boundary_mean_map = compute_ccfs_map(
            boundary_image, window_size=roi_size, use_gpu=False, gpu_id=0
        )
        del boundary_image
        boundary_focus_scores = b_focus[cy, cx].astype(np.float64)
        boundary_intensities = boundary_mean_map[cy, cx].astype(np.float64)
        del b_focus  # boundary_mean_map kept for CCFS
        logging.info("  Boundary maps computed and sampled.")
    else:
        del boundary_image

    # -- IntRNA --
    intrna_focus_scores = None
    intrna_intensities = None
    intrna_mean_map = None
    if has_intrna:
        i_focus, intrna_mean_map = compute_ccfs_map(
            intrna_image, window_size=roi_size, use_gpu=False, gpu_id=0
        )
        del intrna_image
        intrna_focus_scores = i_focus[cy, cx].astype(np.float64)
        intrna_intensities = intrna_mean_map[cy, cx].astype(np.float64)
        del i_focus  # intrna_mean_map kept for CCFS
        logging.info("  IntRNA maps computed and sampled.")
    else:
        del intrna_image

    logging.info("  Per-pixel focus maps computed (incremental CPU path).")

    # -- Normalize focus scores with RobustScaler --
    scaler_dapi = RobustScaler()
    dapi_focus_scores_norm = scaler_dapi.fit_transform(
        dapi_focus_scores.reshape(-1, 1)
    ).flatten()

    boundary_focus_scores_norm = None
    if boundary_focus_scores is not None:
        scaler_b = RobustScaler()
        boundary_focus_scores_norm = scaler_b.fit_transform(
            boundary_focus_scores.reshape(-1, 1)
        ).flatten()

    intrna_focus_scores_norm = None
    if intrna_focus_scores is not None:
        scaler_i = RobustScaler()
        intrna_focus_scores_norm = scaler_i.fit_transform(
            intrna_focus_scores.reshape(-1, 1)
        ).flatten()

    # -- Assemble DataFrame --
    rc = grid["roi_coords_arr"]
    df_data: dict[str, Any] = {
        "roi_id": np.arange(n_rois),
        "x1": rc[:, 0],
        "x2": rc[:, 1],
        "y1": rc[:, 2],
        "y2": rc[:, 3],
        "focus_score": dapi_focus_scores,
        "focus_score_norm": dapi_focus_scores_norm,
        "dapi_focus_score": dapi_focus_scores,
        "dapi_focus_score_norm": dapi_focus_scores_norm,
        "dapi_lap_var": dapi_lap_vars,
        "dapi_intensity": dapi_intensities,
        "raw_intensity": dapi_intensities.copy(),
        "boundary_focus_score": boundary_focus_scores if has_boundary else np.nan,
        "boundary_focus_score_norm": boundary_focus_scores_norm
        if has_boundary
        else np.nan,
        "boundary_intensity": boundary_intensities if has_boundary else np.nan,
        "intrna_focus_score": intrna_focus_scores if has_intrna else np.nan,
        "intrna_focus_score_norm": intrna_focus_scores_norm if has_intrna else np.nan,
        "intrna_intensity": intrna_intensities if has_intrna else np.nan,
        "tissue_coverage": grid["tissue_coverages"],
        "overlaps_tissue": grid["tissue_coverages"] > 0.0,
        "is_boundary_roi": grid["is_boundary_roi"],
    }
    df_grid_roi = pd.DataFrame(df_data)

    if return_pixel_maps:
        focus_maps = {
            "dapi_focus_map": dapi_focus_map,
            "dapi_mean_map": dapi_mean_map,
            "dapi_lap_var_map": None,  # already freed
            "boundary_focus_map": None,  # already freed
            "boundary_mean_map": boundary_mean_map,
            "intrna_focus_map": None,  # already freed
            "intrna_mean_map": intrna_mean_map,
        }
        return df_grid_roi, focus_maps, None
    return df_grid_roi


def calculate_roi_blur_threshold(
    df_grid_roi,
    intensity_threshold=ROI_INTENSITY_THRESHOLD,
    focus_percentile=ROI_FOCUS_SCORE_PERCENTILE,
):
    """
    Calculate tile blur detection threshold from raw focus scores.

    Option B: Exclude tiles with intensity < intensity_threshold, then calculate
    percentile from remaining tissue tiles. This focuses the threshold on tissue regions.

    The threshold is applied to RAW focus scores (not normalized). Classification
    is: blurred if (focus_score <= threshold) OR (intensity < intensity_threshold)

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        DataFrame with 'dapi_focus_score' (raw scores) and 'dapi_intensity'
    intensity_threshold : float, optional
        Minimum intensity to include tiles in percentile calculation (default: ROI_INTENSITY_THRESHOLD)
    focus_percentile : float, optional
        Percentile of raw scores to use as threshold (default: ROI_FOCUS_SCORE_PERCENTILE)

    Returns:
    --------
    float
        Threshold value in raw score units
    """
    # Get intensity column (handle both naming conventions)
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df_grid_roi.columns else "raw_intensity"
    )
    if intensity_col not in df_grid_roi.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    # Get focus score column (handle both naming conventions)
    focus_col = (
        "dapi_focus_score"
        if "dapi_focus_score" in df_grid_roi.columns
        else "focus_score"
    )
    if focus_col not in df_grid_roi.columns:
        raise ValueError(
            "Focus score column not found. Expected 'dapi_focus_score' or 'focus_score'"
        )

    # Exclude low-intensity tiles (background) before calculating percentile
    tissue_rois = df_grid_roi[df_grid_roi[intensity_col] >= intensity_threshold]

    if len(tissue_rois) == 0:
        # Fallback: if no tissue tiles, use all tiles
        logging.warning(
            f"  Warning: No tiles with intensity >= {intensity_threshold}, using all tiles for threshold calculation"
        )
        tissue_rois = df_grid_roi

    raw_scores = tissue_rois[focus_col].values

    # Calculate threshold on raw scores from tissue ROIs only
    threshold_raw = np.percentile(raw_scores, focus_percentile)

    return threshold_raw


def fit_focus_gmm(
    df_grid_roi,
    intensity_threshold: float = ROI_INTENSITY_THRESHOLD,
    focus_col_name: str = "dapi_focus_score",
    n_components: int = 2,
    random_state: int = 0,
):
    """
    Fit a Gaussian Mixture Model (GMM) to tile focus scores from tissue tiles
    (intensity >= intensity_threshold) in raw focus-score space.

    The model learns 2 components that approximately correspond to:
    - Lower-focus (blurred) tiles
    - Higher-focus (in-focus) tiles

    This function does NOT classify tiles directly; it only returns the fitted GMM
    and identifies which component is the "blur" component (lower mean).

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        DataFrame with at least:
        - focus_col_name (e.g. 'dapi_focus_score' or 'focus_score')
        - 'dapi_intensity' or 'raw_intensity'
    intensity_threshold : float, optional
        Minimum intensity to consider a tile as tissue (default: ROI_INTENSITY_THRESHOLD).
        Tiles below this are excluded from GMM training.
    focus_col_name : str, optional
        Name of the focus-score column to use (default: 'dapi_focus_score').
        If not present, 'focus_score' will be used.
    n_components : int, optional
        Number of Gaussian components for the GMM (default: 2).
    random_state : int, optional
        Random seed for reproducibility (default: 0).

    Returns
    -------
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM model on (log1p(focus_score)) of tissue tiles.
    blur_component_idx : int
        Index of the GMM component corresponding to blurred tiles (lower mean).
    """
    # Determine intensity column
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df_grid_roi.columns else "raw_intensity"
    )
    if intensity_col not in df_grid_roi.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    # Determine focus column
    if focus_col_name not in df_grid_roi.columns:
        # Fallback to generic 'focus_score'
        if "focus_score" not in df_grid_roi.columns:
            raise ValueError(
                f"Focus score column '{focus_col_name}' not found and 'focus_score' not present either"
            )
        focus_col_name = "focus_score"

    # Select tissue tiles for training
    tissue_rois = df_grid_roi[df_grid_roi[intensity_col] >= intensity_threshold].copy()
    if len(tissue_rois) == 0:
        raise ValueError(
            f"No tissue tiles found with intensity >= {intensity_threshold}. "
            f"Cannot fit GMM. Check intensity thresholds or image quality."
        )

    raw_scores = tissue_rois[focus_col_name].values.astype(np.float64)

    # Log-transform to reduce skewness (handle zeros safely)
    x_log = np.log1p(raw_scores).reshape(-1, 1)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=random_state
    )
    gmm.fit(x_log)

    # Identify which component corresponds to blurred ROIs:
    # assume lower mean log-focus = blur
    means = gmm.means_.flatten()
    blur_component_idx = int(np.argmin(means))

    logging.info("  GMM focus model fitted on tissue tiles")
    logging.info(f"    Number of tissue tiles used: {len(tissue_rois)}")
    logging.info(f"    Component means (log1p focus): {means}")
    logging.info(f"    Blur component index: {blur_component_idx} (lower mean)")

    return gmm, blur_component_idx


def classify_roi_blur_by_threshold(
    df_grid_roi,
    roi_threshold: float,
    intensity_threshold: float = ROI_INTENSITY_THRESHOLD,
    focus_col_name: str = "dapi_focus_score",
):
    """
    Percentile-threshold fallback blur classification.

    Used when the 1D GMM fit fails (e.g. too few tissue tiles on a very dim
    sample). A tile is classified as blurred if its raw focus score is at or
    below ``roi_threshold`` OR its intensity is below ``intensity_threshold``.
    This mirrors the rule documented in ``calculate_roi_blur_threshold`` and
    produces the same columns as ``classify_roi_blur`` so downstream code is
    unaffected:

    - 'blur_prob_gmm'    : NaN (no posterior probability without a GMM)
    - 'is_blurred_gmm'   : boolean, final classification
    - 'is_low_intensity' : boolean, intensity < intensity_threshold

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        DataFrame with a focus-score column ('dapi_focus_score' or
        'focus_score') and an intensity column ('dapi_intensity' or
        'raw_intensity').
    roi_threshold : float
        Raw focus-score threshold (from ``calculate_roi_blur_threshold``).
    intensity_threshold : float, optional
        Tiles below this are auto-blurred (default: ROI_INTENSITY_THRESHOLD).
    focus_col_name : str, optional
        Focus-score column to use (default: 'dapi_focus_score'; falls back to
        'focus_score').
    """
    df = df_grid_roi.copy()

    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df.columns else "raw_intensity"
    )
    if intensity_col not in df.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    if focus_col_name not in df.columns:
        if "focus_score" not in df.columns:
            raise ValueError(
                f"Focus score column '{focus_col_name}' not found and 'focus_score' not present either"
            )
        focus_col_name = "focus_score"

    is_low_intensity = df[intensity_col] < intensity_threshold
    df["is_low_intensity"] = is_low_intensity
    df["blur_prob_gmm"] = np.nan
    df["is_blurred_gmm"] = (df[focus_col_name] <= roi_threshold) | is_low_intensity

    total_rois = len(df)
    n_low_int = int(is_low_intensity.sum())
    n_blur = int(df["is_blurred_gmm"].sum())
    pct_low = (n_low_int / total_rois * 100) if total_rois > 0 else 0.0
    pct_blur = (n_blur / total_rois * 100) if total_rois > 0 else 0.0

    logging.info("  Percentile-threshold fallback blur classification completed")
    logging.info(f"    Total tiles: {total_rois}")
    logging.info(
        f"    Low-intensity tiles (auto-blurred): {n_low_int} ({pct_low:.1f}%)"
    )
    logging.info(
        f"    Blurred tiles (threshold + intensity): {n_blur} ({pct_blur:.1f}%)"
    )

    return df


def classify_roi_blur(
    df_grid_roi,
    gmm,
    blur_component_idx: int,
    blur_prob_threshold: float = 0.5,
    intensity_threshold: float = ROI_INTENSITY_THRESHOLD,
    focus_col_name: str = "dapi_focus_score",
):
    """
    Classify each tile as blurred or in-focus using a fitted GMM and an intensity safeguard.

    Rules:
    ------
    - Tiles with intensity < intensity_threshold are always marked as blurred
      (low-intensity / background or globally problematic tissue).
    - For tissue tiles (intensity >= threshold), use the GMM posterior probability
      of belonging to the "blur" component:
        - blur_prob = P(component == blur_component_idx | focus_score)
        - Tile is blurred if blur_prob > blur_prob_threshold.

    The classification is added to df_grid_roi in new columns:
    - 'blur_prob_gmm' : posterior probability of being blurred (NaN for low-intensity tiles)
    - 'is_blurred_gmm' : boolean, final classification combining intensity + GMM
    - 'is_low_intensity' : boolean, intensity < intensity_threshold

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        DataFrame with at least:
        - focus_col_name (e.g. 'dapi_focus_score' or 'focus_score')
        - 'dapi_intensity' or 'raw_intensity'
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM model from fit_focus_gmm()
    blur_component_idx : int
        Index of the GMM component corresponding to blurred tiles.
    blur_prob_threshold : float, optional
        Threshold on posterior blur probability to classify a tile as blurred
        (default: 0.5).
    intensity_threshold : float, optional
        Intensity safeguard: tiles below this are auto-blurred (default: ROI_INTENSITY_THRESHOLD).
    focus_col_name : str, optional
        Name of focus-score column used (default: 'dapi_focus_score').

    Returns
    -------
    df_grid_roi : pandas.DataFrame
        Input DataFrame with new columns:
        - 'blur_prob_gmm'
        - 'is_blurred_gmm'
        - 'is_low_intensity'
    """
    df = df_grid_roi.copy()

    # Intensity column
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df.columns else "raw_intensity"
    )
    if intensity_col not in df.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    # Focus column
    if focus_col_name not in df.columns:
        if "focus_score" not in df.columns:
            raise ValueError(
                f"Focus score column '{focus_col_name}' not found and 'focus_score' not present either"
            )
        focus_col_name = "focus_score"

    # Intensity-based low-intensity flag
    is_low_intensity = df[intensity_col] < intensity_threshold
    df["is_low_intensity"] = is_low_intensity

    # Initialize columns
    df["blur_prob_gmm"] = np.nan
    df["is_blurred_gmm"] = False

    # Tissue ROIs: intensity >= threshold
    tissue_mask = ~is_low_intensity
    if tissue_mask.any():
        raw_scores = df.loc[tissue_mask, focus_col_name].values.astype(np.float64)
        x_log = np.log1p(raw_scores).reshape(-1, 1)

        # Posterior probabilities of components
        probs = gmm.predict_proba(x_log)
        blur_prob = probs[:, blur_component_idx]

        df.loc[tissue_mask, "blur_prob_gmm"] = blur_prob
        df.loc[tissue_mask, "is_blurred_gmm"] = blur_prob > blur_prob_threshold

    # Low-intensity tiles are always blurred according to the safeguard
    df.loc[is_low_intensity, "is_blurred_gmm"] = True

    # Summary
    total_rois = len(df)
    n_low_int = int(is_low_intensity.sum())
    n_blur = int(df["is_blurred_gmm"].sum())
    pct_blur = (n_blur / total_rois) * 100 if total_rois > 0 else 0.0

    logging.info("  GMM-based blur classification completed")
    logging.info(f"    Total tiles: {total_rois}")
    logging.info(
        f"    Low-intensity tiles (auto-blurred): {n_low_int} ({n_low_int / total_rois * 100:.1f}%)"
    )
    logging.info(f"    Blurred tiles (GMM + intensity): {n_blur} ({pct_blur:.1f}%)")

    return df


def fit_focus_gmm_2d(
    df_grid_roi,
    intensity_threshold: float = ROI_INTENSITY_THRESHOLD,
    focus_col_name: str = "dapi_focus_score",
    n_components: int = 2,
    random_state: int = 0,
):
    """
    Fit a 2D Gaussian Mixture Model (GMM) to tile focus scores using both
    focus score (std²/mean) and Laplacian variance as features.

    The model uses log1p-transformed features:
    - Feature 1: log1p(dapi_focus_score)
    - Feature 2: log1p(dapi_lap_var)

    This allows the model to use both global contrast (focus_score) and
    high-frequency content (Laplacian variance) to distinguish blurred vs in-focus tiles.

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        DataFrame with at least:
        - focus_col_name (e.g. 'dapi_focus_score' or 'focus_score')
        - 'dapi_lap_var' (Laplacian variance)
        - 'dapi_intensity' or 'raw_intensity'
    intensity_threshold : float, optional
        Minimum intensity to consider a tile as tissue (default: ROI_INTENSITY_THRESHOLD).
        Tiles below this are excluded from GMM training.
    focus_col_name : str, optional
        Name of the focus-score column to use (default: 'dapi_focus_score').
        If not present, 'focus_score' will be used.
    n_components : int, optional
        Number of Gaussian components for the GMM (default: 2).
    random_state : int, optional
        Random seed for reproducibility (default: 0).

    Returns
    -------
    gmm : sklearn.mixture.GaussianMixture
        Fitted 2D GMM model on [log1p(focus_score), log1p(lap_var)] of tissue tiles.
    blur_component_idx : int
        Index of the GMM component corresponding to blurred tiles (lower mean on focus_score dimension).
    """
    # Determine intensity column
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df_grid_roi.columns else "raw_intensity"
    )
    if intensity_col not in df_grid_roi.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    # Determine focus column
    if focus_col_name not in df_grid_roi.columns:
        # Fallback to generic 'focus_score'
        if "focus_score" not in df_grid_roi.columns:
            raise ValueError(
                f"Focus score column '{focus_col_name}' not found and 'focus_score' not present either"
            )
        focus_col_name = "focus_score"

    # Check for Laplacian variance column
    if "dapi_lap_var" not in df_grid_roi.columns:
        raise ValueError(
            "dapi_lap_var column not found. 2D GMM requires Laplacian variance. "
            "Make sure calculate_roi_focusscore() was used (not calculate_roi_focusscore_without_laplace)."
        )

    # Select tissue tiles for training.
    # 2026-06-22 (qc_drift_analysis): select by tissue-mask coverage, not a raw
    # intensity floor. The old `intensity >= ROI_INTENSITY_THRESHOLD` gate broke
    # on dim XOA-4.0 images (~14x dimmer), dropping most real tissue from the
    # training set and biasing the blur/focus split. Coverage is
    # brightness-independent. Falls back to the intensity gate only when
    # tissue_coverage is unavailable. NB: this still trains the GMM on tissue
    # only (it is NOT the reverted "all-tiles" Scope B, commit e8e6731), so the
    # within-tissue blur-vs-focus bimodality is preserved.
    if "tissue_coverage" in df_grid_roi.columns:
        tissue_rois = df_grid_roi[
            df_grid_roi["tissue_coverage"] >= ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC
        ].copy()
        _selection_desc = (
            f"tissue_coverage >= {ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC}"
        )
    else:
        tissue_rois = df_grid_roi[
            df_grid_roi[intensity_col] >= intensity_threshold
        ].copy()
        _selection_desc = f"intensity >= {intensity_threshold}"
    if len(tissue_rois) == 0:
        raise ValueError(
            f"No tissue tiles found with {_selection_desc}. "
            f"Cannot fit GMM. Check tissue mask / intensity thresholds or image quality."
        )

    # Get both features
    focus_scores = tissue_rois[focus_col_name].values.astype(np.float64)
    lap_vars = tissue_rois["dapi_lap_var"].values.astype(np.float64)

    # Filter out NaN values
    valid_mask = ~(np.isnan(focus_scores) | np.isnan(lap_vars))
    if valid_mask.sum() == 0:
        raise ValueError("No valid tile data (all NaN) for 2D GMM training")

    focus_scores_valid = focus_scores[valid_mask]
    lap_vars_valid = lap_vars[valid_mask]

    # Log-transform both features (clamp negatives to 0 before log1p)
    x_focus_log = np.log1p(np.maximum(focus_scores_valid, 0.0))
    x_lap_log = np.log1p(np.maximum(lap_vars_valid, 0.0))

    # Combine into 2D feature matrix
    x_2d = np.column_stack([x_focus_log, x_lap_log])

    # Fit 2D GMM
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=random_state
    )
    gmm.fit(x_2d)

    # Identify which component corresponds to blurred ROIs:
    # Use the focus_score dimension (first column) - lower mean = blur
    means_focus = gmm.means_[:, 0]  # First dimension (focus_score)
    blur_component_idx = int(np.argmin(means_focus))

    logging.info("  2D GMM focus model fitted on tissue tiles")
    logging.info(f"    Number of tissue tiles used: {valid_mask.sum()}")
    logging.info("    Component means (log1p focus_score, log1p lap_var):")
    for i, mean in enumerate(gmm.means_):
        logging.info(f"      Component {i}: [{mean[0]:.4f}, {mean[1]:.4f}]")
    logging.info(
        f"    Blur component index: {blur_component_idx} (lower mean on focus_score dimension)"
    )

    return gmm, blur_component_idx


def classify_roi_blur_2d(
    df_grid_roi,
    gmm,
    blur_component_idx: int,
    blur_prob_threshold: float = 0.5,
    intensity_threshold: float = ROI_INTENSITY_THRESHOLD,
    focus_col_name: str = "dapi_focus_score",
):
    """
    Classify each tile as blurred or in-focus using a fitted 2D GMM and an intensity safeguard.

    Uses both focus_score and Laplacian variance as features.

    Rules:
    ------
    - Tissue is defined by tissue_coverage >= ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC
      when the column is present (brightness-independent); otherwise it falls back
      to intensity >= intensity_threshold.
    - Non-tissue tiles (low coverage / background) are marked as blurred. They are
      excluded from the tissue-filtered blur % regardless.
    - Tissue tiles missing dapi_lap_var are marked as blurred (cannot use 2D GMM).
    - For tissue tiles with valid lap_var, use the 2D GMM posterior probability
      of belonging to the "blur" component:
        - blur_prob = P(component == blur_component_idx | focus_score, lap_var)
        - Tile is blurred if blur_prob > blur_prob_threshold.

    The classification is added to df_grid_roi in new columns:
    - 'blur_prob_gmm_2d' : posterior probability of being blurred (NaN for low-intensity or missing lap_var tiles)
    - 'is_blurred_gmm_2d' : boolean, final classification combining intensity + 2D GMM
    - 'is_low_intensity' : boolean, intensity < intensity_threshold (reused if exists)

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        DataFrame with at least:
        - focus_col_name (e.g. 'dapi_focus_score' or 'focus_score')
        - 'dapi_lap_var' (Laplacian variance)
        - 'dapi_intensity' or 'raw_intensity'
    gmm : sklearn.mixture.GaussianMixture
        Fitted 2D GMM model from fit_focus_gmm_2d()
    blur_component_idx : int
        Index of the GMM component corresponding to blurred tiles.
    blur_prob_threshold : float, optional
        Threshold on posterior blur probability to classify a tile as blurred
        (default: 0.5).
    intensity_threshold : float, optional
        Intensity safeguard: tiles below this are auto-blurred (default: ROI_INTENSITY_THRESHOLD).
    focus_col_name : str, optional
        Name of focus-score column used (default: 'dapi_focus_score').

    Returns
    -------
    df_grid_roi : pandas.DataFrame
        Input DataFrame with new columns:
        - 'blur_prob_gmm_2d'
        - 'is_blurred_gmm_2d'
        - 'is_low_intensity' (if not already present)
    """
    df = df_grid_roi.copy()

    # Intensity column
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df.columns else "raw_intensity"
    )
    if intensity_col not in df.columns:
        raise ValueError(
            "Intensity column not found. Expected 'dapi_intensity' or 'raw_intensity'"
        )

    # Focus column
    if focus_col_name not in df.columns:
        if "focus_score" not in df.columns:
            raise ValueError(
                f"Focus score column '{focus_col_name}' not found and 'focus_score' not present either"
            )
        focus_col_name = "focus_score"

    # Check for Laplacian variance
    if "dapi_lap_var" not in df.columns:
        raise ValueError(
            "dapi_lap_var column not found. 2D GMM classification requires Laplacian variance."
        )

    # Intensity-based low-intensity flag (reuse if exists, otherwise create)
    if "is_low_intensity" not in df.columns:
        is_low_intensity = df[intensity_col] < intensity_threshold
        df["is_low_intensity"] = is_low_intensity
    else:
        is_low_intensity = df["is_low_intensity"]

    # Initialize columns
    df["blur_prob_gmm_2d"] = np.nan
    df["is_blurred_gmm_2d"] = False

    # Tissue ROIs for blur classification.
    # 2026-06-22 (qc_drift_analysis): define tissue by mask coverage, not the
    # intensity floor — matches fit_focus_gmm_2d. On dim XOA-4.0 images many real
    # tissue tiles fall below ROI_INTENSITY_THRESHOLD; the old rule force-blurred
    # them and inflated the blur rate (~40% on v4). Coverage is
    # brightness-independent. is_low_intensity is still computed above for
    # intensity QC / reporting, just not used to gate blur here.
    if "tissue_coverage" in df.columns:
        tissue_mask = df["tissue_coverage"] >= ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC
    else:
        tissue_mask = ~is_low_intensity
    has_lap_var = df["dapi_lap_var"].notna()
    valid_mask = tissue_mask & has_lap_var

    if valid_mask.any():
        # Get both features for valid ROIs
        focus_scores = df.loc[valid_mask, focus_col_name].values.astype(np.float64)
        lap_vars = df.loc[valid_mask, "dapi_lap_var"].values.astype(np.float64)

        # Filter out any remaining NaN (shouldn't happen, but safety check)
        valid_data_mask = ~(np.isnan(focus_scores) | np.isnan(lap_vars))
        if valid_data_mask.sum() > 0:
            focus_scores_valid = focus_scores[valid_data_mask]
            lap_vars_valid = lap_vars[valid_data_mask]

            # Log-transform (clamp negatives to 0, matching fit_focus_gmm_2d)
            x_focus_log = np.log1p(np.maximum(focus_scores_valid, 0.0))
            x_lap_log = np.log1p(np.maximum(lap_vars_valid, 0.0))

            # Combine into 2D feature matrix
            x_2d = np.column_stack([x_focus_log, x_lap_log])

            # Posterior probabilities of components
            probs = gmm.predict_proba(x_2d)
            blur_prob = probs[:, blur_component_idx]

            # Map back to original valid_mask indices
            valid_indices = df.index[valid_mask][valid_data_mask]
            df.loc[valid_indices, "blur_prob_gmm_2d"] = blur_prob
            df.loc[valid_indices, "is_blurred_gmm_2d"] = blur_prob > blur_prob_threshold

    # Non-tissue tiles (low mask coverage / background) are not treated as
    # focused. They are excluded from the tissue-filtered blur % anyway; this
    # only affects the unfiltered count, preserving the prior convention.
    df.loc[~tissue_mask, "is_blurred_gmm_2d"] = True

    # Tissue tiles without lap_var cannot be classified by the 2D GMM → blurred
    missing_lap_var = ~has_lap_var & tissue_mask
    if missing_lap_var.any():
        df.loc[missing_lap_var, "is_blurred_gmm_2d"] = True

    # Summary
    total_rois = len(df)
    n_low_int = int(is_low_intensity.sum())
    n_missing_lap = int(missing_lap_var.sum()) if missing_lap_var.any() else 0
    n_blur = int(df["is_blurred_gmm_2d"].sum())
    pct_blur = (n_blur / total_rois) * 100 if total_rois > 0 else 0.0

    logging.info("  2D GMM-based blur classification completed")
    logging.info(f"    Total tiles: {total_rois}")
    logging.info(
        f"    Low-intensity tiles (auto-blurred): {n_low_int} ({n_low_int / total_rois * 100:.1f}%)"
    )
    if n_missing_lap > 0:
        logging.info(
            f"    Tiles missing lap_var (auto-blurred): {n_missing_lap} ({n_missing_lap / total_rois * 100:.1f}%)"
        )
    logging.info(f"    Blurred tiles (2D GMM + intensity): {n_blur} ({pct_blur:.1f}%)")

    return df


_MAX_SCATTER_POINTS = 10_000


def _subsample_idx(mask, max_points=None, rng_seed=42):
    """Return indices where *mask* is True, randomly subsampled to *max_points*."""
    if max_points is None:
        max_points = _MAX_SCATTER_POINTS
    idx = np.where(mask)[0]
    if max_points <= 0 or len(idx) <= max_points:
        return idx
    rng = np.random.default_rng(rng_seed)
    return rng.choice(idx, size=max_points, replace=False)


def plot_grid_roi_focus_heatmap(
    df_grid_roi,
    small0,
    figures_dir,
    figures_source_dir,
    threshold=-1.0,
    focus_maps=None,
    focus_heatmap=None,
):
    """
    Create heatmap of grid tile focus scores across the whole tissue.

    When pixel-level ``focus_maps`` are provided, renders smooth per-pixel
    heatmaps via imshow (much faster and higher resolution than Rectangle
    patches). Falls back to the legacy Rectangle-patch approach otherwise.

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: roi_id, x1, x2, y1, y2, focus_score,
        focus_score_norm, raw_intensity, tissue_coverage
    small0 : numpy.ndarray
        Downsampled DAPI image (level 3, 8x downsampling)
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    threshold : float, optional
        Threshold for normalized focus score (default: -1.0)
    focus_maps : dict or None, optional
        Pixel-level focus maps from compute_all_focus_maps(). If provided,
        uses imshow for smooth rendering instead of Rectangle patches.
    """
    from skimage.transform import downscale_local_mean
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    downsample_factor = 8
    img_height, img_width = small0.shape
    img_aspect = img_height / img_width if img_width > 0 else 1.0
    panel_width = 6
    # Floor at 50% of width so very wide slides (e.g. brain) don't squash titles / colorbars
    panel_height = max(panel_width * 0.5, panel_width * img_aspect)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width + 2, panel_height))
    fig.suptitle(
        "Spatial focus-score map across the Xenium region",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    # DAPI background p99 is identical for both panels -- compute once over the full
    # ~86 Mpx small0. Draw the background via _imshow_thumb (block-mean to the ~2000px
    # panel) not a full-res imshow: the raw imshow was ~50 s/panel x2 panels x2 saves
    # and was the entire residual cost of this figure (the binned panels are ~5 s).
    _bg_vmax = np.percentile(small0, 99)

    # Plot 1: Focus score heatmap
    ax = axes[0]
    _imshow_thumb(
        ax,
        small0,
        cmap="Greys_r",
        vmax=_bg_vmax,
        alpha=0.5,
        aspect="auto",
        extent=[0, img_width, img_height, 0],
        origin="upper",
    )

    # Streaming mode supplies the down-sampled canvas directly, and it reproduces
    # downscale_local_mean bit-exactly: write boundaries are aligned to the block size so
    # each block is reduced by a single reshape in skimage's own order. No
    # full-resolution plane is read (22 GB, plus another 22 GB for skimage's pad).
    heatmap = focus_heatmap
    if heatmap is None and focus_maps is not None:
        dapi_focus = focus_maps.get("dapi_focus_map")
        if dapi_focus is not None:
            heatmap = downscale_local_mean(
                dapi_focus, (downsample_factor, downsample_factor)
            )
    if heatmap is not None:
        # Binned redesign: an imshow of the full ~86-megapixel per-pixel field cost
        # ~126 s (it measured 4459.8 s on a 102045x53908 sample, runs 1ZyVIlaKBYxJrQ /
        # 4c0HFuivKDkWXr -- the largest single cost in the step) and rendered millions
        # of unreadable per-pixel dots. Aggregating to a coarse grid (~180 cells on the
        # long axis) keeps the regional signal and draws in ~1 s.
        t_h = time.perf_counter()
        # Clip to small0 dimensions (in case of rounding: the canvas can be a row/column
        # larger, ceil vs floor).
        focus_ds = np.asarray(heatmap)[:img_height, :img_width]
        positive = focus_ds > 0
        has_positive = bool(positive.any())
        # Color range comes from the FULL-resolution positive field, not the binned
        # means, so viridis maps exactly as the per-pixel figure did.
        vmin = np.percentile(focus_ds[positive], 1) if has_positive else 0
        vmax = np.percentile(focus_ds[positive], 99) if has_positive else 1
        # Per-bin MEAN focus over tissue pixels (focus_ds > 0). Non-tissue pixels are
        # NaN'd so they do not drag the mean down; all-non-tissue bins come back NaN and
        # are set to vmin so they render as viridis-min -- the same dark background the
        # per-pixel field showed where focus_ds == 0.
        step = max(1, -(-max(img_height, img_width) // _FOCUS_HEATMAP_BINS_LONG))
        tissue_focus = np.where(positive, focus_ds, np.nan)
        del positive
        focus_binned = _bin_nanmean(tissue_focus, step)
        focus_binned = np.where(np.isnan(focus_binned), vmin, focus_binned)
        ax.imshow(
            focus_binned,
            cmap="viridis",
            alpha=0.6,
            aspect="auto",
            extent=[0, img_width, img_height, 0],
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        logging.info(
            f"    [TIMING] fig5 left bin+imshow (step={step}, "
            f"{focus_binned.shape}): {time.perf_counter() - t_h:.1f}s"
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Focus Score (var/mean)", fontsize=12)
        ax.set_title("Focus Score (binned)", fontsize=14)
    else:
        # Legacy Rectangle-patch approach
        from matplotlib.patches import Rectangle

        for _, roi in df_grid_roi.iterrows():
            x1_ds = roi["x1"] / downsample_factor
            x2_ds = roi["x2"] / downsample_factor
            y1_ds = roi["y1"] / downsample_factor
            y2_ds = roi["y2"] / downsample_factor
            w = x2_ds - x1_ds
            h = y2_ds - y1_ds
            if w <= 0 or h <= 0:
                continue
            if x1_ds < 0 or y1_ds < 0 or x2_ds > img_width or y2_ds > img_height:
                continue
            norm_score = np.clip((roi["focus_score_norm"] + 3) / 6, 0, 1)
            color = plt.cm.viridis(norm_score)
            rect = Rectangle(
                (x1_ds, y1_ds), w, h, facecolor=color, alpha=0.6, edgecolor="none"
            )
            ax.add_patch(rect)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=-3, vmax=3))
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Focus Score (Normalized)", fontsize=12)
        cbar.ax.axhline(y=float(threshold), color="red", linewidth=1.5, linestyle="--")
        ax.set_title("Grid Tile Focus Score Heatmap (Normalized)", fontsize=14)

    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Plot 2: GMM 2D classification (blurred vs in-focus)
    ax = axes[1]
    _imshow_thumb(
        ax,
        small0,
        cmap="Greys_r",
        vmax=_bg_vmax,
        alpha=0.5,
        aspect="auto",
        extent=[0, img_width, img_height, 0],
        origin="upper",
    )

    has_gmm_2d = "is_blurred_gmm_2d" in df_grid_roi.columns

    # Gated on the GMM column alone, NOT on focus_maps: this branch reads no pixel
    # data. It rasterises `is_blurred_gmm_2d` from the ROI table's own x1/x2/y1/y2
    # columns at down-sampled resolution, so the focus_maps check was only ever a
    # proxy for "not the legacy path".
    #
    # It mattered because streaming passes focus_maps=None, which sent this panel to
    # the Rectangle fallback below -- one patch per ROI through df.iterrows(). On the
    # 102045x53908 sample that is 4,490,640 patches, and it took Figure 5 from 15.9 s
    # to 4388-4460 s. Measured on runs 47jub5CwOHb82v (streamed, 436.8 s at 429 k
    # ROIs) against 3xiurC3181Zgwc (planes, 15.9 s, same bundle).
    #
    # `has_gmm_2d` reproduces the old behaviour exactly where it mattered: the 2D GMM
    # needs dapi_lap_var, which --legacy-focus does not produce, so that path still
    # falls through to the Rectangle branch.
    if has_gmm_2d:
        from matplotlib.colors import LinearSegmentedColormap

        t_h = time.perf_counter()
        # Rasterise the in-focus indicator at downsampled resolution: 1.0 for an
        # in-focus ROI pixel, 0.0 for a blurred one, NaN off-tissue (no ROI). ALL ROIs
        # are painted -- not just blurred ones -- because a bin needs its tissue
        # denominator; the fraction in focus is meaningless without it.
        infocus_ds = np.full((img_height, img_width), np.nan, dtype=np.float32)
        x1_arr = (df_grid_roi["x1"].values // downsample_factor).astype(int)
        x2_arr = np.minimum(
            df_grid_roi["x2"].values // downsample_factor, img_width
        ).astype(int)
        y1_arr = (df_grid_roi["y1"].values // downsample_factor).astype(int)
        y2_arr = np.minimum(
            df_grid_roi["y2"].values // downsample_factor, img_height
        ).astype(int)
        infocus_val = np.where(
            df_grid_roi["is_blurred_gmm_2d"].values, 0.0, 1.0
        ).astype(np.float32)
        for i in range(len(x1_arr)):
            infocus_ds[y1_arr[i] : y2_arr[i], x1_arr[i] : x2_arr[i]] = infocus_val[i]
        # Per-bin FRACTION in focus: mean of the 0/1 indicator over tissue pixels, on
        # the SAME coarse grid as the left panel. A continuous 0..1 field is smooth
        # under binning (no per-pixel speckle) -- the whole point of the redesign.
        # All-non-tissue bins stay NaN and render transparent (set_bad alpha 0), so the
        # DAPI background shows through off-tissue.
        step = max(1, -(-max(img_height, img_width) // _FOCUS_HEATMAP_BINS_LONG))
        frac_infocus = _bin_nanmean(infocus_ds, step)
        # Continuous red(0 = blurred) -> blue(1 = in focus): the same two colors the
        # per-tile ListedColormap used, now as the endpoints of a smooth map.
        cmap_focus = LinearSegmentedColormap.from_list("focus_frac", ["red", "blue"])
        cmap_focus.set_bad(alpha=0.0)
        ax.imshow(
            frac_infocus,
            cmap=cmap_focus,
            alpha=0.5,
            aspect="auto",
            extent=[0, img_width, img_height, 0],
            origin="upper",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap_focus, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("In-focus fraction (0 = blurred, 1 = in focus)", fontsize=12)
        logging.info(
            f"    [TIMING] fig5 right bin+imshow (step={step}, "
            f"{frac_infocus.shape}): {time.perf_counter() - t_h:.1f}s"
        )
        ax.set_title("Focus Classification (2D GMM)", fontsize=14)
    else:
        # Legacy Rectangle-patch approach
        from matplotlib.patches import Rectangle

        for _, roi in df_grid_roi.iterrows():
            x1_ds = roi["x1"] / downsample_factor
            x2_ds = roi["x2"] / downsample_factor
            y1_ds = roi["y1"] / downsample_factor
            y2_ds = roi["y2"] / downsample_factor
            w = x2_ds - x1_ds
            h = y2_ds - y1_ds
            if w <= 0 or h <= 0:
                continue
            if x1_ds < 0 or y1_ds < 0 or x2_ds > img_width or y2_ds > img_height:
                continue
            if has_gmm_2d:
                color = "blue" if not roi["is_blurred_gmm_2d"] else "red"
            else:
                color = "blue" if roi["focus_score_norm"] > threshold else "red"
            rect = Rectangle(
                (x1_ds, y1_ds), w, h, facecolor=color, alpha=0.6, edgecolor="none"
            )
            ax.add_patch(rect)
        if has_gmm_2d:
            ax.set_title(
                "Grid Tile Focus Score (2D GMM: Blue=In-Focus, Red=Blurred)",
                fontsize=14,
            )
        else:
            ax.set_title(f"Grid Tile Focus Score (Threshold={threshold})", fontsize=14)

    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # The 2D-GMM branch gives the right panel its own real colorbar. Only the legacy
    # (no-GMM) branch has none, so add a phantom cax there to keep its plotting region
    # the same width as the left panel (whose width is shrunk by its colorbar).
    if not has_gmm_2d:
        cax_r = make_axes_locatable(axes[1]).append_axes("right", size="3%", pad=0.05)
        cax_r.axis("off")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "grid_roi_focus_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data as CSV
    if figures_source_dir is not None:
        df_grid_roi_scaled = df_grid_roi.copy()
        df_grid_roi_scaled["x1_ds"] = df_grid_roi_scaled["x1"] / downsample_factor
        df_grid_roi_scaled["x2_ds"] = df_grid_roi_scaled["x2"] / downsample_factor
        df_grid_roi_scaled["y1_ds"] = df_grid_roi_scaled["y1"] / downsample_factor
        df_grid_roi_scaled["y2_ds"] = df_grid_roi_scaled["y2"] / downsample_factor
        df_grid_roi_scaled.to_csv(
            figures_source_dir / "grid_roi_focus_heatmap.csv", index=False
        )


def plot_snr_roi_heatmap(
    df_grid_roi,
    small0,
    figures_dir,
    figures_source_dir,
    snr_thresholds=None,
):
    """Paint per-tile transcript SNR spatially on the DAPI background.

    Two-panel figure:
      Left  – neg_pct (fraction of negative-control transcripts per tile)
      Right – roi_tx_snr_ratio (real / negative transcript ratio, log scale)

    Colorbars autoscale to the data's p99 (no fixed floor). When provided,
    WARN/FAIL threshold lines from `snr_thresholds` are drawn on each
    colorbar. Lines outside the autoscaled range are clipped by matplotlib —
    on a clean slide where data is far below FAIL, the line simply doesn't
    appear, which is the intended visual cue.

    Parameters
    ----------
    df_grid_roi : pandas.DataFrame
        Must contain columns: x1, x2, y1, y2, neg_pct, roi_tx_snr_ratio,
        snr_total_tx. Optionally: snr_real_tx, snr_neg_tx — when present, the
        right panel paints (real+1)/(neg+1) (Laplace pseudocount) so tiles
        with neg=0 or real=0 (currently NaN/0 under raw ratio) render in
        their correct extremes. The canonical roi_tx_snr_ratio column stays
        raw — pseudocount applies to *display only*, not to verdicts/JSON.
    small0 : numpy.ndarray
        Downsampled DAPI image (level 3, 8× downsampling).
    figures_dir, figures_source_dir : Path
        Output directories.
    snr_thresholds : dict, optional
        YAML ``snr.roi_tx`` block. Keys: neg_pct_warn, neg_pct_fail,
        ratio_warn, ratio_fail.
    """
    from matplotlib.colors import LogNorm

    _t = snr_thresholds or {}

    downsample_factor = 8
    img_height, img_width = small0.shape

    # Phase v5: scope to within-tissue tiles with transcripts. Tissue tiles
    # WITHOUT transcripts (alveolar / bronchiolar airspaces in lung, etc.)
    # stay transparent so the DAPI background still shows through them.
    if "tissue_coverage" in df_grid_roi.columns:
        df_plot = df_grid_roi[
            (df_grid_roi["snr_total_tx"] > 0) & (df_grid_roi["tissue_coverage"] > 0.5)
        ].copy()
    else:
        df_plot = df_grid_roi[df_grid_roi["snr_total_tx"] > 0].copy()
    if df_plot.empty:
        logging.warning("No tiles with transcripts — skipping SNR heatmap.")
        return

    # Compute figure size from image aspect ratio to avoid empty white space
    img_aspect = img_height / img_width if img_width > 0 else 1.0
    panel_width = 6  # width per panel in inches (matches morphology overview)
    # Floor at 50% of width so very wide slides (e.g. brain) don't squash titles / colorbars
    panel_height = max(panel_width * 0.5, panel_width * img_aspect)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width + 2, panel_height))

    # Pre-compute downsampled tile coordinates (vectorized)
    x1_arr = np.clip(
        (df_plot["x1"].values / downsample_factor).astype(int), 0, img_width
    )
    x2_arr = np.clip(
        (df_plot["x2"].values / downsample_factor).astype(int), 0, img_width
    )
    y1_arr = np.clip(
        (df_plot["y1"].values / downsample_factor).astype(int), 0, img_height
    )
    y2_arr = np.clip(
        (df_plot["y2"].values / downsample_factor).astype(int), 0, img_height
    )

    def _draw_background(ax):
        # _imshow_thumb: downsample the ~86 Mpx DAPI background to the display
        # resolution before imshow (explicit extent is preserved, so the panel is
        # visually identical). Cuts ~16 s/panel of full-res resampling.
        _imshow_thumb(
            ax,
            small0,
            cmap="Greys_r",
            vmax=np.percentile(small0, 99),
            alpha=0.5,
            aspect="auto",
            extent=[0, img_width, img_height, 0],
            origin="upper",
        )

    def _fill_tile_image(values):
        """Build a 2D float32 array with tile values; NaN = transparent."""
        img = np.full((img_height, img_width), np.nan, dtype=np.float32)
        for i in range(len(df_plot)):
            if x2_arr[i] <= x1_arr[i] or y2_arr[i] <= y1_arr[i]:
                continue
            val = values[i]
            if not np.isfinite(val):
                continue
            img[y1_arr[i] : y2_arr[i], x1_arr[i] : x2_arr[i]] = val
        return img

    def _finish_ax(ax):
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Panel 1: neg_pct ─────────────────────────────────────────────────
    ax = axes[0]
    _draw_background(ax)

    # Fixed colorbar (0 → 0.40) — enables cross-sample comparison and ensures
    # WARN (0.15) / FAIL (0.30) threshold lines always render in frame.
    # extend="max" tags tiles above 0.40 with the deepest red + a triangle marker.
    _NEG_VMAX = 0.40
    neg_vals = df_plot["neg_pct"].values

    neg_img = _fill_tile_image(neg_vals)
    # _imshow_thumb: NaN-safe block-mean downsample of the full-res tile overlay
    # (tiles are large blocks, so the mean is visually identical). ~16 s -> ~0.5 s.
    _imshow_thumb(
        ax,
        neg_img,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=_NEG_VMAX,
        alpha=0.6,
        aspect="auto",
        extent=[0, img_width, img_height, 0],
        origin="upper",
        interpolation="nearest",
    )
    _finish_ax(ax)
    ax.set_title("Negative Probe Fraction per Tile", fontsize=14)

    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=_NEG_VMAX)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, extend="max")
    cbar.set_label("neg_pct (fraction)", fontsize=12)
    # Threshold lines (now always in frame thanks to fixed colorbar range)
    _neg_warn = _t.get("neg_pct_warn")
    _neg_fail = _t.get("neg_pct_fail")
    if isinstance(_neg_warn, (int, float)):
        cbar.ax.axhline(
            y=float(_neg_warn), color="orange", linewidth=1.5, linestyle="--"
        )
    if isinstance(_neg_fail, (int, float)):
        cbar.ax.axhline(y=float(_neg_fail), color="red", linewidth=1.5)

    # ── Panel 2: roi_tx_snr_ratio (log scale) ───────────────────────────
    ax = axes[1]
    _draw_background(ax)

    # Display-only pseudocount: paint (real+1)/(neg+1) instead of raw
    # real/neg. The raw ratio NaN's whenever neg=0 (divide-by-zero in
    # snr_metrics.py) AND zeros out whenever real=0 — exactly the extreme
    # tiles a reader most wants to see. Laplace α=1 regularises both
    # extremes onto the colorbar without affecting the canonical
    # roi_tx_snr_ratio column used for sample-level verdicts / JSON.
    _PSEUDOCOUNT = 1
    if "snr_real_tx" in df_plot.columns and "snr_neg_tx" in df_plot.columns:
        ratio_vals = (
            (df_plot["snr_real_tx"].astype(np.float64) + _PSEUDOCOUNT)
            / (df_plot["snr_neg_tx"].astype(np.float64) + _PSEUDOCOUNT)
        ).values
        _ratio_cbar_label = "(real+1) / (neg+1) — pseudocount-smoothed"
        _ratio_title_suffix = ", α=1 pseudocount"
    else:
        ratio_vals = df_plot["roi_tx_snr_ratio"].values
        _ratio_cbar_label = "roi_tx_snr_ratio (real / neg)"
        _ratio_title_suffix = ""
    # Fixed log colorbar (1× → 1000×) — enables cross-sample comparison and
    # ensures WARN (30×) / FAIL (10×) threshold lines always render in frame.
    # extend="min" tags tiles below 1× (the no-signal regime) with the deepest
    # red + a triangle marker. High end stays uncapped visually — very-good
    # samples saturate at dark green, which is fine for QC purposes.
    _RATIO_VMIN = 1.0
    _RATIO_VMAX = 1000.0

    # For LogNorm, clamp values to [vmin, vmax] range; ≤0 stays NaN
    ratio_img = _fill_tile_image(ratio_vals)
    # Replace non-positive finite values with NaN (LogNorm requires > 0)
    ratio_img[ratio_img <= 0] = np.nan
    # _imshow_thumb: NaN-safe block-mean downsample (visually identical block overlay).
    _imshow_thumb(
        ax,
        ratio_img,
        cmap="RdYlGn",
        norm=LogNorm(vmin=_RATIO_VMIN, vmax=_RATIO_VMAX),
        alpha=0.6,
        aspect="auto",
        extent=[0, img_width, img_height, 0],
        origin="upper",
        interpolation="nearest",
    )
    _finish_ax(ax)
    ax.set_title(
        f"Transcript SNR Ratio per Tile (log scale{_ratio_title_suffix})",
        fontsize=14,
    )

    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn", norm=LogNorm(vmin=_RATIO_VMIN, vmax=_RATIO_VMAX)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, extend="min")
    cbar.set_label(_ratio_cbar_label, fontsize=12)
    # Threshold lines (always in frame thanks to fixed colorbar range)
    _ratio_warn = _t.get("ratio_warn")
    _ratio_fail = _t.get("ratio_fail")
    if isinstance(_ratio_warn, (int, float)) and float(_ratio_warn) > 0:
        cbar.ax.axhline(
            y=float(_ratio_warn), color="orange", linewidth=1.5, linestyle="--"
        )
    if isinstance(_ratio_fail, (int, float)) and float(_ratio_fail) > 0:
        cbar.ax.axhline(y=float(_ratio_fail), color="red", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "snr_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save source data
    _src_cols = [
        "roi_id",
        "x1",
        "x2",
        "y1",
        "y2",
        "neg_pct",
        "roi_tx_snr_ratio",
        "roi_tx_snr_log",
        "tissue_coverage",
    ]
    df_src = df_plot[[c for c in _src_cols if c in df_plot.columns]].copy()
    df_src["x1_ds"] = df_src["x1"] / downsample_factor
    df_src["x2_ds"] = df_src["x2"] / downsample_factor
    df_src["y1_ds"] = df_src["y1"] / downsample_factor
    df_src["y2_ds"] = df_src["y2"] / downsample_factor
    if figures_source_dir is not None:
        df_src.to_csv(figures_source_dir / "snr_heatmap.csv", index=False)


def plot_cross_section_concordance(
    df_grid_roi,
    figures_dir,
    figures_source_dir,
    snr_thresholds=None,
):
    """Cross-section concordance: image quality vs transcript SNR per tile.

    Two-panel scatter showing how different quality lenses relate:
      Left  – focus_score vs roi_tx_snr_ratio (optical quality → transcript quality)
      Right – dapi_intensity vs neg_pct (signal strength → noise contamination)

    Only tissue tiles with transcripts are included.
    """
    required = {"focus_score", "roi_tx_snr_ratio", "neg_pct", "dapi_intensity"}
    missing = required - set(df_grid_roi.columns)
    if missing:
        logging.warning(
            "Skipping cross-section concordance plot: missing columns %s", missing
        )
        return

    # Filter to tissue ROIs with transcripts
    df = df_grid_roi.copy()
    if "overlaps_tissue" in df.columns:
        df = df[df["overlaps_tissue"]]
    if "snr_total_tx" in df.columns:
        df = df[df["snr_total_tx"] > 0]
    mask = (
        df["focus_score"].notna()
        & df["roi_tx_snr_ratio"].notna()
        & df["neg_pct"].notna()
        & df["dapi_intensity"].notna()
    )
    df = df[mask]
    if len(df) < 10:
        logging.warning(
            "Skipping cross-section concordance: too few valid tiles (%d)", len(df)
        )
        return

    from scipy.stats import spearmanr

    _t = snr_thresholds or {}

    # Three stacked panels, all identical size for visual consistency.
    # top = Focus vs TxSNR, middle = Image-SNR (Otsu) vs TxSNR, bottom =
    # DAPI vs neg_pct. Each panel 8×4 — matches the §3.2 Focus
    # distribution / Focus-vs-DAPI scatter dimensions for visual rhythm
    # across §3.x figures. Middle panel falls back to a placeholder when
    # per-tile Otsu data is unavailable (legacy samples or upstream SNR
    # skipped).
    fig, axes = plt.subplots(3, 1, figsize=(6, 11))
    plt.subplots_adjust(hspace=0.35)

    # --- Left panel: focus_score vs roi_tx_snr_ratio ---
    ax = axes[0]
    focus_vals = df["focus_score"].values
    snr_vals = df["roi_tx_snr_ratio"].values

    sidx = _subsample_idx(np.ones(len(df), dtype=bool))
    # Color by GMM 2D classification if available
    if "is_blurred_gmm_2d" in df.columns and df["is_blurred_gmm_2d"].notna().any():
        colors = np.where(
            df["is_blurred_gmm_2d"].values[sidx].astype(bool), "#E57373", "#64B5F6"
        )
    else:
        colors = "#64B5F6"

    ax.scatter(
        np.log1p(focus_vals[sidx]),
        np.log1p(snr_vals[sidx]),
        s=8,
        alpha=0.4,
        c=colors,
        edgecolors="none",
        rasterized=True,
    )
    rho_fs, pval_fs = spearmanr(focus_vals, snr_vals)
    ax.set_xlabel("log(1 + Focus Score)", fontsize=12)
    ax.set_ylabel("log(1 + Transcript SNR Ratio)", fontsize=12)
    ax.set_title("Optical Quality vs Transcript Quality", fontsize=13)
    ax.text(
        0.05,
        0.95,
        f"Spearman ρ = {rho_fs:.3f}\nn = {len(df):,} tiles",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )
    # Threshold lines
    ratio_warn = float(_t.get("ratio_warn", 3.0))
    ratio_fail = float(_t.get("ratio_fail", 1.5))
    ax.axhline(
        np.log1p(ratio_warn),
        ls="--",
        lw=1,
        color="orange",
        alpha=0.8,
        label=f"WARN = {ratio_warn}",
    )
    ax.axhline(
        np.log1p(ratio_fail),
        ls="--",
        lw=1,
        color="red",
        alpha=0.8,
        label=f"FAIL = {ratio_fail}",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Middle panel: Image SNR (Otsu) per-tile vs roi_tx_snr_ratio (NEW) ---
    # Direct test of the section's premise: does per-tile image SNR predict
    # per-tile transcript SNR? Otsu-split dB is the per-tile Image-SNR axis
    # (also surfaced as a slide-level metric in §3.4 Detailed metrics).
    ax = axes[1]
    rho_o = float("nan")
    n_valid_otsu = 0
    if "snr_image_otsu_db" in df.columns and df["snr_image_otsu_db"].notna().any():
        _otsu_mask = df["snr_image_otsu_db"].notna() & df["roi_tx_snr_ratio"].notna()
        n_valid_otsu = int(_otsu_mask.sum())
        if n_valid_otsu >= 10:
            otsu_db = df["snr_image_otsu_db"].values
            snr_ratio_arr = df["roi_tx_snr_ratio"].values
            sidx_o = _subsample_idx(_otsu_mask.values)
            if (
                "is_blurred_gmm_2d" in df.columns
                and df["is_blurred_gmm_2d"].notna().any()
            ):
                colors_o = np.where(
                    df["is_blurred_gmm_2d"].values[sidx_o].astype(bool),
                    "#E57373",
                    "#64B5F6",
                )
            else:
                colors_o = "#64B5F6"
            ax.scatter(
                otsu_db[sidx_o],
                np.log1p(snr_ratio_arr[sidx_o]),
                s=8,
                alpha=0.4,
                c=colors_o,
                edgecolors="none",
                rasterized=True,
            )
            rho_o, _ = spearmanr(otsu_db[_otsu_mask], snr_ratio_arr[_otsu_mask])
            ax.set_xlabel("Image SNR — Otsu split (dB)", fontsize=12)
            ax.set_ylabel("log(1 + Transcript SNR Ratio)", fontsize=12)
            ax.set_title("Image SNR vs Transcript Quality", fontsize=13)
            ax.text(
                0.05,
                0.95,
                f"Spearman ρ = {rho_o:.3f}\nn = {n_valid_otsu:,} tiles",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )
            # Reuse Transcript-SNR ratio thresholds from the left panel —
            # both panels share the same y-axis metric.
            ax.axhline(
                np.log1p(ratio_warn),
                ls="--",
                lw=1,
                color="orange",
                alpha=0.8,
                label=f"WARN = {ratio_warn}",
            )
            ax.axhline(
                np.log1p(ratio_fail),
                ls="--",
                lw=1,
                color="red",
                alpha=0.8,
                label=f"FAIL = {ratio_fail}",
            )
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(True, ls="--", alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Image SNR (Otsu) per-tile data\nunavailable (< 10 valid tiles)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.text(
            0.5,
            0.5,
            "Image SNR (Otsu) per-tile data\nunavailable (legacy sample or SNR not computed)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Right panel: dapi_intensity vs neg_pct ---
    ax = axes[2]
    int_vals = df["dapi_intensity"].values
    neg_vals = df["neg_pct"].values

    if "is_blurred_gmm_2d" in df.columns and df["is_blurred_gmm_2d"].notna().any():
        colors_r = np.where(
            df["is_blurred_gmm_2d"].values[sidx].astype(bool), "#E57373", "#64B5F6"
        )
    else:
        colors_r = "#64B5F6"

    ax.scatter(
        np.log1p(int_vals[sidx]),
        neg_vals[sidx],
        s=8,
        alpha=0.4,
        c=colors_r,
        edgecolors="none",
        rasterized=True,
    )
    rho_in, pval_in = spearmanr(int_vals, neg_vals)
    ax.set_xlabel("log(1 + DAPI Intensity)", fontsize=12)
    ax.set_ylabel("Negative Probe Fraction", fontsize=12)
    ax.set_title("Signal Strength vs Noise Contamination", fontsize=13)
    ax.text(
        0.05,
        0.95,
        f"Spearman ρ = {rho_in:.3f}\nn = {len(df):,} tiles",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )
    neg_warn = float(_t.get("neg_pct_warn", 0.15))
    neg_fail = float(_t.get("neg_pct_fail", 0.30))
    ax.axhline(
        neg_warn, ls="--", lw=1, color="orange", alpha=0.8, label=f"WARN = {neg_warn}"
    )
    ax.axhline(
        neg_fail, ls="--", lw=1, color="red", alpha=0.8, label=f"FAIL = {neg_fail}"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)

    # Add GMM legend if coloured
    if "is_blurred_gmm_2d" in df.columns and df["is_blurred_gmm_2d"].notna().any():
        from matplotlib.patches import Patch

        for a in axes:
            handles = a.get_legend_handles_labels()[0]
            handles.extend(
                [
                    Patch(facecolor="#64B5F6", label="In-Focus (GMM)"),
                    Patch(facecolor="#E57373", label="Blurred (GMM)"),
                ]
            )
            a.legend(
                handles=handles,
                fontsize=8,
                loc="lower right" if a is axes[0] else "upper right",
            )

    plt.tight_layout()
    plt.savefig(figures_dir / "cross_section_concordance.png", dpi=200)
    plt.savefig(
        figures_dir / "cross_section_concordance.pdf", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Source data
    src_cols = [
        "roi_id",
        "focus_score",
        "dapi_intensity",
        "roi_tx_snr_ratio",
        "neg_pct",
        "snr_image_otsu_db",
    ]
    if "is_blurred_gmm_2d" in df.columns:
        src_cols.append("is_blurred_gmm_2d")
    if "tissue_coverage" in df.columns:
        src_cols.append("tissue_coverage")
    if figures_source_dir is not None:
        df[[c for c in src_cols if c in df.columns]].to_csv(
            figures_source_dir / "cross_section_concordance.csv", index=False
        )
    logging.info(
        "Cross-section concordance: focus-vs-SNR ρ=%.3f, otsu-vs-SNR ρ=%.3f (n=%d), intensity-vs-neg ρ=%.3f (%d tiles)",
        rho_fs,
        rho_o,
        n_valid_otsu,
        rho_in,
        len(df),
    )


def plot_roi_focus_vs_intensity(
    df_grid_roi, figures_dir, figures_source_dir, threshold=-1.0
):
    """
    Create scatter plot showing focus score vs raw DAPI intensity for tile threshold analysis.
    Uses log scale for intensity to better visualize the relationship.

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: focus_score, focus_score_norm, raw_intensity (or dapi_intensity)
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    threshold : float, optional
        Threshold for normalized focus score (default: -1.0)
    """
    # Use dapi_intensity if available, otherwise raw_intensity
    intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df_grid_roi.columns else "raw_intensity"
    )

    # Calculate log10 of intensity (add small epsilon to avoid log(0))
    intensity_values = df_grid_roi[intensity_col].values
    log_intensity = np.log10(
        intensity_values + 1e-10
    )  # Add small epsilon to handle any zeros

    # Phase v5: robust outlier handling for the rendered scatter.
    # (A) Filter to tissue tiles (tissue_coverage > 0.5) — empty / non-tissue
    #     tiles have intensity ≈ 0 → log10(1e-10) = -10, which dominates the
    #     auto-scaled xlim. CSV still writes unfiltered data.
    # (B) Compute 1st-99th percentile xlim bounds as a robustness floor so a
    #     single saturated tissue tile (fold, debris) doesn't compress the rest.
    if "tissue_coverage" in df_grid_roi.columns:
        _tissue_mask = df_grid_roi["tissue_coverage"].values > 0.5
    else:
        _tissue_mask = np.ones(len(df_grid_roi), dtype=bool)

    # Single-panel figure: normalized focus score vs log DAPI intensity,
    # coloured by GMM 2D blurry/in-focus classification. Figsize matches the
    # Focus score distribution histogram (figsize=(8, 4)) below for visual
    # rhythm. Previously this was a 2-panel figure where the left panel
    # showed raw CCFS vs intensity coloured by *normalised* focus score —
    # redundant with the right panel since the y-axis there directly
    # encodes the same metric.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # Filter out invalid values
    valid_mask_plot2 = (
        np.isfinite(log_intensity)
        & np.isfinite(df_grid_roi["focus_score_norm"].values)
        & _tissue_mask
    )

    if valid_mask_plot2.sum() == 0:
        logging.warning("  Warning: No valid data points for plot 2")
        ax.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        # Color by GMM 2D classification if available, otherwise fall back to threshold
        has_gmm_2d = "is_blurred_gmm_2d" in df_grid_roi.columns

        if has_gmm_2d:
            # Color by GMM 2D classification
            valid_gmm_mask = valid_mask_plot2 & df_grid_roi["is_blurred_gmm_2d"].notna()
            if valid_gmm_mask.sum() > 0:
                sidx = _subsample_idx(valid_gmm_mask)
                is_blurred_sub = (
                    df_grid_roi["is_blurred_gmm_2d"].values[sidx].astype(bool)
                )
                colors = np.where(is_blurred_sub, "red", "blue")
                ax.scatter(
                    log_intensity[sidx],
                    df_grid_roi["focus_score_norm"].values[sidx],
                    alpha=0.5,
                    s=10,
                    c=colors,
                    edgecolors="none",
                    rasterized=True,
                )

                # Phase v5: percentile-clipped xlim (see comment at top of function).
                _x_lo, _x_hi = np.nanpercentile(log_intensity[valid_gmm_mask], [1, 99])
                ax.set_xlim(_x_lo - 0.1, _x_hi + 0.1)
                ax.set_ylim(
                    df_grid_roi.loc[valid_gmm_mask, "focus_score_norm"].min() - 0.5,
                    df_grid_roi.loc[valid_gmm_mask, "focus_score_norm"].max() + 0.5,
                )

                # Add text annotation for GMM 2D (counts from full data, not subsample)
                n_blurred = df_grid_roi.loc[valid_gmm_mask, "is_blurred_gmm_2d"].sum()
                n_in_focus = (
                    ~df_grid_roi.loc[valid_gmm_mask, "is_blurred_gmm_2d"]
                ).sum()
                pct_blurred = (
                    n_blurred / valid_gmm_mask.sum() * 100
                    if valid_gmm_mask.sum() > 0
                    else 0
                )
                ax.text(
                    0.05,
                    0.95,
                    f"2D GMM Classification\nRed: Blurred ({n_blurred:,}, {pct_blurred:.1f}%)\nBlue: In-Focus ({n_in_focus:,}, {100 - pct_blurred:.1f}%)",
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                )
                ax.set_title(
                    "Normalized Focus Score vs Log10(Raw DAPI Intensity) (2D GMM Classification)",
                    fontsize=12,
                )
        else:
            # Fallback to threshold method
            sidx = _subsample_idx(valid_mask_plot2)
            scores_sub = df_grid_roi["focus_score_norm"].values[sidx]
            colors = np.where(scores_sub <= threshold, "red", "blue")
            ax.scatter(
                log_intensity[sidx],
                scores_sub,
                alpha=0.5,
                s=10,
                c=colors,
                edgecolors="none",
                rasterized=True,
            )

            # Phase v5: percentile-clipped xlim (see comment at top of function).
            _x_lo, _x_hi = np.nanpercentile(log_intensity[valid_mask_plot2], [1, 99])
            ax.set_xlim(_x_lo - 0.1, _x_hi + 0.1)
            ax.set_ylim(
                df_grid_roi.loc[valid_mask_plot2, "focus_score_norm"].min() - 0.5,
                df_grid_roi.loc[valid_mask_plot2, "focus_score_norm"].max() + 0.5,
            )

            # Add threshold line
            ax.axhline(
                y=threshold,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({threshold})",
            )

            # Add text annotation for threshold
            ax.text(
                0.05,
                0.95,
                f"Threshold: {threshold}\nRed: Blurred (≤{threshold})\nBlue: In-Focus (>{threshold})",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )
            ax.set_title(
                "Normalized Focus Score vs Log10(Raw DAPI Intensity) (Thresholded)",
                fontsize=12,
            )

    ax.set_xlabel("log₁₀(Raw DAPI Intensity, 16-bit counts)", fontsize=12)
    ax.set_ylabel("Normalised focus score", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)

    # Pin axes-box position so the rendered plot box matches the §3.2 Focus
    # score distribution figure exactly. Both figures use figsize=(8, 4) and
    # the same subplots_adjust margins; identical (left, right, top, bottom)
    # ⇒ identical plot-box position regardless of y-tick label width.
    plt.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.18)
    plt.savefig(
        figures_dir / "roi_focus_vs_intensity.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(figures_dir / "roi_focus_vs_intensity.png", dpi=300)
    plt.close(fig)

    # Save data as CSV (include both raw and log intensity)
    df_scatter = df_grid_roi[
        ["focus_score", "focus_score_norm", intensity_col, "tissue_coverage"]
    ].copy()
    df_scatter["log10_intensity"] = log_intensity
    df_scatter["is_low_nuclear_texture"] = df_scatter["focus_score_norm"] <= threshold
    if figures_source_dir is not None:
        df_scatter.to_csv(
            figures_source_dir / "roi_focus_vs_intensity.csv", index=False
        )

    # Print correlation statistics (using log intensity)
    correlation = np.corrcoef(log_intensity, df_grid_roi["focus_score_norm"])[0, 1]
    logging.info(
        f"  Correlation (log10(intensity) vs normalized focus score): {correlation:.4f}"
    )


def plot_roi_focus_distribution(
    df_grid_roi, figures_dir, figures_source_dir, threshold=-1.0
):
    """Histogram of normalized focus scores on tissue-filtered tiles, with
    GMM 2D binary classification overlaid (red = blurred, blue = in-focus).

    Phase v5 (2026-05-15): simplified from the previous dual-panel (raw +
    normalized) × dual-variant (all-tiles + tissue-filtered) layout. The
    all-tiles view was misleading — background tiles are force-classified
    blurry by the intensity-floor rule, so the all-tiles histogram showed
    a "blurry mass" that was not actually a GMM decision. The raw-score
    panel was redundant with the normalized panel for QC interpretation.
    Single panel: tissue-filtered, normalized scores, GMM-colored.

    Emits roi_focus_distribution_tissue.png — only when tissue_coverage
    column is present with at least one tile above the 0.5 threshold.

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: focus_score, focus_score_norm,
        is_blurred_gmm_2d, tissue_coverage (required).
    figures_dir : Path
        Directory to save figures.
    figures_source_dir : Path
        Directory to save source data.
    threshold : float, optional
        Normalized-score fallback threshold (default: -1.0). Used only when
        is_blurred_gmm_2d column is absent.
    """
    if "tissue_coverage" not in df_grid_roi.columns:
        logging.info("  No tissue_coverage column; skipping focus score distribution.")
        return

    df_in = df_grid_roi[df_grid_roi["tissue_coverage"] > 0.5]
    if len(df_in) == 0:
        logging.info(
            "  No tissue tiles (tissue_coverage > 0.5); skipping focus score distribution."
        )
        return

    save_stem = "roi_focus_distribution_tissue"
    # figsize chosen to be more landscape-y than the previous (10, 6) so the
    # histogram doesn't dominate the §3.2 visual flow against the adjacent
    # spatial focus heatmap. Width reduced ~20%, height reduced ~33%.
    fig, ax = plt.subplots(figsize=(7, 4))

    has_gmm_2d = "is_blurred_gmm_2d" in df_in.columns
    if has_gmm_2d:
        blurred = df_in[df_in["is_blurred_gmm_2d"]]
        in_focus = df_in[~df_in["is_blurred_gmm_2d"]]
        title_suffix = "2D GMM Classification"
    else:
        blurred = df_in[df_in["focus_score_norm"] <= threshold]
        in_focus = df_in[df_in["focus_score_norm"] > threshold]
        title_suffix = f"Threshold: {threshold}"

    ax.hist(
        [blurred["focus_score_norm"], in_focus["focus_score_norm"]],
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color=["red", "blue"],
        label=["Blurred", "In-Focus"],
        stacked=False,
    )

    if not has_gmm_2d:
        ax.axvline(
            x=threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )

    ax.set_xlabel("Focus Score (Normalized)", fontsize=12)
    ax.set_ylabel("Number of tiles", fontsize=12)
    ax.set_title(
        f"Distribution of Normalized Focus Scores ({title_suffix})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    mean_norm = df_in["focus_score_norm"].mean()
    median_norm = df_in["focus_score_norm"].median()
    pct_blurred = len(blurred) / len(df_in) * 100 if len(df_in) > 0 else 0
    ax.text(
        0.95,
        0.95,
        f"Mean: {mean_norm:.4f}\nMedian: {median_norm:.4f}\nBlurred: {pct_blurred:.1f}%\nn = {len(df_in):,} tiles",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )

    # Pin axes-box position so the rendered plot box matches the §3.2 Focus
    # score vs DAPI intensity figure exactly (same figsize, same margins).
    plt.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.18)
    plt.savefig(figures_dir / f"{save_stem}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / f"{save_stem}.png", dpi=300)
    plt.close(fig)

    # Source CSV mirrors the rendered subset.
    df_dist = df_in[["focus_score", "focus_score_norm"]].copy()
    df_dist["is_low_nuclear_texture"] = df_dist["focus_score_norm"] <= threshold
    if figures_source_dir is not None:
        df_dist.to_csv(figures_source_dir / f"{save_stem}.csv", index=False)

    logging.info("  Focus score distribution summary (tissue tiles):")
    logging.info(
        f"    Normalized focus score - Mean: {mean_norm:.4f}, Median: {median_norm:.4f}"
    )
    logging.info(f"    Tiles blurred: {len(blurred)} ({pct_blurred:.1f}%)")
    logging.info(f"    Tiles in-focus: {len(in_focus)} ({100 - pct_blurred:.1f}%)")


def calculate_roi_intensities(xoa_morphology_files, df_grid_roi):
    """
    Calculate mean intensity per tile for Boundary and IntRNA channels.

    Note: If intensities are already calculated in calculate_roi_focusscore() (new behavior),
    this function will detect that and return the DataFrame unchanged.

    Parameters:
    -----------
    xoa_morphology_files : list
        List of paths to morphology image files
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: roi_id, x1, x2, y1, y2, raw_intensity (DAPI)
        If dapi_intensity, boundary_intensity, intrna_intensity already exist, returns unchanged.

    Returns:
    --------
    pandas DataFrame
        DataFrame with columns:
        - dapi_intensity: Mean DAPI intensity per tile
        - boundary_intensity: Mean boundary intensity per tile (if available)
        - intrna_intensity: Mean IntRNA intensity per tile (if available)
    """

    # Check if intensities are already calculated (new behavior in calculate_roi_focusscore)
    # Note: boundary_intensity may be NaN if only DAPI channel is available, but column should exist
    if "dapi_intensity" in df_grid_roi.columns:
        # Intensities already calculated, just return the DataFrame
        logging.info(
            "  Intensities already calculated in calculate_roi_focusscore(), skipping recalculation"
        )
        return df_grid_roi.copy()

    # Legacy behavior: calculate intensities if not already present
    # Load channels robustly from multi-channel stack or split channel files.
    dapi_image, boundary_image, intrna_image = _load_morphology_channels(
        xoa_morphology_files, level=0
    )
    has_boundary = boundary_image is not None
    has_intrna = intrna_image is not None

    # Create output DataFrame
    df_roi_intensities = df_grid_roi.copy()
    if "dapi_intensity" not in df_roi_intensities.columns:
        df_roi_intensities["dapi_intensity"] = df_roi_intensities.get(
            "raw_intensity", np.nan
        )  # Rename for consistency

    # Calculate intensities for each ROI
    n_rois = len(df_grid_roi)

    # Pre-extract ROI coordinate arrays for vectorized access
    x1_arr = df_grid_roi["x1"].values.astype(int)
    x2_arr = df_grid_roi["x2"].values.astype(int)
    y1_arr = df_grid_roi["y1"].values.astype(int)
    y2_arr = df_grid_roi["y2"].values.astype(int)

    # Boundary intensity
    if has_boundary:
        boundary_intensities = np.empty(n_rois, dtype=np.float64)
        for i in range(n_rois):
            boundary_intensities[i] = np.mean(
                boundary_image[y1_arr[i] : y2_arr[i], x1_arr[i] : x2_arr[i]]
            )
        df_roi_intensities["boundary_intensity"] = boundary_intensities
        del boundary_image
    else:
        df_roi_intensities["boundary_intensity"] = np.nan
        logging.warning(
            "Warning: Boundary channel not available, setting boundary_intensity to NaN"
        )

    # IntRNA intensity
    if has_intrna:
        intrna_intensities = np.empty(n_rois, dtype=np.float64)
        for i in range(n_rois):
            intrna_intensities[i] = np.mean(
                intrna_image[y1_arr[i] : y2_arr[i], x1_arr[i] : x2_arr[i]]
            )
        df_roi_intensities["intrna_intensity"] = intrna_intensities
    else:
        df_roi_intensities["intrna_intensity"] = np.nan
        logging.warning(
            "Warning: IntRNA channel not available, setting intrna_intensity to NaN"
        )

    # Clean up
    del dapi_image

    return df_roi_intensities


def assess_raw_intensity_quality(
    df_roi_intensities,
    dapi_threshold_critical=500,
    boundary_threshold_critical=100,
    intrna_threshold_critical=300,
    min_tissue_coverage=ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC,
    channel_pct_thresholds=None,
):
    """
    Assess raw intensity quality from per-tile intensities.
    Handles missing channels gracefully (NaN values).

    Note: This function can apply an additional tissue coverage filter before
    intensity QC. If ``tissue_coverage`` is present, only tiles with
    ``tissue_coverage >= min_tissue_coverage`` are used for threshold-based
    percentage calculations. This reduces false FAILs from low-content/background
    edge tiles that technically overlap tissue but are mostly non-tissue.

    Threshold Method:
    -----------------
    Each channel has a single absolute minimum intensity threshold (``intensity_critical``
    from YAML).  The metric is the fraction of tissue tiles whose mean intensity falls
    below that threshold: ``pct_tissue_roi_below_critical``.

    Critical thresholds (defaults):
      - DAPI: 500  — ~0.76% of 16-bit max; typical well-stained range 2000-8000
      - Boundary: 100  — membrane markers 5-10× lower than DAPI
      - IntRNA: 300  — rRNA markers 2-3× lower than DAPI

    Quality Status Determination (WARN-only since 2026-06-22):
    ----------------------------------------------------------
    Per-channel verdict is driven by ``pct_tissue_roi_below_critical`` compared
    against the YAML ``intensity_warn`` fraction (converted to %). There is no
    FAIL tier — intensity does not track quality after XOA 4.0, so a dim sample
    is flagged for review, never hard-failed on intensity alone:
      - **warn**: pct_tissue_roi_below_critical > warn%  OR  mean < critical_threshold
      - **pass**: otherwise
    The ``critical_threshold`` is XOA-version-specific (selected by the caller).

    Parameters:
    -----------
    df_roi_intensities : pandas DataFrame
        DataFrame with per-tile intensities (dapi_intensity, boundary_intensity, intrna_intensity).
    dapi_threshold_critical : float, optional
        Critical DAPI threshold (default: 500).
    boundary_threshold_critical : float, optional
        Critical boundary threshold (default: 100).
    intrna_threshold_critical : float, optional
        Critical IntRNA threshold (default: 300).

    Returns:
    --------
    dict
        Per-channel dict with:
        - mean, median, p10, p25, p75, p90: Intensity statistics
        - critical_threshold: Absolute minimum intensity threshold
        - n_tissue_rois_below_critical: Count of tissue tiles below threshold
        - pct_tissue_roi_below_critical: Percentage of tissue tiles below threshold
        - pct_warn_threshold, pct_fail_threshold: YAML-derived population thresholds (%)
        - quality_status: 'pass', 'warn', 'fail', or 'not_available'
    """
    stats = {}
    _cpct = channel_pct_thresholds or {}
    df_work = df_roi_intensities
    n_rois_input = int(len(df_roi_intensities))
    if "tissue_coverage" in df_roi_intensities.columns:
        df_work = df_roi_intensities[
            df_roi_intensities["tissue_coverage"] >= float(min_tissue_coverage)
        ]
    n_rois_used = int(len(df_work))
    stats["_intensity_qc_scope"] = {
        "n_rois_input": n_rois_input,
        "n_rois_used": n_rois_used,
        "min_tissue_coverage": float(min_tissue_coverage),
        "uses_tissue_coverage_filter": "tissue_coverage" in df_roi_intensities.columns,
    }

    # YAML key → function-local name mapping
    _yaml_keys = {"dapi": "DAPI", "boundary": "boundary", "intrna": "intRNA"}

    for channel, critical_threshold in [
        ("dapi", dapi_threshold_critical),
        ("boundary", boundary_threshold_critical),
        ("intrna", intrna_threshold_critical),
    ]:
        # Per-channel WARN fraction from YAML (or default). WARN-only since
        # 2026-06-22 (qc_drift_analysis): intensity does not track quality
        # post-XOA-4.0, so there is no FAIL tier — `intensity_fail` is no longer
        # read or applied.
        _ch = _cpct.get(_yaml_keys.get(channel, channel)) or {}
        pct_warn_frac = float(_ch.get("intensity_warn", 0.15))
        col_name = f"{channel}_intensity"
        intensities = df_work[col_name].values

        # Check if channel is available (not all NaN)
        if np.all(np.isnan(intensities)):
            stats[channel] = {
                "mean": np.nan,
                "median": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "critical_threshold": float(critical_threshold),
                "n_tissue_rois_below_critical": 0,
                "pct_tissue_roi_below_critical": 0.0,
                "quality_status": "not_available",
            }
            continue

        # Filter out NaN values for calculations
        intensities_valid = intensities[~np.isnan(intensities)]

        if len(intensities_valid) == 0:
            stats[channel] = {
                "mean": np.nan,
                "median": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "critical_threshold": float(critical_threshold),
                "n_tissue_rois_below_critical": 0,
                "pct_tissue_roi_below_critical": 0.0,
                "quality_status": "not_available",
            }
            continue

        # Calculate statistics (using valid values only)
        mean_int = np.mean(intensities_valid)
        median_int = np.median(intensities_valid)
        p10 = np.percentile(intensities_valid, 10)
        p25 = np.percentile(intensities_valid, 25)
        p75 = np.percentile(intensities_valid, 75)
        p90 = np.percentile(intensities_valid, 90)

        # Count tissue tiles below the single critical intensity threshold
        n_below = int(np.sum(intensities_valid < critical_threshold))
        pct_tissue_roi_below_critical = (n_below / len(intensities_valid)) * 100

        # Determine quality status — WARN-only (no FAIL tier). Both the
        # prevalence case (too many tiles below the floor) and the mean-below-floor
        # case cap at WARN: a dim sample is flagged for review, never hard-failed
        # on intensity alone.
        pct_warn = pct_warn_frac * 100.0
        if pct_tissue_roi_below_critical > pct_warn or mean_int < critical_threshold:
            quality_status = "warn"
        else:
            quality_status = "pass"

        stats[channel] = {
            "mean": float(mean_int),
            "median": float(median_int),
            "p10": float(p10),
            "p25": float(p25),
            "p75": float(p75),
            "p90": float(p90),
            "critical_threshold": float(critical_threshold),
            "pct_warn_threshold": float(pct_warn),
            # None signals "advisory / no FAIL tier" to the report renderer.
            "pct_fail_threshold": None,
            "n_tissue_rois_below_critical": n_below,
            "pct_tissue_roi_below_critical": float(pct_tissue_roi_below_critical),
            "quality_status": quality_status,
        }

    # Overall quality — WARN-only (intensity never fails the sample on its own).
    statuses = [stats[ch]["quality_status"] for ch in ["dapi", "boundary", "intrna"]]
    if n_rois_used == 0:
        overall_quality = "not_available"
    elif "warn" in statuses:
        overall_quality = "warn"
    else:
        overall_quality = "pass"

    stats["overall_quality"] = overall_quality

    return stats


def plot_focus_score_vs_laplacian(df_grid_roi, figures_dir, figures_source_dir):
    """
    Compare Laplacian variance vs original focus score (std²/mean).

    Creates scatter plot and distribution histograms comparing the two focus metrics.

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: dapi_focus_score, dapi_lap_var
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    """
    # Check if required columns exist
    if "dapi_lap_var" not in df_grid_roi.columns:
        logging.warning(
            "  Warning: dapi_lap_var column not found, skipping Laplacian comparison plots"
        )
        return

    if "dapi_focus_score" not in df_grid_roi.columns:
        # Fallback to generic focus_score
        if "focus_score" not in df_grid_roi.columns:
            logging.warning(
                "  Warning: No focus score column found, skipping Laplacian comparison plots"
            )
            return
        focus_col = "focus_score"
    else:
        focus_col = "dapi_focus_score"

    # Filter out NaN values
    valid_mask = df_grid_roi["dapi_lap_var"].notna() & df_grid_roi[focus_col].notna()
    df_valid = df_grid_roi[valid_mask].copy()

    if len(df_valid) == 0:
        logging.warning(
            "  Warning: No valid data for Laplacian comparison, skipping plots"
        )
        return

    # Scatter plot: log1p(focus_score) vs log1p(laplacian_variance)
    logging.info("Generating Figure: Focus score vs Laplacian variance comparison...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Color by GMM 2D classification if available
    has_gmm_2d = "is_blurred_gmm_2d" in df_valid.columns
    focus_vals = df_valid[focus_col].values
    lap_vals = df_valid["dapi_lap_var"].values
    if has_gmm_2d:
        # Filter to valid mask for GMM 2D
        valid_gmm_mask = df_valid["is_blurred_gmm_2d"].notna().values
        if valid_gmm_mask.sum() > 0:
            sidx = _subsample_idx(valid_gmm_mask)
            is_blurred_sub = df_valid["is_blurred_gmm_2d"].values[sidx].astype(bool)
            colors = np.where(is_blurred_sub, "red", "blue")
            ax.scatter(
                np.log1p(focus_vals[sidx]),
                np.log1p(lap_vals[sidx]),
                s=5,
                alpha=0.3,
                c=colors,
                rasterized=True,
            )
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="blue", label="In-Focus (2D GMM)"),
                Patch(facecolor="red", label="Blurred (2D GMM)"),
            ]
            ax.legend(handles=legend_elements, fontsize=10)
        else:
            sidx = _subsample_idx(np.ones(len(df_valid), dtype=bool))
            ax.scatter(
                np.log1p(focus_vals[sidx]),
                np.log1p(lap_vals[sidx]),
                s=5,
                alpha=0.3,
                rasterized=True,
            )
    else:
        sidx = _subsample_idx(np.ones(len(df_valid), dtype=bool))
        ax.scatter(
            np.log1p(focus_vals[sidx]),
            np.log1p(lap_vals[sidx]),
            s=5,
            alpha=0.3,
            rasterized=True,
        )

    ax.set_xlabel("log(1 + std²/mean) (DAPI)", fontsize=12)
    ax.set_ylabel("log(1 + Laplacian variance) (DAPI)", fontsize=12)
    if has_gmm_2d:
        ax.set_title(
            "Focus Score vs Laplacian Variance (2D GMM Classification)", fontsize=14
        )
    else:
        ax.set_title("Focus Score vs Laplacian Variance", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save to methodology assessment folder (subfolder of figures_dir)
    methodology_figures_dir = figures_dir / "figures_methodology_assessment"
    methodology_figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        methodology_figures_dir / "focus_score_vs_laplacian.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        methodology_figures_dir / "focus_score_vs_laplacian.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df_valid[focus_col], bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_title("std²/mean (DAPI Focus Score)", fontsize=12)
    axes[0].set_xlabel("Focus Score (std²/mean)", fontsize=11)
    axes[0].set_ylabel("Number of tiles", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(df_valid["dapi_lap_var"], bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_title("Laplacian Variance (DAPI)", fontsize=12)
    axes[1].set_xlabel("Laplacian Variance", fontsize=11)
    axes[1].set_ylabel("Number of tiles", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    # Save to methodology assessment folder
    plt.savefig(
        methodology_figures_dir / "focus_score_vs_laplacian_distributions.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        methodology_figures_dir / "focus_score_vs_laplacian_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save source data
    df_comparison = df_valid[[focus_col, "dapi_lap_var"]].copy()
    df_comparison["log1p_focus_score"] = np.log1p(df_comparison[focus_col])
    df_comparison["log1p_lap_var"] = np.log1p(df_comparison["dapi_lap_var"])
    if figures_source_dir is not None:
        df_comparison.to_csv(
            figures_source_dir / "focus_score_vs_laplacian.csv", index=False
        )

    # Calculate correlation
    correlation = np.corrcoef(
        np.log1p(df_valid[focus_col]), np.log1p(df_valid["dapi_lap_var"])
    )[0, 1]
    logging.info(f"  Correlation (log1p): {correlation:.4f}")


def plot_intensity_assessment(
    df_roi_intensities, intensity_stats, small0, figures_dir, figures_source_dir
):
    """
    Create visualization of per-tile intensity distributions and spatial heatmaps.

    Note: Intensity distributions are calculated from tissue-filtered tiles only
    (tiles with any tissue overlap, i.e., coverage > 0%). Background/empty regions
    of the image are excluded from the distributions.

    Parameters:
    -----------
    df_roi_intensities : pandas DataFrame
        DataFrame with per-tile intensities and coordinates (tissue-filtered tiles only)
    intensity_stats : dict
        Dictionary from assess_raw_intensity_quality()
    small0 : numpy.ndarray
        Downsampled DAPI image (for spatial overlay)
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    """
    downsample_factor = 8
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Calculate ROI centers
    df_roi_intensities["center_x"] = (
        df_roi_intensities["x1"] + df_roi_intensities["x2"]
    ) / 2
    df_roi_intensities["center_y"] = (
        df_roi_intensities["y1"] + df_roi_intensities["y2"]
    ) / 2

    channels = [
        ("dapi", "DAPI", intensity_stats["dapi"]),
        ("boundary", "Boundary", intensity_stats["boundary"]),
        ("intrna", "IntRNA", intensity_stats["intrna"]),
    ]

    # p99 of the DAPI background is the same for all three spatial panels;
    # compute it once over the full ~86 Mpx small0 instead of once per channel
    # inside the loop. PIXEL-IDENTICAL.
    _small0_vmax = np.percentile(small0, 99)

    for col_idx, (channel, channel_name, stats) in enumerate(channels):
        intensity_col = f"{channel}_intensity"
        intensities = df_roi_intensities[intensity_col].values

        # Check if channel is available
        is_available = not np.all(np.isnan(intensities))
        intensities_valid = (
            intensities[~np.isnan(intensities)] if is_available else np.array([])
        )

        # Row 1: Distribution plots
        ax = axes[0, col_idx]

        if is_available and len(intensities_valid) > 0:
            # Phase v5: clip the histogram x-range to the 99.9th percentile
            # to avoid a single saturated/outlier tile compressing the bulk
            # distribution. Threshold lines (intensity_critical) and stats
            # text (mean/median/p10/p90) are pulled from the FULL-data
            # `stats` dict and are unchanged by this clip.
            _p_hi_intensity = np.nanpercentile(intensities_valid, 99.9)
            _clipped_intensity = intensities_valid[intensities_valid <= _p_hi_intensity]
            ax.hist(_clipped_intensity, bins=50, alpha=0.7, edgecolor="black")

            # Add threshold lines (only if thresholds are valid)
            if not np.isnan(stats["critical_threshold"]):
                ax.axvline(
                    stats["critical_threshold"],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Min intensity ({stats['critical_threshold']:.0f})",
                )

            # No green "optimal range" band — only the red critical threshold
            # line (from YAML intensity_critical) is shown to avoid implying an
            # uncalibrated optimal window.

            # Add statistics text
            if not np.isnan(stats["mean"]):
                stats_text = f"Mean: {stats['mean']:.0f}\nMedian: {stats['median']:.0f}\nP10: {stats['p10']:.0f}\nP90: {stats['p90']:.0f}"
                ax.text(
                    0.98,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                f"{channel_name} channel\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )

        ax.set_xlabel(f"{channel_name} Intensity", fontsize=12)
        ax.set_ylabel("Number of tiles", fontsize=12)
        status_text = (
            stats["quality_status"].upper()
            if stats["quality_status"] != "not_available"
            else "NOT AVAILABLE"
        )
        ax.set_title(
            f"{channel_name} Intensity Distribution\n(Status: {status_text})",
            fontsize=14,
        )
        if is_available and len(intensities_valid) > 0:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 2: Spatial heatmaps
        ax = axes[1, col_idx]

        # Get image dimensions for setting axes limits
        img_height, img_width = small0.shape

        # Show downsampled DAPI as background (optional, light)
        _imshow_thumb(
            ax,
            small0,
            cmap="Greys_r",
            vmax=_small0_vmax,
            alpha=0.3,
            aspect="auto",
            origin="upper",
            extent=[0, img_width, img_height, 0],
        )

        # Scale coordinates to downsampled space
        x_scaled = df_roi_intensities["center_x"] / downsample_factor
        y_scaled = df_roi_intensities["center_y"] / downsample_factor

        if is_available and len(intensities_valid) > 0:
            # Create scatter plot heatmap (only for valid intensities)
            valid_mask = ~np.isnan(intensities)

            # Filter coordinates to be within image bounds
            x_vals = x_scaled.values if hasattr(x_scaled, "values") else x_scaled
            y_vals = y_scaled.values if hasattr(y_scaled, "values") else y_scaled
            in_bounds = (
                (x_vals >= 0)
                & (x_vals < img_width)
                & (y_vals >= 0)
                & (y_vals < img_height)
            )
            valid_mask = valid_mask & in_bounds

            if valid_mask.sum() > 0:
                # Phase v6: fixed per-channel colorbar caps (cross-sample
                # comparable). Calibrated against 9 tissues — see the
                # _INTENSITY_DISPLAY_CAP module constant. vmin=0 anchors the
                # dark end to absolute zero so dim samples render dim and
                # bright samples fill the range; mirrors the fixed-vmin/vmax
                # convention already used by plot_snr_roi_heatmap.
                _vmax_intensity = _INTENSITY_DISPLAY_CAP[channel]
                # Use hexbin for spatial heatmap — bins ALL data, renders O(bins) not O(N)
                hb = ax.hexbin(
                    x_vals[valid_mask],
                    y_vals[valid_mask],
                    C=intensities[valid_mask],
                    reduce_C_function=np.mean,
                    gridsize=100,
                    cmap="viridis",
                    mincnt=1,
                    vmin=0,
                    vmax=_vmax_intensity,
                    # rasterized: the PDF embeds a raster of the hex grid instead of
                    # ~30-60k vector polygons per panel. Pixel-identical at dpi 300,
                    # but avoids re-rendering the vector layer for BOTH .pdf and .png.
                    rasterized=True,
                )

                # extend="max" — upper triangle marks hex cells whose mean
                # exceeds the cap (bright tissues will saturate routinely).
                cbar = plt.colorbar(hb, ax=ax, extend="max")
                cbar.set_label(f"{channel_name} Intensity", fontsize=10)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data points\nwithin image bounds",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
                )
        else:
            ax.text(
                0.5,
                0.5,
                f"{channel_name} channel\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )

        # Set axes limits explicitly
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Reversed because origin='upper'
        ax.set_xlabel("X coordinate (downsampled)", fontsize=12)
        ax.set_ylabel("Y coordinate (downsampled)", fontsize=12)
        ax.set_title(f"{channel_name} Intensity Spatial Heatmap", fontsize=14)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(figures_dir / "intensity_assessment.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "intensity_assessment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save source data
    if figures_source_dir is None:
        return
    df_roi_intensities.to_csv(
        figures_source_dir / "intensity_assessment.csv", index=False
    )

    # Save statistics summary
    df_stats = pd.DataFrame(
        {
            "channel": ["dapi", "boundary", "intrna"],
            "mean_intensity": [
                intensity_stats["dapi"]["mean"],
                intensity_stats["boundary"]["mean"],
                intensity_stats["intrna"]["mean"],
            ],
            "median_intensity": [
                intensity_stats["dapi"]["median"],
                intensity_stats["boundary"]["median"],
                intensity_stats["intrna"]["median"],
            ],
            "p10_intensity": [
                intensity_stats["dapi"]["p10"],
                intensity_stats["boundary"]["p10"],
                intensity_stats["intrna"]["p10"],
            ],
            "p90_intensity": [
                intensity_stats["dapi"]["p90"],
                intensity_stats["boundary"]["p90"],
                intensity_stats["intrna"]["p90"],
            ],
            "critical_threshold": [
                intensity_stats["dapi"]["critical_threshold"],
                intensity_stats["boundary"]["critical_threshold"],
                intensity_stats["intrna"]["critical_threshold"],
            ],
            "pct_tissue_roi_below_critical": [
                intensity_stats["dapi"]["pct_tissue_roi_below_critical"],
                intensity_stats["boundary"]["pct_tissue_roi_below_critical"],
                intensity_stats["intrna"]["pct_tissue_roi_below_critical"],
            ],
            "quality_status": [
                intensity_stats["dapi"]["quality_status"],
                intensity_stats["boundary"]["quality_status"],
                intensity_stats["intrna"]["quality_status"],
            ],
        }
    )
    df_stats.to_csv(
        figures_source_dir / "intensity_assessment_statistics.csv", index=False
    )


def generate_roi_figures(
    data,
    small0,
    small1,
    small2,
    distance_map,
    distance_map2,
    whole_sample,
    holes,
    dense_intensity_regions,
    df_grid_roi,
    df_roi_intensities,
    intensity_stats,
    xoa_morphology_files,
    focus_maps=None,
    focus_heatmap=None,
    snr_thresholds=None,
    multistain_whole_sample=None,
    multistain_distance_map=None,
    multistain_distance_map2=None,
    figure_source_tables=False,
):
    """
    Generate cell-independent tile-based figures using multithreading.

    Parameters:
    -----------
    data : dict
        Dictionary with 'figures_dir' key
    small0, small1, small2 : numpy.ndarray
        Downsampled morphology images
    distance_map, distance_map2 : numpy.ndarray
        Distance maps (edge and holes)
    whole_sample, holes, dense_intensity_regions : numpy.ndarray
        Tissue masks
    df_grid_roi : pandas DataFrame
        Grid ROI focus scores
    df_roi_intensities : pandas DataFrame
        Tile intensity measurements
    intensity_stats : dict
        Intensity quality assessment statistics
    xoa_morphology_files : list
        List of morphology file paths
    focus_maps : dict, optional
        Focus map arrays for heatmap overlay
    snr_thresholds : dict, optional
        YAML ``snr.roi_tx`` block for SNR heatmap threshold lines
    """
    figures_dir = data["figures_dir"]
    figures_source_dir = (
        (figures_dir / "figures_source") if figure_source_tables else None
    )
    if figures_source_dir is not None:
        figures_source_dir.mkdir(parents=True, exist_ok=True)
    # Create methodology assessment folder for comparison figures
    methodology_figures_dir = figures_dir / "figures_methodology_assessment"
    methodology_figures_dir.mkdir(parents=True, exist_ok=True)

    # §2.4 distance figures use the multi-stain extent mask + its distance maps when present,
    # matching the edge/hole burden metrics in save_roi_qc_metrics. DAPI fallback (None on
    # DAPI-only bundles) keeps those bundles byte-identical. distance_map/distance_map2 are
    # referenced only by the distance closures, so rebinding them here is safe; the mask is
    # aliased as _dm_mask because whole_sample is also used by the masks figure below.
    if multistain_distance_map is not None:
        distance_map = multistain_distance_map
    if multistain_distance_map2 is not None:
        distance_map2 = multistain_distance_map2
    _dm_mask = (
        whole_sample if multistain_whole_sample is None else multistain_whole_sample
    )

    def _fig1_distance_edge():
        logging.info("Generating Figure 1: Distance map (edge)...")
        t_fig = time.time()
        _h, _w = distance_map.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        # Floor the height at 50% of width so very wide slides (e.g. brain) don't get squashed
        fig, ax = plt.subplots(1, 1, figsize=(6, max(3, 6 * _aspect)))
        # Phase v5: distance-to-edge readability fix.
        # - viridis so near-edge tissue (low |distance|) renders as bright
        #   yellow against the white background — visible diagnostic region.
        # - Absolute distance in µm: signed maurer is negative inside;
        #   |·| × 8 × 0.2125 → 0-at-boundary → max-deep-inside gradient.
        # - NaN outside tissue mask → white via cmap.set_bad.
        # - Thin black tissue outline as unambiguous boundary marker.
        # TODO: 8 (downsample factor) and 0.2125 (Xenium native µm/px) are
        # hardcoded here and in three other sites. See task #15 — future
        # plumbing reads pixel_size from the bundle's experiment.xenium.
        from copy import copy as _copy_cmap

        _cmap_edge = _copy_cmap(plt.cm.viridis)
        _cmap_edge.set_bad(color="white")
        # Cap the colorscale at 300 µm so the near-edge band uses most of the
        # spectrum; tiles further than 300 µm from the boundary saturate at
        # yellow and the colorbar shows an "extend max" arrow. Beyond 300 µm
        # the tile is unambiguously deep-tissue and not edge-affected.
        _EDGE_VMAX_UM = 300.0
        if _dm_mask.shape == distance_map.shape:
            _dist_um = np.abs(distance_map) * 8 * 0.2125
            _dm_edge = np.where(_dm_mask > 0, _dist_um, np.nan)
            im = _imshow_thumb(
                ax, _dm_edge, cmap=_cmap_edge, vmin=0.0, vmax=_EDGE_VMAX_UM
            )
            _edge_cbar_extend = "max"
        else:
            _dm_edge = (
                distance_map  # fallback: shapes mismatch, preserve prior behaviour
            )
            im = _imshow_thumb(ax, _dm_edge, cmap=_cmap_edge)
            _edge_cbar_extend = "neither"
        if _dm_mask.shape == distance_map.shape:
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
        ax.set_title("Distance to Edge")
        ax.set_aspect("equal")
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_edge_cbar_extend
        )
        cbar.set_label("Distance from edge (µm)")
        # Explicit "300+" label at the cap when extend="max" is active
        if _edge_cbar_extend == "max":
            cbar.set_ticks([0, 50, 100, 150, 200, 250, _EDGE_VMAX_UM])
            cbar.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_EDGE_VMAX_UM)}+"]
            )
        plt.tight_layout()
        plt.savefig(figures_dir / "distance_map_edge.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "distance_map_edge.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # Full-resolution export. These raster figures_source CSVs are gated behind
        # --figure-source-tables (off by default), so the ~86-Mpx / ~1 GB dump only
        # happens when a user explicitly asks for the raw plotting data — and then it
        # must match the analysis exactly, not a thumbnail. The rendered PNG still uses
        # the _imshow_thumb / _thumb display-resolution path for speed regardless.
        if figures_source_dir is not None:
            pd.DataFrame(distance_map).to_csv(
                figures_source_dir / "distance_map_edge.csv",
                index=False,
                header=False,
            )
        logging.info(
            f"[TIMING] Figure 1 (distance map edge): {time.time() - t_fig:.1f}s"
        )

    def _fig2_distance_holes():
        logging.info("Generating Figure 2: Distance map (holes)...")
        t_fig = time.time()
        _h, _w = distance_map2.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        # Floor the height at 50% of width so very wide slides (e.g. brain) don't get squashed
        fig, ax = plt.subplots(1, 1, figsize=(6, max(3, 6 * _aspect)))
        # Phase v5: distance-to-holes readability fix.
        # - viridis colormap (same as edge map) for visual consistency.
        # - Absolute distance in µm: |signed_maurer_distance| × 8 × 0.2125.
        # - Linear vmin=0 / vmax=300 µm — matches the edge map for visual
        #   consistency across both panels of §2.4 (edge + holes). Tiles beyond 300 µm saturate
        #   at yellow with an "extend max" arrow on the colorbar.
        # - NaN outside tissue mask → white via cmap.set_bad.
        # - Thin black tissue outline as boundary marker.
        # TODO: pixel-size conversion hardcoded — see task #15.
        from copy import copy as _copy_cmap

        _cmap_holes = _copy_cmap(plt.cm.viridis)
        _cmap_holes.set_bad(color="white")
        _HOLES_VMAX_UM = 300.0
        if _dm_mask.shape == distance_map2.shape:
            _dist_um_h = np.abs(distance_map2) * 8 * 0.2125
            _dm_holes = np.where(_dm_mask > 0, _dist_um_h, np.nan)
            im = _imshow_thumb(
                ax, _dm_holes, cmap=_cmap_holes, vmin=0.0, vmax=_HOLES_VMAX_UM
            )
            _holes_cbar_extend = "max"
        else:
            _dm_holes = distance_map2  # fallback: shapes mismatch
            im = _imshow_thumb(ax, _dm_holes, cmap=_cmap_holes)
            _holes_cbar_extend = "neither"
        if _dm_mask.shape == distance_map2.shape:
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
        ax.set_title("Distance to Nearest Hole")
        ax.set_aspect("equal")
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_holes_cbar_extend
        )
        cbar.set_label("Distance from nearest hole (µm)")
        # Explicit "300+" label at the cap when extend="max" is active.
        if _holes_cbar_extend == "max":
            cbar.set_ticks([0, 50, 100, 150, 200, 250, _HOLES_VMAX_UM])
            cbar.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_HOLES_VMAX_UM)}+"]
            )
        plt.tight_layout()
        plt.savefig(
            figures_dir / "distance_map_holes.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            figures_dir / "distance_map_holes.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # Full-resolution export (see distance_map_edge.csv note): gated behind
        # --figure-source-tables (off by default); rendering uses the display-res thumbnail.
        if figures_source_dir is not None:
            pd.DataFrame(distance_map2).to_csv(
                figures_source_dir / "distance_map_holes.csv",
                index=False,
                header=False,
            )
        logging.info(
            f"[TIMING] Figure 2 (distance map holes): {time.time() - t_fig:.1f}s"
        )

    def _fig2b_distance_combined():
        """Two-panel combined figure: distance to edge (left) + distance to
        holes (right), sharing coordinate system and colorbar conventions.
        Standalone _fig1_distance_edge / _fig2_distance_holes continue to
        render the individual PNGs as latent artefacts; the combined figure
        is what the QMD §2.4 embeds.
        """
        logging.info("Generating Figure 2b: Distance combined (edge + holes)...")
        t_fig = time.time()
        _h, _w = distance_map.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        _panel_width = 6
        _panel_height = max(3, _panel_width * _aspect)
        fig, axes = plt.subplots(1, 2, figsize=(2 * _panel_width + 2, _panel_height))

        from copy import copy as _copy_cmap

        _VMAX_UM = 300.0
        _cmap = _copy_cmap(plt.cm.viridis)
        _cmap.set_bad(color="white")

        # ── Left panel: distance to edge ───────────────────────────────
        ax = axes[0]
        if _dm_mask.shape == distance_map.shape:
            _dist_um = np.abs(distance_map) * 8 * 0.2125
            _dm_edge = np.where(_dm_mask > 0, _dist_um, np.nan)
            im_e = _imshow_thumb(ax, _dm_edge, cmap=_cmap, vmin=0.0, vmax=_VMAX_UM)
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
            _edge_extend = "max"
        else:
            im_e = _imshow_thumb(ax, distance_map, cmap=_cmap)
            _edge_extend = "neither"
        ax.set_title("Distance to Edge", fontsize=14)
        ax.set_aspect("equal")
        cbar_e = fig.colorbar(
            im_e, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_edge_extend
        )
        cbar_e.set_label("Distance from edge (µm)")
        if _edge_extend == "max":
            cbar_e.set_ticks([0, 50, 100, 150, 200, 250, _VMAX_UM])
            cbar_e.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_VMAX_UM)}+"]
            )

        # ── Right panel: distance to holes ─────────────────────────────
        ax = axes[1]
        if _dm_mask.shape == distance_map2.shape:
            _dist_um_h = np.abs(distance_map2) * 8 * 0.2125
            _dm_holes = np.where(_dm_mask > 0, _dist_um_h, np.nan)
            im_h = _imshow_thumb(ax, _dm_holes, cmap=_cmap, vmin=0.0, vmax=_VMAX_UM)
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
            _holes_extend = "max"
        else:
            im_h = _imshow_thumb(ax, distance_map2, cmap=_cmap)
            _holes_extend = "neither"
        ax.set_title("Distance to Nearest Hole", fontsize=14)
        ax.set_aspect("equal")
        cbar_h = fig.colorbar(
            im_h, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_holes_extend
        )
        cbar_h.set_label("Distance from nearest hole (µm)")
        if _holes_extend == "max":
            cbar_h.set_ticks([0, 50, 100, 150, 200, 250, _VMAX_UM])
            cbar_h.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_VMAX_UM)}+"]
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "distance_maps.png", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "distance_maps.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(
            f"[TIMING] Figure 2b (distance combined): {time.time() - t_fig:.1f}s"
        )

    def _fig3_morphology_overview():
        logging.info("Generating Figure 3: Morphology overview...")
        t_fig = time.time()
        # Phase v5 TODO #12: 1×3 layout (DAPI + Boundary + Interior only). The
        # artefact / dense-intensity-regions panel that used to live in this
        # figure's bottom-right is also rendered in imageqc_masks.png (§2.3 Masks),
        # so showing it here too duplicates the same plot — dropped here.
        _img_aspect = small0.shape[0] / small0.shape[1] if small0.shape[1] > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        fig, ax = plt.subplots(1, 3, figsize=(3 * _panel_width, _panel_height))
        _imshow_thumb(ax[0], small0, cmap="Greys_r", vmax=np.percentile(small0, 99))
        ax[0].set_title("DAPI", fontsize=14)
        ax[0].set_aspect("equal")
        ax[0].axis("off")
        _imshow_thumb(ax[1], small1, cmap="Greys_r", vmax=np.percentile(small1, 99))
        ax[1].set_title("Boundary", fontsize=14)
        ax[1].set_aspect("equal")
        ax[1].axis("off")
        _imshow_thumb(ax[2], small2, cmap="Greys_r", vmax=np.percentile(small2, 99))
        ax[2].set_title("Interior", fontsize=14)
        ax[2].set_aspect("equal")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "morphology_overview.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            figures_dir / "morphology_overview.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # Full-resolution export (see distance_map_edge.csv note): four ~86-Mpx channel
        # grids, gated behind --figure-source-tables (off by default); rendering uses
        # the display-res thumbnail so the figure wall time is unaffected.
        if figures_source_dir is not None:
            pd.DataFrame(small0).to_csv(
                figures_source_dir / "morphology_overview_DAPI.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(small1).to_csv(
                figures_source_dir / "morphology_overview_Boundary.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(small2).to_csv(
                figures_source_dir / "morphology_overview_Interior.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(dense_intensity_regions).to_csv(
                figures_source_dir / "morphology_overview_DenseIntensityRegions.csv",
                index=False,
                header=False,
            )
        logging.info(
            f"[TIMING] Figure 3 (morphology overview): {time.time() - t_fig:.1f}s"
        )

    def _fig4_imageqc_masks():
        logging.info("Generating Figure 4: ImageQC masks...")
        t_fig = time.time()
        # Phase v5: 1x3 triplet — three mask panels only. DAPI morphology
        # image was dropped (it's a staining, already shown under §2.4
        # Stainings via morphology_overview.png).
        _img_aspect = small0.shape[0] / small0.shape[1] if small0.shape[1] > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        # §2.3 tissue mask shows the EXTENT mask (all available stains) so it matches the
        # reported tissue coverage; falls back to the DAPI mask on single-stain slides.
        _extent_ws = (
            multistain_whole_sample
            if multistain_whole_sample is not None
            else whole_sample
        )
        fig, ax = plt.subplots(1, 3, figsize=(3 * _panel_width, _panel_height))
        ax[0].set_title("Tissue mask (all stains)", fontsize=14)
        _imshow_thumb(ax[0], _extent_ws, rgb=lambda d: color.label2rgb(d, bg_label=0))
        ax[0].set_aspect("equal")
        ax[0].axis("off")
        ax[1].set_title("Holes in sample", fontsize=14)
        _imshow_thumb(ax[1], holes, rgb=lambda d: color.label2rgb(d, bg_label=0))
        ax[1].set_aspect("equal")
        ax[1].axis("off")
        ax[2].set_title("Optically dense regions", fontsize=14)
        _imshow_thumb(
            ax[2], dense_intensity_regions, rgb=lambda d: color.label2rgb(d, bg_label=0)
        )
        ax[2].set_aspect("equal")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(figures_dir / "imageqc_masks.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "imageqc_masks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # DAPI csv export dropped — same data exposed by §2.4 Stainings.
        # Full-resolution export (see distance_map_edge.csv note): three ~86-Mpx mask
        # grids, gated behind --figure-source-tables (off by default); rendering uses
        # the display-res thumbnail so the figure wall time is unaffected.
        if figures_source_dir is not None:
            pd.DataFrame(_extent_ws).to_csv(
                figures_source_dir / "imageqc_masks_WholeSample.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(holes).to_csv(
                figures_source_dir / "imageqc_masks_Holes.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(dense_intensity_regions).to_csv(
                figures_source_dir / "imageqc_masks_DenseIntensityRegions.csv",
                index=False,
                header=False,
            )
        logging.info(f"[TIMING] Figure 4 (ImageQC masks): {time.time() - t_fig:.1f}s")

    def _fig5_focus_heatmap():
        logging.info("Generating Figure 5: Grid tile focus heatmap...")
        t_fig = time.time()
        plot_grid_roi_focus_heatmap(
            df_grid_roi,
            small0,
            figures_dir,
            figures_source_dir,
            threshold=-1.0,
            focus_maps=focus_maps,
            focus_heatmap=focus_heatmap,
        )
        logging.info(f"[TIMING] Figure 5 (focus heatmap): {time.time() - t_fig:.1f}s")

    def _fig5b_focus_vs_intensity():
        logging.info("Generating Figure 5b: Tile focus score vs intensity...")
        t_fig = time.time()
        plot_roi_focus_vs_intensity(
            df_grid_roi, figures_dir, figures_source_dir, threshold=-1.0
        )
        logging.info(
            f"[TIMING] Figure 5b (focus vs intensity): {time.time() - t_fig:.1f}s"
        )

    def _fig5c_focus_distribution():
        logging.info("Generating Figure 5c: Tile focus score distribution...")
        t_fig = time.time()
        plot_roi_focus_distribution(
            df_grid_roi, figures_dir, figures_source_dir, threshold=-1.0
        )
        logging.info(
            f"[TIMING] Figure 5c (focus distribution): {time.time() - t_fig:.1f}s"
        )

    def _fig5d_focus_vs_laplacian():
        logging.info(
            "Generating Figure 5d: Focus score vs Laplacian variance comparison..."
        )
        t_fig = time.time()
        plot_focus_score_vs_laplacian(df_grid_roi, figures_dir, figures_source_dir)
        logging.info(
            f"[TIMING] Figure 5d (focus vs Laplacian): {time.time() - t_fig:.1f}s"
        )

    def _fig6_intensity_assessment():
        logging.info("Generating Figure 6: Intensity assessment...")
        t_fig = time.time()
        plot_intensity_assessment(
            df_roi_intensities, intensity_stats, small0, figures_dir, figures_source_dir
        )
        logging.info(
            f"[TIMING] Figure 6 (intensity assessment): {time.time() - t_fig:.1f}s"
        )

    def _fig7_snr_heatmap():
        if "neg_pct" not in df_grid_roi.columns:
            logging.info("Skipping SNR heatmap: no SNR columns in df_grid_roi")
            return
        logging.info("Generating Figure 7: SNR spatial heatmap...")
        t_fig = time.time()
        plot_snr_roi_heatmap(
            df_grid_roi,
            small0,
            figures_dir,
            figures_source_dir,
            snr_thresholds=snr_thresholds,
        )
        logging.info(f"[TIMING] Figure 7 (SNR heatmap): {time.time() - t_fig:.1f}s")

    def _fig8_concordance():
        if "roi_tx_snr_ratio" not in df_grid_roi.columns:
            logging.info("Skipping concordance plot: no SNR columns in df_grid_roi")
            return
        logging.info("Generating Figure 8: Cross-section concordance...")
        t_fig = time.time()
        plot_cross_section_concordance(
            df_grid_roi,
            figures_dir,
            figures_source_dir,
            snr_thresholds=snr_thresholds,
        )
        logging.info(f"[TIMING] Figure 8 (concordance): {time.time() - t_fig:.1f}s")

    tasks = [
        _fig1_distance_edge,
        _fig2_distance_holes,
        _fig2b_distance_combined,
        _fig3_morphology_overview,
        _fig4_imageqc_masks,
        _fig5_focus_heatmap,
        _fig5b_focus_vs_intensity,
        _fig5c_focus_distribution,
        # _fig5d_focus_vs_laplacian disabled 2026-05-20 — produces
        # focus_score_vs_laplacian.png + focus_score_vs_laplacian_distributions.png
        # in methodology_figures_dir; neither is embedded in the report.
        # Function definition retained above as latent code.
        _fig6_intensity_assessment,
        _fig7_snr_heatmap,
        _fig8_concordance,
    ]

    _run_figure_pool(tasks, phase="tile-based figures")

    logging.info("All tile-based figures generated successfully!")
    logging.info(f"Saved figures to {figures_dir}")
    if figures_source_dir is not None:
        logging.info(f"Saved source data to {figures_source_dir}")


def compute_whole_grid_stain_percentiles(df_grid_roi):
    """Mask-independent stain percentiles (p95 / p99) per channel over the WHOLE
    tile grid (no tissue-mask filter), so they survive a mask-generation failure
    (qc_threshold_refinement §5.5). DIAGNOSTIC ONLY — used to triage why a mask
    failed (collapsed p99 = dim stain like skin; healthy p99 = a structural
    mask-detection failure on bright tissue). NOT a gate: an absolute p99 does
    not separate mask-PASS from mask-FAIL (XOA-version confound).

    Returns ``{channel: {"p95": float|None, "p99": float|None}}`` for dapi,
    boundary, intrna. None when the channel column is absent or all-NaN.
    """
    out = {}
    for ch, col in (
        ("dapi", "dapi_intensity"),
        ("boundary", "boundary_intensity"),
        ("intrna", "intrna_intensity"),
    ):
        vals = None
        if col in df_grid_roi.columns:
            vals = pd.to_numeric(df_grid_roi[col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
        if vals is not None and vals.size > 0:
            out[ch] = {
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }
        else:
            out[ch] = {"p95": None, "p99": None}
    return out


def save_roi_qc_metrics(
    df_grid_roi,
    intensity_stats,
    outdir,
    roi_size=None,
    snr_summary=None,
    distance_map=None,
    distance_map2=None,
    multistain_whole_sample=None,
    multistain_distance_map=None,
    multistain_distance_map2=None,
    edge_distance_threshold: float = -25.0,
    hole_distance_threshold: float = -25.0,
    min_tissue_coverage_for_qc: float = ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC,
    qc_thresholds: dict | None = None,
    lap_sigma: float | None = None,
    segmentation_software: str | None = None,
    xoa_version: str | None = None,
):
    """
    Save tile-based QC metrics to JSON file.

    Parameters
    ----------
    snr_summary : dict or None
        If provided, stored under ``snr`` (from :mod:`snr_metrics`).
    segmentation_software : str or None
        Human-readable label for the segmentation software that produced the
        bundle this report describes (e.g. ``"Xenium Onboard Analysis v4.0.1"``).
    """
    import json

    # Tissue EXTENT coverage: from the multi-stain mask (all available stains) when provided,
    # else the DAPI tissue_coverage column. EXTENT metrics (coverage stats, mask status,
    # edge/hole) use this; DAPI-QUALITY metrics (focus, blur, usable, cluster) keep the DAPI
    # tissue_coverage column. DAPI-only bundles pass multistain_whole_sample=None, so extent
    # == DAPI and the output is identical. See plans/2026-06-26_PLAN_multistain-mask.md.
    _has_xy = {"x1", "x2", "y1", "y2"}.issubset(df_grid_roi.columns)
    if multistain_whole_sample is not None and _has_xy:
        _ext_cov = _per_tile_coverage(
            multistain_whole_sample,
            df_grid_roi["x1"].to_numpy(np.int64, copy=False),
            df_grid_roi["x2"].to_numpy(np.int64, copy=False),
            df_grid_roi["y1"].to_numpy(np.int64, copy=False),
            df_grid_roi["y2"].to_numpy(np.int64, copy=False),
        )
        _ext_dmap = (
            distance_map if multistain_distance_map is None else multistain_distance_map
        )
        _ext_dmap2 = (
            distance_map2
            if multistain_distance_map2 is None
            else multistain_distance_map2
        )
    else:
        _ext_cov = (
            df_grid_roi["tissue_coverage"].to_numpy(np.float64, copy=False)
            if "tissue_coverage" in df_grid_roi.columns
            else None
        )
        _ext_dmap, _ext_dmap2 = distance_map, distance_map2
    _ext_rois_in_tissue = int((_ext_cov > 0).sum()) if _ext_cov is not None else 0

    # Existing stats
    roi_metrics = {
        "roi_size_pixels": roi_size,
        "xoa_version": xoa_version,
        "segmentation_software": segmentation_software,
        "total_rois": len(df_grid_roi),
        "rois_in_tissue": _ext_rois_in_tissue,
        "focus_score": {
            "mean": float(df_grid_roi["focus_score"].mean()),
            "median": float(df_grid_roi["focus_score"].median()),
            "std": float(df_grid_roi["focus_score"].std()),
            "min": float(df_grid_roi["focus_score"].min()),
            "max": float(df_grid_roi["focus_score"].max()),
            "tissue_median": float(
                df_grid_roi.loc[
                    df_grid_roi["tissue_coverage"] >= min_tissue_coverage_for_qc,
                    "focus_score",
                ].median()
            )
            if "tissue_coverage" in df_grid_roi.columns
            else float(df_grid_roi["focus_score"].median()),
        },
        "focus_score_norm": {
            "mean": float(df_grid_roi["focus_score_norm"].mean()),
            "median": float(df_grid_roi["focus_score_norm"].median()),
            "std": float(df_grid_roi["focus_score_norm"].std()),
            "min": float(df_grid_roi["focus_score_norm"].min()),
            "max": float(df_grid_roi["focus_score_norm"].max()),
        },
        "raw_intensity": {
            "mean": float(df_grid_roi["raw_intensity"].mean()),
            "median": float(df_grid_roi["raw_intensity"].median()),
            "std": float(df_grid_roi["raw_intensity"].std()),
            "min": float(df_grid_roi["raw_intensity"].min()),
            "max": float(df_grid_roi["raw_intensity"].max()),
        },
        # tissue_coverage (extent) reflects all available stains (multi-stain mask).
        "tissue_coverage": {
            "mean": float(np.mean(_ext_cov)) if _ext_cov is not None else 0.0,
            "median": float(np.median(_ext_cov)) if _ext_cov is not None else 0.0,
            "min": float(np.min(_ext_cov)) if _ext_cov is not None else 0.0,
            "max": float(np.max(_ext_cov)) if _ext_cov is not None else 0.0,
        },
        "intensity_quality": intensity_stats,
    }
    total_rois = int(len(df_grid_roi))
    rois_in_tissue = _ext_rois_in_tissue  # extent: any tissue tile (all stains)
    roi_metrics["tissue_mask_qc"] = {
        "tissue_mask_generated": bool(rois_in_tissue > 0),
        "status": "PASS" if rois_in_tissue > 0 else "FAIL",
        "rois_in_tissue": rois_in_tissue,
        "total_rois": total_rois,
        "tissue_roi_fraction": float(rois_in_tissue / total_rois)
        if total_rois > 0
        else 0.0,
    }

    # Mask-independent stain percentiles (p95 / p99 over the WHOLE tile grid,
    # 2026-06-23, qc_threshold_refinement §5.5). See
    # compute_whole_grid_stain_percentiles for the diagnostic-only rationale.
    roi_metrics["stain_percentiles_whole_grid"] = compute_whole_grid_stain_percentiles(
        df_grid_roi
    )

    # Optional: add GMM-based blur summary if columns are present
    if (
        "is_blurred_gmm" in df_grid_roi.columns
        and "is_low_intensity" in df_grid_roi.columns
    ):
        total_rois = len(df_grid_roi)
        n_blurred = int(df_grid_roi["is_blurred_gmm"].sum())
        n_low_int = int(df_grid_roi["is_low_intensity"].sum())
        if "tissue_coverage" in df_grid_roi.columns:
            tissue_mask_qc = (
                df_grid_roi["tissue_coverage"] >= min_tissue_coverage_for_qc
            )
            total_rois_tissue = int(tissue_mask_qc.sum())
            n_blurred_tissue = int(
                df_grid_roi.loc[tissue_mask_qc, "is_blurred_gmm"].sum()
            )
            n_low_int_tissue = int(
                df_grid_roi.loc[tissue_mask_qc, "is_low_intensity"].sum()
            )
        else:
            total_rois_tissue = total_rois
            n_blurred_tissue = n_blurred
            n_low_int_tissue = n_low_int

        roi_metrics["blur_gmm_1d"] = {
            "total_rois": total_rois,
            "rois_blurred_gmm": n_blurred,
            "rois_low_intensity": n_low_int,
            "pct_blurred_gmm": float(n_blurred / total_rois * 100.0)
            if total_rois > 0
            else 0.0,
            "pct_low_intensity": float(n_low_int / total_rois * 100.0)
            if total_rois > 0
            else 0.0,
            # Tissue-aware denominator for report-level interpretation
            "total_rois_tissue_filtered": total_rois_tissue,
            "rois_blurred_gmm_tissue_filtered": n_blurred_tissue,
            "rois_low_intensity_tissue_filtered": n_low_int_tissue,
            "pct_blurred_gmm_tissue_filtered": float(
                n_blurred_tissue / total_rois_tissue * 100.0
            )
            if total_rois_tissue > 0
            else 0.0,
            "pct_low_intensity_tissue_filtered": float(
                n_low_int_tissue / total_rois_tissue * 100.0
            )
            if total_rois_tissue > 0
            else 0.0,
            "tissue_coverage_min_for_qc": float(min_tissue_coverage_for_qc),
        }

        if "blur_prob_gmm" in df_grid_roi.columns:
            valid_probs = df_grid_roi["blur_prob_gmm"].dropna()
            if len(valid_probs) > 0:
                roi_metrics["blur_gmm_1d"]["blur_prob_mean"] = float(valid_probs.mean())
                roi_metrics["blur_gmm_1d"]["blur_prob_median"] = float(
                    valid_probs.median()
                )

    # Optional: add 2D GMM-based blur summary if columns are present
    if "is_blurred_gmm_2d" in df_grid_roi.columns:
        total_rois = len(df_grid_roi)
        n_blurred_2d = int(df_grid_roi["is_blurred_gmm_2d"].sum())
        n_low_int = (
            int(df_grid_roi["is_low_intensity"].sum())
            if "is_low_intensity" in df_grid_roi.columns
            else 0
        )
        if "tissue_coverage" in df_grid_roi.columns:
            tissue_mask_qc = (
                df_grid_roi["tissue_coverage"] >= min_tissue_coverage_for_qc
            )
            total_rois_tissue = int(tissue_mask_qc.sum())
            n_blurred_tissue = int(
                df_grid_roi.loc[tissue_mask_qc, "is_blurred_gmm_2d"].sum()
            )
            n_low_int_tissue = (
                int(df_grid_roi.loc[tissue_mask_qc, "is_low_intensity"].sum())
                if "is_low_intensity" in df_grid_roi.columns
                else 0
            )
        else:
            total_rois_tissue = total_rois
            n_blurred_tissue = n_blurred_2d
            n_low_int_tissue = n_low_int

        roi_metrics["blur_gmm_2d"] = {
            "total_rois": total_rois,
            "rois_blurred_gmm": n_blurred_2d,
            "rois_low_intensity": n_low_int,
            "pct_blurred_gmm": float(n_blurred_2d / total_rois * 100.0)
            if total_rois > 0
            else 0.0,
            "pct_low_intensity": float(n_low_int / total_rois * 100.0)
            if total_rois > 0
            else 0.0,
            # Tissue-aware denominator for report-level interpretation
            "total_rois_tissue_filtered": total_rois_tissue,
            "rois_blurred_gmm_tissue_filtered": n_blurred_tissue,
            "rois_low_intensity_tissue_filtered": n_low_int_tissue,
            "pct_blurred_gmm_tissue_filtered": float(
                n_blurred_tissue / total_rois_tissue * 100.0
            )
            if total_rois_tissue > 0
            else 0.0,
            "pct_low_intensity_tissue_filtered": float(
                n_low_int_tissue / total_rois_tissue * 100.0
            )
            if total_rois_tissue > 0
            else 0.0,
            "tissue_coverage_min_for_qc": float(min_tissue_coverage_for_qc),
        }

        if "blur_prob_gmm_2d" in df_grid_roi.columns:
            valid_probs = df_grid_roi["blur_prob_gmm_2d"].dropna()
            if len(valid_probs) > 0:
                roi_metrics["blur_gmm_2d"]["blur_prob_mean"] = float(valid_probs.mean())
                roi_metrics["blur_gmm_2d"]["blur_prob_median"] = float(
                    valid_probs.median()
                )

        # Compare 1D vs 2D if both exist
        if "is_blurred_gmm" in df_grid_roi.columns:
            both_blurred = (
                df_grid_roi["is_blurred_gmm"] & df_grid_roi["is_blurred_gmm_2d"]
            ).sum()
            both_in_focus = (
                (~df_grid_roi["is_blurred_gmm"]) & (~df_grid_roi["is_blurred_gmm_2d"])
            ).sum()
            agreement = (
                (both_blurred + both_in_focus) / total_rois * 100.0
                if total_rois > 0
                else 0.0
            )
            roi_metrics["gmm_comparison"] = {
                "agreement_pct": float(agreement),
                "both_blurred": int(both_blurred),
                "both_in_focus": int(both_in_focus),
                "only_1d_blurred": int(
                    (
                        df_grid_roi["is_blurred_gmm"]
                        & ~df_grid_roi["is_blurred_gmm_2d"]
                    ).sum()
                ),
                "only_2d_blurred": int(
                    (
                        ~df_grid_roi["is_blurred_gmm"]
                        & df_grid_roi["is_blurred_gmm_2d"]
                    ).sum()
                ),
            }

    if snr_summary is not None:
        roi_metrics["snr"] = snr_summary

    # Optional: morphology ROI summary metrics (vectorized, memory-friendly).
    if (
        distance_map is not None
        and distance_map2 is not None
        and {"x1", "x2", "y1", "y2", "tissue_coverage"}.issubset(df_grid_roi.columns)
    ):
        x1 = df_grid_roi["x1"].to_numpy(np.int64, copy=False)
        x2 = df_grid_roi["x2"].to_numpy(np.int64, copy=False)
        y1 = df_grid_roi["y1"].to_numpy(np.int64, copy=False)
        y2 = df_grid_roi["y2"].to_numpy(np.int64, copy=False)
        # Two coverage arrays: DAPI for QUALITY (usable, cluster), multi-stain for EXTENT
        # (edge/hole, tissue-tile count). ext_cov == dapi_cov on DAPI-only bundles.
        dapi_cov = df_grid_roi["tissue_coverage"].to_numpy(np.float64, copy=False)
        ext_cov = _ext_cov if _ext_cov is not None else dapi_cov

        # EXTENT distance maps (multi-stain when present) sampled at ROI centroids.
        cx_ds = np.clip(((x1 + x2) // 2) // 8, 0, _ext_dmap.shape[1] - 1)
        cy_ds = np.clip(((y1 + y2) // 2) // 8, 0, _ext_dmap.shape[0] - 1)
        dist_edge = _ext_dmap[cy_ds, cx_ds]
        dist_hole = _ext_dmap2[cy_ds, cx_ds]

        # DAPI-quality masks (usable, cluster) and EXTENT masks (edge/hole, counts).
        dapi_tissue_mask = dapi_cov > 0.0
        qc_tissue_mask = dapi_cov >= float(min_tissue_coverage_for_qc)
        n_qc_tissue = int(qc_tissue_mask.sum())
        ext_tissue_mask = ext_cov > 0.0
        n_ext_tissue = int(ext_tissue_mask.sum())

        edge_zone_frac = (
            float(
                (ext_tissue_mask & (dist_edge > float(edge_distance_threshold))).sum()
                / n_ext_tissue
            )
            if n_ext_tissue > 0
            else None
        )
        hole_area_frac = (
            float(
                (ext_tissue_mask & (dist_hole > float(hole_distance_threshold))).sum()
                / n_ext_tissue
            )
            if n_ext_tissue > 0
            else None
        )

        if "is_blurred_gmm_2d" in df_grid_roi.columns:
            blurred_mask = df_grid_roi["is_blurred_gmm_2d"].to_numpy(bool, copy=False)
        elif "is_blurred_gmm" in df_grid_roi.columns:
            blurred_mask = df_grid_roi["is_blurred_gmm"].to_numpy(bool, copy=False)
        else:
            blurred_mask = np.zeros(len(df_grid_roi), dtype=bool)
        # usable_tissue uses its OWN low-intensity floor (DAPI_LOW_INTENSITY_FLOOR=50),
        # independent of the is_low_intensity column (which stays at ROI_INTENSITY_THRESHOLD
        # for the 1D-GMM tissue scope). A dim-but-real tissue tile shouldn't count as
        # unusable on absolute brightness alone. See 2026-06-26_PLAN_tissue-mask-recalibration.
        _intensity_col = (
            "dapi_intensity"
            if "dapi_intensity" in df_grid_roi.columns
            else ("raw_intensity" if "raw_intensity" in df_grid_roi.columns else None)
        )
        if _intensity_col is not None:
            low_intensity_mask = (
                df_grid_roi[_intensity_col].to_numpy() < DAPI_LOW_INTENSITY_FLOOR
            )
        else:
            low_intensity_mask = np.zeros(len(df_grid_roi), dtype=bool)
        bad_mask = blurred_mask | low_intensity_mask

        usable_tissue_frac = (
            float((qc_tissue_mask & (~bad_mask)).sum() / n_qc_tissue)
            if n_qc_tissue > 0
            else None
        )

        # Largest contiguous bad zone (4-neighbour) over ROI grid.
        x_unique = np.unique(x1)
        y_unique = np.unique(y1)
        x_idx = np.searchsorted(x_unique, x1)
        y_idx = np.searchsorted(y_unique, y1)
        bad_grid = np.zeros((len(y_unique), len(x_unique)), dtype=bool)
        tissue_grid = np.zeros((len(y_unique), len(x_unique)), dtype=bool)
        bad_grid[y_idx, x_idx] = dapi_tissue_mask & bad_mask
        tissue_grid[y_idx, x_idx] = dapi_tissue_mask
        n_tissue_grid = int(tissue_grid.sum())
        if n_tissue_grid > 0 and bad_grid.any():
            labels, n_comp = ndimage.label(
                bad_grid,
                structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            )
            component_sizes = np.bincount(labels.ravel())
            largest_bad_component = int(component_sizes[1:].max()) if n_comp > 0 else 0
            cluster_zone_bad_fraction = float(largest_bad_component / n_tissue_grid)
        else:
            cluster_zone_bad_fraction = 0.0 if n_tissue_grid > 0 else None

        roi_metrics["morphology"] = {
            "edge_zone_frac": edge_zone_frac,
            "hole_area_frac": hole_area_frac,
            "usable_tissue_frac": usable_tissue_frac,
            "cluster_zone_bad_fraction": cluster_zone_bad_fraction,
            "edge_distance_threshold_px_ds": float(edge_distance_threshold),
            "hole_distance_threshold_px_ds": float(hole_distance_threshold),
            "min_tissue_coverage_for_qc": float(min_tissue_coverage_for_qc),
            "n_tissue_rois": n_ext_tissue,
            "n_qc_tissue_rois": n_qc_tissue,
        }

    # Optional: Laplacian sharpness summary for section-6 report metrics.
    _qc = qc_thresholds or {}
    _ch_dapi = (_qc.get("channels") or {}).get("DAPI") or {}
    _focus_cut = _qc.get("focus") or {}
    # The standalone absolute Laplacian-variance floor verdict was dropped
    # (2026-06-22, qc_drift_analysis): lap_var scales with brightness² (rho≈0.91
    # with DAPI) and collapses to near-zero on dim XOA-4.0 images, over-flagging
    # good dim tissue. It is redundant with the GMM-2D classifier, which already
    # uses lap_var as a feature. lap_var_median_raw is still emitted as an
    # informational calibration stat below.

    if {"dapi_lap_var", "tissue_coverage"}.issubset(df_grid_roi.columns):
        lap_var = df_grid_roi["dapi_lap_var"].to_numpy(np.float64, copy=False)
        tissue_cov = df_grid_roi["tissue_coverage"].to_numpy(np.float64, copy=False)
        focus_col = (
            "dapi_focus_score"
            if "dapi_focus_score" in df_grid_roi.columns
            else ("focus_score" if "focus_score" in df_grid_roi.columns else None)
        )
        focus_vals = (
            df_grid_roi[focus_col].to_numpy(np.float64, copy=False)
            if focus_col is not None
            else np.full_like(lap_var, np.nan, dtype=np.float64)
        )

        mask = (
            (tissue_cov >= float(min_tissue_coverage_for_qc))
            & np.isfinite(lap_var)
            & np.isfinite(focus_vals)
            & (lap_var >= 0.0)
        )
        # P5: Exclude boundary ROIs (centre within window_size//2 of image edge)
        if "is_boundary_roi" in df_grid_roi.columns:
            boundary_mask = df_grid_roi["is_boundary_roi"].to_numpy(bool, copy=False)
            mask = mask & ~boundary_mask
            n_boundary_excluded = int(boundary_mask.sum())
        else:
            n_boundary_excluded = 0

        n_used = int(mask.sum())
        if n_used > 1:
            lap_used = lap_var[mask]
            focus_used = focus_vals[mask]

            med = float(np.nanmedian(lap_used))
            q25, q75 = np.nanpercentile(lap_used, [25, 75])
            iqr = float(q75 - q25)

            iqr_uniform = iqr <= 1e-12

            # P4: Guard Spearman correlation on uniform slides
            cv_focus = float(np.std(focus_used) / (np.mean(focus_used) + 1e-12))
            cv_lap = float(np.std(lap_used) / (np.mean(lap_used) + 1e-12))
            if cv_focus < 0.1 and cv_lap < 0.1:
                focus_lap_corr = None
                focus_lap_corr_status = "uniform"
            else:
                focus_lap_corr = float(
                    pd.Series(focus_used).corr(pd.Series(lap_used), method="spearman")
                )
                focus_lap_corr_status = None

            # Separation of 2D-GMM blur components in log1p(lap_var) space (Cohen's d).
            component_separation = None
            if "is_blurred_gmm_2d" in df_grid_roi.columns:
                blur2d = df_grid_roi["is_blurred_gmm_2d"].to_numpy(bool, copy=False)[
                    mask
                ]
                if blur2d.any() and (~blur2d).any():
                    lap_log = np.log1p(np.maximum(lap_used, 0.0))
                    lap_blur = lap_log[blur2d]
                    lap_focus = lap_log[~blur2d]
                    if lap_blur.size > 1 and lap_focus.size > 1:
                        n_b, n_f = lap_blur.size, lap_focus.size
                        v_blur = float(np.var(lap_blur, ddof=1))
                        v_focus = float(np.var(lap_focus, ddof=1))
                        pooled_sd = np.sqrt(
                            max(
                                ((n_b - 1) * v_blur + (n_f - 1) * v_focus)
                                / (n_b + n_f - 2),
                                0.0,
                            )
                            + 1e-12
                        )
                        component_separation = float(
                            abs(float(np.mean(lap_focus)) - float(np.mean(lap_blur)))
                            / pooled_sd
                        )

            roi_metrics["laplacian_sharpness"] = {
                "lap_sigma": float(lap_sigma) if lap_sigma is not None else None,
                "n_rois_used": n_used,
                "n_boundary_excluded": n_boundary_excluded,
                "min_tissue_coverage_for_qc": float(min_tissue_coverage_for_qc),
                "iqr_uniform": iqr_uniform,
                "lap_var_median_raw": med,
                "focus_lap_spearman_corr": focus_lap_corr,
                "focus_lap_corr_status": focus_lap_corr_status,
                "component_separation": component_separation,
            }

    # Save to JSON
    metrics_file = Path(outdir) / "roi_qc_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(roi_metrics, f, indent=2, default=str)
    logging.info(f"[OK] Saved tile QC metrics to {metrics_file}")


def save_pixel_focus_maps(focus_maps, outdir, compress=True):
    """
    Save per-pixel focus maps as compressed tiled TIFF files.

    Uses lz4 compression (5-10x faster than zlib) and parallel I/O via
    ThreadPoolExecutor to minimize wall-clock time on multi-map datasets.

    Args:
        focus_maps: dict returned by compute_all_focus_maps()
        outdir: Path to output directory
        compress: Use lz4 compression (default: True)
    """
    from concurrent.futures import ThreadPoolExecutor

    outdir = Path(outdir)
    items = [(name, arr) for name, arr in focus_maps.items() if arr is not None]

    if not items:
        logging.info("  No focus maps to save.")
        return

    def _save_one(name, array, outdir_path):
        t_start = time.time()
        path = outdir_path / f"{name}.tif"
        tile = (256, 256) if array.shape[0] >= 256 and array.shape[1] >= 256 else None
        tifffile.imwrite(
            str(path),
            np.asarray(array, dtype=np.float32),
            compression="zstd" if compress else None,
            tile=tile,
        )
        elapsed = time.time() - t_start
        return name, path, array.shape, elapsed

    with ThreadPoolExecutor(max_workers=min(len(items), 4)) as executor:
        futures = [executor.submit(_save_one, name, arr, outdir) for name, arr in items]
        for future in futures:
            name, path, shape, elapsed = future.result()
            logging.info(f"  Saved {name}: {path} ({shape}, float32, {elapsed:.1f}s)")


def save_versions_file(outdir):
    """Save package versions to YAML file"""
    import matplotlib

    def get_version(package, package_name=None):
        """Safely get version of a package"""
        if package_name is None:
            package_name = (
                package.__name__ if hasattr(package, "__name__") else str(package)
            )

        try:
            if hasattr(package, "__version__"):
                return package.__version__
            else:
                # Try to get version via importlib
                import importlib.metadata

                return importlib.metadata.version(package_name)
        except Exception:
            return "unknown"

    # Get Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # Get package versions safely
    packages = {
        "python": python_version,
        "numpy": get_version(np, "numpy"),
        "pandas": get_version(pd, "pandas"),
        "matplotlib": get_version(matplotlib, "matplotlib"),
        "seaborn": get_version(sns, "seaborn"),
        "click": get_version(click, "click"),
        "pathlib": "built-in",
    }

    # Get versions for other packages
    try:
        import skimage

        packages["scikit-image"] = get_version(skimage, "scikit-image")
    except Exception:
        packages["scikit-image"] = "unknown"

    try:
        packages["tifffile"] = get_version(tifffile, "tifffile")
    except Exception:
        packages["tifffile"] = "unknown"

    try:
        packages["zarr"] = get_version(zarr, "zarr")
    except Exception:
        packages["zarr"] = "unknown"

    try:
        import napari_skimage_regionprops

        packages["napari-skimage-regionprops"] = get_version(
            napari_skimage_regionprops, "napari-skimage-regionprops"
        )
    except Exception:
        packages["napari-skimage-regionprops"] = "unknown"

    try:
        packages["napari-simpleitk-image-processing"] = get_version(
            nsitk, "napari-simpleitk-image-processing"
        )
    except Exception:
        packages["napari-simpleitk-image-processing"] = "unknown"

    # Create YAML content
    yaml_content = f"""XENIUM_IMAGE_QC:
  python: {packages["python"]}
  numpy: {packages["numpy"]}
  pandas: {packages["pandas"]}
  matplotlib: {packages["matplotlib"]}
  seaborn: {packages["seaborn"]}
  scikit-image: {packages["scikit-image"]}
  tifffile: {packages["tifffile"]}
  zarr: {packages["zarr"]}
  napari-skimage-regionprops: {packages["napari-skimage-regionprops"]}
  napari-simpleitk-image-processing: {packages["napari-simpleitk-image-processing"]}
  click: {packages["click"]}
  pathlib: {packages["pathlib"]}
"""

    # Save to file
    versions_file = outdir / "versions.yml"
    with open(versions_file, "w") as f:
        f.write(yaml_content)

    logging.info(f"[OK] Saved package versions to {versions_file}")


# ===== FUNCTIONS FROM image_qc_mapping_to_cells.py =====


def load_spatial_data(data):
    """Load and prepare spatial data. This only works for the original Xenium bundle generated from XOA"""

    # Load cell masks
    cell_masks_zarr = open_zarr(data["cell_masks_path"])
    # cell_masks_zarr will only have one channel if its the result from xeniumranger import-segmentation
    # Lazily sliced, not materialised: np.array() here cost 22 GB of uint32 on a
    # 5.5 gigapixel sample. Its consumers are row-block reductions
    # (_labeled_sums_chunked, LabeledSumAccumulator) and scattered point lookups,
    # both of which LazyLabelPlane serves straight from zarr. The legacy
    # --legacy-focus path calls regionprops_table, which needs a real array, and
    # materialises it explicitly there.
    cellseg_mask = LazyLabelPlane(cell_masks_zarr.get("masks").get("1"))

    # Load spatial data
    df_spatial = pd.read_parquet(
        data["cells_parquet_path"],
        columns=[
            "x_centroid",
            "y_centroid",
            "transcript_counts",
            "cell_area",
            "nucleus_area",
            "nucleus_count",
            "segmentation_method",
            "cell_id",
        ],
    )

    # Rename columns for simplicity
    df_spatial.rename(columns={"x_centroid": "x", "y_centroid": "y"}, inplace=True)

    # Convert to pixel coordinates
    df_spatial["x"] = (df_spatial.x / XENIUM_PIXEL_SIZE_UM).astype(int)
    df_spatial["y"] = (df_spatial.y / XENIUM_PIXEL_SIZE_UM).astype(int)

    # Measure CellID - OPTIMIZED: vectorized NumPy indexing (50-100x faster)
    x_coords = np.clip(df_spatial["x"].values, 0, cellseg_mask.shape[1] - 1)
    y_coords = np.clip(df_spatial["y"].values, 0, cellseg_mask.shape[0] - 1)
    df_spatial["CellID"] = cellseg_mask[y_coords, x_coords]

    # Load and merge clustering and UMAP data
    df_clusters = pd.read_csv(data["clusters_csv_path"]).rename(
        columns={"Barcode": "cell_id", "Cluster": "Cluster_kmeans10"}
    )
    df_UMAP = pd.read_csv(data["umap_path"]).rename(columns={"Barcode": "cell_id"})
    df_spatial = df_spatial.merge(df_clusters, on="cell_id").merge(
        df_UMAP, on="cell_id"
    )
    return df_spatial, cellseg_mask, cell_masks_zarr


def map_distances_to_cells(
    df_spatial,
    distance_map,
    distance_map2,
    dense_intensity_regions,
    downsample_factor=8,
):
    """
    Map distance maps and dense intensity region masks to cells based on cell centroid locations.

    Extracted from generate_sample_masks_and_distances() for cell-independent workflow.

    Parameters:
    -----------
    df_spatial : pandas DataFrame
        Cell DataFrame with 'x' and 'y' centroid columns
    distance_map : numpy.ndarray
        Distance to edge map (downsampled)
    distance_map2 : numpy.ndarray
        Distance to nearest hole map (downsampled)
    dense_intensity_regions : numpy.ndarray
        Dense intensity region mask (downsampled)
    downsample_factor : int, optional
        Downsampling factor (default: 8)

    Returns:
    --------
    pandas DataFrame
        df_spatial with added columns:
        - Distance-to-edge: Distance to sample edge for each cell
        - Distance-to-nearest-hole: Distance to nearest hole for each cell
        - In-Area-with-Dense-Intensity-Region: Binary indicator if cell is in dense intensity region area
    """
    # Vectorized coordinate mapping -- compute pixel indices once, clipped to array bounds
    X = np.clip(
        (df_spatial["x"].values / downsample_factor).astype(int),
        0,
        distance_map.shape[1] - 1,
    )
    Y = np.clip(
        (df_spatial["y"].values / downsample_factor).astype(int),
        0,
        distance_map.shape[0] - 1,
    )

    # Vectorized lookups -- three lines instead of three loops
    df_spatial["Distance-to-edge"] = distance_map[Y, X]
    df_spatial["Distance-to-nearest-hole"] = distance_map2[Y, X]
    df_spatial["Dense-Intensity-Region-ID"] = dense_intensity_regions[Y, X].astype(int)

    return df_spatial


def map_grid_roi_to_cells(df_grid_roi, df_cells, overlapping=False):
    """
    Map cell-independent grid tile focus scores to cells based on cell centroid location.

    This function replaces the previous cell-based tile approach with a cell-independent
    grid-based tile approach that provides uniform coverage across the tissue.

    For each cell, finds which tile its centroid (x, y) falls into and assigns
    that tile's focus scores to the cell.

    Parameters:
    -----------
    df_grid_roi : pandas DataFrame
        Grid ROI DataFrame with columns: roi_id, x1, x2, y1, y2, focus_score,
        focus_score_norm, raw_intensity, tissue_coverage, and channel-specific columns
    df_cells : pandas DataFrame
        Cell DataFrame with columns: x, y (centroid coordinates)
    overlapping : bool, optional
        Whether grid is overlapping. If True and cell falls into multiple ROIs,
        uses ROI with highest tissue_coverage (default: False)

    Returns:
    --------
    pandas DataFrame
        Cell DataFrame with added columns:
        - roi_id: Integer ID of ROI containing cell centroid (0, 1, 2, ...) or None if cell outside all ROIs
        - DAPI_mean_roi: Mean intensity from grid ROI (NaN if cell outside all ROIs)
        - DAPI_RFS_roi: Raw focus score from grid ROI (NaN if cell outside all ROIs)
        - DAPI_RFSnorm_roi: Normalized focus score from grid ROI (NaN if cell outside all ROIs)
        - roi_tissue_coverage: Tissue coverage of assigned ROI (NaN if cell outside all ROIs)
        - Additional channel-specific columns if available (boundary_*, intrna_*)
    """
    # Create output DataFrame
    df_cells_mapped = df_cells.copy()

    # Initialize columns using old naming convention for compatibility
    df_cells_mapped["roi_id"] = None
    df_cells_mapped["DAPI_mean_roi"] = np.nan
    df_cells_mapped["DAPI_RFS_roi"] = np.nan
    df_cells_mapped["DAPI_RFSnorm_roi"] = np.nan
    df_cells_mapped["roi_tissue_coverage"] = np.nan

    # Check for additional channel columns
    has_boundary = "boundary_focus_score" in df_grid_roi.columns
    has_intrna = "intrna_focus_score" in df_grid_roi.columns

    if has_boundary:
        df_cells_mapped["boundary_focus_score_roi"] = np.nan
        df_cells_mapped["boundary_focus_score_norm_roi"] = np.nan
        df_cells_mapped["boundary_intensity_roi"] = np.nan

    if has_intrna:
        df_cells_mapped["intrna_focus_score_roi"] = np.nan
        df_cells_mapped["intrna_focus_score_norm_roi"] = np.nan
        df_cells_mapped["intrna_intensity_roi"] = np.nan

    # OPTIMIZED: Fast vectorized lookup using 2D grid
    logging.info(f"  Mapping {len(df_cells):,} cells to {len(df_grid_roi):,} tiles...")

    # Get cell coordinates as arrays (faster than DataFrame access)
    cell_x = df_cells["x"].values
    cell_y = df_cells["y"].values
    n_cells = len(df_cells)

    # Create arrays to store matches
    cell_roi_matches = np.full(n_cells, -1, dtype=np.int32)  # -1 means no match
    cell_roi_tissue_coverage_arr = np.full(n_cells, np.nan, dtype=np.float64)

    # Always try the fast vectorized approach: build a 2D lookup grid
    is_regular_grid = False
    if len(df_grid_roi) > 1 and not overlapping:
        roi_size = df_grid_roi.iloc[0]["x2"] - df_grid_roi.iloc[0]["x1"]
        stride = roi_size  # non-overlapping grid

        x_min = df_grid_roi["x1"].min()
        y_min = df_grid_roi["y1"].min()
        x_max = df_grid_roi["x1"].max()
        y_max = df_grid_roi["y1"].max()

        n_cols = int(round((x_max - x_min) / stride)) + 1
        n_rows = int(round((y_max - y_min) / stride)) + 1

        # Build 2D grid: grid[row, col] -> index in df_grid_roi
        grid_lookup = np.full((n_rows, n_cols), -1, dtype=np.int32)
        roi_x1_arr = df_grid_roi["x1"].values
        roi_y1_arr = df_grid_roi["y1"].values
        gx_arr = np.round((roi_x1_arr - x_min) / stride).astype(int)
        gy_arr = np.round((roi_y1_arr - y_min) / stride).astype(int)

        # Check if this is a valid grid (no out-of-bounds)
        valid = (gx_arr >= 0) & (gx_arr < n_cols) & (gy_arr >= 0) & (gy_arr < n_rows)
        if valid.all():
            # Populate grid (vectorized)
            grid_lookup[gy_arr, gx_arr] = np.arange(len(df_grid_roi))

            # Vectorized lookup for all cells -- O(n_cells)
            cell_gx = np.clip(
                np.round((cell_x - x_min) / stride).astype(int), 0, n_cols - 1
            )
            cell_gy = np.clip(
                np.round((cell_y - y_min) / stride).astype(int), 0, n_rows - 1
            )
            cell_roi_idx = grid_lookup[cell_gy, cell_gx]  # vectorized!

            matched_mask = cell_roi_idx >= 0
            cell_roi_matches[matched_mask] = df_grid_roi["roi_id"].values[
                cell_roi_idx[matched_mask]
            ]
            cell_roi_tissue_coverage_arr[matched_mask] = df_grid_roi[
                "tissue_coverage"
            ].values[cell_roi_idx[matched_mask]]

            is_regular_grid = True
            logging.info(
                f"  Using fast grid lookup (stride={stride}px, grid={n_cols}x{n_rows})..."
            )

    if not is_regular_grid:
        # SLOW PATH: Irregular/filtered grid - fallback for grids that don't validate
        if len(df_grid_roi) > 1000:
            logging.info(
                f"  Processing {len(df_cells):,} cells against {len(df_grid_roi):,} tiles (irregular grid)..."
            )

        # Convert ROI boundaries to arrays for fast lookup
        roi_x1 = df_grid_roi["x1"].values
        roi_x2 = df_grid_roi["x2"].values
        roi_y1 = df_grid_roi["y1"].values
        roi_y2 = df_grid_roi["y2"].values
        roi_ids = df_grid_roi["roi_id"].values
        roi_tissue_coverage = df_grid_roi["tissue_coverage"].values

        for cell_idx in range(n_cells):
            x_cell = cell_x[cell_idx]
            y_cell = cell_y[cell_idx]

            matches = (
                (roi_x1 <= x_cell)
                & (x_cell < roi_x2)
                & (roi_y1 <= y_cell)
                & (y_cell < roi_y2)
            )

            if np.any(matches):
                match_idx = np.where(matches)[0][0]
                cell_roi_matches[cell_idx] = roi_ids[match_idx]
                cell_roi_tissue_coverage_arr[cell_idx] = roi_tissue_coverage[match_idx]

    # Check for GMM columns
    has_gmm_1d = "is_blurred_gmm" in df_grid_roi.columns
    has_gmm_2d = "is_blurred_gmm_2d" in df_grid_roi.columns
    has_blur_prob_1d = "blur_prob_gmm" in df_grid_roi.columns
    has_blur_prob_2d = "blur_prob_gmm_2d" in df_grid_roi.columns

    # Initialize GMM columns if available
    if has_gmm_1d:
        df_cells_mapped["is_blurred_gmm_roi"] = False
    if has_blur_prob_1d:
        df_cells_mapped["blur_prob_gmm_roi"] = np.nan
    if has_gmm_2d:
        df_cells_mapped["is_blurred_gmm_2d_roi"] = False
    if has_blur_prob_2d:
        df_cells_mapped["blur_prob_gmm_2d_roi"] = np.nan

    # Build vectorized column arrays from df_grid_roi for direct indexing
    # Map roi_id -> index in df_grid_roi for O(1) lookup
    roi_id_to_idx = pd.Series(
        range(len(df_grid_roi)), index=df_grid_roi["roi_id"].values
    )

    # Resolve column names (handle legacy naming)
    _dapi_intensity_col = (
        "dapi_intensity" if "dapi_intensity" in df_grid_roi.columns else "raw_intensity"
    )
    _dapi_focus_col = (
        "dapi_focus_score"
        if "dapi_focus_score" in df_grid_roi.columns
        else "focus_score"
    )
    _dapi_focus_norm_col = (
        "dapi_focus_score_norm"
        if "dapi_focus_score_norm" in df_grid_roi.columns
        else "focus_score_norm"
    )

    dapi_intensity_arr = df_grid_roi[_dapi_intensity_col].values
    dapi_focus_arr = df_grid_roi[_dapi_focus_col].values
    dapi_focus_norm_arr = df_grid_roi[_dapi_focus_norm_col].values

    # Assign ROI values to cells using vectorized operations
    matched_mask = cell_roi_matches >= 0
    matched_roi_ids = cell_roi_matches[matched_mask]

    # Map matched roi_ids to df_grid_roi indices for vectorized column access
    matched_df_indices = roi_id_to_idx[matched_roi_ids].values

    # Assign ROI ID and tissue coverage
    df_cells_mapped.loc[matched_mask, "roi_id"] = matched_roi_ids
    df_cells_mapped.loc[matched_mask, "roi_tissue_coverage"] = (
        cell_roi_tissue_coverage_arr[matched_mask]
    )

    # Assign DAPI data via direct array indexing (no dict lookups)
    df_cells_mapped.loc[matched_mask, "DAPI_mean_roi"] = dapi_intensity_arr[
        matched_df_indices
    ]
    df_cells_mapped.loc[matched_mask, "DAPI_RFS_roi"] = dapi_focus_arr[
        matched_df_indices
    ]
    df_cells_mapped.loc[matched_mask, "DAPI_RFSnorm_roi"] = dapi_focus_norm_arr[
        matched_df_indices
    ]

    # Assign GMM classifications if available
    if has_gmm_1d:
        df_cells_mapped.loc[matched_mask, "is_blurred_gmm_roi"] = df_grid_roi[
            "is_blurred_gmm"
        ].values[matched_df_indices]
    if has_blur_prob_1d:
        df_cells_mapped.loc[matched_mask, "blur_prob_gmm_roi"] = df_grid_roi[
            "blur_prob_gmm"
        ].values[matched_df_indices]
    if has_gmm_2d:
        df_cells_mapped.loc[matched_mask, "is_blurred_gmm_2d_roi"] = df_grid_roi[
            "is_blurred_gmm_2d"
        ].values[matched_df_indices]
    if has_blur_prob_2d:
        df_cells_mapped.loc[matched_mask, "blur_prob_gmm_2d_roi"] = df_grid_roi[
            "blur_prob_gmm_2d"
        ].values[matched_df_indices]

    # Assign additional channel data if available
    if has_boundary:
        df_cells_mapped.loc[matched_mask, "boundary_focus_score_roi"] = df_grid_roi[
            "boundary_focus_score"
        ].values[matched_df_indices]
        df_cells_mapped.loc[matched_mask, "boundary_focus_score_norm_roi"] = (
            df_grid_roi["boundary_focus_score_norm"].values[matched_df_indices]
        )
        df_cells_mapped.loc[matched_mask, "boundary_intensity_roi"] = df_grid_roi[
            "boundary_intensity"
        ].values[matched_df_indices]

    if has_intrna:
        df_cells_mapped.loc[matched_mask, "intrna_focus_score_roi"] = df_grid_roi[
            "intrna_focus_score"
        ].values[matched_df_indices]
        df_cells_mapped.loc[matched_mask, "intrna_focus_score_norm_roi"] = df_grid_roi[
            "intrna_focus_score_norm"
        ].values[matched_df_indices]
        df_cells_mapped.loc[matched_mask, "intrna_intensity_roi"] = df_grid_roi[
            "intrna_intensity"
        ].values[matched_df_indices]

    # Handle overlapping grids: if cell falls into multiple ROIs, use highest tissue_coverage
    if overlapping:
        # Find cells with multiple ROI assignments
        cell_roi_counts = df_cells_mapped.groupby(df_cells_mapped.index)[
            "roi_id"
        ].count()
        cells_with_multiple = cell_roi_counts[cell_roi_counts > 1].index

        if len(cells_with_multiple) > 0:
            # For each cell with multiple ROIs, find the one with highest tissue_coverage
            for cell_idx in cells_with_multiple:
                # Find all ROIs this cell falls into
                x_cell = df_cells.loc[cell_idx, "x"]
                y_cell = df_cells.loc[cell_idx, "y"]

                matching_rois = df_grid_roi[
                    (df_grid_roi["x1"] <= x_cell)
                    & (x_cell < df_grid_roi["x2"])
                    & (df_grid_roi["y1"] <= y_cell)
                    & (y_cell < df_grid_roi["y2"])
                ]

                if len(matching_rois) > 0:
                    # Select ROI with highest tissue_coverage
                    best_roi = matching_rois.loc[
                        matching_rois["tissue_coverage"].idxmax()
                    ]

                    # Update cell with best ROI
                    df_cells_mapped.loc[cell_idx, "roi_id"] = best_roi["roi_id"]
                    df_cells_mapped.loc[cell_idx, "DAPI_mean_roi"] = best_roi.get(
                        "dapi_intensity", best_roi.get("raw_intensity", np.nan)
                    )
                    df_cells_mapped.loc[cell_idx, "DAPI_RFS_roi"] = best_roi.get(
                        "dapi_focus_score", best_roi.get("focus_score", np.nan)
                    )
                    df_cells_mapped.loc[cell_idx, "DAPI_RFSnorm_roi"] = best_roi.get(
                        "dapi_focus_score_norm",
                        best_roi.get("focus_score_norm", np.nan),
                    )
                    df_cells_mapped.loc[cell_idx, "roi_tissue_coverage"] = best_roi[
                        "tissue_coverage"
                    ]

                    # Update GMM classifications if available
                    if has_gmm_1d:
                        df_cells_mapped.loc[cell_idx, "is_blurred_gmm_roi"] = (
                            best_roi.get("is_blurred_gmm", False)
                        )
                    if has_blur_prob_1d:
                        df_cells_mapped.loc[cell_idx, "blur_prob_gmm_roi"] = (
                            best_roi.get("blur_prob_gmm", np.nan)
                        )
                    if has_gmm_2d:
                        df_cells_mapped.loc[cell_idx, "is_blurred_gmm_2d_roi"] = (
                            best_roi.get("is_blurred_gmm_2d", False)
                        )
                    if has_blur_prob_2d:
                        df_cells_mapped.loc[cell_idx, "blur_prob_gmm_2d_roi"] = (
                            best_roi.get("blur_prob_gmm_2d", np.nan)
                        )

                    if has_boundary:
                        df_cells_mapped.loc[cell_idx, "boundary_focus_score_roi"] = (
                            best_roi.get("boundary_focus_score", np.nan)
                        )
                        df_cells_mapped.loc[
                            cell_idx, "boundary_focus_score_norm_roi"
                        ] = best_roi.get("boundary_focus_score_norm", np.nan)
                        df_cells_mapped.loc[cell_idx, "boundary_intensity_roi"] = (
                            best_roi.get("boundary_intensity", np.nan)
                        )

                    if has_intrna:
                        df_cells_mapped.loc[cell_idx, "intrna_focus_score_roi"] = (
                            best_roi.get("intrna_focus_score", np.nan)
                        )
                        df_cells_mapped.loc[cell_idx, "intrna_focus_score_norm_roi"] = (
                            best_roi.get("intrna_focus_score_norm", np.nan)
                        )
                        df_cells_mapped.loc[cell_idx, "intrna_intensity_roi"] = (
                            best_roi.get("intrna_intensity", np.nan)
                        )

    return df_cells_mapped


def load_roi_blur_threshold(outdir):
    """
    Load ROI blur threshold configuration from JSON file saved by image_qc_roi_processing.py.

    Parameters:
    -----------
    outdir : Path
        Output directory where roi_blur_threshold.json should be located

    Returns:
    --------
    tuple
        (roi_focus_score_threshold, roi_intensity_threshold) or (None, None) if not found
    """
    threshold_json = Path(outdir) / "roi_blur_threshold.json"
    if not threshold_json.exists():
        return None, None

    try:
        with open(threshold_json, "r") as f:
            threshold_config = json.load(f)
        roi_threshold = threshold_config.get("roi_focus_score_threshold")
        intensity_threshold = threshold_config.get("roi_intensity_threshold")
        return roi_threshold, intensity_threshold
    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"  Warning: Could not load threshold configuration: {e}")
        return None, None


def create_final_merged_data(
    df_spatial,
    myData,
    roi_data=None,
    ccfs_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
    edge_distance_threshold=-25,
    hole_distance_threshold=-25,
    roi_threshold=None,
    roi_intensity_threshold=None,
):
    """Create final merged dataset with boolean columns for thresholds"""

    # Link the Spatial Cell matrix (df_spatial) with the FocusScore table
    new_df = pd.merge(df_spatial, myData, how="inner", on="CellID")

    # Create new segmentation names with colour palette for plotting.
    # XR import-segmentation (e.g. from the CROP subworkflow) writes
    # `segmentation_method = 'Imported Cell Segmentation'` into cells.parquet,
    # which doesn't substring-match any of the three OBA categories
    # ("boundary"/"interior"/"nucleus"). Without this 4th row the match-filter
    # below drops every cell and downstream figures fail on empty arrays.
    seg = pd.DataFrame(
        {
            "segmentation": [
                "boundary",
                "interior",
                "nucleus",
                "Imported Cell Segmentation",
            ],
            "segPal": ["#FFABC3", "#A9A800", "#A9CEFF", "#7FBFFF"],
            "join": [1, 1, 1, 1],
        }
    )

    # Link the Spatial Cell matrix to the new segmentation colour palette
    new_df["join"] = 1
    new_df = new_df.merge(seg, on="join").drop("join", axis=1)
    seg.drop("join", axis=1, inplace=True)
    new_df["match"] = new_df.apply(
        lambda x: x.segmentation_method.find(x.segmentation), axis=1
    ).ge(0)
    new_df = new_df[new_df["match"]]

    # Add boolean columns for various thresholds (nuclei-based method)
    new_df["is_low_nuclear_texture"] = new_df["CCFS_DAPI"] <= ccfs_threshold
    new_df["is_high_nuclear_texture"] = new_df["CCFS_DAPI"] > ccfs_threshold
    new_df["is_near_edge"] = new_df["Distance-to-edge"] > edge_distance_threshold
    new_df["is_far_from_edge"] = new_df["Distance-to-edge"] <= edge_distance_threshold
    new_df["is_near_hole"] = (
        new_df["Distance-to-nearest-hole"] > hole_distance_threshold
    )
    new_df["is_far_from_hole"] = (
        new_df["Distance-to-nearest-hole"] <= hole_distance_threshold
    )
    # Create boolean indicator for cells in any dense intensity region (backward compatibility)
    new_df["has_dense_intensity_regions"] = new_df["Dense-Intensity-Region-ID"] > 0

    # Merge ROI data if provided
    calculated_roi_threshold = None
    if roi_data is not None:
        # Merge ROI data using x and y coordinates
        new_df = pd.merge(
            new_df, roi_data, on=["x", "y"], how="left", suffixes=("", "_roi_merge")
        )

        # Use new threshold approach: raw score percentile + intensity threshold
        # If roi_threshold is provided (e.g., for backward compatibility), use it
        # Otherwise, calculate from raw scores using configured parameters
        if roi_threshold is None:
            # roi_threshold should have been calculated from df_grid_roi and passed here
            # If not provided, we can't calculate it here (need df_grid_roi)
            # This should not happen in normal flow, but provide fallback
            logging.warning(
                "  Warning: roi_threshold not provided, using default calculation"
            )
            roi_threshold = -1.0  # Old default for backward compatibility

        # Use intensity threshold from configuration if not provided
        # If None, it means threshold config wasn't loaded (backward compatibility)
        # In that case, skip intensity check and only use focus score threshold
        if roi_intensity_threshold is None:
            roi_intensity_threshold = (
                None  # Will skip intensity check in classification
            )

        calculated_roi_threshold = roi_threshold

        # Add boolean columns for tile-based blur detection
        # New approach: blurred if (raw_focus_score <= threshold) OR (intensity < intensity_threshold)
        # Use raw scores (DAPI_RFS_roi) not normalized (DAPI_RFSnorm_roi)
        has_intensity = "DAPI_mean_roi" in new_df.columns
        has_raw_focus = "DAPI_RFS_roi" in new_df.columns

        if has_raw_focus:
            # Combined threshold: focus score OR intensity (if intensity threshold provided)
            if has_intensity and roi_intensity_threshold is not None:
                new_df["is_blurred_roi"] = (new_df["DAPI_RFS_roi"] <= roi_threshold) | (
                    new_df["DAPI_mean_roi"] < roi_intensity_threshold
                )
            else:
                # Fallback: just use focus score if intensity not available or threshold not provided
                new_df["is_blurred_roi"] = new_df["DAPI_RFS_roi"] <= roi_threshold
        else:
            # Fallback: use normalized scores if raw scores not available (backward compatibility)
            new_df["is_blurred_roi"] = new_df["DAPI_RFSnorm_roi"] <= roi_threshold

        new_df["is_high_focus_roi"] = ~new_df["is_blurred_roi"]

    return new_df, calculated_roi_threshold


# ===== CELL-LEVEL PLOTTING FUNCTIONS =====


def plot_nuclear_texture_proportions(
    new_df,
    figures_dir,
    figures_source_dir,
    texture_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
    GROUP_BY_COLUMN="Cluster_kmeans10",
):
    """
    Create a stacked bar plot showing proportion of cells with high and low blur scores per cluster.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing CCFS_DAPI and Cluster_kmeans10 columns
    texture_threshold : float, optional
        Threshold to classify cells as high/low nuclear texture (default: DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD)
    """
    # Create nuclear texture categories (create temporary column to avoid modifying original)
    df_temp = new_df.copy()
    # Low CCFS_DAPI values (<= threshold) = low nuclear texture quality
    # High CCFS_DAPI values (> threshold) = high nuclear texture quality
    df_temp["texture_category"] = np.where(
        df_temp["CCFS_DAPI"] <= texture_threshold, "Low Quality", "High Quality"
    )

    # Calculate proportions
    proportions = (
        df_temp.groupby([GROUP_BY_COLUMN, "texture_category"])
        .size()
        .unstack(fill_value=0)
    )
    proportions = (
        proportions.div(proportions.sum(axis=1), axis=0) * 100
    )  # Convert to percentages

    # Reorder so 'Low Quality' (red) is plotted first → at the bottom of
    # each stacked bar. Matches the convention in plot_tile_blur_proportions_roi
    # (Blurred at bottom). pandas stacks columns from bottom up in the order
    # they appear in the DataFrame.
    proportions = proportions[["Low Quality", "High Quality"]]

    # Set up the plot
    fig = plt.figure(figsize=(12, 6))

    # Create stacked bar plot with specified colors
    # Column order is Low Quality (red, bottom) then High Quality (blue, top).
    proportions.plot(
        kind="bar",
        stacked=True,
        color=[
            "#d62728",
            "#1f77b4",
        ],  # Red for low quality (bottom), blue for high quality (top)
        figsize=(12, 6),
    )

    # Customize the plot
    plt.title(
        f"Proportion of Cells by Nuclear Texture Quality (Threshold: {texture_threshold})",
        fontsize=14,
        pad=20,
    )
    plt.xlabel(GROUP_BY_COLUMN, fontsize=12)
    plt.ylabel("Percentage of Cells", fontsize=12)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Add percentage labels on the bars
    for c in plt.gca().containers:
        # Add labels
        plt.gca().bar_label(c, fmt="%.1f%%", label_type="center")

    # Add a grid for better readability
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure instead of showing
    plt.savefig(
        figures_dir / "nuclear_texture_proportions.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data as CSV
    df_texture_proportions = (
        df_temp.groupby([GROUP_BY_COLUMN, "texture_category"])
        .size()
        .reset_index(name="count")
    )
    if figures_source_dir is not None:
        df_texture_proportions.to_csv(
            figures_source_dir / "nuclear_texture_proportions.csv", index=False
        )

    # Print summary statistics
    logging.info("Summary statistics by cluster:")
    summary_stats = (
        df_temp.groupby([GROUP_BY_COLUMN, "texture_category"])
        .size()
        .unstack(fill_value=0)
    )
    logging.info("Count of cells by texture category:")
    logging.info(summary_stats)
    logging.info("Percentage of cells by texture category:")
    logging.info(summary_stats.div(summary_stats.sum(axis=1), axis=0) * 100)


def plot_tile_blur_proportions_roi(
    new_df,
    figures_dir,
    figures_source_dir,
    roi_threshold=-1.0,
    GROUP_BY_COLUMN="Cluster_kmeans10",
):
    """
    Create a stacked bar plot showing proportion of cells with high and low tile-based blur scores per cluster.
    Uses GMM 2D classification if available, otherwise falls back to threshold-based method.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing DAPI_RFSnorm_roi, is_blurred_roi (or is_blurred_gmm_2d_roi), and Cluster_kmeans10 columns
    roi_threshold : float, optional
        Threshold to classify cells as high/low blur (default: -1.0) - used only if GMM 2D not available
    """
    # Check if ROI data is available - prefer GMM 2D, fall back to threshold-based
    has_gmm_2d = "is_blurred_gmm_2d_roi" in new_df.columns
    has_threshold = "is_blurred_roi" in new_df.columns

    if not has_gmm_2d and not has_threshold:
        logging.warning(
            "Warning: Tile-based blur scores not available. Skipping tile blur score proportions plot."
        )
        return

    # Create blur score categories (create temporary column to avoid modifying original)
    df_temp = new_df.copy()

    # Use GMM 2D if available, otherwise use threshold-based
    if has_gmm_2d:
        df_temp["blur_category"] = np.where(
            df_temp["is_blurred_gmm_2d_roi"], "Blurred", "In Focus"
        )
        method_name = "2D GMM"
    else:
        df_temp["blur_category"] = np.where(
            df_temp["is_blurred_roi"], "Blurred", "In Focus"
        )
        method_name = "Threshold"

    # Calculate proportions
    proportions = (
        df_temp.groupby([GROUP_BY_COLUMN, "blur_category"]).size().unstack(fill_value=0)
    )
    proportions = (
        proportions.div(proportions.sum(axis=1), axis=0) * 100
    )  # Convert to percentages

    # Set up the plot
    fig = plt.figure(figsize=(12, 6))

    # Create stacked bar plot with specified colors
    # Colors are applied in alphabetical order: 'Blurred' (red), 'In Focus' (blue)
    proportions.plot(
        kind="bar",
        stacked=True,
        color=["#d62728", "#1f77b4"],  # Red for blurred, blue for in focus
        figsize=(12, 6),
    )

    # Customize the plot
    if has_gmm_2d:
        plt.title(
            "Proportion of Cells by Tile-based Blur Score (2D GMM Classification)",
            fontsize=14,
            pad=20,
        )
    else:
        # Note: roi_threshold is in raw score units, classification uses combined approach
        # Try to get intensity threshold from the threshold config if available
        intensity_threshold_display = getattr(
            plot_tile_blur_proportions_roi, "_intensity_threshold", None
        )
        if intensity_threshold_display is None:
            intensity_threshold_display = 20.0  # Default
        plt.title(
            f"Proportion of Cells by Tile-based Blur Score\n(Raw score threshold: {roi_threshold:.2f}, Intensity threshold: {intensity_threshold_display})",
            fontsize=14,
            pad=20,
        )
    plt.xlabel(GROUP_BY_COLUMN, fontsize=12)
    plt.ylabel("Percentage of Cells", fontsize=12)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")

    # Add percentage labels on the bars
    for c in plt.gca().containers:
        # Add labels
        plt.gca().bar_label(c, fmt="%.1f%%", label_type="center")

    # Add a grid for better readability
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure instead of showing
    plt.savefig(
        figures_dir / "tile_blur_proportions_roi.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data as CSV
    df_blur_proportions = (
        df_temp.groupby([GROUP_BY_COLUMN, "blur_category"])
        .size()
        .reset_index(name="count")
    )
    if figures_source_dir is not None:
        df_blur_proportions.to_csv(
            figures_source_dir / "tile_blur_proportions_roi.csv", index=False
        )

    # Print summary statistics
    logging.info(f"Summary statistics by cluster (tile-based, {method_name}):")
    summary_stats = (
        df_temp.groupby([GROUP_BY_COLUMN, "blur_category"]).size().unstack(fill_value=0)
    )
    logging.info("Count of cells by blur category:")
    logging.info(summary_stats)
    logging.info("Percentage of cells by blur category:")
    logging.info(summary_stats.div(summary_stats.sum(axis=1), axis=0) * 100)


def plot_cell_focus_distribution(
    new_df,
    figures_dir,
    figures_source_dir,
    GROUP_BY_COLUMN="Cluster_kmeans10",
):
    """Plot cell-level focus score distributions using tile-propagated metrics.

    Creates a 1×2 panel (v4 r5 trim):
      - Left: histogram of DAPI_RFSnorm_roi (tile focus score per cell)
      - Right: histogram coloured by GMM blur classification

    The previous bottom row (focus density per cluster, blur prob density per
    cluster) was clustering-related and overlapped with figures in the
    Per-cluster subsection of 8.B. The blur-prob-density-per-cluster view
    moved to its own dedicated function `plot_blur_prob_density_by_cluster`;
    the focus-density-per-cluster view was dropped as redundant with the
    nuclear texture density per cluster figure.
    """
    has_focus = "DAPI_RFSnorm_roi" in new_df.columns
    has_gmm_prob = (
        "blur_prob_gmm_2d_roi" in new_df.columns
    )  # used for source-CSV column inclusion below
    has_gmm_class = "is_blurred_gmm_2d_roi" in new_df.columns

    if not has_focus:
        logging.warning(
            "No DAPI_RFSnorm_roi column — skipping cell focus distribution plot"
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    focus_vals = new_df["DAPI_RFSnorm_roi"].dropna()

    # --- Left: overall focus score histogram ---
    ax = axes[0]
    ax.hist(focus_vals, bins=60, alpha=0.7, edgecolor="black", color="steelblue")
    ax.set_xlabel("Tile focus score (DAPI_RFSnorm_roi)", fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=11)
    ax.set_title("Overall histogram of tile focus scores across all cells", fontsize=12)
    stats_text = f"n={len(focus_vals):,}\nMedian={focus_vals.median():.4f}\nMean={focus_vals.mean():.4f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # --- Right: histogram split by GMM blur class ---
    ax = axes[1]
    if has_gmm_class:
        gmm_col = new_df["is_blurred_gmm_2d_roi"].astype(bool)
        valid = new_df["DAPI_RFSnorm_roi"].notna()
        blurred = new_df.loc[valid & gmm_col, "DAPI_RFSnorm_roi"]
        sharp = new_df.loc[valid & ~gmm_col, "DAPI_RFSnorm_roi"]
        ax.hist(
            sharp,
            bins=60,
            alpha=0.6,
            label=f"In focus ({len(sharp):,})",
            color="#1f77b4",
        )
        ax.hist(
            blurred,
            bins=60,
            alpha=0.6,
            label=f"Blurred ({len(blurred):,})",
            color="#d62728",
        )
        ax.legend(fontsize=10)
        ax.set_title(
            "Histogram split by GMM blur classification (blue = in focus, red = blurred)",
            fontsize=12,
        )
    else:
        ax.hist(focus_vals, bins=60, alpha=0.7, color="steelblue")
        ax.set_title(
            "Histogram split by GMM blur classification (no GMM data)", fontsize=12
        )
    ax.set_xlabel("Tile focus score (DAPI_RFSnorm_roi)", fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "cell_focus_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save source data
    src_cols = ["DAPI_RFSnorm_roi"]
    if has_gmm_class:
        src_cols.append("is_blurred_gmm_2d_roi")
    if has_gmm_prob:
        src_cols.append("blur_prob_gmm_2d_roi")
    if GROUP_BY_COLUMN in new_df.columns:
        src_cols.append(GROUP_BY_COLUMN)
    src = new_df[[c for c in src_cols if c in new_df.columns]].dropna(
        subset=["DAPI_RFSnorm_roi"]
    )
    if figures_source_dir is not None:
        src.to_csv(figures_source_dir / "cell_focus_distribution.csv", index=False)


def plot_tile_focus_gmm_spatial(new_df, figures_dir, figures_source_dir):
    """Single-panel spatial map of tile-based focus, coloured by GMM 2D blur
    classification (red = blurred, blue = in-focus). Falls back to RFSnorm
    viridis colormap when GMM 2D classification is unavailable.

    Replaces the right panel of the now-latent `plot_spatial_comparison` 2-panel
    figure (v4 r5: spatial concordance dropped per user feedback — known low
    concordance not informative; tile-blur spatial overview kept on its own).
    """
    if "DAPI_RFSnorm_roi" not in new_df.columns:
        logging.warning("No DAPI_RFSnorm_roi — skipping tile-focus GMM spatial plot")
        return
    has_gmm_2d = "is_blurred_gmm_2d_roi" in new_df.columns
    valid_mask = new_df["DAPI_RFSnorm_roi"].notna()
    if valid_mask.sum() == 0:
        logging.warning(
            "No valid DAPI_RFSnorm_roi values — skipping tile-focus GMM spatial plot"
        )
        return

    # Phase 11 (v5): aspect-adaptive figure size matching the slide proportions
    # rather than a fixed (8, 8) square. Mirrors plot_grid_roi_focus_heatmap and
    # plot_snr_roi_heatmap so all whole-sample maps render at consistent width.
    _x = new_df.loc[valid_mask, "x"]
    _y = new_df.loc[valid_mask, "y"]
    _x_range = float(_x.max() - _x.min()) if len(_x) else 1.0
    _y_range = float(_y.max() - _y.min()) if len(_y) else 1.0
    _img_aspect = _y_range / _x_range if _x_range > 0 else 1.0
    _panel_width = 6
    _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
    fig, ax = plt.subplots(1, 1, figsize=(_panel_width, _panel_height))
    if has_gmm_2d:
        gmm_valid = valid_mask & new_df["is_blurred_gmm_2d_roi"].notna()
        if gmm_valid.sum() > 0:
            colors = [
                "red" if blurred else "blue"
                for blurred in new_df.loc[gmm_valid, "is_blurred_gmm_2d_roi"]
            ]
            ax.scatter(
                new_df.loc[gmm_valid, "x"],
                -new_df.loc[gmm_valid, "y"],
                s=0.1,
                c=colors,
                rasterized=True,
            )
            ax.set_title(
                "Focus score, coloured by GMM 2D blur classification\n"
                "(red = blurred, blue = in focus)",
                fontsize=12,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No valid GMM 2D classification data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
    else:
        roi_values = new_df.loc[valid_mask, "DAPI_RFSnorm_roi"]
        scatter = ax.scatter(
            new_df.loc[valid_mask, "x"],
            -new_df.loc[valid_mask, "y"],
            s=0.1,
            c=roi_values,
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title("Focus score (DAPI_RFSnorm_roi)", fontsize=12)
        plt.colorbar(scatter, ax=ax, label="DAPI_RFSnorm_roi")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "tile_focus_gmm_spatial.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    # Save source data
    src_cols = ["x", "y", "DAPI_RFSnorm_roi"]
    if has_gmm_2d:
        src_cols.append("is_blurred_gmm_2d_roi")
    src = new_df[[c for c in src_cols if c in new_df.columns]].dropna(
        subset=["DAPI_RFSnorm_roi"]
    )
    if figures_source_dir is not None:
        src.to_csv(figures_source_dir / "tile_focus_gmm_spatial.csv", index=False)


def plot_cell_flagged_maps_combined(new_df, figures_dir, figures_source_dir):
    """Two-panel spatial map combining the nuclear texture flagged and
    blurriness flagged cell views into one figure for direct visual
    comparison. Left panel mirrors the standalone ccfs_thresholded.png;
    right panel mirrors the standalone tile_focus_gmm_spatial.png. Both
    panels share the sample's coordinate system and use the aspect-adaptive
    sizing pattern from plot_snr_roi_heatmap so the rendered figure aligns
    with the §3.4 SNR heatmap panel layout.
    """
    needed = ("is_low_nuclear_texture", "is_blurred_gmm_2d_roi")
    if all(c not in new_df.columns for c in needed):
        logging.warning(
            "No nuclear texture or blurry-(GMM) columns — skipping combined cell-flagged maps"
        )
        return
    if "x" not in new_df.columns or "y" not in new_df.columns:
        logging.warning("No x / y coordinates — skipping combined cell-flagged maps")
        return

    _x = new_df["x"].dropna()
    _y = new_df["y"].dropna()
    if len(_x) == 0 or len(_y) == 0:
        logging.warning("Empty x / y — skipping combined cell-flagged maps")
        return
    _x_range = float(_x.max() - _x.min()) if _x.max() > _x.min() else 1.0
    _y_range = float(_y.max() - _y.min()) if _y.max() > _y.min() else 1.0
    _img_aspect = _y_range / _x_range if _x_range > 0 else 1.0
    panel_width = 6
    panel_height = max(panel_width * 0.5, panel_width * _img_aspect)
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width + 2, panel_height))

    # ── Left panel: nuclear texture-flagged ────────────────────────────
    ax = axes[0]
    if "is_low_nuclear_texture" in new_df.columns:
        _mask_low = new_df["is_low_nuclear_texture"].fillna(False).astype(bool)
        _high = new_df[~_mask_low]
        _low = new_df[_mask_low]
        if len(_high) > 0:
            ax.scatter(_high["x"], -_high["y"], s=0.1, color="#64B5F6", rasterized=True)
        if len(_low) > 0:
            # Slightly larger red points so flagged cells remain visible
            # against the dense blue background and on the white figure bg.
            ax.scatter(_low["x"], -_low["y"], s=0.5, color="red", rasterized=True)
    ax.set_title(
        "Spatial distribution of low nuclear texture cells\n(red = low nuclear texture, blue = high)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for _spine in ax.spines.values():
        _spine.set_edgecolor("black")
        _spine.set_linewidth(1.2)

    # ── Right panel: blurriness-flagged (tile-GMM inherited) ───────────
    ax = axes[1]
    if "is_blurred_gmm_2d_roi" in new_df.columns:
        _valid_mask = new_df["is_blurred_gmm_2d_roi"].notna()
        if _valid_mask.any():
            _valid_df = new_df.loc[_valid_mask]
            _blur_mask = _valid_df["is_blurred_gmm_2d_roi"].astype(bool)
            _in_focus = _valid_df[~_blur_mask]
            _blurred = _valid_df[_blur_mask]
            if len(_in_focus) > 0:
                ax.scatter(
                    _in_focus["x"],
                    -_in_focus["y"],
                    s=0.1,
                    color="#64B5F6",
                    rasterized=True,
                )
            if len(_blurred) > 0:
                # Same size as in-focus: GMM blur flags typically cover contiguous
                # regions, so enlargement is not needed for visibility and would
                # swamp the panel.
                ax.scatter(
                    _blurred["x"], -_blurred["y"], s=0.1, color="red", rasterized=True
                )
    ax.set_title(
        "Spatial distribution of blurry cells\n(red = blurred, blue = in focus)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for _spine in ax.spines.values():
        _spine.set_edgecolor("black")
        _spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(figures_dir / "cell_flagged_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_blur_prob_density_by_cluster(
    new_df,
    figures_dir,
    figures_source_dir,
    GROUP_BY_COLUMN="Cluster_kmeans10",
):
    """Single-panel KDE of `blur_prob_gmm_2d_roi` per expression cluster, with
    a vertical line at the blur threshold (0.5). Parallels the existing
    nuclear texture density by cluster figure (CCFS density per cluster) but
    for tile-blur probability — sits in 8.B's Per-cluster subsection.

    Extracted from the bottom-right panel of the v4-r4 `plot_cell_focus_distribution`
    (which was a 2×2 grid; trimmed to 1×2 in v4 r5 because the bottom row
    was clustering-related and belonged in the Per-cluster subsection).
    """
    if "blur_prob_gmm_2d_roi" not in new_df.columns:
        logging.warning(
            "No blur_prob_gmm_2d_roi column — skipping per-cluster blur probability density"
        )
        return

    # Aesthetic harmonization with plot_nuclear_texture_density: same
    # `palette="husl"` + `fill=True` + `alpha=0.3` so cluster colours match
    # across the two density figures (same Cluster_kmeans10 hue → same colour
    # assignment by seaborn) and both have semi-transparent fill.
    fig = plt.figure(figsize=(12, 6))
    if GROUP_BY_COLUMN in new_df.columns:
        df_kde = new_df[[GROUP_BY_COLUMN, "blur_prob_gmm_2d_roi"]].dropna()
        if len(df_kde) > 0:
            ax = sns.kdeplot(
                data=df_kde,
                x="blur_prob_gmm_2d_roi",
                hue=GROUP_BY_COLUMN,
                palette="husl",
                common_norm=False,
                fill=True,
                alpha=0.3,
            )
            plt.axvline(
                x=0.5,
                color="black",
                linestyle="--",
                alpha=0.5,
                label="Blur threshold (0.5)",
            )
            # Reformat legend labels to "Cluster N" prefix and include the
            # threshold line — matches plot_nuclear_texture_density legend.
            legend = ax.get_legend()
            if legend is not None:
                handles = legend.legend_handles
                labels = [f"Cluster {label.get_text()}" for label in legend.get_texts()]
                threshold_line = plt.Line2D(
                    [0], [0], color="black", linestyle="--", alpha=0.5
                )
                handles = [threshold_line] + handles
                labels = ["Blur threshold (0.5)"] + labels
                plt.legend(
                    handles,
                    labels,
                    title=GROUP_BY_COLUMN,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
        plt.title(
            f"Distribution of GMM blur probability by {GROUP_BY_COLUMN}",
            fontsize=14,
            pad=20,
        )
    else:
        ax = plt.gca()
        new_df["blur_prob_gmm_2d_roi"].dropna().plot.kde(ax=ax, color="steelblue")
        plt.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
        plt.title(
            "GMM blur probability density (threshold at 0.5)", fontsize=14, pad=20
        )
    plt.xlabel("P(blur component)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "blur_prob_density_by_cluster.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    src_cols = ["blur_prob_gmm_2d_roi"]
    if GROUP_BY_COLUMN in new_df.columns:
        src_cols.append(GROUP_BY_COLUMN)
    src = new_df[[c for c in src_cols if c in new_df.columns]].dropna(
        subset=["blur_prob_gmm_2d_roi"]
    )
    if figures_source_dir is not None:
        src.to_csv(figures_source_dir / "blur_prob_density_by_cluster.csv", index=False)


def plot_intensity_transcript_correlation(new_df, figures_dir, figures_source_dir):
    """Spearman correlation heatmap between per-cell quality metrics and
    transcript counts.

    Includes six per-cell variables: nuclear texture (CCFS_DAPI), focus
    score (tile-level, propagated to cell), three channel intensities
    (DAPI / Boundary / IntRNA), and transcript counts.

    Diagnostic for "which quality axis is critical for transcript yield
    in THIS sample's tissue type?". Per-channel intensity_critical cutoffs
    (DAPI=500, Boundary=100, IntRNA=300) are calibrated on lung/liver and
    may not be appropriate for tissues like brain (where DAPI is
    unreliable due to large neurons with naturally lower nuclear
    contrast). The heatmap shows which axis actually correlates with
    transcript yield, helping the analyst weight per-channel verdicts in
    §4.1 appropriately.

    Saves `figures/intensity_transcript_correlation.png` + source CSV.
    Skips silently if fewer than 2 of the input columns are available.
    """
    # All columns are read from `new_df`, the post-merge cell-aligned
    # frame produced at line ~5502 by `pd.merge(df_spatial, myData, on="CellID")`.
    # `mean_intensity` (DAPI) comes in via the merge from `myData`. Reading
    # all columns from a single index-aligned DataFrame guarantees that
    # row N is the SAME cell across all series — critical for the
    # correlation to be scientifically meaningful (reading half from `myData`
    # and half from `new_df` would pair values across DIFFERENT cells).
    cols = {}
    # Image-quality metrics first (nuclear texture + focus) — these are the
    # axes a reader is checking "is X the operative quality signal for my
    # tissue?". Stain intensities follow; transcript counts last.
    if "CCFS_DAPI" in new_df.columns:
        cols["Nuclear texture"] = new_df["CCFS_DAPI"]
    if "DAPI_RFSnorm_roi" in new_df.columns:
        cols["Focus score"] = new_df["DAPI_RFSnorm_roi"]
    if "mean_intensity" in new_df.columns:
        cols["DAPI"] = new_df["mean_intensity"]
    if "mean_intensity_Boundary" in new_df.columns:
        cols["Boundary"] = new_df["mean_intensity_Boundary"]
    if "mean_intensity_IntRNA" in new_df.columns:
        cols["IntRNA"] = new_df["mean_intensity_IntRNA"]
    if "transcript_counts" in new_df.columns:
        cols["transcripts"] = new_df["transcript_counts"]

    if len(cols) < 2:
        logging.warning(
            "Fewer than 2 quality / transcript columns available — skipping "
            "per-cell correlation heatmap."
        )
        return

    df_corr_input = pd.DataFrame(cols).dropna()
    if len(df_corr_input) < 30:
        logging.warning(
            "Fewer than 30 cells with all quality / transcript columns — "
            "skipping per-cell correlation heatmap (rho unstable)."
        )
        return

    # Spearman (rank-based) — robust to non-normal distributions and outliers
    # which intensity / transcript count data often exhibits.
    corr_matrix = df_corr_input.corr(method="spearman")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        f"Per-cell quality metrics vs transcript count correlation (Spearman ρ; n={len(df_corr_input):,} cells)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / "intensity_transcript_correlation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    # Save the input data (long form: cell + 4 columns) for reproducibility.
    if figures_source_dir is not None:
        df_corr_input.to_csv(
            figures_source_dir / "intensity_transcript_correlation.csv", index=False
        )


def plot_nuclear_texture_density(
    new_df,
    figures_dir,
    figures_source_dir,
    GROUP_BY_COLUMN="Cluster_kmeans10",
    ccfs_low_texture_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
):
    """
    Create a density plot of CCFS_DAPI values grouped by Cluster_kmeans10.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing CCFS_DAPI and Cluster_kmeans10 columns
    """
    # Set up the plot
    fig = plt.figure(figsize=(12, 6))

    # Create the density plot
    ax = sns.kdeplot(
        data=new_df,
        x="CCFS_DAPI",
        hue=GROUP_BY_COLUMN,
        palette="husl",
        common_norm=False,  # Normalize each cluster separately
        fill=True,
        alpha=0.3,
    )  # Make the fill semi-transparent

    # Add a vertical line for the blur threshold
    plt.axvline(
        x=ccfs_low_texture_threshold,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Low Texture Threshold ({ccfs_low_texture_threshold})",
    )

    # Customize the plot
    plt.title(
        f"Distribution of Nuclear Texture Scores (CCFS_DAPI) by {GROUP_BY_COLUMN}",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("CCFS_DAPI (Nuclear Texture Score)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Add a grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Get the current legend
    legend = ax.get_legend()

    # Get the handles and labels
    handles = legend.legend_handles
    labels = [f"Cluster {label.get_text()}" for label in legend.get_texts()]

    # Add the threshold line to the legend
    threshold_line = plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.5)
    handles = [threshold_line] + handles
    labels = ["Default Threshold"] + labels

    # Create new legend
    plt.legend(
        handles,
        labels,
        title=GROUP_BY_COLUMN,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure instead of showing
    plt.savefig(
        figures_dir / "nuclear_texture_density.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data as CSV
    if figures_source_dir is not None:
        df_density_data = new_df[["CCFS_DAPI", GROUP_BY_COLUMN]].copy()
        df_density_data.to_csv(
            figures_source_dir / "nuclear_texture_density.csv", index=False
        )


def plot_nuclear_texture_vs_transcripts(
    new_df,
    figures_dir,
    figures_source_dir,
    log_scale=False,
    ccfs_low_texture_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
):
    """
    Create a scatter plot of nuclear texture scores (CCFS_DAPI) vs transcript counts.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing CCFS_DAPI and transcript_counts columns
    log_scale : bool, optional
        Whether to use log scale for x-axis (default: False)
    """
    # Set up the plot
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    _valid = new_df[["transcript_counts", "CCFS_DAPI"]].dropna()
    x_vals = _valid["transcript_counts"].values.astype(np.float64)
    y_vals = _valid["CCFS_DAPI"].values.astype(np.float64)

    # Density-coloured scatter (same idiom as §4.4.b and the CCFS rank
    # comparison plot at line ~6692). High-overlap regions paint at the
    # high-density end of viridis, revealing trend / correlation that
    # pure alpha-blending obscures at large N.
    try:
        from scipy.stats import gaussian_kde

        # KDE in display coordinates: log space when axes are log-scaled
        # so density reflects what the reader sees.
        if log_scale:
            x_kde = np.log10(np.clip(x_vals, 1e-9, None))
            y_kde = np.log10(np.clip(y_vals, 1e-9, None))
        else:
            x_kde, y_kde = x_vals, y_vals

        max_kde_pts = 8000
        n_pts = len(x_kde)
        if n_pts > max_kde_pts:
            rng = np.random.default_rng(42)
            sidx = rng.choice(n_pts, max_kde_pts, replace=False)
            kde = gaussian_kde(np.vstack([x_kde[sidx], y_kde[sidx]]))
        else:
            kde = gaussian_kde(np.vstack([x_kde, y_kde]))
        # Subsample the PLOTTED points to the standard cap (see
        # plot_per_cell_intensity_vs_transcripts): ~15M cells overplot into a solid
        # cloud, so ~10k render identically in ~1s vs ~35s. Density colour is still
        # from the full-data KDE fit.
        pidx = _subsample_idx(np.ones(n_pts, dtype=bool))
        zc = _kde_density_grid(kde, x_kde[pidx], y_kde[pidx])
        order = zc.argsort()  # draw high-density points last (on top)
        scatter = ax.scatter(
            x_vals[pidx][order],
            y_vals[pidx][order],
            c=zc[order],
            s=8,
            cmap="viridis",
            alpha=0.6,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Cell density (relative)")
    except Exception as _kde_err:
        logging.warning(
            f"Density coloring failed for nuclear texture scatter ({_kde_err}); "
            "falling back to alpha-blended scatter."
        )
        fidx = _subsample_idx(np.ones(len(x_vals), dtype=bool))
        ax.scatter(
            x_vals[fidx],
            y_vals[fidx],
            alpha=0.3,
            s=8,
            color="C0",
            rasterized=True,
        )

    # Add a horizontal line for the blur threshold
    plt.axhline(
        y=ccfs_low_texture_threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Low Texture Threshold ({ccfs_low_texture_threshold})",
    )

    # Customize the plot
    title = "Per-cell nuclear texture score vs transcript count"
    if log_scale:
        title += " (log scale)"
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("CCFS_DAPI (Nuclear Texture Score)", fontsize=12)
    plt.xlabel("Transcript Counts", fontsize=12)

    # Set log scale for both axes if requested
    if log_scale:
        plt.xscale("log")
        plt.xlabel("Transcript Counts (log scale)", fontsize=12)
        plt.yscale("log")
        plt.ylabel("CCFS_DAPI (Nuclear Texture Score) (log scale)", fontsize=12)

    # Add a grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add legend
    plt.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure instead of showing
    suffix = "_log" if log_scale else ""
    plt.savefig(
        figures_dir / f"nuclear_texture_vs_transcripts{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save data as CSV
    if figures_source_dir is not None:
        df_scatter_data = new_df[["transcript_counts", "CCFS_DAPI"]].copy()
        df_scatter_data.to_csv(
            figures_source_dir / f"nuclear_texture_vs_transcripts{suffix}.csv",
            index=False,
        )

    # Print summary statistics
    logging.info("Summary statistics:")
    logging.info("Nuclear Texture Score (CCFS_DAPI):")
    logging.info(new_df["CCFS_DAPI"].describe().round(4))
    logging.info("Transcript Counts:")
    logging.info(new_df["transcript_counts"].describe())


def plot_per_cell_intensity_vs_transcripts(
    new_df,
    figures_dir,
    figures_source_dir,
    log_scale=False,
):
    """Per-cell mean intensity (DAPI / Boundary / IntRNA) vs transcript counts.

    Three separate scatter plots, same format as
    `plot_nuclear_texture_vs_transcripts` so readers learn the axes once.
    Used in §4.4.b "Stain intensity vs transcript count" alongside the
    higher-level `intensity_transcript_correlation.png` heatmap.

    Skips silently when a channel intensity column is missing (e.g. DAPI-only
    bundles produce no Boundary / IntRNA columns).

    Parameters
    ----------
    new_df : pandas.DataFrame
        Per-cell DataFrame; expected columns: `transcript_counts`,
        `mean_intensity`, `mean_intensity_Boundary`, `mean_intensity_IntRNA`.
    log_scale : bool, optional
        If True, render both axes on a log scale (default False).
    """
    channels = [
        ("DAPI", "mean_intensity"),
        ("Boundary", "mean_intensity_Boundary"),
        ("IntRNA", "mean_intensity_IntRNA"),
    ]

    for ch_label, col in channels:
        if col not in new_df.columns:
            logging.warning(
                f"Column {col!r} not in per-cell DataFrame — "
                f"skipping {ch_label} intensity vs transcripts scatter."
            )
            continue

        _valid = new_df[[col, "transcript_counts"]].dropna()
        if len(_valid) == 0:
            logging.warning(
                f"No valid (non-NaN) cells for {ch_label} intensity vs transcripts — skipping."
            )
            continue

        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        x_vals = _valid["transcript_counts"].values.astype(np.float64)
        y_vals = _valid[col].values.astype(np.float64)

        # Density-colored scatter — high-overlap regions paint in the
        # high-density end of the colormap, making trend / correlation
        # readable even at large N where simple alpha blending saturates.
        # Pattern matches plot_ccfs_vs_roi_comparison (line ~6692).
        try:
            from scipy.stats import gaussian_kde

            # Compute KDE in display coordinates: log space when the axes
            # are log-scaled, so the density gradient reflects what the
            # reader sees rather than the raw coordinate distance.
            if log_scale:
                x_kde = np.log10(np.clip(x_vals, 1e-9, None))
                y_kde = np.log10(np.clip(y_vals, 1e-9, None))
            else:
                x_kde, y_kde = x_vals, y_vals

            max_kde_pts = 8000
            n_pts = len(x_kde)
            if n_pts > max_kde_pts:
                rng = np.random.default_rng(42)
                sidx = rng.choice(n_pts, max_kde_pts, replace=False)
                kde = gaussian_kde(np.vstack([x_kde[sidx], y_kde[sidx]]))
            else:
                kde = gaussian_kde(np.vstack([x_kde, y_kde]))
            # Subsample the PLOTTED points to the standard scatter cap. A
            # density-colored scatter of ~15M cells overplots into a solid cloud at
            # display resolution, so ~10k points render the identical cloud in ~1s
            # vs ~32s/channel (measured 96s total). Density colour is still computed
            # per plotted point from the full-data KDE fit; matches _subsample_idx as
            # used by roi_focus_vs_intensity and the other density scatters.
            pidx = _subsample_idx(np.ones(n_pts, dtype=bool))
            zc = _kde_density_grid(kde, x_kde[pidx], y_kde[pidx])
            order = zc.argsort()  # draw high-density points last (on top)
            scatter = ax.scatter(
                x_vals[pidx][order],
                y_vals[pidx][order],
                c=zc[order],
                s=8,
                cmap="viridis",
                alpha=0.6,
                rasterized=True,
            )
            plt.colorbar(scatter, ax=ax, label="Cell density (relative)")
        except Exception as _kde_err:
            logging.warning(
                f"Density coloring failed for {ch_label} ({_kde_err}); "
                "falling back to alpha-blended scatter."
            )
            fidx = _subsample_idx(np.ones(len(x_vals), dtype=bool))
            ax.scatter(
                x_vals[fidx],
                y_vals[fidx],
                alpha=0.3,
                s=8,
                color="C0",
                rasterized=True,
            )

        title = f"Per-cell {ch_label} mean intensity vs transcript count"
        if log_scale:
            title += " (log scale)"
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel(f"{ch_label} mean intensity (16-bit counts)", fontsize=12)
        plt.xlabel("Transcript counts", fontsize=12)

        if log_scale:
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Transcript counts (log scale)", fontsize=12)
            plt.ylabel(
                f"{ch_label} mean intensity (16-bit counts, log scale)", fontsize=12
            )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        suffix = "_log" if log_scale else ""
        fname = f"{ch_label.lower()}_intensity_vs_transcripts{suffix}"
        plt.savefig(figures_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        if figures_source_dir is not None:
            _valid.to_csv(figures_source_dir / f"{fname}.csv", index=False)

        logging.info(
            f"Saved {fname}.png (n={len(_valid):,}, "
            f"median intensity={_valid[col].median():.0f})"
        )


def plot_gmm_focus_vs_transcripts(
    new_df,
    figures_dir,
    figures_source_dir,
):
    """Scatter plot of tile-level GMM focus score vs transcript counts, coloured by blur class.

    Mirrors plot_nuclear_texture_vs_transcripts but uses the tile-level Laplacian
    GMM focus score (DAPI_RFSnorm_roi) propagated to cells, with points coloured
    by GMM blur classification.
    """
    focus_col = "DAPI_RFSnorm_roi"
    tx_col = "transcript_counts"
    gmm_col = "is_blurred_gmm_2d_roi"

    if focus_col not in new_df.columns or tx_col not in new_df.columns:
        logging.warning(
            "Missing %s or %s — skipping GMM focus vs transcripts plot.",
            focus_col,
            tx_col,
        )
        return

    df = new_df[
        [tx_col, focus_col] + ([gmm_col] if gmm_col in new_df.columns else [])
    ].dropna()
    if df.empty:
        return

    has_gmm = gmm_col in df.columns
    fig, ax = plt.subplots(figsize=(12, 6))

    if has_gmm:
        sharp = df[~df[gmm_col].astype(bool)]
        blurred = df[df[gmm_col].astype(bool)]
        ax.scatter(
            sharp[tx_col],
            sharp[focus_col],
            alpha=0.3,
            s=8,
            color="#1f77b4",
            label=f"In Focus ({len(sharp):,})",
            rasterized=True,
        )
        ax.scatter(
            blurred[tx_col],
            blurred[focus_col],
            alpha=0.3,
            s=8,
            color="#d62728",
            label=f"Blurred ({len(blurred):,})",
            rasterized=True,
        )
        ax.legend(fontsize=10)
    else:
        ax.scatter(df[tx_col], df[focus_col], alpha=0.3, s=8, rasterized=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Transcript Counts (log scale)", fontsize=12)
    ax.set_ylabel("Tile Focus Score (DAPI_RFSnorm_roi, log scale)", fontsize=12)
    ax.set_title("GMM Tile Focus Score vs Transcript Counts", fontsize=14, pad=20)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "gmm_focus_vs_transcripts.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save source data
    if figures_source_dir is not None:
        src_cols = [tx_col, focus_col]
        if has_gmm:
            src_cols.append(gmm_col)
        df[src_cols].to_csv(
            figures_source_dir / "gmm_focus_vs_transcripts.csv", index=False
        )

    logging.info("GMM focus vs transcripts: %d cells plotted.", len(df))


def plot_ccfs_vs_roi_comparison(new_df, figures_dir, figures_source_dir):
    """
    Create comparison plots between CCFS (nuclei-based) and cell-independent tile-based focus scores.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing both CCFS_DAPI and DAPI_RFSnorm_roi columns
        (DAPI_RFSnorm_roi contains cell-independent grid ROI focus scores transferred to cells)
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot 1: Ranked scatter plot with density coloring
    ax = axes[0, 0]
    # Calculate ranks (higher rank = higher focus score)
    ccfs_ranks = new_df["CCFS_DAPI"].rank(method="average")
    roi_ranks = new_df["DAPI_RFSnorm_roi"].rank(method="average")

    # Create scatter plot with density coloring (subsample for KDE performance)
    try:
        from scipy.stats import gaussian_kde

        valid = ccfs_ranks.dropna().index.intersection(roi_ranks.dropna().index)
        x_vals, y_vals = ccfs_ranks[valid].values, roi_ranks[valid].values
        max_kde_pts = 8000
        if len(x_vals) > max_kde_pts:
            rng = np.random.default_rng(42)
            sidx = rng.choice(len(x_vals), max_kde_pts, replace=False)
            xy_sub = np.vstack([x_vals[sidx], y_vals[sidx]])
            kde = gaussian_kde(xy_sub)
            z = _kde_density_grid(kde, x_vals, y_vals)
        else:
            z = gaussian_kde(np.vstack([x_vals, y_vals]))(np.vstack([x_vals, y_vals]))
        idx = z.argsort()
        scatter = ax.scatter(
            x_vals[idx],
            y_vals[idx],
            c=z[idx],
            s=1,
            cmap="viridis",
            alpha=0.6,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Density")
    except (ImportError, Exception):
        # Fallback if scipy not available or density calculation fails
        ax.scatter(ccfs_ranks, roi_ranks, alpha=0.3, s=1, marker="o", rasterized=True)

    ax.set_xlabel("CCFS_DAPI Rank (Nuclei-based)", fontsize=12)
    ax.set_ylabel("DAPI_RFSnorm_roi Rank (Cell-Independent Tile-based)", fontsize=12)
    ax.set_title("CCFS vs Cell-Independent Tile Focus Score (Ranked)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add correlation coefficient (Spearman rank-based correlation)
    correlation = new_df["CCFS_DAPI"].corr(
        new_df["DAPI_RFSnorm_roi"], method="spearman"
    )
    ax.text(
        0.05,
        0.95,
        f"Spearman Correlation: {correlation:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 2: Density plot overlay
    ax = axes[0, 1]
    ax.scatter(
        new_df["CCFS_DAPI"],
        new_df["DAPI_RFSnorm_roi"],
        alpha=0.1,
        s=0.5,
        marker="o",
        c="blue",
        label="Data points",
        rasterized=True,
    )
    # Add 2D density contour (subsample for KDE performance)
    try:
        from scipy.stats import gaussian_kde

        valid_mask = new_df["CCFS_DAPI"].notna() & new_df["DAPI_RFSnorm_roi"].notna()
        x_vals = new_df.loc[valid_mask, "CCFS_DAPI"].values
        y_vals = new_df.loc[valid_mask, "DAPI_RFSnorm_roi"].values
        max_kde_pts = 8000
        if len(x_vals) > max_kde_pts:
            rng = np.random.default_rng(42)
            sidx = rng.choice(len(x_vals), max_kde_pts, replace=False)
            kde = gaussian_kde(np.vstack([x_vals[sidx], y_vals[sidx]]))
            z = _kde_density_grid(kde, x_vals, y_vals)
        else:
            z = gaussian_kde(np.vstack([x_vals, y_vals]))(np.vstack([x_vals, y_vals]))
        idx = z.argsort()
        ax.scatter(
            x_vals[idx],
            y_vals[idx],
            c=z[idx],
            s=1,
            cmap="viridis",
            alpha=0.6,
            rasterized=True,
        )
    except (ImportError, Exception):
        pass  # Skip density overlay if unavailable
    ax.set_xlabel("CCFS_DAPI (Nuclei-based)", fontsize=12)
    ax.set_ylabel("DAPI_RFSnorm_roi (Cell-Independent Tile-based)", fontsize=12)
    ax.set_title("CCFS vs Cell-Independent Tile Focus Score (Density)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Plot 3: Agreement/disagreement classification
    ax = axes[1, 0]
    # Prefer GMM 2D if available, otherwise use threshold-based
    has_gmm_2d = "is_blurred_gmm_2d_roi" in new_df.columns
    has_threshold = "is_blurred_roi" in new_df.columns

    if "is_low_nuclear_texture" in new_df.columns and (has_gmm_2d or has_threshold):
        # Use GMM 2D if available, otherwise threshold-based
        if has_gmm_2d:
            roi_blurred_col = "is_blurred_gmm_2d_roi"
            roi_high_focus = ~new_df["is_blurred_gmm_2d_roi"]
            method_label = "2D GMM"
        else:
            roi_blurred_col = "is_blurred_roi"
            roi_high_focus = new_df["is_high_focus_roi"]
            method_label = "Threshold"

        # Create agreement categories
        agreement = new_df["is_low_nuclear_texture"] == new_df[roi_blurred_col]
        agree_high = new_df["is_high_nuclear_texture"] & roi_high_focus
        agree_low = new_df["is_low_nuclear_texture"] & new_df[roi_blurred_col]
        disagree = ~agreement

        # Plot
        ax.scatter(
            new_df.loc[agree_high, "CCFS_DAPI"],
            new_df.loc[agree_high, "DAPI_RFSnorm_roi"],
            alpha=0.3,
            s=1,
            c="green",
            label="Both high focus",
            marker="o",
            rasterized=True,
        )
        ax.scatter(
            new_df.loc[agree_low, "CCFS_DAPI"],
            new_df.loc[agree_low, "DAPI_RFSnorm_roi"],
            alpha=0.3,
            s=1,
            c="red",
            label="Both blurred",
            marker="o",
            rasterized=True,
        )
        ax.scatter(
            new_df.loc[disagree, "CCFS_DAPI"],
            new_df.loc[disagree, "DAPI_RFSnorm_roi"],
            alpha=0.5,
            s=2,
            c="orange",
            label="Disagree",
            marker="x",
            rasterized=True,
        )
        ax.set_xlabel("CCFS_DAPI (Nuclei-based)", fontsize=12)
        ax.set_ylabel("DAPI_RFSnorm_roi (Cell-Independent Tile-based)", fontsize=12)
        ax.set_title(
            f"Classification Agreement (CCFS vs Tile-based {method_label})", fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add agreement percentage
        agreement_rate = agreement.sum() / len(new_df) * 100
        ax.text(
            0.05,
            0.95,
            f"Agreement: {agreement_rate:.2f}%",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 4: Distribution comparison
    ax = axes[1, 1]
    ax.hist(
        new_df["CCFS_DAPI"].dropna(),
        bins=50,
        alpha=0.5,
        label="CCFS_DAPI (Nuclei-based)",
        color="blue",
        density=True,
    )
    ax.hist(
        new_df["DAPI_RFSnorm_roi"].dropna(),
        bins=50,
        alpha=0.5,
        label="DAPI_RFSnorm_roi (Cell-Independent Tile)",
        color="red",
        density=True,
    )
    ax.set_xlabel("Focus Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution Comparison (CCFS vs Cell-Independent Tile)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "ccfs_vs_roi_comparison.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        figures_dir / "ccfs_vs_roi_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save data as CSV (including ranks)
    df_comparison = new_df[["CCFS_DAPI", "DAPI_RFS_roi", "DAPI_RFSnorm_roi"]].copy()
    # Add ranks for analysis
    df_comparison["CCFS_DAPI_rank"] = new_df["CCFS_DAPI"].rank(method="average")
    df_comparison["DAPI_RFSnorm_roi_rank"] = new_df["DAPI_RFSnorm_roi"].rank(
        method="average"
    )

    # Add classification columns - prefer GMM 2D, include threshold-based if available
    if "is_low_nuclear_texture" in new_df.columns:
        df_comparison["is_low_nuclear_texture"] = new_df["is_low_nuclear_texture"]

        # GMM 2D classification (preferred)
        if "is_blurred_gmm_2d_roi" in new_df.columns:
            df_comparison["is_blurred_gmm_2d_roi"] = new_df["is_blurred_gmm_2d_roi"]
            df_comparison["classification_agreement_gmm_2d"] = (
                new_df["is_low_nuclear_texture"] == new_df["is_blurred_gmm_2d_roi"]
            )
            if "blur_prob_gmm_2d_roi" in new_df.columns:
                df_comparison["blur_prob_gmm_2d_roi"] = new_df["blur_prob_gmm_2d_roi"]

        # Threshold-based classification (for comparison)
        if "is_blurred_roi" in new_df.columns:
            df_comparison["is_blurred_roi"] = new_df["is_blurred_roi"]
            df_comparison["classification_agreement"] = (
                new_df["is_low_nuclear_texture"] == new_df["is_blurred_roi"]
            )

    if figures_source_dir is not None:
        df_comparison.to_csv(
            figures_source_dir / "ccfs_vs_roi_comparison.csv", index=False
        )


def plot_spatial_comparison(new_df, myData, figures_dir, figures_source_dir):
    """
    Create spatial comparison plots showing both nuclei-based and tile-based focus scores.

    Parameters:
    -----------
    new_df : pandas DataFrame
        DataFrame containing spatial and focus score data
    myData : pandas DataFrame
        DataFrame with nuclei-based measurements (for spatial coordinates)
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    """
    # Compute figure size from data extents to avoid empty white space
    _x_range = (
        myData["centroid-1"].max() - myData["centroid-1"].min()
        if "centroid-1" in myData.columns
        else 1
    )
    _y_range = (
        myData["centroid-0"].max() - myData["centroid-0"].min()
        if "centroid-0" in myData.columns
        else 1
    )
    _data_aspect = _y_range / _x_range if _x_range > 0 else 1.0
    _pw = 7
    _ph = max(4, _pw * _data_aspect)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _pw + 2, _ph))

    # Plot 1: Nuclei-based CCFS spatial
    ax = axes[0]
    if "centroid-1" in myData.columns and "centroid-0" in myData.columns:
        # Filter out NaN values for plotting
        valid_mask = myData["CCFS_DAPI"].notna()
        if valid_mask.sum() > 0:
            # Use CCFS_DAPI directly (not negated) with auto-scaling
            ccfs_values = myData.loc[valid_mask, "CCFS_DAPI"]
            scatter = ax.scatter(
                myData.loc[valid_mask, "centroid-1"],
                -myData.loc[valid_mask, "centroid-0"],
                s=0.1,
                c=ccfs_values,
                cmap="viridis",
                rasterized=True,
            )
            ax.set_title("Nuclei-based Focus Score (CCFS_DAPI)", fontsize=14)
            ax.set_facecolor("black")
            ax.set_aspect("equal")
            plt.colorbar(scatter, ax=ax, label="CCFS_DAPI")
        else:
            ax.text(
                0.5,
                0.5,
                "No valid CCFS_DAPI data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )
            ax.set_facecolor("black")
            ax.set_aspect("equal")

    # Plot 2: Tile-based RFS spatial (colored by GMM 2D classification if available)
    ax = axes[1]
    if "DAPI_RFSnorm_roi" in new_df.columns:
        # Prefer GMM 2D classification for coloring, otherwise use raw focus scores
        has_gmm_2d = "is_blurred_gmm_2d_roi" in new_df.columns
        valid_mask = new_df["DAPI_RFSnorm_roi"].notna()

        if has_gmm_2d:
            # Color by GMM 2D classification (blurred vs in-focus)
            gmm_valid_mask = valid_mask & new_df["is_blurred_gmm_2d_roi"].notna()
            if gmm_valid_mask.sum() > 0:
                # Color: red for blurred, blue for in-focus
                colors = [
                    "red" if blurred else "blue"
                    for blurred in new_df.loc[gmm_valid_mask, "is_blurred_gmm_2d_roi"]
                ]
                ax.scatter(
                    new_df.loc[gmm_valid_mask, "x"],
                    -new_df.loc[gmm_valid_mask, "y"],
                    s=0.1,
                    c=colors,
                    alpha=0.5,
                    rasterized=True,
                )
                ax.set_title(
                    "Tile-based Focus Score (GMM 2D Classification)", fontsize=14
                )
                # Add legend
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor="blue", label="In-Focus (2D GMM)"),
                    Patch(facecolor="red", label="Blurred (2D GMM)"),
                ]
                ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid GMM 2D classification data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
                )
        else:
            # Fallback: use raw focus scores with color scale
            if valid_mask.sum() > 0:
                roi_values = new_df.loc[valid_mask, "DAPI_RFSnorm_roi"]
                scatter = ax.scatter(
                    new_df.loc[valid_mask, "x"],
                    -new_df.loc[valid_mask, "y"],
                    s=0.1,
                    c=roi_values,
                    cmap="viridis",
                    rasterized=True,
                )
                ax.set_title("Tile-based Focus Score (DAPI_RFSnorm_roi)", fontsize=14)
                plt.colorbar(scatter, ax=ax, label="DAPI_RFSnorm_roi")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid DAPI_RFSnorm_roi data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
                )

        ax.set_facecolor("black")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "spatial_comparison_nuclei_vs_roi.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        figures_dir / "spatial_comparison_nuclei_vs_roi.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save data as CSV
    if "centroid-1" in myData.columns:
        df_spatial_comparison = pd.DataFrame(
            {
                "x_nuclei": myData["centroid-1"],
                "y_nuclei": -myData["centroid-0"],
                "CCFS_DAPI": myData["CCFS_DAPI"],
            }
        )
        if "DAPI_RFSnorm_roi" in new_df.columns:
            # Try to merge on coordinates
            df_spatial_comparison = df_spatial_comparison.merge(
                new_df[["x", "y", "DAPI_RFSnorm_roi"]],
                left_on=["x_nuclei", "y_nuclei"],
                right_on=["x", "y"],
                how="left",
            )
        if figures_source_dir is not None:
            df_spatial_comparison.to_csv(
                figures_source_dir / "spatial_comparison_nuclei_vs_roi.csv",
                index=False,
            )


def save_cell_qc_metrics(new_df, outdir, roi_size=None):
    """Save QC metrics to CSV and JSON"""

    # Move cell_id to first position
    cell_id_col = new_df.pop("cell_id")
    new_df.insert(0, "cell_id", cell_id_col)
    # Save full dataset
    new_df.to_csv(outdir / "image_qc_cell_metrics.csv", index=False)

    # Generate QC summary statistics
    qc_metrics = {
        "total_cells": len(new_df),
        "cells_with_transcripts": len(new_df[new_df["transcript_counts"] > 0]),
        "mean_transcript_count": float(new_df["transcript_counts"].mean()),
        "median_transcript_count": float(new_df["transcript_counts"].median()),
        "std_transcript_count": float(new_df["transcript_counts"].std()),
        "mean_ccfs_dapi": float(new_df["CCFS_DAPI"].mean()),
        "cells_high_nuclear_texture": len(new_df[new_df["is_high_nuclear_texture"]]),
        "cells_low_nuclear_texture": len(new_df[new_df["is_low_nuclear_texture"]]),
        "cells_near_edge": len(new_df[new_df["is_near_edge"]]),
        "cells_near_holes": len(new_df[new_df["is_near_hole"]]),
        "cells_with_dense_intensity_regions": len(
            new_df[new_df["has_dense_intensity_regions"]]
        ),
        "unique_dense_intensity_regions": sorted(
            new_df["Dense-Intensity-Region-ID"].unique().tolist()
        )
        if "Dense-Intensity-Region-ID" in new_df.columns
        else [],
        "total_dense_intensity_regions": len(
            new_df[new_df["Dense-Intensity-Region-ID"] > 0][
                "Dense-Intensity-Region-ID"
            ].unique()
        )
        if "Dense-Intensity-Region-ID" in new_df.columns
        else 0,
        "clusters_present": sorted(new_df["Cluster_kmeans10"].unique().tolist()),
        "segmentation_methods": sorted(new_df["segmentation_method"].unique().tolist()),
    }

    # Add tile-based metrics if available
    if "DAPI_RFS_roi" in new_df.columns:
        # Save ROI size used for this analysis
        if roi_size is not None:
            qc_metrics["roi_size_used"] = int(roi_size)
        qc_metrics["mean_dapi_rfs_roi"] = float(new_df["DAPI_RFS_roi"].mean())
        qc_metrics["median_dapi_rfs_roi"] = float(new_df["DAPI_RFS_roi"].median())
        qc_metrics["mean_dapi_rfsnorm_roi"] = float(new_df["DAPI_RFSnorm_roi"].mean())
        qc_metrics["median_dapi_rfsnorm_roi"] = float(
            new_df["DAPI_RFSnorm_roi"].median()
        )

        # GMM 2D metrics (preferred method)
        if "is_blurred_gmm_2d_roi" in new_df.columns:
            qc_metrics["cells_high_focus_gmm_2d_roi"] = len(
                new_df[~new_df["is_blurred_gmm_2d_roi"]]
            )
            qc_metrics["cells_blurred_gmm_2d_roi"] = len(
                new_df[new_df["is_blurred_gmm_2d_roi"]]
            )
            if "blur_prob_gmm_2d_roi" in new_df.columns:
                valid_probs = new_df["blur_prob_gmm_2d_roi"].dropna()
                if len(valid_probs) > 0:
                    qc_metrics["mean_blur_prob_gmm_2d_roi"] = float(valid_probs.mean())
                    qc_metrics["median_blur_prob_gmm_2d_roi"] = float(
                        valid_probs.median()
                    )

        # Threshold-based metrics (for comparison/backward compatibility)
        if "is_blurred_roi" in new_df.columns:
            qc_metrics["cells_high_focus_roi"] = len(
                new_df[new_df["is_high_focus_roi"]]
            )
            qc_metrics["cells_blurred_roi"] = len(new_df[new_df["is_blurred_roi"]])

        # Comparison metrics between nuclei-based and tile-based methods
        # Calculate Spearman rank-based correlation between CCFS_DAPI and DAPI_RFSnorm_roi
        # Spearman is better suited for different scales and non-linear relationships
        if "CCFS_DAPI" in new_df.columns and "DAPI_RFSnorm_roi" in new_df.columns:
            correlation = new_df["CCFS_DAPI"].corr(
                new_df["DAPI_RFSnorm_roi"], method="spearman"
            )
            qc_metrics["ccfs_vs_rfs_correlation"] = (
                float(correlation) if not np.isnan(correlation) else None
            )
            qc_metrics["ccfs_vs_rfs_correlation_method"] = "spearman"

            # Calculate agreement rate - prefer GMM 2D, include threshold-based for comparison
            if "is_low_nuclear_texture" in new_df.columns:
                # GMM 2D agreement (preferred)
                if "is_blurred_gmm_2d_roi" in new_df.columns:
                    agreement_gmm_2d = (
                        new_df["is_low_nuclear_texture"]
                        == new_df["is_blurred_gmm_2d_roi"]
                    ).sum()
                    agreement_rate_gmm_2d = agreement_gmm_2d / len(new_df)
                    qc_metrics["classification_agreement_gmm_2d"] = float(
                        agreement_rate_gmm_2d
                    )
                    qc_metrics["classification_agreement_gmm_2d_count"] = int(
                        agreement_gmm_2d
                    )
                    qc_metrics["classification_disagreement_gmm_2d_count"] = int(
                        len(new_df) - agreement_gmm_2d
                    )

                # Threshold-based agreement (for comparison)
                if "is_blurred_roi" in new_df.columns:
                    agreement = (
                        new_df["is_low_nuclear_texture"] == new_df["is_blurred_roi"]
                    ).sum()
                    agreement_rate = agreement / len(new_df)
                    qc_metrics["classification_agreement"] = float(agreement_rate)
                    qc_metrics["classification_agreement_count"] = int(agreement)
                    qc_metrics["classification_disagreement_count"] = int(
                        len(new_df) - agreement
                    )

    # Save metrics
    with open(outdir / "image_qc_cell_metrics.json", "w") as f:
        json.dump(qc_metrics, f, indent=2)

    # Save ROI size to simple text file for easy retrieval
    if roi_size is not None:
        with open(outdir / "roi_size.txt", "w") as f:
            f.write(f"{roi_size}\n")

    logging.info("\n=== IMAGE QC SUMMARY ===")
    logging.info(f"Total cells analyzed: {qc_metrics['total_cells']:,}")
    logging.info(f"Cells with transcripts: {qc_metrics['cells_with_transcripts']:,}")
    logging.info(f"Mean transcript count: {qc_metrics['mean_transcript_count']:.1f}")
    logging.info(f"Mean CCFS DAPI: {qc_metrics['mean_ccfs_dapi']:.6f}")
    logging.info(
        f"Cells high nuclear texture: {qc_metrics['cells_high_nuclear_texture']:,}"
    )
    logging.info(
        f"Cells low nuclear texture: {qc_metrics['cells_low_nuclear_texture']:,}"
    )
    logging.info(f"Cells near edge: {qc_metrics['cells_near_edge']:,}")
    logging.info(f"Cells near holes: {qc_metrics['cells_near_holes']:,}")
    logging.info(
        f"Cells with dense intensity regions: {qc_metrics['cells_with_dense_intensity_regions']:,}"
    )
    if (
        "total_dense_intensity_regions" in qc_metrics
        and qc_metrics["total_dense_intensity_regions"] > 0
    ):
        logging.info(
            f"Total unique dense intensity regions detected: {qc_metrics['total_dense_intensity_regions']}"
        )
        logging.info(
            f"  (Region IDs: {qc_metrics['unique_dense_intensity_regions'][:10]}{'...' if len(qc_metrics['unique_dense_intensity_regions']) > 10 else ''})"
        )
    logging.info(f"Clusters present: {qc_metrics['clusters_present']}")
    logging.info(f"Segmentation methods: {qc_metrics['segmentation_methods']}")

    # Print tile-based metrics if available
    if "mean_dapi_rfs_roi" in qc_metrics:
        logging.info("\n=== TILE-BASED FOCUS SCORE SUMMARY ===")
        logging.info(f"Mean DAPI RFS (tile): {qc_metrics['mean_dapi_rfs_roi']:.6f}")
        logging.info(
            f"Mean DAPI RFS normalized (tile): {qc_metrics['mean_dapi_rfsnorm_roi']:.6f}"
        )

        # GMM 2D metrics (preferred)
        if "cells_blurred_gmm_2d_roi" in qc_metrics:
            logging.info("\n--- 2D GMM Classification (Preferred) ---")
            logging.info(
                f"Cells with high focus (2D GMM): {qc_metrics['cells_high_focus_gmm_2d_roi']:,}"
            )
            logging.info(
                f"Cells blurred (2D GMM): {qc_metrics['cells_blurred_gmm_2d_roi']:,}"
            )
            if "mean_blur_prob_gmm_2d_roi" in qc_metrics:
                logging.info(
                    f"Mean blur probability (2D GMM): {qc_metrics['mean_blur_prob_gmm_2d_roi']:.4f}"
                )

        # Threshold-based metrics (for comparison)
        if "cells_blurred_roi" in qc_metrics:
            logging.info("\n--- Threshold-based Classification (Comparison) ---")
            logging.info(
                f"Cells with high focus (Threshold): {qc_metrics['cells_high_focus_roi']:,}"
            )
            logging.info(
                f"Cells blurred (Threshold): {qc_metrics['cells_blurred_roi']:,}"
            )

        if (
            "ccfs_vs_rfs_correlation" in qc_metrics
            and qc_metrics["ccfs_vs_rfs_correlation"] is not None
        ):
            logging.info("\n=== METHOD COMPARISON ===")
            logging.info(
                f"Spearman Correlation (CCFS vs RFS): {qc_metrics['ccfs_vs_rfs_correlation']:.4f}"
            )

            # GMM 2D agreement (preferred)
            if "classification_agreement_gmm_2d" in qc_metrics:
                logging.info("\n--- CCFS vs 2D GMM Agreement (Preferred) ---")
                logging.info(
                    f"Classification agreement: {qc_metrics['classification_agreement_gmm_2d'] * 100:.2f}%"
                )
                logging.info(
                    f"  - Agreeing cells: {qc_metrics['classification_agreement_gmm_2d_count']:,}"
                )
                logging.info(
                    f"  - Disagreeing cells: {qc_metrics['classification_disagreement_gmm_2d_count']:,}"
                )

            # Threshold-based agreement (for comparison)
            if "classification_agreement" in qc_metrics:
                logging.info("\n--- CCFS vs Threshold Agreement (Comparison) ---")
                logging.info(
                    f"Classification agreement: {qc_metrics['classification_agreement'] * 100:.2f}%"
                )
                logging.info(
                    f"  - Agreeing cells: {qc_metrics['classification_agreement_count']:,}"
                )
                logging.info(
                    f"  - Disagreeing cells: {qc_metrics['classification_disagreement_count']:,}"
                )


def generate_cell_figures(
    data,
    new_df,
    myData,
    figures_dir,
    figures_source_dir,
    roi_threshold=None,
    roi_intensity_threshold=None,
    ccfs_low_texture_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
):
    """
    Generate cell-based figures from merged data using multithreading.

    Parameters:
    -----------
    data : dict
        Dictionary with paths and directories
    new_df : pandas DataFrame
        Merged DataFrame with cell data and ROI mappings
    myData : pandas DataFrame
        Nuclei-based measurements DataFrame
    figures_dir : Path
        Directory to save figures
    figures_source_dir : Path
        Directory to save source data
    roi_threshold : float, optional
        Tile-based blur threshold for display
    roi_intensity_threshold : float, optional
        Tile intensity threshold for display
    """
    if figures_source_dir is not None:
        figures_source_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Generating cell-based figures...")

    def _plot1_nuclear_texture_proportions():
        logging.info("  - Nuclear texture proportions (CCFS)...")
        plot_nuclear_texture_proportions(
            new_df,
            figures_dir,
            figures_source_dir,
            texture_threshold=ccfs_low_texture_threshold,
        )

    def _plot2_blur_proportions_roi():
        if (
            "is_blurred_gmm_2d_roi" in new_df.columns
            or "is_blurred_roi" in new_df.columns
        ):
            logging.info("  - Blur score proportions (tile-based)...")
            roi_threshold_display = roi_threshold if roi_threshold is not None else -1.0
            plot_tile_blur_proportions_roi._intensity_threshold = (
                roi_intensity_threshold if roi_intensity_threshold is not None else 20.0
            )
            plot_tile_blur_proportions_roi(
                new_df,
                figures_dir,
                figures_source_dir,
                roi_threshold=roi_threshold_display,
            )

    def _plot3_nuclear_texture_density():
        logging.info("  - Nuclear texture density...")
        plot_nuclear_texture_density(
            new_df,
            figures_dir,
            figures_source_dir,
            ccfs_low_texture_threshold=ccfs_low_texture_threshold,
        )

    def _plot5_ccfs_vs_roi():
        if "DAPI_RFSnorm_roi" in new_df.columns:
            logging.info("  - CCFS vs tile comparison...")
            plot_ccfs_vs_roi_comparison(new_df, figures_dir, figures_source_dir)

    def _plot6_spatial_comparison():
        logging.info("  - Spatial comparison...")
        plot_spatial_comparison(new_df, myData, figures_dir, figures_source_dir)

    def _plot7_cell_focus_distribution():
        logging.info("  - Cell-level focus score distribution...")
        plot_cell_focus_distribution(new_df, figures_dir, figures_source_dir)

    def _plot8_gmm_focus_vs_transcripts():
        logging.info("  - GMM focus vs transcripts...")
        plot_gmm_focus_vs_transcripts(new_df, figures_dir, figures_source_dir)

    def _plot9_tile_focus_gmm_spatial():
        logging.info("  - Tile-focus GMM spatial...")
        plot_tile_focus_gmm_spatial(new_df, figures_dir, figures_source_dir)

    def _plot9b_cell_flagged_maps_combined():
        logging.info("  - Cell-flagged maps (combined)...")
        plot_cell_flagged_maps_combined(new_df, figures_dir, figures_source_dir)

    def _plot10_blur_prob_density_by_cluster():
        logging.info("  - Blur probability density by cluster...")
        plot_blur_prob_density_by_cluster(new_df, figures_dir, figures_source_dir)

    def _plot11_intensity_transcript_correlation():
        logging.info("  - Intensity-transcript correlation heatmap...")
        plot_intensity_transcript_correlation(new_df, figures_dir, figures_source_dir)

    def _plot12_per_cell_intensity_vs_transcripts():
        logging.info(
            "  - Per-cell intensity vs transcripts (DAPI / Boundary / IntRNA)..."
        )
        plot_per_cell_intensity_vs_transcripts(
            new_df, figures_dir, figures_source_dir, log_scale=True
        )

    tasks = [
        _plot1_nuclear_texture_proportions,
        _plot2_blur_proportions_roi,
        # Latent figures disabled 2026-05-20 (user request) — function defs
        # retained above as latent code: _plot3_nuclear_texture_density,
        # _plot5_ccfs_vs_roi, _plot6_spatial_comparison,
        # _plot7_cell_focus_distribution, _plot10_blur_prob_density_by_cluster.
        _plot8_gmm_focus_vs_transcripts,
        _plot9_tile_focus_gmm_spatial,
        _plot9b_cell_flagged_maps_combined,
        _plot11_intensity_transcript_correlation,
        _plot12_per_cell_intensity_vs_transcripts,
    ]

    _run_figure_pool(tasks, phase="cell-based figures")

    logging.info("All cell-based figures generated successfully!")


# ===== QUARTO FIGURE GENERATION (from image_qc_processing.py) =====


def generate_all_figures(
    data,
    df_spatial,
    new_df,
    myData,
    small0,
    small1,
    small2,
    distance_map,
    distance_map2,
    whole_sample,
    holes,
    artefacts,
    ccfs_low_texture_threshold=DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD,
    multistain_whole_sample=None,
    multistain_distance_map=None,
    multistain_distance_map2=None,
    figure_source_tables=False,
):
    """Generate ALL 13 Quarto-required figures using multithreading, and save the data used for each plot as CSV."""
    figures_dir = data["figures_dir"]
    figures_source_dir = (
        (figures_dir / "figures_source") if figure_source_tables else None
    )
    if figures_source_dir is not None:
        figures_source_dir.mkdir(parents=True, exist_ok=True)
    # if df_spatial['In-Area-with-Artefact'] have a single unique value, use that value, otherwise use 0.5
    if len(df_spatial["In-Area-with-Artefact"].unique()) == 1:
        iawa_vmax = df_spatial["In-Area-with-Artefact"].unique()[0]
    else:
        iawa_vmax = 0.5

    # §2.4 distance figures use the multi-stain extent mask + its distance maps when present,
    # matching the edge/hole burden metrics in save_roi_qc_metrics. DAPI fallback (None on
    # DAPI-only bundles) keeps those bundles byte-identical. distance_map/distance_map2 are
    # referenced only by the distance closures, so rebinding them here is safe; the mask is
    # aliased as _dm_mask because whole_sample is also used by the masks figure below.
    if multistain_distance_map is not None:
        distance_map = multistain_distance_map
    if multistain_distance_map2 is not None:
        distance_map2 = multistain_distance_map2
    _dm_mask = (
        whole_sample if multistain_whole_sample is None else multistain_whole_sample
    )

    def _fig1_distance_edge():
        logging.info("Generating Figure 1: Distance map (edge)...")
        _h, _w = distance_map.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        # Floor the height at 50% of width so very wide slides (e.g. brain) don't get squashed
        fig, ax = plt.subplots(1, 1, figsize=(6, max(3, 6 * _aspect)))
        # Phase v5: distance-to-edge readability fix.
        # - viridis so near-edge tissue (low |distance|) renders as bright
        #   yellow against the white background — visible diagnostic region.
        # - Absolute distance in µm: signed maurer is negative inside;
        #   |·| × 8 × 0.2125 → 0-at-boundary → max-deep-inside gradient.
        # - NaN outside tissue mask → white via cmap.set_bad.
        # - Thin black tissue outline as unambiguous boundary marker.
        # TODO: 8 (downsample factor) and 0.2125 (Xenium native µm/px) are
        # hardcoded here and in three other sites. See task #15 — future
        # plumbing reads pixel_size from the bundle's experiment.xenium.
        from copy import copy as _copy_cmap

        _cmap_edge = _copy_cmap(plt.cm.viridis)
        _cmap_edge.set_bad(color="white")
        # Cap the colorscale at 300 µm so the near-edge band uses most of the
        # spectrum; tiles further than 300 µm from the boundary saturate at
        # yellow and the colorbar shows an "extend max" arrow. Beyond 300 µm
        # the tile is unambiguously deep-tissue and not edge-affected.
        _EDGE_VMAX_UM = 300.0
        if _dm_mask.shape == distance_map.shape:
            _dist_um = np.abs(distance_map) * 8 * 0.2125
            _dm_edge = np.where(_dm_mask > 0, _dist_um, np.nan)
            im = _imshow_thumb(
                ax, _dm_edge, cmap=_cmap_edge, vmin=0.0, vmax=_EDGE_VMAX_UM
            )
            _edge_cbar_extend = "max"
        else:
            _dm_edge = (
                distance_map  # fallback: shapes mismatch, preserve prior behaviour
            )
            im = _imshow_thumb(ax, _dm_edge, cmap=_cmap_edge)
            _edge_cbar_extend = "neither"
        if _dm_mask.shape == distance_map.shape:
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
        ax.set_title("Distance to Edge")
        ax.set_aspect("equal")
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_edge_cbar_extend
        )
        cbar.set_label("Distance from edge (µm)")
        # Explicit "300+" label at the cap when extend="max" is active
        if _edge_cbar_extend == "max":
            cbar.set_ticks([0, 50, 100, 150, 200, 250, _EDGE_VMAX_UM])
            cbar.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_EDGE_VMAX_UM)}+"]
            )
        plt.tight_layout()
        plt.savefig(figures_dir / "distance_map_edge.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "distance_map_edge.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # Full-resolution export. These raster figures_source CSVs are gated behind
        # --figure-source-tables (off by default), so the ~86-Mpx / ~1 GB dump only
        # happens when a user explicitly asks for the raw plotting data — and then it
        # must match the analysis exactly, not a thumbnail. The rendered PNG still uses
        # the _imshow_thumb / _thumb display-resolution path for speed regardless.
        if figures_source_dir is not None:
            pd.DataFrame(distance_map).to_csv(
                figures_source_dir / "distance_map_edge.csv",
                index=False,
                header=False,
            )

    def _fig2_distance_holes():
        logging.info("Generating Figure 2: Distance map (holes)...")
        _h, _w = distance_map2.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        # Floor the height at 50% of width so very wide slides (e.g. brain) don't get squashed
        fig, ax = plt.subplots(1, 1, figsize=(6, max(3, 6 * _aspect)))
        # Phase v5: distance-to-holes readability fix.
        # - viridis colormap (same as edge map) for visual consistency.
        # - Absolute distance in µm: |signed_maurer_distance| × 8 × 0.2125.
        # - Linear vmin=0 / vmax=300 µm — matches the edge map for visual
        #   consistency across both panels of §2.4 (edge + holes). Tiles beyond 300 µm saturate
        #   at yellow with an "extend max" arrow on the colorbar.
        # - NaN outside tissue mask → white via cmap.set_bad.
        # - Thin black tissue outline as boundary marker.
        # TODO: pixel-size conversion hardcoded — see task #15.
        from copy import copy as _copy_cmap

        _cmap_holes = _copy_cmap(plt.cm.viridis)
        _cmap_holes.set_bad(color="white")
        _HOLES_VMAX_UM = 300.0
        if _dm_mask.shape == distance_map2.shape:
            _dist_um_h = np.abs(distance_map2) * 8 * 0.2125
            _dm_holes = np.where(_dm_mask > 0, _dist_um_h, np.nan)
            im = _imshow_thumb(
                ax, _dm_holes, cmap=_cmap_holes, vmin=0.0, vmax=_HOLES_VMAX_UM
            )
            _holes_cbar_extend = "max"
        else:
            _dm_holes = distance_map2  # fallback: shapes mismatch
            im = _imshow_thumb(ax, _dm_holes, cmap=_cmap_holes)
            _holes_cbar_extend = "neither"
        if _dm_mask.shape == distance_map2.shape:
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
        ax.set_title("Distance to Nearest Hole")
        ax.set_aspect("equal")
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_holes_cbar_extend
        )
        cbar.set_label("Distance from nearest hole (µm)")
        # Explicit "300+" label at the cap when extend="max" is active.
        if _holes_cbar_extend == "max":
            cbar.set_ticks([0, 50, 100, 150, 200, 250, _HOLES_VMAX_UM])
            cbar.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_HOLES_VMAX_UM)}+"]
            )
        plt.tight_layout()
        plt.savefig(
            figures_dir / "distance_map_holes.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            figures_dir / "distance_map_holes.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # Full-resolution export (see distance_map_edge.csv note): gated behind
        # --figure-source-tables (off by default); rendering uses the display-res thumbnail.
        if figures_source_dir is not None:
            pd.DataFrame(distance_map2).to_csv(
                figures_source_dir / "distance_map_holes.csv",
                index=False,
                header=False,
            )

    def _fig2b_distance_combined():
        """Two-panel combined figure: distance to edge (left) + distance to
        holes (right), sharing coordinate system and colorbar conventions.
        Standalone _fig1_distance_edge / _fig2_distance_holes continue to
        render the individual PNGs as latent artefacts; the combined figure
        is what the QMD §2.4 embeds.
        """
        logging.info("Generating Figure 2b: Distance combined (edge + holes)...")
        _h, _w = distance_map.shape
        _aspect = (_h / _w) if _w > 0 else 1.0
        _panel_width = 6
        _panel_height = max(3, _panel_width * _aspect)
        fig, axes = plt.subplots(1, 2, figsize=(2 * _panel_width + 2, _panel_height))

        from copy import copy as _copy_cmap

        _VMAX_UM = 300.0
        _cmap = _copy_cmap(plt.cm.viridis)
        _cmap.set_bad(color="white")

        # ── Left panel: distance to edge ───────────────────────────────
        ax = axes[0]
        if _dm_mask.shape == distance_map.shape:
            _dist_um = np.abs(distance_map) * 8 * 0.2125
            _dm_edge = np.where(_dm_mask > 0, _dist_um, np.nan)
            im_e = _imshow_thumb(ax, _dm_edge, cmap=_cmap, vmin=0.0, vmax=_VMAX_UM)
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
            _edge_extend = "max"
        else:
            im_e = _imshow_thumb(ax, distance_map, cmap=_cmap)
            _edge_extend = "neither"
        ax.set_title("Distance to Edge", fontsize=14)
        ax.set_aspect("equal")
        cbar_e = fig.colorbar(
            im_e, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_edge_extend
        )
        cbar_e.set_label("Distance from edge (µm)")
        if _edge_extend == "max":
            cbar_e.set_ticks([0, 50, 100, 150, 200, 250, _VMAX_UM])
            cbar_e.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_VMAX_UM)}+"]
            )

        # ── Right panel: distance to holes ─────────────────────────────
        ax = axes[1]
        if _dm_mask.shape == distance_map2.shape:
            _dist_um_h = np.abs(distance_map2) * 8 * 0.2125
            _dm_holes = np.where(_dm_mask > 0, _dist_um_h, np.nan)
            im_h = _imshow_thumb(ax, _dm_holes, cmap=_cmap, vmin=0.0, vmax=_VMAX_UM)
            _contour_thumb(
                ax,
                (_dm_mask > 0).astype(np.uint8),
                levels=[0.5],
                colors="black",
                linewidths=0.5,
            )
            _holes_extend = "max"
        else:
            im_h = _imshow_thumb(ax, distance_map2, cmap=_cmap)
            _holes_extend = "neither"
        ax.set_title("Distance to Nearest Hole", fontsize=14)
        ax.set_aspect("equal")
        cbar_h = fig.colorbar(
            im_h, ax=ax, fraction=0.035, pad=0.04, shrink=0.45, extend=_holes_extend
        )
        cbar_h.set_label("Distance from nearest hole (µm)")
        if _holes_extend == "max":
            cbar_h.set_ticks([0, 50, 100, 150, 200, 250, _VMAX_UM])
            cbar_h.set_ticklabels(
                ["0", "50", "100", "150", "200", "250", f"{int(_VMAX_UM)}+"]
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "distance_maps.png", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "distance_maps.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _fig3_morphology_overview():
        logging.info("Generating Figure 3: Morphology overview...")
        # Phase v5 TODO #12: 1×3 layout (DAPI + Boundary + Interior only). The
        # artefacts panel that used to live in this figure's bottom-right is
        # also rendered in imageqc_masks.png (§2.3 Masks), so showing it here
        # duplicates the same plot — dropped here.
        _img_aspect = small0.shape[0] / small0.shape[1] if small0.shape[1] > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        fig, ax = plt.subplots(1, 3, figsize=(3 * _panel_width, _panel_height))
        _imshow_thumb(ax[0], small0, cmap="Greys_r", vmax=np.percentile(small0, 99))
        ax[0].set_title("DAPI", fontsize=14)
        ax[0].set_aspect("equal")
        ax[0].axis("off")
        _imshow_thumb(ax[1], small1, cmap="Greys_r", vmax=np.percentile(small1, 99))
        ax[1].set_title("Boundary", fontsize=14)
        ax[1].set_aspect("equal")
        ax[1].axis("off")
        _imshow_thumb(ax[2], small2, cmap="Greys_r", vmax=np.percentile(small2, 99))
        ax[2].set_title("Interior", fontsize=14)
        ax[2].set_aspect("equal")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "morphology_overview.pdf", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            figures_dir / "morphology_overview.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # Full-resolution export (see distance_map_edge.csv note): four ~86-Mpx channel
        # grids, gated behind --figure-source-tables (off by default); rendering uses
        # the display-res thumbnail so the figure wall time is unaffected.
        if figures_source_dir is not None:
            pd.DataFrame(small0).to_csv(
                figures_source_dir / "morphology_overview_DAPI.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(small1).to_csv(
                figures_source_dir / "morphology_overview_Boundary.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(small2).to_csv(
                figures_source_dir / "morphology_overview_Interior.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(artefacts).to_csv(
                figures_source_dir / "morphology_overview_Artefacts.csv",
                index=False,
                header=False,
            )

    def _fig4_sample_qc_metrics():
        logging.info("Generating Figure 4: Sample QC metrics...")
        idx4 = _subsample_idx(np.ones(len(df_spatial), dtype=bool))
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        _imshow_thumb(ax[0, 0], small0, cmap="Greys_r", vmax=np.percentile(small0, 99))
        ax[0, 0].set_title("DAPI", fontsize=14)
        ax[0, 0].set_aspect("equal")
        ax[0, 1].scatter(
            df_spatial["x"].iloc[idx4],
            -df_spatial["y"].iloc[idx4],
            c=df_spatial["Distance-to-edge"].iloc[idx4],
            s=0.1,
            marker="o",
            rasterized=True,
        )
        ax[0, 1].set_title("Distance to sample Edge")
        ax[0, 1].set_aspect("equal")
        ax[0, 1].set_facecolor("black")
        ax[1, 0].scatter(
            df_spatial["x"].iloc[idx4],
            -df_spatial["y"].iloc[idx4],
            c=df_spatial["Distance-to-nearest-hole"].iloc[idx4],
            s=0.1,
            marker="o",
            rasterized=True,
        )
        ax[1, 0].set_title("Distance to nearest hole")
        ax[1, 0].set_aspect("equal")
        ax[1, 0].set_facecolor("black")
        ax[1, 1].scatter(
            df_spatial["x"].iloc[idx4],
            -df_spatial["y"].iloc[idx4],
            c=df_spatial["In-Area-with-Artefact"].iloc[idx4],
            cmap="Blues_r",
            s=0.1,
            marker="o",
            vmax=iawa_vmax,
            rasterized=True,
        )
        ax[1, 1].set_title("Overlap with sample artefact")
        ax[1, 1].set_aspect("equal")
        ax[1, 1].set_facecolor("black")
        plt.tight_layout()
        plt.savefig(figures_dir / "sample_qc_metrics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        df_sample_qc_metrics = pd.DataFrame(
            {
                "x": df_spatial["x"],
                "y": df_spatial["y"],
                "Distance-to-edge": df_spatial["Distance-to-edge"],
                "Distance-to-nearest-hole": df_spatial["Distance-to-nearest-hole"],
                "In-Area-with-Artefact": df_spatial["In-Area-with-Artefact"],
            }
        )
        if figures_source_dir is not None:
            df_sample_qc_metrics.to_csv(
                figures_source_dir / "sample_qc_metrics.csv", index=False
            )

    def _fig5_imageqc_masks():
        logging.info("Generating Figure 5: ImageQC masks...")
        # Phase v5: 1x3 triplet — three mask panels only. DAPI morphology
        # image was dropped (already shown under §2.4 Stainings).
        _img_aspect = small0.shape[0] / small0.shape[1] if small0.shape[1] > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        # §2.3 tissue mask shows the EXTENT mask (all available stains) so it matches the
        # reported tissue coverage; falls back to the DAPI mask on single-stain slides.
        _extent_ws = (
            multistain_whole_sample
            if multistain_whole_sample is not None
            else whole_sample
        )
        fig, ax = plt.subplots(1, 3, figsize=(3 * _panel_width, _panel_height))
        ax[0].set_title("Tissue mask (all stains)", fontsize=14)
        _imshow_thumb(ax[0], _extent_ws, rgb=lambda d: color.label2rgb(d, bg_label=0))
        ax[0].set_aspect("equal")
        ax[0].axis("off")
        ax[1].set_title("Holes in sample", fontsize=14)
        _imshow_thumb(ax[1], holes, rgb=lambda d: color.label2rgb(d, bg_label=0))
        ax[1].set_aspect("equal")
        ax[1].axis("off")
        ax[2].set_title("Optically dense regions", fontsize=14)
        _imshow_thumb(ax[2], artefacts, rgb=lambda d: color.label2rgb(d, bg_label=0))
        ax[2].set_aspect("equal")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(figures_dir / "imageqc_masks.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "imageqc_masks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # DAPI csv export dropped — same data exposed by §2.4 Stainings.
        # Full-resolution export (see distance_map_edge.csv note): three ~86-Mpx mask
        # grids, gated behind --figure-source-tables (off by default); rendering uses
        # the display-res thumbnail so the figure wall time is unaffected.
        if figures_source_dir is not None:
            pd.DataFrame(_extent_ws).to_csv(
                figures_source_dir / "imageqc_masks_WholeSample.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(holes).to_csv(
                figures_source_dir / "imageqc_masks_Holes.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(artefacts).to_csv(
                figures_source_dir / "imageqc_masks_Artefacts.csv",
                index=False,
                header=False,
            )

    def _fig6_ccfs_spatial():
        logging.info("Generating Figure 6: CCFS Spatial...")
        idx6 = _subsample_idx(np.ones(len(myData), dtype=bool))
        # Phase 11 (v5): aspect-adaptive figure size matching slide proportions
        # (was a fixed 8x8 square). Mirrors plot_grid_roi_focus_heatmap.
        _x = myData["centroid-1"]
        _y = myData["centroid-0"]
        _x_range = float(_x.max() - _x.min()) if len(_x) else 1.0
        _y_range = float(_y.max() - _y.min()) if len(_y) else 1.0
        _img_aspect = _y_range / _x_range if _x_range > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        fig, ax = plt.subplots(1, 1, figsize=(_panel_width, _panel_height))
        ax.scatter(
            myData["centroid-1"].iloc[idx6],
            -myData["centroid-0"].iloc[idx6],
            s=0.1,
            c=-myData["CCFS_DAPI"].iloc[idx6],
            cmap="viridis",
            vmin=-0.012,
            rasterized=True,
        )
        ax.set_title("Calculated Cell Focus Score")
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(figures_dir / "ccfs_spatial.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        df_ccfs_spatial = pd.DataFrame(
            {
                "centroid-1": myData["centroid-1"],
                "centroid-0": myData["centroid-0"],
                "CCFS_DAPI": myData["CCFS_DAPI"],
            }
        )
        if figures_source_dir is not None:
            df_ccfs_spatial.to_csv(figures_source_dir / "ccfs_spatial.csv", index=False)

    def _fig7_ccfs_thresholded():
        logging.info("Generating Figure 7: CCFS Thresholded...")
        highC = myData[myData["is_high_nuclear_texture"]]
        lowC = myData[myData["is_low_nuclear_texture"]]
        idx7h = _subsample_idx(np.ones(len(highC), dtype=bool))
        idx7l = _subsample_idx(np.ones(len(lowC), dtype=bool))
        # Phase 11 (v5): aspect-adaptive figure size matching slide proportions
        # (was a fixed 8x8 square). Uses full myData coordinate range so both
        # high/low subsets share the same axis layout.
        _x = myData["centroid-1"]
        _y = myData["centroid-0"]
        _x_range = float(_x.max() - _x.min()) if len(_x) else 1.0
        _y_range = float(_y.max() - _y.min()) if len(_y) else 1.0
        _img_aspect = _y_range / _x_range if _x_range > 0 else 1.0
        _panel_width = 6
        _panel_height = max(_panel_width * 0.5, _panel_width * _img_aspect)
        fig, ax = plt.subplots(1, 1, figsize=(_panel_width, _panel_height))
        # High-texture cells were `#1B2631` (near-black) on black background
        # — invisible. Lightened to `#999999` (mid-grey) for clear contrast
        # against the black facecolor while keeping the red low-texture cells
        # visually dominant.
        ax.scatter(
            highC["centroid-1"].iloc[idx7h],
            -highC["centroid-0"].iloc[idx7h],
            s=0.1,
            color="#999999",
            rasterized=True,
        )
        ax.scatter(
            lowC["centroid-1"].iloc[idx7l],
            -lowC["centroid-0"].iloc[idx7l],
            s=0.1,
            color="red",
            rasterized=True,
        )
        ax.set_title("Thresholded CCFS (red = low nuclear texture cells)")
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(figures_dir / "ccfs_thresholded.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        df_ccfs_thresholded = pd.DataFrame(
            {
                "centroid-1": myData["centroid-1"],
                "centroid-0": myData["centroid-0"],
                "CCFS_DAPI": myData["CCFS_DAPI"],
                "is_low_nuclear_texture": myData["is_low_nuclear_texture"],
            }
        )
        if figures_source_dir is not None:
            df_ccfs_thresholded.to_csv(
                figures_source_dir / "ccfs_thresholded.csv", index=False
            )

    def _fig8_umap_multiple_metrics():
        logging.info("Generating Figure 8: UMAP by multiple metrics...")
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0, 0].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=-new_df["CCFS_DAPI"],
            marker="o",
            cmap="viridis",
            vmin=-0.012,
            rasterized=True,
        )
        ax[0, 0].set_title("UMAP by Nuclear Texture Score")
        ax[0, 0].set_facecolor("black")
        ax[0, 0].set_aspect("equal")
        ax[0, 1].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=new_df["Cluster_kmeans10"],
            marker="o",
            rasterized=True,
        )
        ax[0, 1].set_title("UMAP by Cluster Allocation")
        ax[0, 1].set_facecolor("black")
        ax[0, 1].set_aspect("equal")
        ax[1, 0].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=new_df["segPal"],
            marker="o",
            rasterized=True,
        )
        ax[1, 0].set_title("UMAP by Segmentation method")
        ax[1, 0].set_facecolor("black")
        ax[1, 0].set_aspect("equal")
        ax[1, 1].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=new_df["transcript_counts"],
            marker="o",
            rasterized=True,
        )
        ax[1, 1].set_title("UMAP by Transcript counts")
        ax[1, 1].set_facecolor("black")
        ax[1, 1].set_aspect("equal")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "umap_multiple_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        df_umap_multiple_metrics = new_df[
            [
                "UMAP-1",
                "UMAP-2",
                "CCFS_DAPI",
                "Cluster_kmeans10",
                "segPal",
                "transcript_counts",
            ]
        ]
        if figures_source_dir is not None:
            df_umap_multiple_metrics.to_csv(
                figures_source_dir / "umap_multiple_metrics.csv", index=False
            )

    def _fig9_umap_distance_metrics():
        logging.info("Generating Figure 9: UMAP by distance metrics...")
        highC_new = new_df[new_df["is_high_nuclear_texture"]]
        lowC_new = new_df[new_df["is_low_nuclear_texture"]]
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0, 0].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=-new_df["Distance-to-edge"],
            marker="o",
            rasterized=True,
        )
        ax[0, 0].set_title("UMAP by Distance to edge")
        ax[0, 0].set_facecolor("black")
        ax[0, 0].set_aspect("equal")
        ax[0, 1].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=-new_df["Distance-to-nearest-hole"],
            marker="o",
            rasterized=True,
        )
        ax[0, 1].set_title("UMAP by Distance to nearest hole")
        ax[0, 1].set_facecolor("black")
        ax[0, 1].set_aspect("equal")
        ax[1, 0].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=new_df["In-Area-with-Artefact"],
            cmap="Blues_r",
            marker="o",
            vmax=iawa_vmax,
            rasterized=True,
        )
        ax[1, 0].set_title("UMAP by Overlap with artefact")
        ax[1, 0].set_facecolor("black")
        ax[1, 0].set_aspect("equal")
        ax[1, 1].scatter(
            highC_new["UMAP-1"],
            highC_new["UMAP-2"],
            s=0.1,
            color="#1B2631",
            rasterized=True,
        )
        ax[1, 1].scatter(
            lowC_new["UMAP-1"], lowC_new["UMAP-2"], s=0.1, color="red", rasterized=True
        )
        ax[1, 1].set_title("UMAP by Blurred cells (in red)")
        ax[1, 1].set_facecolor("black")
        ax[1, 1].set_aspect("equal")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "umap_distance_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        df_umap_distance_metrics = new_df[
            [
                "UMAP-1",
                "UMAP-2",
                "Distance-to-edge",
                "Distance-to-nearest-hole",
                "In-Area-with-Artefact",
                "CCFS_DAPI",
                "is_low_nuclear_texture",
            ]
        ]
        if figures_source_dir is not None:
            df_umap_distance_metrics.to_csv(
                figures_source_dir / "umap_distance_metrics.csv", index=False
            )

    def _fig10_umap_thresholded_metrics():
        logging.info("Generating Figure 10: UMAP thresholded metrics...")
        highE = new_df[new_df["is_far_from_edge"]]
        lowE = new_df[new_df["is_near_edge"]]
        highH = new_df[new_df["is_far_from_hole"]]
        lowH = new_df[new_df["is_near_hole"]]
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0, 0].scatter(
            highE["UMAP-1"], highE["UMAP-2"], s=0.1, color="#1B2631", rasterized=True
        )
        ax[0, 0].scatter(
            lowE["UMAP-1"], lowE["UMAP-2"], s=0.1, color="red", rasterized=True
        )
        ax[0, 0].set_title("UMAP by Distance to edge")
        ax[0, 0].set_facecolor("black")
        ax[0, 0].set_aspect("equal")
        ax[0, 1].scatter(
            highH["UMAP-1"], highH["UMAP-2"], s=0.1, color="#1B2631", rasterized=True
        )
        ax[0, 1].scatter(
            lowH["UMAP-1"], lowH["UMAP-2"], s=0.1, color="red", rasterized=True
        )
        ax[0, 1].set_title("UMAP by Distance to nearest hole")
        ax[0, 1].set_facecolor("black")
        ax[0, 1].set_aspect("equal")
        ax[1, 0].scatter(
            new_df["UMAP-1"],
            new_df["UMAP-2"],
            s=0.1,
            c=new_df["In-Area-with-Artefact"],
            cmap="Blues_r",
            marker="o",
            vmax=iawa_vmax,
            rasterized=True,
        )
        ax[1, 0].set_title("UMAP by Overlap with artefact")
        ax[1, 0].set_facecolor("black")
        ax[1, 0].set_aspect("equal")
        ax[1, 1].scatter(
            new_df[new_df["is_high_nuclear_texture"]]["UMAP-1"],
            new_df[new_df["is_high_nuclear_texture"]]["UMAP-2"],
            s=0.1,
            color="#1B2631",
            rasterized=True,
        )
        ax[1, 1].scatter(
            new_df[new_df["is_low_nuclear_texture"]]["UMAP-1"],
            new_df[new_df["is_low_nuclear_texture"]]["UMAP-2"],
            s=0.1,
            color="red",
            rasterized=True,
        )
        ax[1, 1].set_title("UMAP by Blurred cells (in red)")
        ax[1, 1].set_facecolor("black")
        ax[1, 1].set_aspect("equal")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "umap_thresholded_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # Save data as CSV
        df_umap_thresholded_metrics = new_df[
            [
                "UMAP-1",
                "UMAP-2",
                "Distance-to-edge",
                "Distance-to-nearest-hole",
                "In-Area-with-Artefact",
                "CCFS_DAPI",
                "is_near_edge",
                "is_near_hole",
                "is_low_nuclear_texture",
            ]
        ]
        if figures_source_dir is not None:
            df_umap_thresholded_metrics.to_csv(
                figures_source_dir / "umap_thresholded_metrics.csv", index=False
            )

    def _fig11_nuclear_texture_proportions():
        logging.info("Generating Figure 11: Nuclear Texture Proportions by Cluster...")
        plot_nuclear_texture_proportions(
            new_df,
            figures_dir,
            figures_source_dir,
            texture_threshold=ccfs_low_texture_threshold,
            GROUP_BY_COLUMN="Cluster_kmeans10",
        )

    def _fig12_nuclear_texture_density():
        logging.info("Generating Figure 12: Nuclear Texture Density by Cluster...")
        plot_nuclear_texture_density(
            new_df,
            figures_dir,
            figures_source_dir,
            GROUP_BY_COLUMN="Cluster_kmeans10",
            ccfs_low_texture_threshold=ccfs_low_texture_threshold,
        )

    def _fig13_nuclear_texture_vs_transcripts():
        logging.info(
            "Generating Figure 13: Nuclear Texture vs Transcripts (Log Scale)..."
        )
        plot_nuclear_texture_vs_transcripts(
            new_df,
            figures_dir,
            figures_source_dir,
            log_scale=True,
            ccfs_low_texture_threshold=ccfs_low_texture_threshold,
        )

    def _fig14_cell_focus_distribution():
        logging.info("Generating Figure 14: Cell-level focus score distribution...")
        plot_cell_focus_distribution(new_df, figures_dir, figures_source_dir)

    def _fig15_gmm_blur_proportions_by_cluster():
        if (
            "is_blurred_gmm_2d_roi" in new_df.columns
            or "is_blurred_roi" in new_df.columns
        ):
            logging.info("Generating Figure 15: GMM Blur Proportions by Cluster...")
            plot_tile_blur_proportions_roi(
                new_df,
                figures_dir,
                figures_source_dir,
                GROUP_BY_COLUMN="Cluster_kmeans10",
            )

    def _fig16_gmm_focus_vs_transcripts():
        logging.info("Generating Figure 16: GMM Focus vs Transcripts...")
        plot_gmm_focus_vs_transcripts(new_df, figures_dir, figures_source_dir)

    def _fig17_tile_focus_gmm_spatial():
        logging.info("Generating Figure 17: Tile-focus GMM spatial...")
        plot_tile_focus_gmm_spatial(new_df, figures_dir, figures_source_dir)

    def _fig17b_cell_flagged_maps_combined():
        logging.info("Generating Figure 17b: Cell-flagged maps (combined)...")
        plot_cell_flagged_maps_combined(new_df, figures_dir, figures_source_dir)

    def _fig18_blur_prob_density_by_cluster():
        logging.info("Generating Figure 18: Blur probability density by cluster...")
        plot_blur_prob_density_by_cluster(new_df, figures_dir, figures_source_dir)

    def _fig19_intensity_transcript_correlation():
        logging.info(
            "Generating Figure 19: Intensity-transcript correlation heatmap..."
        )
        plot_intensity_transcript_correlation(new_df, figures_dir, figures_source_dir)

    def _fig20_per_cell_intensity_vs_transcripts():
        logging.info(
            "Generating Figure 20: Per-cell intensity vs transcripts (DAPI / Boundary / IntRNA)..."
        )
        plot_per_cell_intensity_vs_transcripts(
            new_df, figures_dir, figures_source_dir, log_scale=True
        )

    tasks = [
        # Deduped 2026-08-01: distance_map_edge/holes, distance_maps,
        # morphology_overview and imageqc_masks are already rendered by the
        # always-run tile path (generate_roi_figures); render once, not twice.
        _fig7_ccfs_thresholded,
        # Latent figures disabled 2026-05-20 (user request) — function defs
        # retained above as latent code: _fig4_sample_qc_metrics,
        # _fig6_ccfs_spatial, _fig8/_fig9/_fig10 UMAP, _fig12_nuclear_texture_density,
        # _fig14_cell_focus_distribution, _fig18_blur_prob_density_by_cluster.
        _fig11_nuclear_texture_proportions,
        _fig13_nuclear_texture_vs_transcripts,
        _fig15_gmm_blur_proportions_by_cluster,
        _fig16_gmm_focus_vs_transcripts,
        _fig17_tile_focus_gmm_spatial,
        _fig17b_cell_flagged_maps_combined,
        _fig19_intensity_transcript_correlation,
        _fig20_per_cell_intensity_vs_transcripts,
    ]

    _run_figure_pool(tasks, phase="figures")

    logging.info("All figures generated successfully!")
    logging.info(f"[OK] Saved figures to {figures_dir}")
    if figures_source_dir is not None:
        logging.info(f"[OK] Saved source data to {figures_source_dir}")


# ===== SIMPLE QC METRICS (from image_qc_processing.py) =====


def save_simple_qc_metrics(new_df, outdir):
    """Save QC metrics to CSV and JSON"""

    # Move cell_id to first position
    cell_id_col = new_df.pop("cell_id")
    new_df.insert(0, "cell_id", cell_id_col)
    # Save full dataset
    new_df.to_csv(outdir / "image_qc_metrics.csv", index=False)

    # Generate QC summary statistics
    qc_metrics = {
        "total_cells": len(new_df),
        "cells_with_transcripts": len(new_df[new_df["transcript_counts"] > 0]),
        "mean_transcript_count": float(new_df["transcript_counts"].mean()),
        "median_transcript_count": float(new_df["transcript_counts"].median()),
        "std_transcript_count": float(new_df["transcript_counts"].std()),
        "mean_ccfs_dapi": float(new_df["CCFS_DAPI"].mean()),
        "cells_high_nuclear_texture": len(new_df[new_df["is_high_nuclear_texture"]]),
        "cells_low_nuclear_texture": len(new_df[new_df["is_low_nuclear_texture"]]),
        "cells_near_edge": len(new_df[new_df["is_near_edge"]]),
        "cells_near_holes": len(new_df[new_df["is_near_hole"]]),
        "cells_with_artifacts": len(new_df[new_df["has_artifacts"]]),
        "clusters_present": sorted(new_df["Cluster_kmeans10"].unique().tolist()),
        "segmentation_methods": sorted(new_df["segmentation_method"].unique().tolist()),
    }

    # Save metrics
    with open(outdir / "image_qc_metrics.json", "w") as f:
        json.dump(qc_metrics, f, indent=2)

    logging.info("\n=== IMAGE QC SUMMARY ===")
    logging.info(f"Total cells analyzed: {qc_metrics['total_cells']:,}")
    logging.info(f"Cells with transcripts: {qc_metrics['cells_with_transcripts']:,}")
    logging.info(f"Mean transcript count: {qc_metrics['mean_transcript_count']:.1f}")
    logging.info(f"Mean CCFS DAPI: {qc_metrics['mean_ccfs_dapi']:.6f}")
    logging.info(
        f"Cells high nuclear texture: {qc_metrics['cells_high_nuclear_texture']:,}"
    )
    logging.info(
        f"Cells low nuclear texture: {qc_metrics['cells_low_nuclear_texture']:,}"
    )
    logging.info(f"Cells near edge: {qc_metrics['cells_near_edge']:,}")
    logging.info(f"Cells near holes: {qc_metrics['cells_near_holes']:,}")
    logging.info(f"Cells with artifacts: {qc_metrics['cells_with_artifacts']:,}")
    logging.info(f"Clusters present: {qc_metrics['clusters_present']}")
    logging.info(f"Segmentation methods: {qc_metrics['segmentation_methods']}")


# ===== NEW: PIXEL-MAP AGGREGATION FOR CCFS =====


# Pixels per row block for labeled reductions.  32 M px keeps each transient
# float64/intp cast at ~256 MB, so a full pass costs ~1.3 GB regardless of how
# large the image is.
_LABEL_CHUNK_PIXELS = 32 * 1024 * 1024


def _grow_to(arr: NDArray[Any], n: int) -> NDArray[Any]:
    """Zero-extend *arr* to length *n*; no-op when it is already long enough."""
    if arr.size >= n:
        return arr
    out = np.zeros(n, dtype=arr.dtype)
    out[: arr.size] = arr
    return out


def _labeled_sums_chunked(
    labels_img,
    value_planes: dict[str, Any],
    include_coords: bool = False,
    rows_per_chunk: int | None = None,
) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]:
    """Per-label pixel counts and value sums, accumulated in row blocks.

    Equivalent to ``scipy.ndimage.sum`` over every label, but it never casts a
    full-resolution plane.  ``scipy.ndimage.mean`` reduces via ``np.bincount``,
    which requires ``intp`` labels and ``float64`` weights, so a whole-image
    call materialises an int64 copy of the label plane *and* a float64 copy of
    the value plane — 88 GB combined on a 5.5 gigapixel sample, invisible in the
    source.  Blocking by rows bounds both casts to one block.

    Counts and sums are additive, so a label straddling a block boundary
    accumulates partial contributions from each block and its final
    ``sum / count`` is exact — not an approximation.

    Args:
        labels_img: 2-D integer label plane.  May be a lazily-sliced handle
            (zarr array, memmap); only one row block is materialised at a time.
        value_planes: Named 2-D value planes aligned with *labels_img*.  May be
            empty to collect counts only.
        include_coords: Also accumulate coordinate sums, returned under the
            ``centroid_y_sum`` / ``centroid_x_sum`` keys.  ``sum / count`` of
            these is exactly ``skimage.measure.regionprops`` ``centroid``.
        rows_per_chunk: Rows per block.  Defaults to ~32 M pixels per block.

    Returns:
        ``(counts, sums)`` indexed by raw label value, index 0 = background.
        ``counts[i]`` is the pixel count for label ``i``; ``sums[name][i]`` the
        summed value.  Both are sized to the largest label actually seen.
    """
    height, width = labels_img.shape
    if rows_per_chunk is None:
        rows_per_chunk = max(1, _LABEL_CHUNK_PIXELS // max(width, 1))

    counts = np.zeros(1, dtype=np.int64)
    sums: dict[str, NDArray[np.float64]] = {
        name: np.zeros(1, dtype=np.float64) for name in value_planes
    }
    if include_coords:
        sums["centroid_y_sum"] = np.zeros(1, dtype=np.float64)
        sums["centroid_x_sum"] = np.zeros(1, dtype=np.float64)

    def _accumulate(key: str, block_sums: NDArray[np.float64]) -> None:
        sums[key] = _grow_to(sums[key], block_sums.size)
        sums[key][: block_sums.size] += block_sums

    for y0 in range(0, height, rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, height)
        lab = np.asarray(labels_img[y0:y1]).ravel()

        block_counts = np.bincount(lab)
        counts = _grow_to(counts, block_counts.size)
        counts[: block_counts.size] += block_counts
        n = block_counts.size

        for name, plane in value_planes.items():
            vals = np.asarray(plane[y0:y1], dtype=np.float64).ravel()
            _accumulate(name, np.bincount(lab, weights=vals, minlength=n))
            del vals

        if include_coords:
            n_rows = y1 - y0
            rows = np.repeat(np.arange(y0, y1, dtype=np.float64), width)
            _accumulate("centroid_y_sum", np.bincount(lab, weights=rows, minlength=n))
            del rows
            cols = np.tile(np.arange(width, dtype=np.float64), n_rows)
            _accumulate("centroid_x_sum", np.bincount(lab, weights=cols, minlength=n))
            del cols

        del lab, block_counts

    return counts, sums


def calculate_ccfs_from_focus_maps(
    focus_maps, cell_masks_zarr, cellseg_mask, xoa_morphology_files, streamed=None
):
    """
    Derive per-cell CCFS from pixel focus maps using scipy.ndimage.mean.

    This replaces the old regionprops-based calculate_ccfs_measurements() for
    DAPI focus scores, but still uses regionprops for boundary/RNA per-cell
    intensities (different channels).

    Args:
        focus_maps: dict from compute_all_focus_maps() with at least 'dapi_focus_map' and 'dapi_mean_map'
        cell_masks_zarr: zarr group with 'masks/0' (nuclear mask) and 'masks/1' (cell mask)
        cellseg_mask: numpy array of cell segmentation mask (from masks/1)
        xoa_morphology_files: list of morphology file paths (for boundary/RNA channels)

    Returns:
        pandas DataFrame with columns: CellID, centroid-0, centroid-1,
        CCFS_DAPI, mean_intensity, area_nucleus, area_cell,
        mean_intensity_Boundary, mean_intensity_IntRNA
    """
    if streamed is None:
        focus_map = focus_maps.get("dapi_focus_map")
        mean_map = focus_maps.get("dapi_mean_map")
        if focus_map is None or mean_map is None:
            raise ValueError(
                "focus_maps must contain 'dapi_focus_map' and 'dapi_mean_map'"
            )
    elif not streamed.has_per_cell:
        raise ValueError(
            "streamed results carry no per-cell reduction; "
            "cell_masks_path was not passed to calculate_roi_focusscore"
        )

    # The nuclear label plane stays lazy: _labeled_sums_chunked slices row
    # blocks straight out of zarr, so the full-res uint32 plane is never
    # materialised (22 GB on a 5.5 GP sample).  One blocked pass replaces
    # np.unique() (a 22 GB flatten copy), two whole-image ndimage.mean() calls
    # (88 GB of intp/float64 casts each), and regionprops() (a whole-plane
    # pass) — centroids and areas fall out of the same accumulators.
    if streamed is not None:
        # The tile pass already folded every tile into these, so there is no plane
        # left to read. Keys are the accumulator's, mapped onto this function's.
        logging.info("  Per-nucleus sums reduced during the tile pass (streamed)")
        nuc_counts = streamed.nuclear_counts
        nuc_sums = {
            "focus": streamed.nuclear_sums["focus_map"],
            "intensity": streamed.nuclear_sums["mean_map"],
            "centroid_y_sum": streamed.nuclear_sums["centroid_y_sum"],
            "centroid_x_sum": streamed.nuclear_sums["centroid_x_sum"],
        }
    else:
        nuclear_mask = cell_masks_zarr.get("masks").get("0")
        logging.info(
            "  Aggregating focus/intensity over nuclear masks (row-blocked)..."
        )
        t0 = time.time()
        nuc_counts, nuc_sums = _labeled_sums_chunked(
            nuclear_mask,
            {"focus": focus_map, "intensity": mean_map},
            include_coords=True,
        )
        logging.info(f"  [TIMING] labeled nuclear aggregation: {time.time() - t0:.1f}s")

    # Labels present, ascending, background excluded — same set and order as
    # regionprops(), which yields one region per distinct label value.
    labels = np.nonzero(nuc_counts)[0]
    labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError("nuclear mask contains no labelled pixels")
    counts = nuc_counts[labels].astype(np.float64)

    cell_focus = nuc_sums["focus"][labels] / counts
    cell_intensity = nuc_sums["intensity"][labels] / counts
    # sum(coord)/count is exactly regionprops' centroid definition.
    centroid_y = nuc_sums["centroid_y_sum"][labels] / counts
    centroid_x = nuc_sums["centroid_x_sum"][labels] / counts

    # Normalize: 99th percentile of mean intensity
    dapi_norm = np.percentile(cell_intensity, 99)
    if dapi_norm == 0:
        dapi_norm = 1.0
    ccfs_dapi = cell_focus / dapi_norm

    # Map each nucleus centroid to a cell ID (point lookups only)
    iy = np.minimum(centroid_y.astype(np.int64), cellseg_mask.shape[0] - 1)
    ix = np.minimum(centroid_x.astype(np.int64), cellseg_mask.shape[1] - 1)
    cell_ids = np.asarray(cellseg_mask[iy, ix]).astype(np.int64)

    nucleus_props = pd.DataFrame(
        {
            "label": labels,
            "centroid-0": centroid_y,
            "centroid-1": centroid_x,
            "area_nucleus": nuc_counts[labels],
            "CCFS_DAPI": ccfs_dapi,
            "mean_intensity": cell_intensity,
            "CellID": cell_ids,
        }
    )
    del nuc_counts, nuc_sums

    # One blocked pass over the cell mask yields cell areas *and* the per-cell
    # boundary/IntRNA means, replacing np.bincount() on the full-res uint32
    # plane (a 44 GB intp cast) plus one ndimage.mean() per channel.
    if streamed is not None:
        logging.info("  Per-cell sums reduced during the tile pass (streamed)")
        cell_counts = streamed.cell_counts
        cell_sums = streamed.cell_sums or {}
    else:
        cell_value_planes: dict[str, Any] = {}
        boundary_mean = focus_maps.get("boundary_mean_map")
        intrna_mean = focus_maps.get("intrna_mean_map")
        if boundary_mean is not None:
            cell_value_planes["boundary"] = boundary_mean
        if intrna_mean is not None:
            cell_value_planes["intrna"] = intrna_mean

        logging.info(
            "  Aggregating cell areas/intensities over cell masks (row-blocked)..."
        )
        t0 = time.time()
        cell_counts, cell_sums = _labeled_sums_chunked(cellseg_mask, cell_value_planes)
        logging.info(f"  [TIMING] labeled cell aggregation: {time.time() - t0:.1f}s")

    cids = nucleus_props["CellID"].to_numpy()
    in_range = cids < cell_counts.size

    # area_cell: index by raw CellID, 0 when out of range.  CellID 0 keeps
    # resolving to the background pixel count, matching the previous mapping.
    area_cell = np.zeros(cids.size, dtype=np.int64)
    area_cell[in_range] = cell_counts[cids[in_range]]
    nucleus_props["area_cell"] = area_cell

    # Per-cell channel means.  CellID 0 stays NaN: the old code built its
    # lookup from the >0 labels only, so background mapped to NaN.
    labelled = in_range & (cids > 0)
    for name, column in (
        ("boundary", "mean_intensity_Boundary"),
        ("intrna", "mean_intensity_IntRNA"),
    ):
        if name not in cell_sums:
            nucleus_props[column] = np.nan
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell = cell_sums[name] / cell_counts
        values = np.full(cids.size, np.nan, dtype=np.float64)
        values[labelled] = per_cell[cids[labelled]]
        nucleus_props[column] = values

    return nucleus_props


# ===== FALLBACK: REGIONPROPS-BASED CCFS (for legacy mode) =====


def calculate_ccfs_measurements(xoa_morphology_files, cellseg_mask, cell_masks_zarr):
    """
    Calculate CCFS measurements with improved variable naming and memory management.
    Loads data just before use and deletes it immediately after.

    Legacy path (--legacy-focus). regionprops_table needs a materialised label
    array, so a LazyLabelPlane is realised here rather than in the caller -- the
    streaming path never does this.
    """
    if isinstance(cellseg_mask, LazyLabelPlane):
        logging.info(
            "  Materialising the cell mask for regionprops (legacy focus path)..."
        )
        # A full-height slice; LazyLabelPlane already returns numpy. Not
        # np.asarray(...): this function has its own local `import numpy as np`
        # further down, so `np` is a local name here and referencing it before
        # that import raises UnboundLocalError.
        cellseg_mask = cellseg_mask[0 : cellseg_mask.shape[0]]
    import numpy as np
    import pandas as pd

    # Load full resolution image channels only when needed
    fullres_channels = imread(
        xoa_morphology_files[0], is_ome=False, level=0, aszarr=False
    )

    # Check number of channels (could be 2D array for single channel, or 3D array for multi-channel)
    if len(fullres_channels.shape) == 2:
        # Single channel (2D array)
        dapi_image = fullres_channels
        boundary_image = None
        rna_image = None
    else:
        # Multi-channel (3D array: channels, height, width)
        dapi_image = fullres_channels[0]
        boundary_image = fullres_channels[1] if fullres_channels.shape[0] > 1 else None
        rna_image = fullres_channels[2] if fullres_channels.shape[0] > 2 else None
    del fullres_channels

    # Load nuclear segmentation mask
    nuclear_mask = np.array(cell_masks_zarr.get("masks").get("0"))

    # Per-nucleus measurements on DAPI image
    nucleus_props = pd.DataFrame(
        regionprops_table(dapi_image, nuclear_mask, position=True)
    )
    # Calculate CCFS_DAPI
    mean_intensity = nucleus_props["mean_intensity"]
    std_intensity = nucleus_props["standard_deviation_intensity"]
    dapi_norm = np.percentile(mean_intensity, 99)
    ccfs_dapi = (std_intensity * std_intensity) / (mean_intensity * dapi_norm)
    nucleus_props["CCFS_DAPI"] = ccfs_dapi
    del dapi_image

    # Get CellID for each nucleus by measuring mean intensity in cellseg_mask
    cellid_props = pd.DataFrame(
        regionprops_table(cellseg_mask, nuclear_mask, position=True)
    )
    nucleus_props["CellID"] = cellid_props["mean_intensity"].astype(int)
    del nuclear_mask, cellid_props

    # Measure boundary (red channel) intensity per cell
    if boundary_image is not None:
        boundary_props = pd.DataFrame(regionprops_table(boundary_image, cellseg_mask))
        boundary_props["CellID"] = boundary_props["label"].astype(int)
        del boundary_image
    else:
        # Create empty boundary_props if boundary channel not available
        boundary_props = pd.DataFrame(
            {"CellID": nucleus_props["CellID"], "mean_intensity": np.nan}
        )

    # Measure RNA (interior) intensity per cell
    if rna_image is not None:
        rna_props = pd.DataFrame(regionprops_table(rna_image, cellseg_mask))
        rna_props["CellID"] = rna_props["label"].astype(int)
        rna_props["mean_intensity_IntRNA"] = rna_props["mean_intensity"]
        del rna_image
    else:
        # Create empty rna_props if RNA channel not available
        rna_props = pd.DataFrame(
            {
                "CellID": nucleus_props["CellID"],
                "area": np.nan,
                "mean_intensity_IntRNA": np.nan,
            }
        )

    # Merge all measurements into a single DataFrame
    merged_data = pd.merge(
        nucleus_props,
        rna_props[["CellID", "area"]],
        how="inner",
        on="CellID",
        suffixes=("_nucleus", "_cell"),
    )
    merged_data = pd.merge(
        merged_data,
        boundary_props[["CellID", "mean_intensity"]],
        how="inner",
        on="CellID",
        suffixes=("_DAPI", "_Boundary"),
    )
    merged_data = pd.merge(
        merged_data,
        rna_props[["CellID", "mean_intensity_IntRNA"]],
        how="inner",
        on="CellID",
    )

    return merged_data


# ===== COMBINED MAIN =====


def _check_cell_data_exists(xenium_bundle_dir):
    """
    Check if cell data files exist in the Xenium bundle directory.

    Args:
        xenium_bundle_dir: Path to Xenium bundle directory

    Returns:
        bool: True if required cell data files exist
    """
    xenium_bundle_dir = Path(xenium_bundle_dir)
    required_files = [
        xenium_bundle_dir / "cells.parquet",
        xenium_bundle_dir / "cells.zarr.zip",
    ]
    # Check clustering path
    clusters_path = (
        xenium_bundle_dir
        / "analysis"
        / "clustering"
        / "gene_expression_kmeans_10_clusters"
        / "clusters.csv"
    )

    # Also check for analysis.tar.gz (test data)
    has_analysis = (
        clusters_path.exists() or (xenium_bundle_dir / "analysis.tar.gz").is_file()
    )

    for f in required_files:
        if not f.exists():
            logging.info(f"  Cell data check: {f.name} not found")
            return False

    if not has_analysis:
        logging.info("  Cell data check: clustering/UMAP data not found")
        return False

    return True


@click.command()
@click.option(
    "--xenium-bundle-dir", required=True, help="Path to Xenium bundle directory"
)
@click.option("--outdir", required=True, help="Output directory for results")
@click.option(
    "--stain-names",
    default=None,
    help="Semicolon-separated list of stain names. If not provided, uses defaults.",
)
@click.option(
    "--roi-size", default=35, type=int, show_default=True, help="Tile size in pixels"
)
@click.option(
    "--max-scatter-points",
    default=10000,
    type=int,
    show_default=True,
    help="Maximum number of points to plot in scatter figures. Set to 0 to plot all points.",
)
@click.option(
    "--legacy-focus",
    is_flag=True,
    default=False,
    help="Use legacy per-tile loop focus scoring instead of the default convolution-based GPU-accelerated method.",
)
@click.option(
    "--sample-id",
    default=None,
    help="Sample identifier for logging and metrics output.",
)
@click.option(
    "--no-snr",
    is_flag=True,
    default=False,
    help="Disable SNR metrics (image Otsu / quartiles, transcripts, slide matrix, neg spatial).",
)
@click.option(
    "--snr-no-roi-tx-table",
    is_flag=True,
    default=False,
    help="Do not write SNR_roi_tx.parquet (or .csv.gz) alongside roi_qc_metrics.",
)
@click.option(
    "--snr-otsu-max-rois",
    type=int,
    default=None,
    help="Cap tiles for per-tile Otsu image SNR (default: all tiles).",
)
@click.option(
    "--snr-with-moran",
    is_flag=True,
    default=False,
    help="SNR only: enable Moran's I for neg spatial (needs PySAL/esda; default off).",
)
@click.option(
    "--stream-tiles/--no-stream-tiles",
    "stream_tiles",
    default=True,
    help=(
        "Reduce each tile as it is computed instead of assembling full-resolution "
        "pixel planes (default: stream). The planes cost ~154 GB of scratch on a "
        "5.5 gigapixel sample and mmap over a FUSE/S3 work directory is "
        "pathological; no downstream metric needs a whole plane. "
        "--no-stream-tiles restores the plane-based path, and "
        "--save-dapi-maps-tiff implies it."
    ),
)
@click.option(
    "--save-dapi-maps-tiff",
    "save_dapi_maps_tiff",
    is_flag=True,
    default=False,
    help=(
        "Write full-resolution per-pixel maps as tiled float32 TIFF "
        "(dapi_focus/mean/lap_var and boundary/intrna focus/mean when present). "
        "Default off — QC and SNR use in-memory arrays only; enable for archival "
        "or external tools (large files)."
    ),
)
@click.option(
    "--max-gpus",
    "max_gpus",
    default=0,
    type=int,
    help=(
        "Cap the number of CUDA devices used (0 = use every device detected). "
        "Nextflow's `accelerator` directive only sizes the Batch request; it does "
        "not restrict CUDA visibility, so a task that asked for one GPU but landed "
        "on a multi-GPU instance would otherwise use all of them."
    ),
)
@click.option(
    "--roi-thresholds-yaml",
    default=None,
    type=click.Path(exists=True),
    help="Path to tile image QC thresholds YAML. Overrides hardcoded defaults.",
)
@click.option(
    "--lap-sigma",
    default=1.0,
    type=float,
    show_default=True,
    help="Gaussian sigma for Laplacian of Gaussian (LoG) pre-smoothing.",
)
@click.option(
    "--pipeline-segmentation",
    default="skip",
    show_default=True,
    help="Pipeline segmentation method (params.segmentation); 'skip' for none.",
)
@click.option(
    "--is-resegmented",
    is_flag=True,
    default=False,
    help="Set when this run analyses a pipeline-resegmented bundle (post-seg).",
)
@click.option(
    "--figure-source-tables/--no-figure-source-tables",
    "figure_source_tables",
    default=False,
    help=(
        "Write the per-figure figures_source/*.csv source-data exports "
        "(unused downstream; default off; ~80s on a 5.5 GP sample)."
    ),
)
@click.option(
    "--figures/--no-figures",
    "figures",
    default=True,
    help=(
        "Generate QC figures (default true; --no-figures skips all figure "
        "rendering for a metrics-only fast run). Metric/JSON/parquet outputs "
        "are always computed regardless of this flag."
    ),
)
def main(
    xenium_bundle_dir,
    outdir,
    stain_names,
    roi_size,
    max_scatter_points,
    legacy_focus,
    sample_id,
    no_snr,
    snr_no_roi_tx_table,
    snr_otsu_max_rois,
    snr_with_moran,
    save_dapi_maps_tiff,
    stream_tiles,
    max_gpus,
    roi_thresholds_yaml,
    lap_sigma,
    pipeline_segmentation,
    is_resegmented,
    figure_source_tables,
    figures,
):
    """
    Combined Xenium Image QC pipeline.

    Performs pixel-level focus analysis (GPU-accelerated), tile-level analysis,
    and optionally cell-level analysis when cell data is available.

    Produces: tile figures, cell figures, image_qc_metrics.json, image_qc_metrics.csv,
    13 Quarto-required PNGs, and versions.yml.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    global _MAX_SCATTER_POINTS
    if max_scatter_points > 0:
        _MAX_SCATTER_POINTS = max_scatter_points

    t_total_start = time.time()
    logging.info("=" * 60)
    logging.info("Starting Combined Xenium Image QC Pipeline")
    if sample_id:
        logging.info("Sample ID: %s", sample_id)
    logging.info("=" * 60)
    logging.info(f"Input directory: {xenium_bundle_dir}")
    logging.info(f"Output directory: {outdir}")

    # Load QC thresholds from YAML (or use hardcoded defaults)
    qc_thresholds = _load_qc_thresholds(roi_thresholds_yaml)
    if roi_thresholds_yaml:
        logging.info(f"Loaded QC thresholds from: {roi_thresholds_yaml}")
    else:
        logging.info("Using built-in default QC thresholds")
    # Resolve lap_sigma from YAML (CLI value is the fallback)
    _yaml_sigma = qc_thresholds.get("lap_sigma", lap_sigma)
    try:
        _lap_sigma = float(_yaml_sigma)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid lap_sigma value %r in YAML, falling back to CLI=%s",
            _yaml_sigma,
            lap_sigma,
        )
        _lap_sigma = float(lap_sigma)
    logging.info(
        f"LoG sigma: {_lap_sigma} (CLI={lap_sigma}, YAML={qc_thresholds.get('lap_sigma', 'not set')})"
    )

    # Resolve top-level operational thresholds from YAML (with hardcoded fallbacks)
    _roi_intensity_threshold = float(
        qc_thresholds.get("roi_intensity_threshold", ROI_INTENSITY_THRESHOLD)
    )
    _min_tissue_cov = float(
        qc_thresholds.get(
            "min_tissue_coverage_for_intensity_qc",
            ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC,
        )
    )
    _roi_focus_pct = float(
        qc_thresholds.get("roi_focus_score_percentile", ROI_FOCUS_SCORE_PERCENTILE)
    )
    _blur_prob_thresh = float(qc_thresholds.get("blur_prob_threshold", 0.5))
    _focus_cfg = qc_thresholds.get("focus") or {}
    _ccfs_low_texture_threshold = float(
        _focus_cfg.get("ccfs_low_texture_threshold", DEFAULT_CCFS_LOW_TEXTURE_THRESHOLD)
    )
    logging.info(
        f"  roi_intensity_threshold={_roi_intensity_threshold}, "
        f"min_tissue_cov={_min_tissue_cov}, "
        f"roi_focus_pct={_roi_focus_pct}, "
        f"blur_prob_thresh={_blur_prob_thresh}, "
        f"ccfs_low_texture_threshold={_ccfs_low_texture_threshold}"
    )

    # ===== PHASE 1: LOAD =====
    logging.info("\n--- PHASE 1: LOAD ---")

    # Handle stain_names (default if None)
    if stain_names is None:
        stain_names_list = [
            "DAPI",
            "Boundary (ATP1A1/E-Cadherin/CD45)",
            "Interior - RNA (18S)",
            "Protein (alphaSMA/Vimentin)",
        ]
        logging.info(f"Using default stain names: {stain_names_list}")
    else:
        logging.info(f"Stain names: {stain_names}")
        if ";" in stain_names:
            stain_names_list = stain_names.split(";")
        else:
            stain_names_list = [stain_names]

    # Load downsampled images for sample QC
    morphology_focus_dir = Path(xenium_bundle_dir) / "morphology_focus"
    if (morphology_focus_dir / "morphology_focus_0000.ome.tif").exists():
        xoa_morphology_files = [
            morphology_focus_dir / "morphology_focus_0000.ome.tif",
            morphology_focus_dir / "morphology_focus_0001.ome.tif",
            morphology_focus_dir / "morphology_focus_0002.ome.tif",
            morphology_focus_dir / "morphology_focus_0003.ome.tif",
        ]
    else:
        xoa_morphology_files = sorted(
            list(morphology_focus_dir.glob("ch000*.ome.tif")),
            key=lambda x: x.stem.split("_")[0],
        )

    # Boundary / IntRNA / Protein channels are optional — DAPI-only morphology
    # bundles are valid (e.g. samples staining nuclei only). The downstream
    # loader `_load_morphology_channels` returns `None` for missing channels;
    # `generate_tissue_mask` and per-cell intensity emissions handle None.
    if len(xoa_morphology_files) < 1:
        raise ValueError("No morphology files found in morphology_focus/ directory")
    if not xoa_morphology_files[0].exists():
        raise ValueError(
            f"Morphology focus file does not exist: {xoa_morphology_files[0]}"
        )

    # Load and prepare data (creates output directories)
    data = load_and_prepare_data(xenium_bundle_dir, outdir)

    # Load morphology images (level 3, downsampled)
    logging.info("Loading morphology images...")
    t0 = time.time()
    # Use the robust loader that already handles 1/2/3-channel inputs
    # (returns None for missing Boundary / IntRNA channels). 5 other call
    # sites already exercise this loader (lines ~1366/1389/1659/1677/3456),
    # so the None-tolerant code path is well-tested for DAPI-only bundles.
    small0, small1, small2 = _load_morphology_channels(xoa_morphology_files, level=3)
    _n_channels_loaded = sum(x is not None for x in (small0, small1, small2))
    _shape_str = f"DAPI={small0.shape}"
    if small1 is not None:
        _shape_str += f", Boundary={small1.shape}"
    if small2 is not None:
        _shape_str += f", IntRNA={small2.shape}"
    logging.info(
        f"Loaded morphology images ({_n_channels_loaded} channel(s)): {_shape_str}"
    )
    logging.info(f"[TIMING] Loading morphology images: {time.time() - t0:.1f}s")
    _log_mem("morphology loaded")

    # Check if cell data exists
    has_cell_data = _check_cell_data_exists(xenium_bundle_dir)
    if has_cell_data:
        logging.info("Cell data detected - will perform cell-level analysis")
    else:
        logging.info("No cell data detected - will skip cell-level analysis")

    # ===== PHASE 2: PIXEL-LEVEL FOCUS MAPS =====
    logging.info("\n--- PHASE 2: PIXEL-LEVEL FOCUS MAPS ---")

    # Generate tissue masks and distance maps (cell-independent)
    logging.info("Generating tissue masks and distance maps...")
    t0 = time.time()
    (
        whole_sample,
        holes,
        dense_intensity_regions,
        distance_map,
        distance_map2,
        ms_whole_sample,
        ms_distance_map,
        ms_distance_map2,
    ) = generate_tissue_mask(xoa_morphology_files, small0, small1, small2)
    logging.info("Generated tissue masks and distance maps")
    logging.info(f"[TIMING] Tissue mask generation: {time.time() - t0:.1f}s")
    _log_mem("tissue mask")

    # Validate ROI size
    if roi_size <= 0:
        roi_size = 35
        logging.info(f"Using default tile size: {roi_size}px")
    else:
        logging.info(f"Using tile size: {roi_size}px")

    # Auto-detect available GPUs
    available_gpus = detect_gpu_ids()
    if available_gpus and max_gpus and len(available_gpus) > max_gpus:
        logging.info(
            f"Detected {len(available_gpus)} GPU(s) {available_gpus} but --max-gpus "
            f"={max_gpus}; using {available_gpus[:max_gpus]}. The accelerator "
            "directive sizes the Batch request only, it does not limit CUDA "
            "visibility."
        )
        available_gpus = available_gpus[:max_gpus]
    if available_gpus:
        logging.info(f"Detected {len(available_gpus)} GPU(s): {available_gpus}")
    else:
        logging.info("No GPUs detected, using CPU backend")

    # Calculate cell-independent grid ROI focus scores
    logging.info("Calculating cell-independent grid tile focus scores...")
    t0 = time.time()
    if legacy_focus:
        logging.info("  Using LEGACY per-tile loop method (--legacy-focus)")
        df_grid_roi = calculate_roi_focusscore_without_laplace(
            xoa_morphology_files,
            roi_size=roi_size,
            stride=None,
            tissue_filter=True,
            min_tissue_coverage=0.0,
        )
        focus_maps = None
        streamed = None
    else:
        logging.info("  Using convolution-based GPU-accelerated method")
        # Streaming folds each tile into small reductions and never assembles a
        # pixel plane. --save-dapi-maps-tiff is the one output that genuinely needs
        # the planes, so asking for it selects the plane-based path.
        _stream = stream_tiles and not save_dapi_maps_tiff
        if stream_tiles and save_dapi_maps_tiff:
            logging.info(
                "  --save-dapi-maps-tiff requires full pixel planes; "
                "streaming disabled for this run"
            )
        logging.info(
            f"  Tile reduction mode: {'streamed' if _stream else 'pixel planes'}"
        )
        df_grid_roi, focus_maps, streamed = calculate_roi_focusscore(
            xoa_morphology_files,
            roi_size=roi_size,
            stride=None,
            tissue_filter=True,
            min_tissue_coverage=0.0,
            gpu_ids=available_gpus if available_gpus else None,
            return_pixel_maps=True,
            lap_sigma=_lap_sigma,
            stream_tiles=_stream,
            # Gated on has_cell_data for the same reason the CCFS block below is:
            # a bundle without cells.zarr.zip has no masks to reduce over, and
            # opening it during the tile pass would fail where the plane path
            # simply skipped the whole cell section.
            cell_masks_path=(
                data["cell_masks_path"] if _stream and has_cell_data else None
            ),
            # Reuse the already-loaded level-3 DAPI plane and the tissue mask
            # generate_tissue_mask just computed from it, instead of re-decoding
            # level 3 and recomputing the mask inside calculate_roi_focusscore.
            # `whole_sample` == compute_tissue_mask(small0)[0] (same small0, same
            # min_size_hole=1500), so the result is bit-identical.
            small0_ds=small0,
            tissue_mask=whole_sample,
        )
    logging.info(f"Calculated grid tile focus scores for {len(df_grid_roi):,} tiles")
    logging.info(f"[TIMING] calculate_roi_focusscore(): {time.time() - t0:.1f}s")
    _log_mem("focus maps built")

    # ===== PHASE 3: ROI-LEVEL ANALYSIS =====
    logging.info("\n--- PHASE 3: TILE-LEVEL ANALYSIS ---")

    # Calculate ROI blur threshold
    logging.info("Calculating tile blur threshold...")
    roi_threshold = calculate_roi_blur_threshold(
        df_grid_roi,
        intensity_threshold=_roi_intensity_threshold,
        focus_percentile=_roi_focus_pct,
    )
    logging.info(f"  Calculated threshold: {roi_threshold:.2f} (raw score units)")
    logging.info(f"  Intensity threshold: {_roi_intensity_threshold}")

    # Fit 1D GMM model. On a very dim sample the GMM can fail to fit (e.g. no
    # tissue tiles clear the intensity gate); fall back to the percentile
    # threshold so the run still completes and produces a report, mirroring the
    # try/except used by the 2D GMM below.
    logging.info("Fitting 1D GMM model for tile focus scores...")
    try:
        gmm, blur_component_idx = fit_focus_gmm(
            df_grid_roi,
            intensity_threshold=_roi_intensity_threshold,
            focus_col_name="dapi_focus_score",
        )
        logging.info("Classifying tiles (1D GMM)...")
        df_grid_roi = classify_roi_blur(
            df_grid_roi,
            gmm=gmm,
            blur_component_idx=blur_component_idx,
            blur_prob_threshold=_blur_prob_thresh,
            intensity_threshold=_roi_intensity_threshold,
            focus_col_name="dapi_focus_score",
        )
    except Exception as e:
        logging.warning(f"  Warning: 1D GMM failed: {e}")
        logging.info("  Using percentile-threshold fallback for 1D blur classification")
        gmm = None
        blur_component_idx = None
        df_grid_roi = classify_roi_blur_by_threshold(
            df_grid_roi,
            roi_threshold=roi_threshold,
            intensity_threshold=_roi_intensity_threshold,
            focus_col_name="dapi_focus_score",
        )
    n_blurred = int(df_grid_roi["is_blurred_gmm"].sum())
    n_in_focus = int((~df_grid_roi["is_blurred_gmm"]).sum())
    logging.info(f"  Blurred: {n_blurred:,}, In-focus: {n_in_focus:,} (1D GMM)")

    # Fit 2D GMM if Laplacian variance available
    gmm_2d = None
    blur_component_idx_2d = None
    if "dapi_lap_var" in df_grid_roi.columns:
        logging.info("\nFitting 2D GMM model (focus_score + Laplacian variance)...")
        try:
            gmm_2d, blur_component_idx_2d = fit_focus_gmm_2d(
                df_grid_roi,
                intensity_threshold=_roi_intensity_threshold,
                focus_col_name="dapi_focus_score",
            )
            logging.info("Classifying tiles (2D GMM)...")
            df_grid_roi = classify_roi_blur_2d(
                df_grid_roi,
                gmm=gmm_2d,
                blur_component_idx=blur_component_idx_2d,
                blur_prob_threshold=_blur_prob_thresh,
                intensity_threshold=_roi_intensity_threshold,
                focus_col_name="dapi_focus_score",
            )
            n_blurred_2d = int(df_grid_roi["is_blurred_gmm_2d"].sum())
            n_in_focus_2d = int((~df_grid_roi["is_blurred_gmm_2d"]).sum())
            logging.info(
                f"  Blurred: {n_blurred_2d:,}, In-focus: {n_in_focus_2d:,} (2D GMM)"
            )
        except Exception as e:
            logging.warning(f"  Warning: 2D GMM failed: {e}")
            logging.info("  Continuing with 1D GMM results only")
            gmm_2d = None
            blur_component_idx_2d = None
    else:
        logging.info("\nSkipping 2D GMM: dapi_lap_var column not found")

    # Save grid ROI data to CSV
    grid_roi_csv = data["outdir"] / "grid_roi_focus_scores.csv"
    df_grid_roi.to_csv(grid_roi_csv, index=False)
    logging.info(f"Saved grid tile data to {grid_roi_csv}")

    # Optional: write full-res per-pixel maps (very large); not needed for QC/SNR in-process
    if focus_maps is not None and save_dapi_maps_tiff:
        logging.info("Saving pixel-level focus maps as TIFF (--save-dapi-maps-tiff)...")
        t0 = time.time()
        save_pixel_focus_maps(focus_maps, data["outdir"])
        logging.info(f"[TIMING] save_pixel_focus_maps(): {time.time() - t0:.1f}s")
    elif focus_maps is not None:
        logging.info(
            "Skipping pixel-map TIFF export (default). "
            "Pass --save-dapi-maps-tiff to write dapi_*/boundary_*/intrna_* map TIFFs."
        )
    elif streamed is not None:
        logging.info(
            "No pixel-map TIFF export: tiles were streamed, so no full-resolution "
            "planes exist. Pass --save-dapi-maps-tiff to build them."
        )

    # Free Laplacian map — no longer needed after TIFF export / GMM fitting.
    # Remaining consumers (SNR, figures, CCFS) only use focus/mean maps.
    if focus_maps is not None:
        for _k in ("dapi_lap_var_map",):
            focus_maps.pop(_k, None)

    # Save threshold configuration
    total_rois_gmm = int(len(df_grid_roi))
    n_blurred_gmm = int(df_grid_roi["is_blurred_gmm"].sum())
    pct_blurred_gmm = (
        (n_blurred_gmm / total_rois_gmm * 100.0) if total_rois_gmm > 0 else 0.0
    )

    threshold_config = {
        "roi_focus_score_threshold": float(roi_threshold),
        "roi_intensity_threshold": float(_roi_intensity_threshold),
        "roi_focus_score_percentile": float(_roi_focus_pct),
        "threshold_method": "Option B: Percentile from tissue tiles (intensity >= threshold) + fixed intensity threshold",
        "units": {
            "roi_focus_score_threshold_percentile": "raw score units",
            "roi_intensity_threshold": "raw pixel intensity (16-bit, 0-65535)",
            "component_means_log1p_focus": "log1p(raw focus score)",
        },
    }

    # The 1D GMM may have failed on a dim sample (gmm is None), in which case
    # the percentile-threshold fallback was used. Guard the dereference the same
    # way the gmm_2d stanza below is guarded, and emit a fallback marker instead.
    if gmm is not None:
        threshold_config["gmm_1d"] = {
            "n_components": int(gmm.n_components),
            "blur_component_index": int(blur_component_idx),
            "component_means_log1p_focus": [float(m) for m in gmm.means_.flatten()],
            "component_weights": [float(w) for w in gmm.weights_.flatten()],
            "blur_prob_threshold": float(_blur_prob_thresh),
            "fraction_rois_blurred_gmm": float(pct_blurred_gmm),
            "features": ["log1p(dapi_focus_score)"],
        }
    else:
        threshold_config["gmm_1d"] = {
            "status": "fallback",
            "method": "percentile_threshold",
            "roi_focus_score_threshold": float(roi_threshold),
            "blur_prob_threshold": float(_blur_prob_thresh),
            "fraction_rois_blurred_gmm": float(pct_blurred_gmm),
            "features": ["dapi_focus_score <= roi_focus_score_threshold"],
        }

    if "is_blurred_gmm_2d" in df_grid_roi.columns and gmm_2d is not None:
        n_blurred_gmm_2d = int(df_grid_roi["is_blurred_gmm_2d"].sum())
        pct_blurred_gmm_2d = (
            (n_blurred_gmm_2d / total_rois_gmm * 100.0) if total_rois_gmm > 0 else 0.0
        )
        threshold_config["gmm_2d"] = {
            "n_components": int(gmm_2d.n_components),
            "blur_component_index": int(blur_component_idx_2d),
            "component_means": [[float(m[0]), float(m[1])] for m in gmm_2d.means_],
            "component_weights": [float(w) for w in gmm_2d.weights_.flatten()],
            "blur_prob_threshold": float(_blur_prob_thresh),
            "fraction_rois_blurred_gmm": float(pct_blurred_gmm_2d),
            "features": ["log1p(dapi_focus_score)", "log1p(dapi_lap_var)"],
        }
        threshold_config["units"]["component_means_2d"] = (
            "[log1p(focus_score), log1p(lap_var)]"
        )

    threshold_json = data["outdir"] / "roi_blur_threshold.json"
    with open(threshold_json, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logging.info(f"Saved tile blur threshold configuration to {threshold_json}")

    # Save ROI count
    roi_count_file = data["outdir"] / "roi_count.txt"
    with open(roi_count_file, "w") as f:
        f.write(str(len(df_grid_roi)))

    # Calculate ROI intensities
    logging.info("Calculating tile intensities...")
    t0 = time.time()
    df_roi_intensities = calculate_roi_intensities(xoa_morphology_files, df_grid_roi)
    logging.info(f"[TIMING] Tile intensity calculation: {time.time() - t0:.1f}s")

    snr_summary = None
    if not no_snr:
        logging.info("Computing SNR metrics (roi_qc / snr_metrics)...")
        t_snr = time.time()
        pix_um = snr_metrics.read_xenium_pixel_size_um(Path(xenium_bundle_dir))
        if pix_um is None:
            logging.info(
                "  experiment.xenium missing or invalid pixel_size — transcript SNR may skip"
            )
        try:
            df_roi_intensities, snr_summary = snr_metrics.compute_snr_summary(
                df_roi_intensities,
                bundle_dir=Path(xenium_bundle_dir),
                outdir=data["outdir"],
                focus_maps=focus_maps,
                # Positional order, not a join: roi_snr_db is indexed by the
                # compute_roi_grid() order that built df_grid_roi, and
                # calculate_roi_intensities() returns df_grid_roi.copy() without
                # filtering or reordering, so row i is the same ROI in both. If that
                # function ever starts filtering, this silently mislabels every tile
                # -- the grid/DataFrame half of the invariant is pinned by
                # test_agrees_with_the_dataframe_coordinates.
                roi_snr_db=streamed.roi_snr_db if streamed else None,
                pixel_size_um=pix_um,
                otsu_max_rois=snr_otsu_max_rois,
                save_roi_tx_table=not snr_no_roi_tx_table,
                snr_include_moran=snr_with_moran,
                # Same stride as grid in calculate_roi_focusscore (stride=None → roi_size).
                roi_grid_stride=(roi_size, roi_size),
                snr_thresholds=qc_thresholds.get("snr") or {},
            )
        except Exception as e:
            logging.warning("SNR metrics failed (continuing image QC): %s", e)
            snr_summary = {
                "status": "error",
                "error": str(e),
                "components": {},
                "verdict": {"overall_snr_verdict": "NOT_COMPUTED"},
            }
        logging.info(f"[TIMING] SNR metrics: {time.time() - t_snr:.1f}s")
        _log_mem("SNR metrics")
    else:
        logging.info("SNR metrics disabled (--no-snr)")
    df_grid_roi = df_roi_intensities

    # Assess raw intensity quality (thresholds from YAML or defaults)
    _ch_cfg = qc_thresholds.get("channels") or {}

    # XOA-version-specific intensity floors (2026-06-22, qc_drift_analysis).
    # XOA 4.0 images are ~14x dimmer than 3.x, so a single floor can't serve both.
    # Pick `intensity_critical_v{major}` when present, else the legacy
    # `intensity_critical`, else the module default. Version unknown → legacy/default.
    _xoa_major = read_xenium_major_version(Path(xenium_bundle_dir))
    logging.info(f"  XOA major version for intensity floors: {_xoa_major}")

    def _pick_critical(ch_key, default):
        ch = _ch_cfg.get(ch_key) or {}
        if _xoa_major is not None:
            v = ch.get(f"intensity_critical_v{_xoa_major}")
            if isinstance(v, (int, float)):
                return v
        return ch.get("intensity_critical", default)

    _ic_dapi = _pick_critical("DAPI", _INTENSITY_CRITICAL_DEFAULTS["dapi"])
    _ic_boundary = _pick_critical("boundary", _INTENSITY_CRITICAL_DEFAULTS["boundary"])
    _ic_intrna = _pick_critical("intRNA", _INTENSITY_CRITICAL_DEFAULTS["intrna"])
    logging.info("Assessing raw intensity quality...")
    intensity_stats = assess_raw_intensity_quality(
        df_roi_intensities,
        dapi_threshold_critical=_ic_dapi,
        boundary_threshold_critical=_ic_boundary,
        intrna_threshold_critical=_ic_intrna,
        min_tissue_coverage=_min_tissue_cov,
        channel_pct_thresholds=_ch_cfg,
    )

    # Save intensity statistics
    intensity_json = data["outdir"] / "intensity_assessment.json"
    with open(intensity_json, "w") as f:
        json.dump(intensity_stats, f, indent=2)

    # Generate ROI figures (cell-independent)
    logging.info("Generating tile-level figures...")
    t0 = time.time()
    if figures:
        generate_roi_figures(
            data,
            small0,
            small1,
            small2,
            distance_map,
            distance_map2,
            whole_sample,
            holes,
            dense_intensity_regions,
            df_grid_roi,
            df_roi_intensities,
            intensity_stats,
            xoa_morphology_files,
            focus_maps=focus_maps,
            focus_heatmap=streamed.focus_heatmap if streamed else None,
            snr_thresholds=(qc_thresholds.get("snr") or {}).get("roi_tx") or {},
            multistain_whole_sample=ms_whole_sample,
            multistain_distance_map=ms_distance_map,
            multistain_distance_map2=ms_distance_map2,
            figure_source_tables=figure_source_tables,
        )
        logging.info(f"[TIMING] generate_roi_figures(): {time.time() - t0:.1f}s")
    _log_mem("tile figures")

    # Free focus-score maps no longer needed.  CCFS only requires
    # dapi_focus_map, dapi_mean_map, boundary_mean_map, intrna_mean_map.
    if focus_maps is not None:
        for _k in ("boundary_focus_map", "intrna_focus_map"):
            focus_maps.pop(_k, None)

    # Save ROI QC metrics
    logging.info("Saving tile QC metrics...")
    save_roi_qc_metrics(
        df_grid_roi,
        intensity_stats,
        data["outdir"],
        roi_size=roi_size,
        snr_summary=snr_summary,
        distance_map=distance_map,
        distance_map2=distance_map2,
        multistain_whole_sample=ms_whole_sample,
        multistain_distance_map=ms_distance_map,
        multistain_distance_map2=ms_distance_map2,
        edge_distance_threshold=-25.0,
        hole_distance_threshold=-25.0,
        min_tissue_coverage_for_qc=_min_tissue_cov,
        qc_thresholds=qc_thresholds,
        lap_sigma=_lap_sigma,
        segmentation_software=resolve_segmentation_software(
            xenium_bundle_dir, pipeline_segmentation, is_resegmented
        ),
        xoa_version=read_xenium_analysis_sw_version(xenium_bundle_dir),
    )

    # ===== PHASE 4: CELL-LEVEL ANALYSIS (conditional on cell data) =====
    if has_cell_data:
        logging.info("\n--- PHASE 4: CELL-LEVEL ANALYSIS ---")

        # Prepare cell-centred output directory
        figures_dir_cell = data["outdir"] / "figures"
        figures_dir_cell.mkdir(parents=True, exist_ok=True)
        if figure_source_tables:
            figures_source_dir_cell = figures_dir_cell / "figures_source"
            figures_source_dir_cell.mkdir(parents=True, exist_ok=True)

        # Prepare data dict for cell-level functions
        cell_data = dict(data)
        cell_data["figures_dir"] = figures_dir_cell

        # Unpack analysis.tar.gz if needed (test data)
        xenium_bundle_dir_path = Path(xenium_bundle_dir)
        if (xenium_bundle_dir_path / "analysis.tar.gz").is_file():
            import shutil

            shutil.unpack_archive(
                xenium_bundle_dir_path / "analysis.tar.gz", extract_dir="."
            )
            cell_data["clusters_csv_path"] = (
                Path("analysis")
                / "clustering"
                / "gene_expression_kmeans_10_clusters"
                / "clusters.csv"
            )
            cell_data["umap_path"] = (
                Path("analysis")
                / "umap"
                / "gene_expression_2_components"
                / "projection.csv"
            )
        else:
            cell_data["clusters_csv_path"] = (
                xenium_bundle_dir_path
                / "analysis"
                / "clustering"
                / "gene_expression_kmeans_10_clusters"
                / "clusters.csv"
            )
            cell_data["umap_path"] = (
                xenium_bundle_dir_path
                / "analysis"
                / "umap"
                / "gene_expression_2_components"
                / "projection.csv"
            )
        cell_data["cells_parquet_path"] = xenium_bundle_dir_path / "cells.parquet"
        cell_data["cell_masks_path"] = xenium_bundle_dir_path / "cells.zarr.zip"

        # Load spatial data
        logging.info("Loading spatial data and masks...")
        t0 = time.time()
        try:
            df_spatial, cellseg_mask, cell_masks_zarr = load_spatial_data(cell_data)
            logging.info(f"Loaded {len(df_spatial):,} cells")
            logging.info(f"[TIMING] Loading spatial data: {time.time() - t0:.1f}s")
        except (FileNotFoundError, KeyError, AttributeError) as e:
            logging.warning(f"Warning: Could not load spatial data: {e}")
            logging.info("Skipping cell-level analysis")
            has_cell_data = False

    if has_cell_data:
        # Map distances to cells
        logging.info("Mapping distances to cells...")
        t0 = time.time()
        df_spatial = map_distances_to_cells(
            df_spatial, distance_map, distance_map2, dense_intensity_regions
        )
        logging.info(f"[TIMING] map_distances_to_cells(): {time.time() - t0:.1f}s")

        # Calculate CCFS measurements
        logging.info("Calculating CCFS measurements...")
        t0 = time.time()
        if streamed is not None and streamed.has_per_cell:
            logging.info("  Using per-cell sums reduced during the tile pass")
            myData = calculate_ccfs_from_focus_maps(
                None,
                cell_masks_zarr,
                cellseg_mask,
                xoa_morphology_files,
                streamed=streamed,
            )
        elif (
            focus_maps is not None
            and "dapi_focus_map" in focus_maps
            and "dapi_mean_map" in focus_maps
        ):
            # NEW: Use pixel-map aggregation
            logging.info("  Using pixel-map aggregation (scipy.ndimage.mean)")
            myData = calculate_ccfs_from_focus_maps(
                focus_maps, cell_masks_zarr, cellseg_mask, xoa_morphology_files
            )
        else:
            # Fallback: use regionprops-based method
            logging.info("  Using regionprops-based method (legacy)")
            myData = calculate_ccfs_measurements(
                xoa_morphology_files, cellseg_mask, cell_masks_zarr
            )
        logging.info(f"Calculated CCFS for {len(myData):,} cells")
        logging.info(f"[TIMING] CCFS calculation: {time.time() - t0:.1f}s")
        _log_mem("per-cell CCFS")

        # All pixel-level focus maps consumed — free remaining memory. The streamed
        # reductions are kept: nothing downstream re-reads them, but they are tens
        # of MB, and dropping the name here would break the `streamed` references
        # in the figure calls that follow.
        del focus_maps
        focus_maps = None

        # Add boolean columns to myData for thresholding (needed by generate_all_figures)
        ccfs_threshold = _ccfs_low_texture_threshold
        myData["is_low_nuclear_texture"] = myData["CCFS_DAPI"] <= ccfs_threshold
        myData["is_high_nuclear_texture"] = myData["CCFS_DAPI"] > ccfs_threshold

        # Map ROI focus scores to cells
        logging.info("Mapping tile focus scores to cells...")
        t0 = time.time()
        roi_mapped = map_grid_roi_to_cells(df_grid_roi, df_spatial, overlapping=False)
        logging.info(
            f"Mapped tile focus scores to {len(roi_mapped[roi_mapped['DAPI_RFSnorm_roi'].notna()]):,} cells"
        )
        logging.info(f"[TIMING] map_grid_roi_to_cells(): {time.time() - t0:.1f}s")

        # Load ROI blur threshold
        roi_threshold_cell, roi_intensity_threshold = load_roi_blur_threshold(
            data["outdir"]
        )
        if roi_threshold_cell is None or roi_intensity_threshold is None:
            roi_threshold_cell = roi_threshold
            roi_intensity_threshold = _roi_intensity_threshold

        # Create merged dataset (superset version with ROI data)
        logging.info("Creating final merged dataset...")
        t0 = time.time()
        new_df, calculated_roi_threshold = create_final_merged_data(
            df_spatial,
            myData,
            roi_data=roi_mapped,
            ccfs_threshold=_ccfs_low_texture_threshold,
            roi_threshold=roi_threshold_cell,
            roi_intensity_threshold=roi_intensity_threshold,
        )
        logging.info(f"Created merged dataset with {len(new_df):,} cells")
        logging.info(f"[TIMING] create_final_merged_data(): {time.time() - t0:.1f}s")

        # Alias dense_intensity_regions as artefacts for generate_all_figures compatibility
        artefacts = dense_intensity_regions

        # Ensure In-Area-with-Artefact column exists (needed by generate_all_figures)
        if "In-Area-with-Artefact" not in df_spatial.columns:
            # Map dense_intensity_regions to the artefact column name
            if "Dense-Intensity-Region-ID" in df_spatial.columns:
                df_spatial["In-Area-with-Artefact"] = df_spatial[
                    "Dense-Intensity-Region-ID"
                ]
            else:
                df_spatial["In-Area-with-Artefact"] = 0

        if "In-Area-with-Artefact" not in new_df.columns:
            if "Dense-Intensity-Region-ID" in new_df.columns:
                new_df["In-Area-with-Artefact"] = new_df["Dense-Intensity-Region-ID"]
            else:
                new_df["In-Area-with-Artefact"] = 0

        # Ensure has_artifacts column exists
        if "has_artifacts" not in new_df.columns:
            new_df["has_artifacts"] = new_df.get(
                "has_dense_intensity_regions",
                new_df.get("In-Area-with-Artefact", 0) > 0,
            )

        # Generate 13 Quarto-required figures
        logging.info("Generating Quarto-required figures (13 PNGs)...")
        t0 = time.time()
        if figures:
            generate_all_figures(
                cell_data,
                df_spatial,
                new_df,
                myData,
                small0,
                small1,
                small2,
                distance_map,
                distance_map2,
                whole_sample,
                holes,
                artefacts,
                ccfs_low_texture_threshold=_ccfs_low_texture_threshold,
                multistain_whole_sample=ms_whole_sample,
                multistain_distance_map=ms_distance_map,
                multistain_distance_map2=ms_distance_map2,
                figure_source_tables=figure_source_tables,
            )
            logging.info(f"[TIMING] generate_all_figures(): {time.time() - t0:.1f}s")

        # Generate cell-centred comparison figures (ROI vs CCFS)
        if figures:
            figures_cell_centred_dir = data["outdir"] / "figures_cell_centred"
            figures_cell_centred_dir.mkdir(parents=True, exist_ok=True)
            figures_cell_centred_source = (
                (figures_cell_centred_dir / "figures_source")
                if figure_source_tables
                else None
            )
            if figures_cell_centred_source is not None:
                figures_cell_centred_source.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            generate_cell_figures(
                cell_data,
                new_df,
                myData,
                figures_cell_centred_dir,
                figures_cell_centred_source,
                roi_threshold=calculated_roi_threshold,
                roi_intensity_threshold=roi_intensity_threshold,
                ccfs_low_texture_threshold=_ccfs_low_texture_threshold,
            )
            logging.info(f"[TIMING] generate_cell_figures(): {time.time() - t0:.1f}s")
            _log_mem("cell figures")

        # Save cell QC metrics (superset version with ROI metrics)
        logging.info("Saving cell QC metrics...")
        t0 = time.time()
        save_cell_qc_metrics(new_df, data["outdir"], roi_size=roi_size)
        logging.info(f"[TIMING] save_cell_qc_metrics(): {time.time() - t0:.1f}s")

        # Save simple image_qc_metrics.json for Quarto compatibility
        logging.info("Saving Quarto-compatible image_qc_metrics.json...")
        n_total = len(new_df)
        n_ccfs_low_texture = int(new_df["is_low_nuclear_texture"].sum())
        qc_metrics = {
            "total_cells": n_total,
            "cells_with_transcripts": len(new_df[new_df["transcript_counts"] > 0]),
            "mean_transcript_count": float(new_df["transcript_counts"].mean()),
            "median_transcript_count": float(new_df["transcript_counts"].median()),
            "mean_ccfs_dapi": float(new_df["CCFS_DAPI"].mean()),
            "median_ccfs_dapi": float(new_df["CCFS_DAPI"].median()),
            "ccfs_low_texture_threshold": float(_ccfs_low_texture_threshold),
            "cells_high_nuclear_texture": len(
                new_df[new_df["is_high_nuclear_texture"]]
            ),
            "cells_low_nuclear_texture": n_ccfs_low_texture,
            "pct_low_nuclear_texture": round(100.0 * n_ccfs_low_texture / n_total, 4)
            if n_total > 0
            else 0.0,
            "cells_near_edge": len(new_df[new_df["is_near_edge"]]),
            "cells_near_holes": len(new_df[new_df["is_near_hole"]]),
            "cells_with_artifacts": int(new_df["has_artifacts"].sum()),
            "clusters_present": sorted(new_df["Cluster_kmeans10"].unique().tolist()),
            "segmentation_methods": sorted(
                new_df["segmentation_method"].unique().tolist()
            ),
        }
        # Phase 2a-revised: per-cell aggregate emissions for Section 9.A's
        # sample-level aggregates table + 9.B's roi_tissue_coverage row.
        # Naming asymmetry note: per-cell DAPI intensity column is `mean_intensity`
        # (no suffix); aggregate JSON key adds `_DAPI` for sibling-channel
        # consistency with mean_intensity_Boundary / mean_intensity_IntRNA. See
        # v3 plan §9 assumption #9.
        for _src_col, _agg_key_base, _decimals in (
            ("mean_intensity", "intensity_DAPI", 4),
            ("mean_intensity_Boundary", "intensity_Boundary", 4),
            ("mean_intensity_IntRNA", "intensity_IntRNA", 4),
            ("area_nucleus", "area_nucleus", 2),
            ("area_cell", "area_cell", 2),
        ):
            if _src_col not in new_df.columns:
                continue
            _series = new_df[_src_col]
            _mean = _series.mean()
            _median = _series.median()
            if pd.notna(_mean):
                qc_metrics[f"mean_{_agg_key_base}"] = round(float(_mean), _decimals)
            if pd.notna(_median):
                qc_metrics[f"median_{_agg_key_base}"] = round(float(_median), _decimals)

        # Phase 13 (v4): % cells below per-channel intensity floor for the
        # 9.A Tier 1 verdict rows. 2026-06-23: use the SAME XOA-version-specific
        # floors as the tile-level intensity QC (via _pick_critical), so a dim
        # XOA-4.0 sample isn't flagged against the bright-era 500/100/300 floors.
        # NaN-intensity cells are excluded from the numerator (NaN < cutoff →
        # False) but stay in the n_total denominator — same convention as
        # `pct_cells_in_low_coverage_tiles`.
        if n_total > 0:
            _cell_intensity_emissions = (
                (
                    "mean_intensity",
                    _pick_critical("DAPI", _INTENSITY_CRITICAL_DEFAULTS["dapi"]),
                    "pct_cells_below_intensity_DAPI",
                ),
                (
                    "mean_intensity_Boundary",
                    _pick_critical(
                        "boundary", _INTENSITY_CRITICAL_DEFAULTS["boundary"]
                    ),
                    "pct_cells_below_intensity_Boundary",
                ),
                (
                    "mean_intensity_IntRNA",
                    _pick_critical("intRNA", _INTENSITY_CRITICAL_DEFAULTS["intrna"]),
                    "pct_cells_below_intensity_IntRNA",
                ),
            )
            for _src_col, _cutoff, _emit_key in _cell_intensity_emissions:
                if _src_col not in new_df.columns:
                    continue
                _n_below = int((new_df[_src_col] < _cutoff).sum())
                qc_metrics[_emit_key] = round(100.0 * _n_below / n_total, 4)

        # 9.B (Phase 2a-revised, folds Phase 11): % cells in low-coverage tiles
        # (roi_tissue_coverage < 0.5). Informational; no PASS/WARN/FAIL pill until
        # calibration. NaN coverage cells are excluded (NaN < 0.5 → False), so the
        # count reflects only cells with assigned ROI tile coverage data.
        # 2026-06-26 (multi-stain): roi_tissue_coverage here is DAPI-based (it is also the
        # denominator of pct_blurred_gmm_2d_roi below, which MUST stay DAPI to match the
        # tile-level DAPI blur figure). A multi-stain cell-coverage view for this
        # informational count is a deferred follow-up; tile-level extent already uses all
        # stains. Report prose marks this as DAPI-derived.
        if "roi_tissue_coverage" in new_df.columns and n_total > 0:
            _n_low_cov = int((new_df["roi_tissue_coverage"] < 0.5).sum())
            qc_metrics["cells_in_low_coverage_tiles"] = _n_low_cov
            qc_metrics["pct_cells_in_low_coverage_tiles"] = round(
                100.0 * _n_low_cov / n_total, 4
            )

        # GMM-ROI blur metrics (if cell-to-ROI mapping was performed).
        # 2026-06-24: pct_blurred_gmm_2d_roi is reported over SOLID-tissue cells
        # (roi_tissue_coverage >= 0.5) so it matches the tile-level "tiles in
        # focus" metric (also tissue-filtered). Cells in low-coverage / edge tiles
        # are force-labelled blurred regardless of optical focus and otherwise
        # inflate this far above the tile figure (e.g. 50% cells vs 9% tiles).
        # The all-cells value is kept for context / the report's "why it differs"
        # note, alongside the existing pct_cells_in_low_coverage_tiles.
        if "is_blurred_gmm_2d_roi" in new_df.columns:
            _blur = new_df["is_blurred_gmm_2d_roi"]
            _n_blur_all = int(_blur.sum())
            qc_metrics["pct_blurred_gmm_2d_roi_all_cells"] = (
                round(100.0 * _n_blur_all / n_total, 4) if n_total > 0 else 0.0
            )
            if "roi_tissue_coverage" in new_df.columns:
                _solid = (
                    new_df["roi_tissue_coverage"]
                    >= ROI_MIN_TISSUE_COVERAGE_FOR_INTENSITY_QC
                )
            else:
                _solid = _blur.notna()
            _n_solid = int(_solid.sum())
            n_gmm = int(_blur[_solid].sum())
            qc_metrics["cells_blurred_gmm_2d_roi"] = n_gmm
            qc_metrics["cells_evaluated_for_blur"] = _n_solid
            qc_metrics["pct_blurred_gmm_2d_roi"] = (
                round(100.0 * n_gmm / _n_solid, 4) if _n_solid > 0 else 0.0
            )
            qc_metrics["pct_blurred_gmm_2d_roi_denominator"] = (
                "solid_tissue_cells_coverage_ge_0.5"
                if "roi_tissue_coverage" in new_df.columns
                else "all_cells"
            )
            agreement = int(
                (
                    new_df["is_low_nuclear_texture"] == new_df["is_blurred_gmm_2d_roi"]
                ).sum()
            )
            qc_metrics["ccfs_gmm_agreement_pct"] = (
                round(100.0 * agreement / n_total, 4) if n_total > 0 else 0.0
            )
        # Per-cluster blur stats for cluster-outlier detection
        if "Cluster_kmeans10" in new_df.columns:
            blur_col = (
                "is_blurred_gmm_2d_roi"
                if "is_blurred_gmm_2d_roi" in new_df.columns
                else "is_low_nuclear_texture"
            )
            cluster_blur = {}
            for cl, grp in new_df.groupby("Cluster_kmeans10"):
                n_cl = len(grp)
                n_blur_cl = int(grp[blur_col].sum())
                pct_blur_cl = round(100.0 * n_blur_cl / n_cl, 2) if n_cl > 0 else 0.0
                cluster_blur[int(cl)] = {
                    "n_cells": n_cl,
                    "n_blurred": n_blur_cl,
                    "pct_blurred": pct_blur_cl,
                    "median_ccfs_dapi": round(float(grp["CCFS_DAPI"].median()), 4)
                    if not pd.isna(grp["CCFS_DAPI"].median())
                    else 0.0,
                }
            qc_metrics["cluster_blur"] = cluster_blur
            qc_metrics["cluster_blur_method"] = blur_col
            # Flag clusters that stand apart from the rest (robust MAD z-score +
            # absolute floor). See detect_cluster_outliers.
            outlier_clusters = detect_cluster_outliers(cluster_blur, "pct_blurred")
            if outlier_clusters:
                qc_metrics["cluster_blur_outliers"] = outlier_clusters

        # Per-cluster CCFS-low-texture stats for cluster-outlier detection.
        # Parallel to cluster_blur above, but always derived from
        # is_low_nuclear_texture (independent of whether tile-mapped GMM blur is
        # available — distinct from cluster_blur, which falls back to
        # is_low_nuclear_texture only when is_blurred_gmm_2d_roi is absent).
        # Consumed by the §4.3 Tier-3 row and the §4.6 per-cluster breakdown in
        # notebooks/xenium_image_qc_report.qmd. Hidden by the qmd until this
        # field is present in the JSON.
        if (
            "Cluster_kmeans10" in new_df.columns
            and "is_low_nuclear_texture" in new_df.columns
        ):
            cluster_ccfs = {}
            for cl, grp in new_df.groupby("Cluster_kmeans10"):
                n_cl = len(grp)
                n_low_cl = int(grp["is_low_nuclear_texture"].sum())
                pct_low_cl = round(100.0 * n_low_cl / n_cl, 2) if n_cl > 0 else 0.0
                cluster_ccfs[int(cl)] = {
                    "n_cells": n_cl,
                    "n_low_texture": n_low_cl,
                    "pct_low_texture": pct_low_cl,
                    "median_ccfs_dapi": round(float(grp["CCFS_DAPI"].median()), 4)
                    if not pd.isna(grp["CCFS_DAPI"].median())
                    else 0.0,
                }
            qc_metrics["cluster_ccfs"] = cluster_ccfs
            # Same robust rule as cluster_blur. The absolute floor backstops the
            # MAD=0 case — most clusters sit near 0% low-texture, so a real spike
            # is caught by the floor even when the spread is degenerate.
            ccfs_outlier_clusters = detect_cluster_outliers(
                cluster_ccfs, "pct_low_texture"
            )
            if ccfs_outlier_clusters:
                qc_metrics["cluster_ccfs_outliers"] = ccfs_outlier_clusters

        with open(data["outdir"] / "image_qc_metrics.json", "w") as f:
            json.dump(qc_metrics, f, indent=2)

        # Save image_qc_metrics.csv
        cell_id_col = new_df.pop("cell_id")
        new_df.insert(0, "cell_id", cell_id_col)
        new_df.to_csv(data["outdir"] / "image_qc_metrics.csv", index=False)

        # Save dense intensity region summary
        if "Dense-Intensity-Region-ID" in new_df.columns:
            region_summary = (
                new_df[new_df["Dense-Intensity-Region-ID"] > 0]
                .groupby("Dense-Intensity-Region-ID")
                .agg({"cell_id": "count", "x": "mean", "y": "mean"})
                .reset_index()
            )
            region_summary.columns = [
                "Dense-Intensity-Region-ID",
                "cell_count",
                "mean_x",
                "mean_y",
            ]
            region_summary = region_summary.sort_values("Dense-Intensity-Region-ID")
            region_summary["annotation"] = ""
            region_summary.to_csv(
                data["outdir"] / "dense_intensity_regions_summary.csv", index=False
            )
    else:
        logging.info("\n--- PHASE 4: SKIPPED (no cell data) ---")

    # ===== PHASE 5: FINALIZE =====
    logging.info("\n--- PHASE 5: FINALIZE ---")

    # Save versions file
    logging.info("Saving versions file...")
    save_versions_file(data["outdir"])

    logging.info(f"\n[TIMING] Total pipeline time: {time.time() - t_total_start:.1f}s")
    _log_mem_summary()
    logging.info("=" * 60)
    logging.info("Combined Image QC Pipeline completed successfully!")
    logging.info(f"  Tile figures: {data['figures_dir']}")
    if has_cell_data:
        logging.info(f"  Cell figures: {data['outdir'] / 'figures'}")
        logging.info(f"  Cell metrics: {data['outdir'] / 'image_qc_metrics.json'}")
        logging.info(f"  Cell CSV: {data['outdir'] / 'image_qc_metrics.csv'}")
    logging.info(f"  Tile metrics: {data['outdir'] / 'roi_qc_metrics.json'}")
    logging.info(f"  Versions: {data['outdir'] / 'versions.yml'}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
