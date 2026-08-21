#!/usr/bin/env python3
"""
SNR metrics for Xenium image QC (ROI image, transcripts, slide matrix, neg spatial).

Used by ``bin/image_qc.py``. See ``plans/image_qc_report/SNR_plan.md``.

Dependencies: numpy, pandas; optional h5py, pyarrow, scipy, libpysal/esda, scikit-image.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# JSON ``summary["components"]`` keys — ``SNR_`` prefix keeps this module separate from
# generic ``roi_qc_metrics`` / image_qc fields when integrated into ``bin/image_qc.py``.
SNR_CKEY_IMAGE_ROI_QUARTILE_DB = "SNR_image_roi_quartile_db"
SNR_CKEY_IMAGE_OTSU = "SNR_image_otsu"
SNR_CKEY_ROI_TX = "SNR_roi_tx"
SNR_CKEY_SLIDE_PLUMMER = "SNR_slide_plummer"
SNR_CKEY_SLIDE_SPATIALQM = "SNR_slide_spatialqm"
SNR_CKEY_ROI_NEG_SPATIAL = "SNR_roi_neg_spatial"

# Per-ROI transcript SNR table (same rows as input grid + count/ratio columns) for histograms / maps.
SNR_ROI_TX_TABLE_BASENAME = "SNR_roi_tx"

# ---------------------------------------------------------------------------
# SNR_plan.md: snr_verdict_to_quality_status
# ---------------------------------------------------------------------------


def snr_verdict_to_quality_status(snr_summary: dict) -> str:
    """
    Convert SNR module verdict to the 'pass'/'warn'/'fail' scale used by
    assess_raw_intensity_quality() so the HTML report scorecard can treat
    SNR like any other channel quality metric.

        PASS  → 'pass'
        WARN  → 'warn'
        FAIL  → 'fail'
        NOT_COMPUTED / ERROR → 'not_available'
    """
    mapping = {
        "PASS": "pass",
        "WARN": "warn",
        "FAIL": "fail",
    }
    overall = snr_summary.get("verdict", {}).get("overall_snr_verdict", "NOT_COMPUTED")
    return mapping.get(overall, "not_available")


# ---------------------------------------------------------------------------
# Neg-control classification (Xenium-style)
# ---------------------------------------------------------------------------

_NEG_PATTERNS = (
    re.compile(r"^NegControlProbe_", re.I),
    re.compile(r"^NegControlCodeword_", re.I),
    re.compile(r"^Blank[-_]", re.I),
    re.compile(r"^BLANK[-_]", re.I),
    re.compile(r"^antisense_", re.I),
)


def is_neg_probe_feature(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    return any(p.search(name) for p in _NEG_PATTERNS)


# ---------------------------------------------------------------------------
# Image SNR — lightweight path (ROI DataFrame columns)
# SNR_plan: top ~75th pct tissue ROIs = signal; bottom quartile = background;
# SNR_dB = 20 * log10(mean_fg / std_bg)
# ---------------------------------------------------------------------------


def compute_image_snr_from_roi_df(
    df_grid_roi: pd.DataFrame,
    intensity_threshold: float = 0.0,
    intensity_col: Optional[str] = None,
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Slide-level image SNR (dB) from pre-aggregated ROI intensities; same value
    conceptually applies to all tissue ROIs (SNR_plan lightweight path).
    """
    if intensity_col is None:
        intensity_col = (
            "dapi_intensity"
            if "dapi_intensity" in df_grid_roi.columns
            else "raw_intensity"
        )
    if intensity_col not in df_grid_roi.columns:
        return {"status": "skipped", "reason": f"missing column {intensity_col!r}"}

    df = df_grid_roi.copy()
    if "overlaps_tissue" in df.columns:
        tissue = df[df["overlaps_tissue"]].copy()
    elif "tissue_coverage" in df.columns:
        tissue = df[df["tissue_coverage"] > 0].copy()
    else:
        tissue = df

    tissue = tissue[tissue[intensity_col] >= intensity_threshold]
    if len(tissue) < 8:
        return {
            "status": "skipped",
            "reason": "too_few_tissue_rois",
            "n": int(len(tissue)),
        }

    vals = tissue[intensity_col].astype(np.float64).values
    q25, q75 = np.percentile(vals, [25, 75])
    bg = vals[vals <= q25]
    fg = vals[vals >= q75]
    if len(bg) < 2 or len(fg) < 2:
        return {"status": "skipped", "reason": "empty_quartile_split"}

    mean_fg = float(np.mean(fg))
    std_bg = float(np.std(bg, ddof=1)) if len(bg) > 1 else float(np.std(bg))
    eps = 1e-12
    if std_bg < eps:
        return {"status": "skipped", "reason": "zero_background_std"}

    ratio = mean_fg / std_bg
    snr_db = 20.0 * math.log10(max(ratio, eps))
    _t = snr_thresholds or {}
    _img = _t.get("image_snr_db") or {}
    warn_db = float(_img.get("warn", 15.0))
    fail_db = float(_img.get("fail", 10.0))
    verdict = "PASS" if snr_db >= warn_db else ("WARN" if snr_db >= fail_db else "FAIL")

    return {
        "status": "ok",
        "method": "roi_df_quartiles",
        "intensity_col": intensity_col,
        "snr_db": snr_db,
        "mean_foreground": mean_fg,
        "std_background": std_bg,
        "n_tissue_rois": int(len(tissue)),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Otsu (numpy-only fallback; optional skimage)
# ---------------------------------------------------------------------------


def _otsu_threshold_uint16(arr: np.ndarray) -> float:
    """Histogram Otsu for non-negative array (2D)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 4:
        return float(np.median(a)) if a.size else 0.0
    # uint16-style bins for Xenium-ish data; cap bins for speed
    a_min, a_max = float(np.min(a)), float(np.max(a))
    if a_max <= a_min:
        return a_min
    nb = min(256, int(a_max - a_min) + 1)
    hist, bin_edges = np.histogram(a, bins=nb, range=(a_min, a_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w = hist.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return float(np.median(a))
    w = w / total
    mu = (w * bin_centers).sum()
    omega = np.cumsum(w)
    mu_t = np.cumsum(w * bin_centers)
    sigma_b2 = (mu_t - mu * omega) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return float(bin_centers[idx])


def roi_snr_db(tile: np.ndarray) -> float:
    """SNR in dB for one ROI window: Otsu foreground mean over background stdev.

    Returns NaN when the window cannot yield a value — too few pixels, an empty
    foreground or background after thresholding, or a degenerate background with
    zero spread. NaN rather than an exception because a slide legitimately
    contains such windows (empty edge tissue) and they are reported as N/A.

    Factored out so the whole-map path and the streaming tile consumer
    (``image_qc.RoiOtsuSnrAccumulator``) share one implementation: equivalence is
    then structural rather than something a test has to keep rediscovering.

    The window is reduced in **float64**. Production maps are float32, but the
    batched (``roi_snr_db_batch``) and numba/GPU paths promote to float64, and a
    float32 masked reduction is order-dependent so it cannot bit-match a batched
    one. Computing every path in float64 makes CPU and GPU results identical to
    float64 rounding instead of differing by ~2e-6 dB with the instance type.
    """
    if tile.size < 4:
        return float("nan")
    tile = np.asarray(tile, dtype=np.float64)

    try:
        from skimage.filters import threshold_otsu as _skimage_threshold_otsu  # type: ignore
    except Exception:
        _skimage_threshold_otsu = None

    if _skimage_threshold_otsu is not None:
        try:
            threshold = float(_skimage_threshold_otsu(tile))
        except Exception:
            threshold = _otsu_threshold_uint16(tile)
    else:
        threshold = _otsu_threshold_uint16(tile)

    foreground = tile[tile >= threshold]
    background = tile[tile < threshold]
    if foreground.size < 2 or background.size < 2:
        return float("nan")

    mean_fg = float(np.mean(foreground))
    std_bg = float(np.std(background, ddof=1))
    eps = 1e-12
    if std_bg < eps:
        return float("nan")
    return 20.0 * math.log10(max(mean_fg / std_bg, eps))


def roi_snr_db_batch(tiles: Any, xp: Any = np) -> Any:
    """Vectorized ``roi_snr_db`` over a stack of equal-size ROI windows.

    ``tiles`` is ``(K, h, w)`` or ``(K, N)``; returns a ``(K,)`` float64 array of
    dB values, one per window, computed as a single batched reduction with **no
    Python per-ROI loop**. Pass ``xp=cupy`` to run entirely on the GPU — every op
    below (min/max, bincount, cumsum, argmax, masked reductions) exists in both
    numpy and cupy, so the same code runs on either device. This is the form the
    streaming tile pass uses when the tile's ``mean_map`` is already resident in
    VRAM (``image_qc._compute_channel_maps_on_gpu``): the Otsu SNR is folded on
    device and only the small ``(K,)`` dB vector returns to host.

    Reproduces ``skimage.filters.threshold_otsu`` (256 bins over each window's own
    min..max, the float-corrected uniform-bin assignment numpy uses) and the
    ``foreground-mean / background-std(ddof=1)`` dB. Equivalence vs the per-ROI
    ``roi_snr_db`` is exact to float64 rounding (max ~1e-14 dB, verified in
    ``tests/test_roi_snr_db_batch.py``).

    Assumes finite, full-size windows. Constant windows (zero span) return NaN,
    matching the scalar path (an all-foreground split leaves the background empty).
    Callers route edge-clipped or non-finite windows to the scalar ``roi_snr_db``.
    """
    nbins = 256
    eps = 1e-12
    flat = xp.asarray(tiles).reshape(xp.asarray(tiles).shape[0], -1).astype(xp.float64)
    K, N = flat.shape[0], flat.shape[1]
    out = xp.full(K, xp.nan, dtype=xp.float64)
    if N < 4:
        return out

    mn = flat.min(axis=1)
    mx = flat.max(axis=1)
    ok = mx > mn  # constant windows -> NaN, as in roi_snr_db
    if not bool(ok.any()):
        return out
    sub = flat[ok]
    mn_s = mn[ok]
    step = (mx[ok] - mn_s) / nbins

    # numpy's uniform-bin assignment for np.histogram(row, nbins, (min, max)),
    # including its float corrections against the arithmetic edge mn + idx*step.
    idx = ((sub - mn_s[:, None]) / step[:, None]).astype(xp.intp)
    idx = xp.where(idx == nbins, nbins - 1, idx)
    idx = xp.clip(idx, 0, nbins - 1)
    left = mn_s[:, None] + idx * step[:, None]
    idx = xp.where(sub < left, idx - 1, idx)
    idx = xp.clip(idx, 0, nbins - 1)
    right = mn_s[:, None] + (idx + 1) * step[:, None]
    idx = xp.where((sub >= right) & (idx != nbins - 1), idx + 1, idx)
    idx = xp.clip(idx, 0, nbins - 1)

    Ks = sub.shape[0]
    flat_idx = (idx + xp.arange(Ks)[:, None] * nbins).reshape(-1)
    counts = (
        xp.bincount(flat_idx, minlength=Ks * nbins)
        .reshape(Ks, nbins)
        .astype(xp.float64)
    )
    centers = mn_s[:, None] + (xp.arange(nbins) + 0.5) * step[:, None]

    # skimage.filters.threshold_otsu's histogram math, per row.
    with np.errstate(invalid="ignore", divide="ignore"):
        w1 = xp.cumsum(counts, axis=1)
        w2 = xp.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
        cb = counts * centers
        mean1 = xp.cumsum(cb, axis=1) / w1
        mean2 = (xp.cumsum(cb[:, ::-1], axis=1) / w2[:, ::-1])[:, ::-1]
    var12 = w1[:, :-1] * w2[:, 1:] * (mean1[:, :-1] - mean2[:, 1:]) ** 2
    thr = centers[xp.arange(Ks), xp.argmax(var12, axis=1)]

    fg = sub >= thr[:, None]
    nfg = fg.sum(axis=1)
    nbg = N - nfg
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_fg = xp.where(fg, sub, 0.0).sum(axis=1) / nfg
        mean_bg = xp.where(~fg, sub, 0.0).sum(axis=1) / nbg
        ss_bg = xp.where(~fg, (sub - mean_bg[:, None]) ** 2, 0.0).sum(axis=1)
        std_bg = xp.sqrt(ss_bg / (nbg - 1))
    good = (nfg >= 2) & (nbg >= 2) & (std_bg >= eps)
    db = xp.full(Ks, xp.nan, dtype=xp.float64)
    ratio = xp.maximum(mean_fg / std_bg, eps)
    db = xp.where(good, 20.0 * xp.log10(ratio), db)
    out[ok] = db
    return out


try:
    import numba as _numba

    _HAS_NUMBA = True
except Exception:  # numba is an optional accelerator; callers fall back otherwise
    _HAS_NUMBA = False

if _HAS_NUMBA:
    # cache=False deliberately: each Nextflow task is a fresh process with an
    # ephemeral work dir, so an on-disk cache never survives to be reused, and
    # numba's cache key for a path-loaded module is "<dynamic>", which poisons a
    # recompile with `ModuleNotFoundError: No module named '<dynamic>'`. Compiling
    # once per task (~seconds) is negligible against the fold.
    @_numba.njit(cache=False, fastmath=False)
    def _roi_snr_db_one_numba(tile_flat, nbins):  # noqa: D401
        """One ROI: skimage-Otsu threshold then fg-mean/bg-std dB, all float64.

        Reproduces ``roi_snr_db`` for a finite, non-constant window; returns NaN
        for the same degenerate cases (too small, constant, empty fg/bg, zero bg
        spread). Compiled + released-GIL so ``prange`` scales across cores.
        """
        n = tile_flat.size
        if n < 4:
            return np.nan
        mn = tile_flat[0]
        mx = tile_flat[0]
        for v in tile_flat:
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        if mx <= mn:
            return np.nan
        step = (mx - mn) / nbins
        counts = np.zeros(nbins, dtype=np.float64)
        for v in tile_flat:
            b = int((v - mn) / step)
            if b == nbins:
                b = nbins - 1
            if v < mn + b * step:
                b -= 1
            elif b != nbins - 1 and v >= mn + (b + 1) * step:
                b += 1
            if b < 0:
                b = 0
            elif b >= nbins:
                b = nbins - 1
            counts[b] += 1.0
        w1 = np.cumsum(counts)
        centers = np.empty(nbins, dtype=np.float64)
        for j in range(nbins):
            centers[j] = mn + (j + 0.5) * step
        m1cum = np.cumsum(counts * centers)
        total = w1[nbins - 1]
        cbtot = m1cum[nbins - 1]
        best_var = -1.0
        best_idx = 0
        for t in range(nbins - 1):
            wa = w1[t]
            wb = total - wa
            if wa == 0.0 or wb == 0.0:
                continue
            ma = m1cum[t] / wa
            mb = (cbtot - m1cum[t]) / wb
            var = wa * wb * (ma - mb) ** 2
            if var > best_var:
                best_var = var
                best_idx = t
        thr = centers[best_idx]
        sfg = 0.0
        nfg = 0
        sbg = 0.0
        nbg = 0
        for v in tile_flat:
            if v >= thr:
                sfg += v
                nfg += 1
            else:
                sbg += v
                nbg += 1
        if nfg < 2 or nbg < 2:
            return np.nan
        mean_fg = sfg / nfg
        mean_bg = sbg / nbg
        ss = 0.0
        for v in tile_flat:
            if v < thr:
                d = v - mean_bg
                ss += d * d
        std_bg = np.sqrt(ss / (nbg - 1))
        if std_bg < 1e-12:
            return np.nan
        ratio = mean_fg / std_bg
        if ratio < 1e-12:
            ratio = 1e-12
        return 20.0 * np.log10(ratio)

    @_numba.njit(cache=False, parallel=True, fastmath=False)  # see cache note above
    def _roi_snr_db_numba_kernel(stack2d, nbins):
        K = stack2d.shape[0]
        out = np.empty(K, dtype=np.float64)
        for k in _numba.prange(K):
            out[k] = _roi_snr_db_one_numba(stack2d[k], nbins)
        return out


def roi_snr_db_numba(tiles: Any) -> np.ndarray:
    """CPU-parallel batched ``roi_snr_db`` (numba ``prange``).

    The efficient path for **CPU** instances: ~50x the Python per-ROI loop on 8
    cores, versus the pure-numpy ``roi_snr_db_batch(xp=np)`` which is memory-bound
    and no faster than the loop. On **GPU** instances use ``roi_snr_db_batch(xp=cp)``
    instead. Equivalent to ``roi_snr_db`` to float64 rounding.

    Thread count follows ``numba.set_num_threads`` / ``NUMBA_NUM_THREADS``; the
    caller must cap it so it does not oversubscribe against the tile worker pool.
    """
    if not _HAS_NUMBA:
        raise RuntimeError("roi_snr_db_numba requires numba, which is not installed")
    arr = np.asarray(tiles)
    flat = np.ascontiguousarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)
    if flat.shape[1] < 4:
        return np.full(flat.shape[0], np.nan, dtype=np.float64)
    return _roi_snr_db_numba_kernel(flat, 256)


def compute_image_snr_from_pixel_maps(
    focus_maps: Optional[Dict[str, Any]],
    df_grid_roi: pd.DataFrame,
    map_key: str = "dapi_mean_map",
    max_rois: Optional[int] = None,
    snr_thresholds: Optional[Dict[str, Any]] = None,
    precomputed_db: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Per-ROI SNR_dB from pixel tiles: Otsu foreground vs background std (SNR_plan accurate path).
    Uses ``dapi_mean_map`` (or ``map_key``) slices [y1:y2, x1:x2] per ROI row.

    ``precomputed_db`` supplies one dB value per ROI when the caller already
    reduced them during a tiled pass, in which case no pixel map is needed. Both
    paths get their dB from :func:`roi_snr_db`, so the aggregate statistics below
    are computed identically either way.
    """
    if precomputed_db is None:
        arr = focus_maps.get(map_key) if focus_maps else None
        if arr is None:
            return {"status": "skipped", "reason": f"missing focus_maps[{map_key!r}]"}

        # Keep the map lazy (do NOT force a whole-array float64 copy — for a
        # full-res / memmap-backed map that materialises tens of GB). Each tile is
        # sliced then converted per-tile below.
        img = np.asarray(arr)
        h, w = img.shape[:2]

    required = {"x1", "x2", "y1", "y2"}
    if not required.issubset(df_grid_roi.columns):
        return {"status": "skipped", "reason": f"df_grid_roi needs columns {required}"}

    df = df_grid_roi.copy()
    if max_rois is not None:
        df = df.iloc[: int(max_rois)]

    x1a = df["x1"].to_numpy(dtype=np.int64)
    x2a = df["x2"].to_numpy(dtype=np.int64)
    y1a = df["y1"].to_numpy(dtype=np.int64)
    y2a = df["y2"].to_numpy(dtype=np.int64)

    # Per-tile dB array aligned with df rows (NaN for skipped tiles).
    # Used both for aggregate stats and to expose per-tile values to df_grid_roi
    # downstream — the §3.4 cross-section-concordance scatter consumes this.
    per_tile_db = np.full(len(df), np.nan, dtype=np.float64)
    if precomputed_db is not None:
        supplied = np.asarray(precomputed_db, dtype=np.float64)
        if supplied.size < len(df):
            return {
                "status": "skipped",
                "reason": f"precomputed_db has {supplied.size} values for {len(df)} tiles",
            }
        per_tile_db = supplied[: len(df)].copy()
        dbs = [float(v) for v in per_tile_db[~np.isnan(per_tile_db)]]
    else:
        dbs = []
        for k in range(len(df)):
            x1, x2, y1, y2 = int(x1a[k]), int(x2a[k]), int(y1a[k]), int(y2a[k])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            db_val = roi_snr_db(img[y1:y2, x1:x2])
            if math.isnan(db_val):
                continue
            per_tile_db[k] = db_val
            dbs.append(db_val)

    # Write per-tile values back to the caller's df_grid_roi (in-place,
    # parallel to how compute_roi_transcript_snr writes roi_tx_snr_ratio).
    # df.index is a subset of df_grid_roi.index when max_rois is set; rows
    # outside that subset retain NaN.
    df_grid_roi.loc[df.index, "snr_image_otsu_db"] = per_tile_db

    if not dbs:
        return {"status": "skipped", "reason": "no_valid_roi_tiles"}

    _t = snr_thresholds or {}
    _img = _t.get("image_snr_db") or {}
    warn_db = float(_img.get("warn", 15.0))
    fail_db = float(_img.get("fail", 10.0))
    med_db = float(np.median(dbs))
    return {
        "status": "ok",
        "method": "per_roi_otsu",
        "map_key": map_key,
        "n_rois_computed": len(dbs),
        "snr_db_median": med_db,
        "snr_db_mean": float(np.mean(dbs)),
        "snr_db_p25": float(np.percentile(dbs, 25)),
        "snr_db_p75": float(np.percentile(dbs, 75)),
        "verdict": "PASS"
        if med_db >= warn_db
        else ("WARN" if med_db >= fail_db else "FAIL"),
    }


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


#: Transcript rows per batch for the per-ROI SNR reduction. Bounded so the
#: per-batch temporaries never approach the frame this replaces.
ROI_TX_BATCH_ROWS = 2_000_000


def load_transcripts(path: Path) -> pd.DataFrame:
    """Load minimal columns: feature_name, x_location, y_location, cell_id (optional).

    Projects the columns at read time rather than after.  A Xenium transcript
    table has ~20 columns and can run to several hundred million rows, and this
    load happens while the full-resolution focus planes are still live — reading
    everything and then slicing cost three overlapping copies of the full frame.
    """
    path = Path(path)
    cols_want = ["feature_name", "x_location", "y_location"]

    if path.suffix.lower() in (".parquet", ".pq"):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [c for c in cols_want if c not in available]
        if missing:
            raise ValueError(f"transcripts missing column(s) {missing!r}")
        keep = cols_want + (["cell_id"] if "cell_id" in available else [])
        return pd.read_parquet(path, columns=keep)

    if path.name.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        available = set(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in cols_want if c not in available]
        if missing:
            raise ValueError(f"transcripts missing column(s) {missing!r}")
        keep = cols_want + (["cell_id"] if "cell_id" in available else [])
        return pd.read_csv(path, usecols=keep)

    raise ValueError(f"Unsupported transcript format: {path}")


def transcripts_um_to_px(df_tx: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    """Convert x_location / y_location from microns to pixels (divide by pixel size)."""
    out = df_tx.copy()
    ps = float(pixel_size_um)
    if ps <= 0:
        raise ValueError("pixel_size_um must be positive")
    out["x_px"] = out["x_location"].astype(np.float64) / ps
    out["y_px"] = out["y_location"].astype(np.float64) / ps
    return out


# ---------------------------------------------------------------------------
# ROI transcript SNR + neg_pct
# ---------------------------------------------------------------------------


def _infer_uniform_grid_strides(
    df_grid_roi: pd.DataFrame,
) -> Optional[Tuple[int, int]]:
    """
    Infer (stride_x, stride_y) for a regular grid like ``image_qc`` (arange(0, W, stride)).

    Returns None if x1/y1 spacing is irregular (fall back to slow geometric assign).
    """
    x1 = df_grid_roi["x1"].to_numpy(dtype=np.int64)
    y1 = df_grid_roi["y1"].to_numpy(dtype=np.int64)
    ux = np.sort(np.unique(x1))
    uy = np.sort(np.unique(y1))
    if len(ux) >= 2:
        dx = np.diff(ux)
        if dx.size == 0 or int(dx[0]) <= 0 or not np.all(dx == dx[0]):
            return None
        sx = int(dx[0])
    else:
        sx = int(df_grid_roi["x2"].iloc[0] - df_grid_roi["x1"].iloc[0])
        if sx <= 0:
            return None
    if len(uy) >= 2:
        dy = np.diff(uy)
        if dy.size == 0 or int(dy[0]) <= 0 or not np.all(dy == dy[0]):
            return None
        sy = int(dy[0])
    else:
        sy = int(df_grid_roi["y2"].iloc[0] - df_grid_roi["y1"].iloc[0])
        if sy <= 0:
            return None
    if not bool(np.all((x1 % sx) == 0)) or not bool(np.all((y1 % sy) == 0)):
        return None
    return sx, sy


class _UniformRoiLookup:
    """Prebuilt (iy, ix) -> roi_id lattice, so the streaming path builds it once.

    `_roi_grid_assign_fast_uniform` derives the lattice from the ROI table on every
    call, and its collision check is an `np.unique` over one entry per ROI -- 424 ms
    for 4.49 M ROIs. Calling it per transcript batch would spend that repeatedly on a
    lookup that cannot change: at 1,325,798,498 transcripts the reference sample is
    663 batches, so ~291 s of pure rebuild.

    Returns None from :meth:`build` when the ROI layout is not a simple lattice, so
    the caller can fall back to the general path.
    """

    __slots__ = ("_grid", "_stride_x", "_stride_y", "_max_ix", "_max_iy")

    def __init__(self, grid, stride_x, stride_y, max_ix, max_iy):
        self._grid = grid
        self._stride_x = stride_x
        self._stride_y = stride_y
        self._max_ix = max_ix
        self._max_iy = max_iy

    @classmethod
    def build(
        cls, df_grid_roi: pd.DataFrame, stride_x: int, stride_y: int
    ) -> Optional["_UniformRoiLookup"]:
        x1 = df_grid_roi["x1"].to_numpy(dtype=np.int64)
        y1 = df_grid_roi["y1"].to_numpy(dtype=np.int64)
        rids = df_grid_roi["roi_id"].to_numpy(dtype=np.int32)
        ix = x1 // stride_x
        iy = y1 // stride_y
        max_ix, max_iy = int(ix.max()) + 1, int(iy.max()) + 1
        flat_idx = iy.astype(np.int64) * max_ix + ix.astype(np.int64)
        if len(np.unique(flat_idx)) != len(flat_idx):
            return None
        grid = np.full(max_iy * max_ix, -1, dtype=np.int32)
        grid[flat_idx] = rids
        return cls(grid.reshape(max_iy, max_ix), stride_x, stride_y, max_ix, max_iy)

    def assign(self, x_px: np.ndarray, y_px: np.ndarray) -> np.ndarray:
        """roi_id per point. Out-of-lattice coordinates clip to the edge ROI, which
        is what `_roi_grid_assign_fast_uniform` has always done."""
        xi = (x_px.astype(np.int64, copy=False) // self._stride_x).clip(
            0, self._max_ix - 1
        )
        yi = (y_px.astype(np.int64, copy=False) // self._stride_y).clip(
            0, self._max_iy - 1
        )
        return self._grid[yi, xi]


def _roi_grid_assign_fast_uniform(
    df_grid_roi: pd.DataFrame,
    x_px: np.ndarray,
    y_px: np.ndarray,
    stride_x: int,
    stride_y: int,
) -> Optional[np.ndarray]:
    """
    O(n_tx) assignment: tile index from pixel coords, lookup pre-filled roi_id grid.

    Returns None if ROI layout does not match a simple (iy, ix) lattice (collisions).
    """
    lookup = _UniformRoiLookup.build(df_grid_roi, stride_x, stride_y)
    if lookup is None:
        return None
    return lookup.assign(x_px, y_px)


def _roi_grid_assign(
    df_grid_roi: pd.DataFrame,
    x_px: np.ndarray,
    y_px: np.ndarray,
    stride_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Assign each transcript pixel to ``roi_id``, or -1.

    **Fast path (typical):** uniform stride grid from ``image_qc`` → O(n_tx) index + lookup.
    Pass ``stride_xy=(stride_x, stride_y)`` from the same ``roi_size`` / stride used to build
    the grid (see ``bin/image_qc.py``) to skip inference on large ROI tables.

    **Slow path:** O(n_roi × n_tx) rectangle tests (irregular ROIs / spikes).
    """
    if stride_xy is not None:
        sx, sy = int(stride_xy[0]), int(stride_xy[1])
        if sx > 0 and sy > 0:
            fast = _roi_grid_assign_fast_uniform(df_grid_roi, x_px, y_px, sx, sy)
            if fast is not None:
                logger.debug(
                    "SNR ROI assignment: fast path (caller stride %d×%d), %d points",
                    sx,
                    sy,
                    int(x_px.shape[0]),
                )
                return fast
            logger.info(
                "SNR ROI assignment: caller stride (%d, %d) incompatible with ROI layout; "
                "inferring or slow path",
                sx,
                sy,
            )

    inferred = _infer_uniform_grid_strides(df_grid_roi)
    if inferred is not None:
        sx, sy = inferred
        fast = _roi_grid_assign_fast_uniform(df_grid_roi, x_px, y_px, sx, sy)
        if fast is not None:
            logger.info(
                "SNR ROI assignment: fast uniform grid (inferred stride %d×%d), %d transcripts",
                sx,
                sy,
                int(x_px.shape[0]),
            )
            return fast
        logger.info(
            "SNR ROI assignment: inferred stride rejected (collision); using slow path"
        )

    rid_out = np.full(x_px.shape[0], -1, dtype=np.int32)
    rids = df_grid_roi["roi_id"].to_numpy(dtype=np.int32)
    x1a = df_grid_roi["x1"].to_numpy(dtype=np.float64)
    x2a = df_grid_roi["x2"].to_numpy(dtype=np.float64)
    y1a = df_grid_roi["y1"].to_numpy(dtype=np.float64)
    y2a = df_grid_roi["y2"].to_numpy(dtype=np.float64)
    n_tx = int(x_px.shape[0])
    n_roi = len(df_grid_roi)
    if n_tx > 2_000_000 and n_roi * n_tx > 5e9:
        logger.warning(
            "Large transcript count (%d) × ROIs (%d): slow ROI assignment O(n_roi×n_tx).",
            n_tx,
            n_roi,
        )
    for i in range(n_roi):
        m = (x_px >= x1a[i]) & (x_px < x2a[i]) & (y_px >= y1a[i]) & (y_px < y2a[i])
        rid_out[m] = int(rids[i])
    return rid_out


def _neg_mask_vectorized(feats: np.ndarray) -> np.ndarray:
    """Same logic as ``is_neg_probe_feature``, vectorised for large transcript tables."""
    s = pd.Series(feats, dtype="string")
    pat = r"^(?:NegControlProbe_|NegControlCodeword_|Blank[-_]|antisense_)"
    return s.str.match(pat, case=False).fillna(False).to_numpy(dtype=bool)


def _neg_mask_for_column(values: pd.Series) -> np.ndarray:
    """Negative-control mask for a transcript ``feature_name`` column.

    When the column is dictionary-encoded -- which is how Xenium writes it, and what
    ``ParquetFile.iter_batches`` hands back -- the regex runs over the ~13 k distinct
    categories and the result is indexed by the codes, instead of materialising one
    Python string per transcript. The mask is identical either way.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        cats = _neg_mask_vectorized(values.cat.categories.to_numpy())
        codes = values.cat.codes.to_numpy()
        out = np.zeros(codes.shape[0], dtype=bool)
        known = codes >= 0
        out[known] = cats[codes[known]]
        return out
    return _neg_mask_vectorized(values.astype(str).to_numpy())


def _accumulate_roi_tx_counts(
    df_grid_roi: pd.DataFrame,
    row_ix: pd.Series,
    x_px: np.ndarray,
    y_px: np.ndarray,
    is_neg: np.ndarray,
    counters: Tuple[np.ndarray, np.ndarray, np.ndarray],
    stride_xy: Optional[Tuple[int, int]],
    lookup: Optional["_UniformRoiLookup"] = None,
    roi_id_is_arange: bool = False,
) -> None:
    """Fold one batch of transcripts into the per-ROI counters.

    ``np.bincount`` rather than ``np.add.at``: both count occurrences exactly, but
    ``add.at`` is an order of magnitude slower, and this runs over every transcript.

    Two count-preserving micro-opts:

    * When ``roi_id_is_arange`` (``roi_id == arange(n_rois)``, how ``image_qc`` builds
      the grid), the ``roi_id -> row index`` map is the identity: a valid id maps to
      itself and an out-of-grid ``-1`` stays ``-1``. The pandas ``reindex`` is then a
      no-op, so ``j`` is ``assign`` directly.
    * ``real = total - neg`` instead of a third ``bincount``. Every counted transcript
      is exactly one of neg / real, so this is the same integer per ROI -- two
      ``bincount`` passes rather than three.
    """
    if lookup is not None:
        assign = lookup.assign(x_px, y_px)
    else:
        assign = _roi_grid_assign(df_grid_roi, x_px, y_px, stride_xy=stride_xy)
    assign = np.asarray(assign, dtype=np.int64)
    if roi_id_is_arange:
        j = assign
    else:
        j = row_ix.reindex(assign, fill_value=-1).to_numpy()
    ok = (j >= 0) & (assign >= 0)
    real_c, neg_c, total_c = counters
    n = real_c.shape[0]
    total_b = np.bincount(j[ok], minlength=n)
    neg_b = np.bincount(j[ok & is_neg], minlength=n)
    total_c += total_b
    neg_c += neg_b
    real_c += total_b - neg_b


def _stream_roi_tx_counts(
    df_grid_roi: pd.DataFrame,
    row_ix: pd.Series,
    path: Path,
    pixel_size_um: float,
    stride_xy: Optional[Tuple[int, int]],
    batch_rows: int = ROI_TX_BATCH_ROWS,
    roi_id_is_arange: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    """Per-ROI real/neg/total transcript counts, read in batches, folded in parallel.

    The whole-frame version materialised ``feature_name``, ``x_location`` and
    ``y_location`` for every transcript, then ``transcripts_um_to_px`` copied the
    frame and added two more float64 columns. On run 1ZyVIlaKBYxJrQ that OOM-killed
    IMAGE_QC (exit 137) at the 180 GB tier *after* the tile pass had completed in
    19.4 minutes. The outputs are three per-ROI int64 arrays, so nothing about this
    needs the transcripts resident.

    Batches are decoded (``batch.to_pandas``, which releases the GIL) and folded
    (``np.bincount``, also GIL-releasing) on a thread pool sized to ``os.cpu_count()``.
    Each worker thread owns a private ``(real, neg, total)`` counter triple and folds
    every batch it draws into it via ``_accumulate_roi_tx_counts``; the per-thread
    partials are summed on the main thread once the pool drains. Counts are additive,
    so the sum is bit-identical to the serial fold no matter how the batches interleave
    across threads -- there is no order dependence in integer addition. In-flight
    batches are capped at the worker count, so peak memory is bounded exactly as the
    serial loop's was (a handful of ``batch_rows`` batches, not the whole table).
    """
    n_rois = len(df_grid_roi)
    ps = float(pixel_size_um)
    if ps <= 0:
        raise ValueError("pixel_size_um must be positive")

    # Built once, not per batch: see _UniformRoiLookup. None means the ROI layout is
    # not a lattice, and each batch falls back to the general assignment.
    lookup = None
    if stride_xy is not None and stride_xy[0] > 0 and stride_xy[1] > 0:
        lookup = _UniformRoiLookup.build(
            df_grid_roi, int(stride_xy[0]), int(stride_xy[1])
        )
    if lookup is None:
        logger.info(
            "SNR ROI assignment: no uniform lattice; assigning per batch (slower)"
        )

    handle = pq.ParquetFile(str(path))
    columns = ["feature_name", "x_location", "y_location"]
    max_workers = max(1, os.cpu_count() or 1)
    logger.info(
        "SNR ROI transcript counts: streaming %s rows in %s-row batches across %d threads",
        f"{handle.metadata.num_rows:,}",
        f"{batch_rows:,}",
        max_workers,
    )

    # One counter triple per worker thread. Registered under a lock the first time a
    # thread runs, so the main thread can sum them after the pool drains.
    tls = threading.local()
    partials: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    partials_lock = threading.Lock()

    def _fold_one(batch) -> int:
        thread_counters = getattr(tls, "counters", None)
        if thread_counters is None:
            thread_counters = (
                np.zeros(n_rois, dtype=np.int64),
                np.zeros(n_rois, dtype=np.int64),
                np.zeros(n_rois, dtype=np.int64),
            )
            tls.counters = thread_counters
            with partials_lock:
                partials.append(thread_counters)
        frame = batch.to_pandas()
        _accumulate_roi_tx_counts(
            df_grid_roi,
            row_ix,
            frame["x_location"].to_numpy(dtype=np.float64) / ps,
            frame["y_location"].to_numpy(dtype=np.float64) / ps,
            _neg_mask_for_column(frame["feature_name"]),
            thread_counters,
            stride_xy,
            lookup=lookup,
            roi_id_is_arange=roi_id_is_arange,
        )
        return len(frame)

    n_used = 0
    n_batches = 0
    batch_iter = handle.iter_batches(batch_size=batch_rows, columns=columns)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Cap outstanding futures at the worker count so at most ~max_workers batches
        # are resident at once (the same memory envelope as the serial loop).
        inflight: deque = deque()
        for batch in batch_iter:
            inflight.append(pool.submit(_fold_one, batch))
            del batch
            if len(inflight) >= max_workers:
                n_used += inflight.popleft().result()
                n_batches += 1
        while inflight:
            n_used += inflight.popleft().result()
            n_batches += 1

    real_c = np.zeros(n_rois, dtype=np.int64)
    neg_c = np.zeros(n_rois, dtype=np.int64)
    total_c = np.zeros(n_rois, dtype=np.int64)
    for p_real, p_neg, p_total in partials:
        real_c += p_real
        neg_c += p_neg
        total_c += p_total
    logger.info(
        "SNR ROI transcript counts: %s rows in %d batches across %d threads",
        f"{n_used:,}",
        n_batches,
        len(partials),
    )
    return (real_c, neg_c, total_c), n_used


def compute_roi_snr(
    df_grid_roi: pd.DataFrame,
    df_tx: Optional[pd.DataFrame] = None,
    x_col: str = "x_px",
    y_col: str = "y_px",
    roi_grid_stride: Optional[Tuple[int, int]] = None,
    snr_thresholds: Optional[Dict[str, Any]] = None,
    *,
    transcripts_path: Optional[Path] = None,
    pixel_size_um: Optional[float] = None,
    batch_rows: int = ROI_TX_BATCH_ROWS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Per-ROI real vs neg transcript counts, ratio, neg_pct; **mutates** *df_grid_roi* in place
    (caller should pass a copy if the original must stay unchanged — ``run_snr_module`` does).

    Give either *df_tx* -- a frame with feature_name and pixel columns x_col, y_col,
    see ``transcripts_um_to_px`` -- or *transcripts_path* plus *pixel_size_um*, in
    which case the table is read in batches and never held whole. The counters are
    additive, so both give identical results; the streaming form exists because the
    whole-frame one OOM-killed IMAGE_QC on a production sample.

    ``roi_grid_stride`` should match the grid used in ``image_qc.py`` (typically
    ``(roi_size, roi_size)`` when stride defaults to roi_size).
    """
    if (df_tx is None) == (transcripts_path is None):
        raise ValueError("pass exactly one of df_tx or transcripts_path")

    df = df_grid_roi
    if "roi_id" not in df.columns:
        df = df.copy()
        df["roi_id"] = np.arange(len(df), dtype=np.int32)

    n_rois = len(df)
    # Map roi_id → row index (avoids O(max(roi_id)) array if ids are sparse)
    row_ix = pd.Series(np.arange(n_rois, dtype=np.int32), index=df["roi_id"].values)

    # image_qc builds the grid with roi_id == arange(n_rois); then row_ix is the
    # identity and the per-batch reindex can be skipped (see _accumulate_roi_tx_counts).
    roi_ids = df["roi_id"].to_numpy()
    roi_id_is_arange = bool(
        roi_ids.dtype.kind in ("i", "u")
        and np.array_equal(roi_ids, np.arange(n_rois, dtype=roi_ids.dtype))
    )

    if transcripts_path is not None:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required with transcripts_path")
        counters, n_used = _stream_roi_tx_counts(
            df,
            row_ix,
            Path(transcripts_path),
            pixel_size_um,
            roi_grid_stride,
            batch_rows=batch_rows,
            roi_id_is_arange=roi_id_is_arange,
        )
    else:
        counters = (
            np.zeros(n_rois, dtype=np.int64),
            np.zeros(n_rois, dtype=np.int64),
            np.zeros(n_rois, dtype=np.int64),
        )
        _accumulate_roi_tx_counts(
            df,
            row_ix,
            df_tx[x_col].to_numpy(dtype=np.float64),
            df_tx[y_col].to_numpy(dtype=np.float64),
            _neg_mask_for_column(df_tx["feature_name"]),
            counters,
            roi_grid_stride,
            roi_id_is_arange=roi_id_is_arange,
        )
        n_used = int(len(df_tx))
    real_c, neg_c, total_c = counters

    df["snr_real_tx"] = real_c
    df["snr_neg_tx"] = neg_c
    df["snr_total_tx"] = total_c
    df["neg_pct"] = np.where(
        total_c > 0, neg_c.astype(np.float64) / total_c.astype(np.float64), np.nan
    )

    ratio = np.divide(
        real_c.astype(np.float64),
        neg_c.astype(np.float64),
        out=np.full(n_rois, np.nan, dtype=np.float64),
        where=neg_c > 0,
    )
    df["roi_tx_snr_ratio"] = ratio
    df["roi_tx_snr_log"] = np.log10(np.maximum(ratio, 1e-12))

    med_ratio = float(np.nanmedian(ratio[total_c > 0]))
    med_neg_pct = float(np.nanmedian(df.loc[total_c > 0, "neg_pct"]))
    _t = snr_thresholds or {}
    _rt = _t.get("roi_tx") or {}
    ratio_warn = float(_rt.get("ratio_warn", 3.0))
    ratio_fail = float(_rt.get("ratio_fail", 1.5))
    neg_pct_warn = float(_rt.get("neg_pct_warn", 0.15))
    neg_pct_fail = float(_rt.get("neg_pct_fail", 0.30))
    verdict = "PASS"
    if med_ratio < ratio_warn or med_neg_pct > neg_pct_warn:
        verdict = "WARN"
    if med_ratio < ratio_fail or med_neg_pct > neg_pct_fail:
        verdict = "FAIL"

    summary = {
        "status": "ok",
        "method": "roi_tx_target_vs_neg",
        "n_transcripts_used": int(n_used),
        "n_rois_with_tx": int(np.sum(total_c > 0)),
        "median_roi_tx_snr_ratio": med_ratio,
        "median_neg_pct": med_neg_pct,
        "verdict": verdict,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Spatial clustering of neg_pct — Moran (optional) + quadrant fallback
# ---------------------------------------------------------------------------


def compute_neg_spatial_autocorrelation(
    df_grid_roi: pd.DataFrame,
    moran_max_rois: int = 25_000,
    moran_permutations: int = 99,
    moran_subsample_seed: int = 42,
    *,
    include_moran: bool = False,
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    SNR_plan: Moran's I with 4-neighbour weights if libpysal/esda available;
    else quadrant variance proxy on neg_pct.

    Moran + permutations on hundreds of thousands of ROIs is impractical; when
    ``len(df) > moran_max_rois``, a fixed-seed random subsample is used **only**
    for the Moran branch. Quadrant summary still uses all finite-``neg_pct`` ROIs.

    If ``include_moran`` is False, the Moran branch is not run (quadrant summary only;
    avoids PySAL/esda and saves time). If True but packages are missing, Moran is
    skipped inside a try/except and ``moran_note`` is set — the run still succeeds.
    """
    if "neg_pct" not in df_grid_roi.columns:
        return {"status": "skipped", "reason": "neg_pct not computed"}

    df = df_grid_roi[np.isfinite(df_grid_roi["neg_pct"])].copy()
    if len(df) < 8:
        return {"status": "skipped", "reason": "too_few_rois"}

    cx = (df["x1"].astype(np.float64) + df["x2"].astype(np.float64)) / 2.0
    cy = (df["y1"].astype(np.float64) + df["y2"].astype(np.float64)) / 2.0
    z = df["neg_pct"].astype(np.float64).values

    # Quadrant proxy
    mx, my = float(np.median(cx)), float(np.median(cy))
    quad = np.zeros(4, dtype=np.float64)
    nq = np.zeros(4, dtype=np.int64)
    for k, mask in enumerate(
        [
            (cx < mx) & (cy < my),
            (cx >= mx) & (cy < my),
            (cx < mx) & (cy >= my),
            (cx >= mx) & (cy >= my),
        ]
    ):
        quad[k] = float(np.mean(z[mask])) if np.any(mask) else 0.0
        nq[k] = int(np.sum(mask))
    spread = float((np.max(quad) - np.min(quad)) / (np.mean(z) + 1e-9))
    _t = snr_thresholds or {}
    _ns = _t.get("neg_spatial") or {}
    qs_warn = float(_ns.get("quadrant_spread_warn", 0.5))
    qs_fail = float(_ns.get("quadrant_spread_fail", 1.0))
    q_verdict = "PASS"
    if spread > qs_warn:
        q_verdict = "WARN"
    if spread > qs_fail:
        q_verdict = "FAIL"

    out: Dict[str, Any] = {
        "status": "ok",
        "method_primary": "quadrant_spread",
        "quadrant_neg_pct_means": quad.tolist(),
        "quadrant_counts": nq.tolist(),
        "quadrant_spread_index": spread,
        "quadrant_verdict": q_verdict,
    }

    # Optional Moran (subsampled when n is large — see docstring)
    if include_moran:
        try:
            from esda.moran import Moran  # type: ignore
            from libpysal.weights import W  # type: ignore
            from scipy.spatial import cKDTree  # type: ignore

            n_all = len(df)
            if n_all > moran_max_rois:
                rng = np.random.default_rng(moran_subsample_seed)
                pick = rng.choice(n_all, size=moran_max_rois, replace=False)
                cx_m = cx.iloc[pick].to_numpy(dtype=np.float64)
                cy_m = cy.iloc[pick].to_numpy(dtype=np.float64)
                z_m = z[pick]
                out["moran_subsample"] = {
                    "n_used": int(moran_max_rois),
                    "n_available": int(n_all),
                    "seed": int(moran_subsample_seed),
                }
            else:
                cx_m = cx.to_numpy(dtype=np.float64)
                cy_m = cy.to_numpy(dtype=np.float64)
                z_m = z

            xy = np.column_stack([cx_m, cy_m])
            # kNN weights k=4 as SNR_plan neighbour count
            tree = cKDTree(xy)
            _d, idx = tree.query(xy, k=min(5, len(xy)))
            neighbors = {i: [int(j) for j in idx[i][1:]] for i in range(len(xy))}

            w = W(neighbors, silence_warnings=True)
            mi = Moran(z_m, w, permutations=int(moran_permutations))
            moran_i = float(mi.I)
            p_sim = float(mi.p_sim) if mi.p_sim is not None else float("nan")
            m_warn = float(_ns.get("moran_warn", 0.5))
            m_fail = float(_ns.get("moran_fail", 1.0))
            m_verdict = "PASS"
            if moran_i > m_warn:
                m_verdict = "WARN"
            if moran_i > m_fail:
                m_verdict = "FAIL"
            if not math.isnan(p_sim) and p_sim >= 0.05:
                m_verdict = "PASS"
            out["method_secondary"] = "moran_knn4"
            out["moran_i"] = moran_i
            out["moran_p_sim"] = p_sim
            out["moran_verdict"] = m_verdict
        except Exception as e:
            out["moran_note"] = f"skipped ({type(e).__name__}: {e})"
    else:
        out["moran_note"] = "skipped (Moran disabled; quadrant summary only)"

    # Combined: escalate per SNR_plan (clustered noise)
    prim = out.get("moran_verdict", q_verdict)
    if prim == "FAIL" or q_verdict == "FAIL":
        out["verdict"] = "FAIL"
    elif prim == "WARN" or q_verdict == "WARN":
        out["verdict"] = "WARN"
    else:
        out["verdict"] = "PASS"

    if out["verdict"] in ("WARN", "FAIL"):
        out["report_note"] = (
            "Elevated spatial clustering of negative-control probes can reflect "
            "genuine biological heterogeneity (e.g. necrosis, adipose tissue, or "
            "varying cell density) rather than a technical artefact. Consider "
            "reviewing the spatial distribution map before concluding a quality issue."
        )
    return out


# ---------------------------------------------------------------------------
# Slide-level matrix SNR (h5) — Plummer-style vs SpatialQM-style
# ---------------------------------------------------------------------------


def load_expression_matrix_h5(
    path: Path,
) -> Tuple[Any, List[str], List[str], Dict[str, Any]]:
    """
    Load 10x-style Xenium h5 sparse matrix [features x cells].

    10x / Xenium may store **CSR** (``len(indptr) == n_features + 1``) or **CSC**
    (``len(indptr) == n_cells + 1``). Returns a **scipy.sparse.csr_matrix** without
    densifying (full matrix can be tens of GB).
    """
    import h5py

    path = Path(path)
    meta: Dict[str, Any] = {"path": str(path)}
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            g = f["matrix"]
            shape_t = tuple(g["shape"][:]) if "shape" in g else None
            if shape_t is None or len(shape_t) != 2:
                raise ValueError("h5 matrix missing valid shape")
            n_feat, n_cell = int(shape_t[0]), int(shape_t[1])
            data = g["data"][:]
            indices = g["indices"][:]
            indptr = g["indptr"][:]
            from scipy.sparse import csc_matrix, csr_matrix  # type: ignore

            if len(indptr) == n_feat + 1:
                mat = csr_matrix(
                    (data, indices, indptr), shape=shape_t, dtype=np.float64
                )
                meta["sparse_layout"] = "csr"
            elif len(indptr) == n_cell + 1:
                mat = csc_matrix(
                    (data, indices, indptr), shape=shape_t, dtype=np.float64
                ).tocsr()
                meta["sparse_layout"] = "csc_assembled_csr"
            else:
                raise ValueError(
                    f"Unrecognized sparse indptr length {len(indptr)} for shape {shape_t}"
                )
            fg = g["features"]
            if "name" in fg:
                raw = fg["name"][:]
            elif "id" in fg:
                raw = fg["id"][:]
            else:
                raise ValueError("h5 matrix/features has neither 'name' nor 'id'")
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
            if "feature_type" in fg:
                ftype = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in fg["feature_type"][:]
                ]
            else:
                ftype = ["Gene Expression"] * len(names)
        else:
            raise ValueError("Unrecognized h5 layout (expected 'matrix' group)")

    meta.update({"n_features": mat.shape[0], "n_cells": mat.shape[1]})
    return mat, names, ftype, meta


PSEUDOCOUNT = 0.1
EPS = 1e-8

# Fallback defaults for slide-level SNR (used when YAML thresholds not provided).
SLIDE_SNR_THRESHOLDS = {
    "plummer_pass_min": 0.12,
    "plummer_fail_below": 0.10,
}


def _mean_all(x: Any) -> float:
    """Mean across all elements; works for scipy sparse and dense arrays."""
    m = x.mean()
    if hasattr(m, "A1"):
        return float(m.A1[0])
    return float(np.asarray(m).reshape(-1)[0])


def _mean_per_feature(feature_by_cell: Any) -> np.ndarray:
    """Row means (features across cells); works for sparse and dense."""
    out = feature_by_cell.mean(axis=1)
    return np.asarray(out, dtype=np.float64).ravel()


def _build_feature_masks(feature_names: List[str]) -> tuple[np.ndarray, np.ndarray]:
    is_neg = np.array([is_neg_probe_feature(n) for n in feature_names], dtype=bool)
    is_real = ~is_neg
    return is_real, is_neg


def compute_slide_snr_plummer_corrected(
    mat: Any,
    feature_names: List[str],
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Slide SNR: log10(mean_real + 0.1) - log10(mean_neg + 0.1) over matrix elements
    (features × cells), real vs neg probe subsets.
    """
    is_real, is_neg = _build_feature_masks(feature_names)
    if not np.any(is_real) or not np.any(is_neg):
        return {"status": "skipped", "reason": "missing real or neg features"}

    real_mat = mat[is_real]
    neg_mat = mat[is_neg]

    mean_real = _mean_all(real_mat)
    mean_neg = _mean_all(neg_mat)
    log_real = math.log10(mean_real + PSEUDOCOUNT)
    log_neg = math.log10(mean_neg + PSEUDOCOUNT)
    snr = float(log_real - log_neg)

    real_feature_means = _mean_per_feature(real_mat)
    dynamic_range = float(
        math.log10(float(real_feature_means.max()) + PSEUDOCOUNT) - log_neg
    )
    noise_floor_pct = float(mean_neg / (mean_real + EPS) * 100.0)

    _t = snr_thresholds or {}
    _pl = _t.get("slide_plummer") or {}
    pmin = float(_pl.get("pass_min", SLIDE_SNR_THRESHOLDS["plummer_pass_min"]))
    fbelow = float(_pl.get("fail_below", SLIDE_SNR_THRESHOLDS["plummer_fail_below"]))
    if snr < fbelow:
        verdict = "FAIL"
    elif snr < pmin:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "status": "ok",
        "method": "plummer_corrected",
        "formula": "log10(mean_real+0.1)-log10(mean_neg+0.1)",
        "snr": snr,
        "dynamic_range": dynamic_range,
        "mean_real": mean_real,
        "mean_neg": mean_neg,
        "n_real_genes": int(np.sum(is_real)),
        "n_neg_probes": int(np.sum(is_neg)),
        "noise_floor_pct": noise_floor_pct,
        "verdict": verdict,
        "thresholds_used": {
            "pass_min": pmin,
            "fail_below": fbelow,
            "note": "PASS if snr >= pass_min; WARN if fail_below <= snr < pass_min; FAIL if snr < fail_below",
        },
    }


def compute_slide_snr_spatialqm_corrected(
    mat: Any,
    feature_names: List[str],
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Per-gene variant: snr_g = log10(mean_gene_g + 0.1) - log10(mean_neg + 0.1); slide_snr = mean(snr_g).
    """
    is_real, is_neg = _build_feature_masks(feature_names)
    if not np.any(is_real) or not np.any(is_neg):
        return {"status": "skipped", "reason": "missing real or neg features"}

    real_mat = mat[is_real]
    neg_mat = mat[is_neg]
    mean_neg = _mean_all(neg_mat)
    log_neg = math.log10(mean_neg + PSEUDOCOUNT)

    real_feature_means = _mean_per_feature(real_mat)
    per_gene_snr = np.log10(real_feature_means + PSEUDOCOUNT) - log_neg
    snr_mean = float(np.mean(per_gene_snr))
    snr_median = float(np.median(per_gene_snr))
    snr_p10 = float(np.percentile(per_gene_snr, 10))
    pct_genes_above_neg = float(np.mean(per_gene_snr > 0.0) * 100.0)
    dynamic_range = float(
        math.log10(float(real_feature_means.max()) + PSEUDOCOUNT) - log_neg
    )

    _t = snr_thresholds or {}
    _sq = _t.get("slide_spatialqm") or {}
    verdict_metric = _sq.get("metric", "pct_genes_above_neg")
    sq_warn = float(_sq.get("warn", 45))
    sq_fail = float(_sq.get("fail", 35))

    if verdict_metric == "pct_genes_above_neg":
        verdict_value = pct_genes_above_neg
    else:
        # Legacy: snr_mean
        verdict_value = snr_mean
    verdict = (
        "PASS"
        if verdict_value >= sq_warn
        else ("WARN" if verdict_value >= sq_fail else "FAIL")
    )

    return {
        "status": "ok",
        "method": "spatialqm_per_gene_corrected",
        "snr_mean": snr_mean,
        "snr_median": snr_median,
        "snr_p10": snr_p10,
        "pct_genes_above_neg": pct_genes_above_neg,
        "dynamic_range": dynamic_range,
        "mean_neg": mean_neg,
        "n_real_genes": int(np.sum(is_real)),
        "n_neg_probes": int(np.sum(is_neg)),
        "verdict": verdict,
        "thresholds_used": {
            "metric": verdict_metric,
            "warn": sq_warn,
            "fail": sq_fail,
        },
    }


# ---------------------------------------------------------------------------
# aggregate_snr_verdict + run_snr_module
# ---------------------------------------------------------------------------


def save_snr_roi_tx_table(
    df: pd.DataFrame,
    outdir: Path,
    *,
    basename: str = SNR_ROI_TX_TABLE_BASENAME,
) -> Optional[Path]:
    """
    Persist the grid with ``compute_roi_snr`` columns (``snr_real_tx``, ``snr_neg_tx``,
    ``snr_total_tx``, ``neg_pct``, ``roi_tx_snr_ratio``, …) for downstream plots.

    Writes ``{basename}.parquet`` when Parquet is available; otherwise ``{basename}.csv.gz``.
    """
    need = {"snr_real_tx", "snr_neg_tx", "snr_total_tx"}
    if not need.issubset(df.columns):
        return None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path_pq = outdir / f"{basename}.parquet"
    try:
        df.to_parquet(path_pq, index=False)
        logger.info("Wrote %s", path_pq)
        return path_pq
    except Exception as e:
        logger.warning("SNR_roi_tx Parquet write failed (%s), trying CSV.gz", e)
    path_gz = outdir / f"{basename}.csv.gz"
    try:
        df.to_csv(path_gz, index=False, compression="gzip")
        logger.info("Wrote %s", path_gz)
        return path_gz
    except Exception as e2:
        logger.warning("Could not write SNR_roi_tx table: %s", e2)
        return None


def aggregate_snr_verdict(parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Any FAIL → overall FAIL; clustered neg (FAIL) escalates WARN → FAIL."""
    overall = "PASS"
    for sub in parts.values():
        if not isinstance(sub, dict):
            continue
        v = sub.get("verdict")
        if v == "FAIL":
            overall = "FAIL"
        elif v == "WARN" and overall != "FAIL":
            overall = "WARN"
    ns = parts.get(SNR_CKEY_ROI_NEG_SPATIAL) or {}
    if ns.get("verdict") == "FAIL" and overall == "WARN":
        overall = "FAIL"
    return {"overall_snr_verdict": overall}


def run_snr_module(
    xenium_bundle_dir: Optional[Path],
    df_grid_roi: pd.DataFrame,
    outdir: Path,
    focus_maps: Optional[Dict[str, Any]] = None,
    roi_snr_db: Optional[Any] = None,
    intensity_threshold: float = 0.0,
    transcripts_path: Optional[Path] = None,
    cell_matrix_h5: Optional[Path] = None,
    pixel_size_um: Optional[float] = None,
    otsu_max_rois: Optional[int] = None,
    save_roi_tx_table: bool = True,
    write_snr_json: bool = True,
    snr_include_moran: bool = False,
    roi_grid_stride: Optional[Tuple[int, int]] = None,
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run all SNR sub-components; returns grid (with transcript columns when computed) and summary dict.

    Writes ``snr_metrics.json`` under outdir when ``write_snr_json`` is True. When ROI transcript
    SNR succeeds and ``save_roi_tx_table`` is True, also writes ``SNR_roi_tx.parquet`` (or
    ``.csv.gz`` fallback) and sets ``components[SNR_roi_tx]["per_roi_table_file"]`` to the basename.

    If ``snr_include_moran`` is False (default), neg-control spatial uses quadrant spread only (no PySAL Moran).

    ``roi_grid_stride``: optional ``(stride_x, stride_y)`` matching ``image_qc`` grid construction
    (same as ``roi_size`` when stride is unset). Enables O(n_tx) transcript-to-ROI assignment
    without scanning unique x1/y1 on huge grids.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = Path(xenium_bundle_dir) if xenium_bundle_dir else None

    df = df_grid_roi.copy()
    parts: Dict[str, Dict[str, Any]] = {}

    thresholds = snr_thresholds or {}

    # 1) Image SNR (ROI df quartiles → dB)
    parts[SNR_CKEY_IMAGE_ROI_QUARTILE_DB] = compute_image_snr_from_roi_df(
        df, intensity_threshold=intensity_threshold, snr_thresholds=thresholds
    )

    # 2) Image SNR (Otsu)
    if focus_maps or roi_snr_db is not None:
        parts[SNR_CKEY_IMAGE_OTSU] = compute_image_snr_from_pixel_maps(
            focus_maps,
            df,
            max_rois=otsu_max_rois,
            snr_thresholds=thresholds,
            precomputed_db=roi_snr_db,
        )
    else:
        parts[SNR_CKEY_IMAGE_OTSU] = {"status": "skipped", "reason": "no focus_maps"}

    # 3) ROI transcript SNR
    tx_path = transcripts_path
    if tx_path is None and bundle:
        cand = bundle / "transcripts.parquet"
        if cand.exists():
            tx_path = cand
        else:
            cg = bundle / "transcripts.csv.gz"
            if cg.exists():
                tx_path = cg
    if tx_path and tx_path.exists():
        try:
            if pixel_size_um is None:
                parts[SNR_CKEY_ROI_TX] = {
                    "status": "skipped",
                    "reason": "pixel_size_um required to convert transcript coordinates to pixels",
                }
            else:
                # Streamed: see _stream_roi_tx_counts. Reading the table whole and
                # then copying it in transcripts_um_to_px OOM-killed this step.
                df, rtx = compute_roi_snr(
                    df,
                    transcripts_path=Path(tx_path),
                    pixel_size_um=pixel_size_um,
                    roi_grid_stride=roi_grid_stride,
                    snr_thresholds=thresholds,
                )
                if save_roi_tx_table and rtx.get("status") == "ok":
                    pth = save_snr_roi_tx_table(df, outdir)
                    if pth is not None:
                        rtx["per_roi_table_file"] = pth.name
                parts[SNR_CKEY_ROI_TX] = rtx
        except Exception as e:
            logger.exception("ROI transcript SNR failed")
            parts[SNR_CKEY_ROI_TX] = {"status": "error", "error": str(e)}
    else:
        parts[SNR_CKEY_ROI_TX] = {"status": "skipped", "reason": "no transcripts file"}

    # 4) Slide SNR
    h5_path = cell_matrix_h5
    if h5_path is None and bundle:
        hp = bundle / "cell_feature_matrix.h5"
        if hp.exists():
            h5_path = hp
    if h5_path and Path(h5_path).exists():
        try:
            mat, names, _ftype, _meta = load_expression_matrix_h5(Path(h5_path))
            parts[SNR_CKEY_SLIDE_PLUMMER] = compute_slide_snr_plummer_corrected(
                mat, names, snr_thresholds=thresholds
            )
            parts[SNR_CKEY_SLIDE_SPATIALQM] = compute_slide_snr_spatialqm_corrected(
                mat, names, snr_thresholds=thresholds
            )
        except Exception as e:
            logger.exception("Slide SNR failed")
            parts[SNR_CKEY_SLIDE_PLUMMER] = {"status": "error", "error": str(e)}
            parts[SNR_CKEY_SLIDE_SPATIALQM] = {"status": "error", "error": str(e)}
    else:
        parts[SNR_CKEY_SLIDE_PLUMMER] = {
            "status": "skipped",
            "reason": "no cell_feature_matrix.h5",
        }
        parts[SNR_CKEY_SLIDE_SPATIALQM] = {
            "status": "skipped",
            "reason": "no cell_feature_matrix.h5",
        }

    # 5) Neg spatial autocorrelation (needs neg_pct)
    if "neg_pct" in df.columns:
        parts[SNR_CKEY_ROI_NEG_SPATIAL] = compute_neg_spatial_autocorrelation(
            df, include_moran=snr_include_moran, snr_thresholds=thresholds
        )
    else:
        parts[SNR_CKEY_ROI_NEG_SPATIAL] = {
            "status": "skipped",
            "reason": "neg_pct not available",
        }

    verdict = aggregate_snr_verdict(parts)
    summary = {
        "components": parts,
        "verdict": verdict,
    }

    if write_snr_json:
        try:
            out_json = outdir / "snr_metrics.json"
            with open(out_json, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Wrote %s", out_json)
        except Exception as e:
            logger.warning("Could not write snr_metrics.json: %s", e)

    return df, summary


def read_xenium_pixel_size_um(bundle_dir: Path) -> Optional[float]:
    """Read ``pixel_size`` (or ``pixel_size_um``) from ``experiment.xenium``."""
    exp = Path(bundle_dir) / "experiment.xenium"
    if not exp.is_file():
        return None
    try:
        with open(exp, encoding="utf-8") as f:
            meta = json.load(f)
        ps = float(meta.get("pixel_size", meta.get("pixel_size_um", 0.0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if ps <= 0 or not math.isfinite(ps):
        return None
    return ps


def compute_snr_summary(
    df_grid_roi: pd.DataFrame,
    *,
    bundle_dir: Path,
    outdir: Path,
    focus_maps: Optional[Dict[str, Any]] = None,
    roi_snr_db: Optional[Any] = None,
    pixel_size_um: Optional[float] = None,
    intensity_threshold: float = 0.0,
    otsu_max_rois: Optional[int] = None,
    save_roi_tx_table: bool = True,
    write_snr_json: bool = True,
    snr_include_moran: bool = False,
    roi_grid_stride: Optional[Tuple[int, int]] = None,
    snr_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """High-level API for :mod:`image_qc` — same as :func:`run_snr_module` with keyword-only opts."""
    return run_snr_module(
        bundle_dir,
        df_grid_roi,
        outdir,
        focus_maps=focus_maps,
        roi_snr_db=roi_snr_db,
        intensity_threshold=intensity_threshold,
        pixel_size_um=pixel_size_um,
        otsu_max_rois=otsu_max_rois,
        save_roi_tx_table=save_roi_tx_table,
        write_snr_json=write_snr_json,
        snr_include_moran=snr_include_moran,
        roi_grid_stride=roi_grid_stride,
        snr_thresholds=snr_thresholds,
    )


__all__ = [
    "SNR_CKEY_IMAGE_OTSU",
    "SNR_CKEY_IMAGE_ROI_QUARTILE_DB",
    "SNR_CKEY_ROI_NEG_SPATIAL",
    "SNR_CKEY_ROI_TX",
    "SNR_CKEY_SLIDE_PLUMMER",
    "SNR_CKEY_SLIDE_SPATIALQM",
    "SNR_ROI_TX_TABLE_BASENAME",
    "aggregate_snr_verdict",
    "compute_image_snr_from_pixel_maps",
    "compute_image_snr_from_roi_df",
    "compute_neg_spatial_autocorrelation",
    "compute_roi_snr",
    "compute_snr_summary",
    "compute_slide_snr_plummer_corrected",
    "compute_slide_snr_spatialqm_corrected",
    "is_neg_probe_feature",
    "load_expression_matrix_h5",
    "load_transcripts",
    "read_xenium_pixel_size_um",
    "run_snr_module",
    "save_snr_roi_tx_table",
    "snr_verdict_to_quality_status",
    "transcripts_um_to_px",
]
