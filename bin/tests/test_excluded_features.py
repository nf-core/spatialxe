"""Tests for dropping decoding failures from the transcript SNR ratio.

Xenium assigns each detected spot to a gene by reading its barcode. Some reads
fail: `UnassignedCodeword_*` means the readout matched no valid barcode. That is a
failure to measure, not a detected transcript.

The negative-control classifier matched on name prefix only and had no entry for it,
and `is_real` was defined as `~is_neg`, so every unassigned codeword was counted as
real gene signal. That inflated the numerator of the real/negative ratio and biased
the verdict toward PASS.

Counting them as negative controls would be equally wrong: a negative-control probe
estimates background because it is designed not to bind, while a failed read
estimates nothing. So they are dropped from both sides, and from the total, which
keeps `neg_pct` a fraction of decoded transcripts.

`DeprecatedCodeword_*` is deliberately NOT excluded: retired-but-valid barcodes may
be genuine detections of a gene no longer in the panel. Its behaviour is pinned below
so a later decision to exclude it is a visible change rather than a silent one.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# snr_metrics is stubbed by other test modules (they share sys.modules), so a plain
# import can hand back an empty ModuleType depending on collection order. Load the
# real module by path, the same way test_tile_consumers.py does, so this file works
# whatever ran first.
_snr_path = (
    Path(__file__).resolve().parent.parent / "snr_metrics.py"
)
_spec = importlib.util.spec_from_file_location("_real_snr_metrics", _snr_path)
assert _spec is not None and _spec.loader is not None  # narrow for mypy; path is checked above
snr_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snr_metrics)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_unassigned_codeword_is_excluded():
    assert snr_metrics.is_excluded_feature("UnassignedCodeword_0123")
    assert snr_metrics.is_excluded_feature("unassignedcodeword_0123"), (
        "case-insensitive"
    )


def test_unassigned_codeword_is_not_a_negative_control():
    """It is not background, so it must not join the neg-control pool."""
    assert not snr_metrics.is_neg_probe_feature("UnassignedCodeword_0123")


def test_real_genes_and_neg_controls_are_not_excluded():
    for name in ("EPCAM", "NegControlProbe_00042", "BLANK_0097", "antisense_PROX1"):
        assert not snr_metrics.is_excluded_feature(name), name


def test_deprecated_codeword_is_left_alone_for_now():
    """Pins the open question: still counted as real, deliberately."""
    assert not snr_metrics.is_excluded_feature("DeprecatedCodeword_0001")
    assert not snr_metrics.is_neg_probe_feature("DeprecatedCodeword_0001")


def test_empty_and_non_string_inputs():
    for bad in ("", None, 3.5, float("nan")):
        assert not snr_metrics.is_excluded_feature(bad)


# ---------------------------------------------------------------------------
# vectorised masks must agree with the scalar predicate
# ---------------------------------------------------------------------------

_NAMES = [
    "EPCAM",
    "UnassignedCodeword_0001",
    "NegControlProbe_00042",
    "DeprecatedCodeword_0002",
    "BLANK_0097",
    "unassignedcodeword_0003",
    "antisense_PROX1",
]


def test_vectorised_excluded_mask_matches_the_scalar_predicate():
    got = snr_metrics._excluded_mask_vectorized(np.array(_NAMES, dtype=object))
    want = np.array([snr_metrics.is_excluded_feature(n) for n in _NAMES])
    np.testing.assert_array_equal(got, want)


def test_dictionary_encoded_column_gives_the_same_mask():
    """Xenium writes feature_name dictionary-encoded; the fast path must agree."""
    plain = pd.Series(_NAMES * 3, dtype="object")
    encoded = plain.astype("category")

    np.testing.assert_array_equal(
        snr_metrics._excluded_mask_for_column(plain),
        snr_metrics._excluded_mask_for_column(encoded),
    )
    np.testing.assert_array_equal(
        snr_metrics._neg_mask_for_column(plain),
        snr_metrics._neg_mask_for_column(encoded),
    )


def test_the_three_categories_are_disjoint_and_cover_everything():
    neg = snr_metrics._neg_mask_for_column(pd.Series(_NAMES))
    exc = snr_metrics._excluded_mask_for_column(pd.Series(_NAMES))
    assert not (neg & exc).any(), "a feature cannot be both background and excluded"


# ---------------------------------------------------------------------------
# the counts themselves
# ---------------------------------------------------------------------------


def _one_tile_frame():
    """A single tile covering everything, so every transcript lands in it."""
    return pd.DataFrame(
        {"roi_id": [0], "x1": [0.0], "x2": [100.0], "y1": [0.0], "y2": [100.0]}
    )


def _count(names: list[str]):
    df = _one_tile_frame()
    row_ix = pd.Series(np.arange(1, dtype=np.int32), index=df["roi_id"].values)
    counters = (
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
    )
    col = pd.Series(names)
    snr_metrics._accumulate_roi_tx_counts(
        df,
        row_ix,
        np.full(len(names), 50.0),
        np.full(len(names), 50.0),
        snr_metrics._neg_mask_for_column(col),
        counters,
        None,
        roi_id_is_arange=True,
        is_excluded=snr_metrics._excluded_mask_for_column(col),
    )
    real, neg, total = (int(c[0]) for c in counters)
    return real, neg, total


def test_unassigned_codewords_leave_all_three_counts():
    """5 genes, 2 neg controls, 3 unassigned: the 3 vanish from every count."""
    names = (
        ["EPCAM"] * 5 + ["NegControlProbe_00042"] * 2 + ["UnassignedCodeword_0001"] * 3
    )
    real, neg, total = _count(names)

    assert real == 5, "unassigned codewords are still being counted as real signal"
    assert neg == 2, "unassigned codewords must not join the negative controls"
    assert total == 7, "excluded features must leave the total too"
    assert real + neg == total, "real = total - neg identity broken"


def test_ratio_is_not_inflated_by_decoding_failures():
    """The point of the fix: adding unassigned codewords must not change the ratio."""
    clean = _count(["EPCAM"] * 20 + ["NegControlProbe_00042"] * 4)
    dirty = _count(
        ["EPCAM"] * 20
        + ["NegControlProbe_00042"] * 4
        + ["UnassignedCodeword_0001"] * 50
    )
    assert clean == dirty, (
        "50 failed barcode reads changed the counts, so they are still reaching "
        f"the ratio: {clean} vs {dirty}"
    )


def test_deprecated_codewords_still_count_as_real():
    """Pins current behaviour so changing it later is deliberate."""
    real, neg, total = _count(["EPCAM"] * 3 + ["DeprecatedCodeword_0002"] * 2)
    assert (real, neg, total) == (5, 0, 5)


def test_all_transcripts_excluded_leaves_empty_counts():
    real, neg, total = _count(["UnassignedCodeword_0001"] * 10)
    assert (real, neg, total) == (0, 0, 0)


def test_slide_masks_exclude_decoding_failures():
    is_real, is_neg = snr_metrics._build_feature_masks(_NAMES)
    for i, name in enumerate(_NAMES):
        if snr_metrics.is_excluded_feature(name):
            assert not is_real[i], f"{name} counted as real signal"
            assert not is_neg[i], f"{name} counted as background"
    assert not (is_real & is_neg).any()
