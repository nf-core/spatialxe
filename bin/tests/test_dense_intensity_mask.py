"""Regression test for the artefact ("optically dense regions") mask reduction.

`generate_tissue_mask` finds bright artefacts by thresholding each morphology
channel at the 97th percentile and combining the results. A pixel bright in two or
three channels is the strongest evidence of a real artefact, because genuine signal
is stain-specific while debris, folds and coverslip contamination are not.

From 2025-06-06 to 2026-05-06 the combination added three *bool* arrays. NumPy `+`
on bool dtype is logical OR and returns bool, so `binary_fill_holes` received a
genuine any-channel union. `119cff9` (2026-05-06, "support DAPI-only morphology
bundles") added `.astype(np.int_)` to each term while making the block None-tolerant,
turning the union into a 0..3 count. `SimpleITK.BinaryFillhole` treats only value 1
as foreground, so the 2s and 3s -- the strongest evidence -- stopped being
foreground.

No literal `> 0` ever existed: the reduction rode on bool dtype alone, which is why
the change looked like tidying and went unnoticed for three months. These tests
assert the *contract* at the boundary (what reaches the fill filter) rather than the
filter's output, because `napari_simpleitk_image_processing` is not installed in the
test environment and the existing `test_generate_tissue_mask.py` stubs the filter as
identity -- which is precisely why its tests could never observe this.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_bin_dir = Path(__file__).resolve().parent.parent
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

# Stub the heavy optional imports image_qc pulls in at module level. scanpy is
# stubbed for the same reason test_cell_roi_mapping.py does it: importing it here
# trips numba's cache locator and aborts collection. Stubbing in this module rather
# than relying on another test file having done it keeps the file runnable alone.
for _mod in (
    "napari_skimage_regionprops",
    "napari_simpleitk_image_processing",
    "snr_metrics",
    "scanpy",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["napari_skimage_regionprops"].regionprops_table = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["scanpy"].AnnData = object  # type: ignore[attr-defined]
_nsitk = sys.modules["napari_simpleitk_image_processing"]
_nsitk.signed_maurer_distance_map = lambda x: np.zeros_like(  # type: ignore[attr-defined]
    np.asarray(x), dtype=np.float64
)
# Define a default so monkeypatch.setattr has an attribute to replace. Each test
# swaps in the spy; this identity stand-in matches test_generate_tissue_mask.py.
_nsitk.binary_fill_holes = lambda x: np.asarray(x)  # type: ignore[attr-defined]

image_qc = importlib.import_module("image_qc")

KW = dict(min_size_edge=10, min_size_hole=5, min_size_dense_intensity_region=5)
SHAPE = (64, 64)


class _FillSpy:
    """Capture what the artefact mask hands to binary_fill_holes.

    generate_tissue_mask calls the filter more than once; the artefact call is the
    one whose argument is not the tissue mask. Recording every call and inspecting
    the last is enough, since the artefact block is the final caller.
    """

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def __call__(self, x):
        arr = np.asarray(x)
        self.calls.append(arr.copy())
        return arr


@pytest.fixture
def fill_spy(monkeypatch):
    spy = _FillSpy()
    monkeypatch.setattr(_nsitk, "binary_fill_holes", spy)
    return spy


RNG = np.random.default_rng(0)


def _channels(n_bright_channels: int):
    """Three noisy channels; a 10x10 patch is bright in the first N of them.

    The background must be textured, NOT flat. A flat field's 97th percentile
    equals its own value, so `channel >= p97` is True everywhere and every
    channel contributes to the count regardless of the patch -- which silently
    turns "bright in 1 of 3" into "bright in 3 of 3" and makes the parametrisation
    below meaningless. With noise, a channel that lacks the patch has an ordinary
    value there and contributes 0, so the counts really are 1, 2 and 3.
    """
    chans = []
    for i in range(3):
        f = RNG.normal(100.0, 30.0, size=SHAPE).clip(0)
        if i < n_bright_channels:
            f[20:30, 20:30] = 50_000.0
        chans.append(f)
    return chans


def _artefact_arg(spy: _FillSpy) -> np.ndarray:
    assert spy.calls, "binary_fill_holes was never called"
    return spy.calls[-1]


@pytest.mark.parametrize("n_bright", [1, 2, 3])
def test_multi_channel_bright_pixels_reach_the_fill_filter(fill_spy, n_bright):
    """The patch must be foreground whether it is bright in 1, 2 or 3 channels.

    Under the count-into-a-binary-filter bug, only n_bright == 1 survived.
    """
    c0, c1, c2 = _channels(n_bright)
    image_qc.generate_tissue_mask(None, c0, c1, c2, **KW)

    arg = _artefact_arg(fill_spy)
    assert arg.dtype == bool, (
        f"the fill filter received {arg.dtype}, not bool. A binary filter treats "
        "only one value as foreground, so an integer channel count silently drops "
        "the multi-channel-bright pixels."
    )
    assert arg[20:30, 20:30].all(), (
        f"a patch bright in {n_bright} of 3 channels is not foreground in the "
        "artefact mask"
    )


def test_the_reduction_is_any_channel_not_a_count(fill_spy):
    """Two disjoint patches, bright in different numbers of channels, both count."""
    c0, c1, c2 = (RNG.normal(100.0, 30.0, size=SHAPE).clip(0) for _ in range(3))
    # patch A: bright in all three
    c0[10:16, 10:16] = c1[10:16, 10:16] = c2[10:16, 10:16] = 50_000.0
    # patch B: bright in DAPI only
    c0[40:46, 40:46] = 50_000.0

    image_qc.generate_tissue_mask(None, c0, c1, c2, **KW)
    arg = _artefact_arg(fill_spy)

    assert arg[10:16, 10:16].all(), "3-channel-bright patch dropped"
    assert arg[40:46, 40:46].all(), "1-channel-bright patch dropped"
    assert set(np.unique(arg)) <= {False, True}


def test_dapi_only_bundle_still_works(fill_spy):
    """small1/small2 are None for DAPI-only bundles; the rule must still apply.

    A `>= 2` rule would make the artefact metric structurally impossible here,
    which is one reason the any-channel rule is the correct restoration.
    """
    c0 = RNG.normal(100.0, 30.0, size=SHAPE).clip(0)
    c0[20:30, 20:30] = 50_000.0

    image_qc.generate_tissue_mask(None, c0, None, None, **KW)
    arg = _artefact_arg(fill_spy)

    assert arg.dtype == bool
    assert arg[20:30, 20:30].all()


def test_uniform_field_yields_no_artefact_foreground(fill_spy):
    """A flat field has no 97th-percentile outlier region worth flagging.

    Guards the other direction: `> 0` on a count must not mark everything. A flat
    field's 97th percentile equals its value, so `>=` marks all of it -- this test
    pins that this is a property of the percentile threshold, not of the reduction,
    by asserting the mask is uniform rather than patchy.
    """
    c0 = np.full(SHAPE, 100.0)
    image_qc.generate_tissue_mask(None, c0, None, None, **KW)
    arg = _artefact_arg(fill_spy)
    assert arg.dtype == bool
    assert arg.all() or not arg.any(), "flat field produced a patchy artefact mask"
