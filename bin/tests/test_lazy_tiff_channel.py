"""Regression test for `_LazyTiffChannel` instances built via `__new__`.

`_open_morphology_lazy` has a branch for TIFFs that store every channel in a single
3-D page. It builds the channel objects with `_LazyTiffChannel.__new__`, which
creates the instance while skipping `__init__`, then sets five attributes by hand:
`_page`, `shape`, `_lock`, `_source`, `_cached_data`.

`_is_tiled` was not among them, and it was only ever assigned inside `__init__`
while `__getitem__` reads it unguarded on every region request. So the first tile
read raised `AttributeError: '_LazyTiffChannel' object has no attribute
'_is_tiled'`, inside a thread pool, killing the step. The pre-cached data does not
save it: the cache is consumed in `_read_region_fallback`, which is only reached
after the `_is_tiled` check.

The fix is a class-level default rather than three more hand-assignments, so a
future `__new__` site cannot reintroduce the bug. These tests assert the object is
usable, not merely that the attribute exists, and cover the profiling counters the
same branch also omits (those are getattr-guarded, so they were never fatal).
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np

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

image_qc = importlib.import_module("image_qc")
_LazyTiffChannel = image_qc._LazyTiffChannel


class _FakePage:
    """Minimal stand-in for a tifffile TiffPage/TiffFrame."""

    def __init__(self, shape, dtype=np.uint16):
        self.shape = shape
        self.dtype = np.dtype(dtype)


def _channel_via_new(data: np.ndarray) -> object:
    """Reproduce exactly what `_open_morphology_lazy`'s 3-D-page branch builds."""
    import threading

    ch = _LazyTiffChannel.__new__(_LazyTiffChannel)
    ch._page = _FakePage(data.shape)
    ch.shape = (data.shape[0], data.shape[1])
    ch._lock = threading.Lock()
    ch._source = None
    ch._cached_data = data
    return ch


def test_is_tiled_is_readable_without_init():
    """The attribute must exist on an instance that never ran __init__."""
    ch = _LazyTiffChannel.__new__(_LazyTiffChannel)
    assert ch._is_tiled is False


def test_is_tiled_is_a_class_level_default():
    """A class attribute, so any future __new__ site inherits it."""
    assert _LazyTiffChannel._is_tiled is False


def test_slicing_a_new_built_channel_returns_the_cached_region():
    """The real failure: the first region read raised AttributeError."""
    data = np.arange(64 * 48, dtype=np.uint16).reshape(64, 48)
    ch = _channel_via_new(data)

    got = ch[10:20, 5:15]

    assert got.shape == (10, 10)
    np.testing.assert_array_equal(got, data[10:20, 5:15])


def test_full_slice_of_a_new_built_channel():
    data = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    ch = _channel_via_new(data)
    np.testing.assert_array_equal(ch[:, :], data)


def test_empty_slice_is_empty_not_an_error():
    """Degenerate request: zero-area slices short-circuit before the tiled check."""
    data = np.zeros((16, 16), dtype=np.uint16)
    ch = _channel_via_new(data)
    assert ch[5:5, 0:10].shape == (0, 0)
    assert ch[0:10, 8:4].shape == (0, 0)


def test_init_still_takes_is_tiled_from_the_page():
    """The class default must not shadow a real tiled page."""

    class _TiledPage(_FakePage):
        is_tiled = True

    ch = _LazyTiffChannel(_TiledPage((32, 32)))
    assert ch._is_tiled is True

    ch_plain = _LazyTiffChannel(_FakePage((32, 32)))
    assert ch_plain._is_tiled is False, "a page without is_tiled means non-tiled"


def test_profiling_counters_are_safe_on_a_new_built_channel():
    """The same branch omits these too; they are getattr-guarded, so non-fatal."""
    ch = _channel_via_new(np.zeros((8, 8), dtype=np.uint16))
    for attr in ("_read_seconds", "_decode_seconds", "_read_bytes"):
        assert getattr(ch, attr, 0.0) is not None
