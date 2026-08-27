"""Worker helpers for the cross-process plane-sharing test.

Lives in its own module because ``multiprocessing`` with the ``spawn`` start
method re-imports the function's module in the child, and a pytest test module
is a fragile thing to re-import.  This mirrors the production arrangement in
``image_qc._tile_worker_init`` / ``_tile_worker_run``: the child receives a
*path*, not an array, and maps it itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_STATE: dict[str, Any] = {}


def init(plane_path: str, shape: tuple[int, int], dtype_str: str) -> None:
    """Map the shared plane file into this worker process."""
    _STATE["plane"] = np.memmap(
        plane_path, dtype=np.dtype(dtype_str), mode="r+", shape=tuple(shape)
    )


def write_tile(task: tuple[dict[str, int], float]) -> float:
    """Write *value* into this tile's disjoint write region and flush."""
    spec, value = task
    plane = _STATE["plane"]
    plane[spec["write_y0"] : spec["write_y1"], spec["write_x0"] : spec["write_x1"]] = (
        value
    )
    plane.flush()
    return value
