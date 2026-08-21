"""Import paths for the pipeline scripts under test.

Ported from nf-xenium-processing `tests/conftest.py` (dev HEAD 5e35cae).

The pipeline's Python lives in the pipeline-level `bin/` directory (staged into
the container by Nextflow), not in an installed package, so tests import it by
path. Each test module used to insert those paths itself, which made collection
order load-bearing: a module that inserted only a partial path list imported
`image_qc` successfully *only* when some other module had already inserted the
real directory. Run alone -- or given its own pytest-xdist worker -- it failed.
conftest.py is imported before any test module, so putting the paths here makes
every module importable in isolation.

ADAPTED FROM UPSTREAM: upstream keeps each script in
`modules/local/<tool>/resources/usr/bin/` plus a `bin/xenium_helpers/src`
package, so `_MODULE_BIN_DIRS` listed four directories. This pipeline ships all
QC scripts in the single pipeline-level `bin/` and does not ship
`xenium_helpers` at all (its helpers are inlined into the scripts), so the list
collapses to one entry.

Stubs for heavy optional imports stay in the individual test modules, since they
differ between modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives in `bin/tests/`, so parent.parent is the pipeline `bin/`.
_bin = Path(__file__).resolve().parent.parent

_MODULE_BIN_DIRS = [
    _bin,
]

for _path in _MODULE_BIN_DIRS:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
