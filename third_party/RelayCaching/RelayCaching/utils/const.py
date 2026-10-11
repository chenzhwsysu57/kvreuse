"""Project root path constant."""

import os
from pathlib import Path

RELAYCACHING_ROOT = Path(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))
