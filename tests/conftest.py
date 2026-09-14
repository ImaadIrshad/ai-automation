"""Test-suite isolation from a developer's local `.env`.

Force safe, light defaults *before* the app is imported, so the suite is
deterministic no matter what `.env` contains: the `echo` model (no torch/index
load at import) and no real API key. Individual tests override as needed.
"""

import os

os.environ["CRS_APPROACH"] = "echo"
