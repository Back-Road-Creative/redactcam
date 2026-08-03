"""PyInstaller entry point for the Windows build.

PyInstaller freezes a *script*. redactcam's console entry point is a module
function (``redactcam = "redactcam.cli:main"`` in pyproject.toml), which pip
turns into a generated shim at install time — there is no file on disk to point
PyInstaller at. This is that file, and it does nothing else.
"""

from __future__ import annotations

import sys

from redactcam.cli import main

if __name__ == "__main__":
    sys.exit(main())
