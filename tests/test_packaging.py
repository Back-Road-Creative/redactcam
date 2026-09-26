"""Packaging contract: which wheel provides ``onnxruntime`` is the installer's
choice, never a default that a later install can silently overturn.

Background (2026-09-25): a pipeline venv had ``onnxruntime-gpu`` installed; a
later ``pip install -e`` of a project that listed plain ``onnxruntime`` (as this
package's base dependencies then did) pulled the CPU wheel over it. Both wheels
own the ``onnxruntime/`` package directory, last install wins, and the blur
detectors ran on the CPU for hours with nothing in the log saying so. A base
dependency on either wheel makes that repeatable; an extra per wheel does not.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import redactcam

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]


def _names(reqs: list[str]) -> list[str]:
    return [r.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for r in reqs]


class TestOnnxruntimeIsAnExtra:
    def test_base_dependencies_carry_neither_wheel(self):
        assert not {"onnxruntime", "onnxruntime-gpu"} & set(_names(_project()["dependencies"]))

    def test_cpu_extra_is_the_cpu_wheel(self):
        assert _names(_project()["optional-dependencies"]["cpu"]) == ["onnxruntime"]

    def test_gpu_extra_is_the_cuda_wheel(self):
        assert _names(_project()["optional-dependencies"]["gpu"]) == ["onnxruntime-gpu"]

    def test_dev_extra_brings_a_runtime_so_the_suite_can_run(self):
        # CI installs ``.[dev]`` on CPU runners; without a runtime nothing imports.
        assert "onnxruntime" in _names(_project()["optional-dependencies"]["dev"])


def test_version_is_declared_once():
    """``__version__`` and pyproject drifted (0.1.0 vs 0.1.1) with nothing to catch it."""
    assert redactcam.__version__ == _project()["version"]
