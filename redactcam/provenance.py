"""Processing identity — which code, which models, which native runtime.

A version string is not an identity: ``0.2.0`` covered the tracker before and
after the one-detection-one-track fix, so two builds that blurred differently
carried the same number. A consumer that pins a git commit cannot tell from the
number whether what it imported is that commit either. This module answers with
things that cannot drift apart from the bytes that ran:

* ``code_sha256`` — a digest of the package's own ``.py`` sources.
* ``installed_revision`` — the commit a VCS install recorded (PEP 610), else the
  git HEAD of a source checkout, else unknown.
* ``model_identity`` — the SHA256 of every model file a run actually resolved.
* ``runtime_identity`` — Python, OpenCV, numpy, the onnxruntime *distribution*
  (CPU and CUDA wheels share one import name), and the ffmpeg build.

``check_current`` is the stale-install gate a consumer runs before calling the
package current. ``redact_video`` writes the same identity into a render receipt
beside every output.

Everything here is bounded: fixed keys, no free-form logs, no URLs (a VCS install
URL can carry credentials, so only the commit is ever read from it).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .apply import tool_status

MIN_REVISION_CHARS = 7


class StaleInstallError(RuntimeError):
    """The installed package is not the one the caller expected."""


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(package_dir: Path | None = None) -> tuple[str | None, int]:
    """``(digest, file_count)`` over every ``.py`` under the package directory.

    Path and content both feed the digest, files are taken in sorted order, and
    CRLF is folded to LF so a Windows autocrlf checkout hashes like the wheel.
    ``(None, 0)`` when there are no sources on disk (a frozen build): an empty
    digest would look like an identity and match every other frozen build.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:
        return None, 0
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest(), len(files)


def installed_revision(source_root: Path | None = None) -> dict:
    """The commit this code came from: ``{"commit": str | None, "source": ...}``.

    ``source`` is ``"direct_url"`` (a ``pip install git+…`` recorded it),
    ``"git"`` (a checkout whose HEAD was read) or ``None`` (a PyPI-style wheel or
    a frozen build records nothing, and this does not guess).
    """
    try:
        text = importlib.metadata.distribution("redactcam").read_text("direct_url.json")
        commit = (json.loads(text).get("vcs_info") or {}).get("commit_id") if text else None
        if commit:
            return {"commit": str(commit), "source": "direct_url"}
    except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError):
        pass
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent.parent
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            head = out.stdout.strip()
            if out.returncode == 0 and len(head) >= 40:
                return {"commit": head, "source": "git"}
        except (OSError, subprocess.SubprocessError):
            pass
    return {"commit": None, "source": None}


def model_identity(paths: Mapping[str, Path | str | None]) -> dict[str, dict | None]:
    """``{kind: {"path", "sha256", "size_bytes"} | None}`` for the models a run
    resolved. ``None`` records a kind that did not resolve, and a ``None`` hash a
    file that could not be read — the receipt says so rather than dropping it. A file shared by two kinds (COCO serves both vehicle
    and person) is hashed once."""
    by_path: dict[str, dict] = {}
    out: dict[str, dict | None] = {}
    for kind, raw in paths.items():
        if not raw:
            out[kind] = None
            continue
        p = Path(raw)
        key = str(p)
        if key not in by_path:
            try:
                by_path[key] = {
                    "path": key,
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            except OSError:
                # An unreadable file is recorded as an unknown hash, never omitted
                # and never faked: the receipt must not claim what it could not see.
                by_path[key] = {"path": key, "sha256": None, "size_bytes": None}
        out[kind] = dict(by_path[key])
    return out


def _dist_version(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        try:
            return name, importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def runtime_identity() -> dict:
    """Bounded description of what is running: this package, and its native stack."""
    import cv2
    import numpy

    import redactcam

    digest, count = code_sha256()
    ort_dist, ort_version = _dist_version("onnxruntime-gpu", "onnxruntime")
    providers: list[str] | None = None
    try:
        import onnxruntime

        providers = list(onnxruntime.get_available_providers())
    except ImportError:
        pass
    return {
        "redactcam": {
            "version": redactcam.__version__,
            "code_sha256": digest,
            "code_files": count,
            "revision": installed_revision(),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "native": {
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "onnxruntime_distribution": ort_dist,
            "onnxruntime": ort_version,
            "onnxruntime_providers": providers,
            "ffmpeg": tool_status("ffmpeg")["version"],
            "ffprobe": tool_status("ffprobe")["version"],
        },
    }


def check_current(
    *,
    expected_version: str | None = None,
    expected_code_sha256: str | None = None,
    expected_revision: str | None = None,
) -> dict:
    """Return the installed identity if it matches every expectation given, else
    raise ``StaleInstallError`` naming each mismatch.

    ``expected_revision`` is a commit id or a prefix of at least 7 characters.
    An installed revision that cannot be determined never passes: an unknown is
    not a match, and treating it as one is how a stale install gets called
    current.
    """
    if expected_revision is not None and len(expected_revision) < MIN_REVISION_CHARS:
        raise ValueError(f"expected_revision needs at least {MIN_REVISION_CHARS} characters")
    if not (expected_version or expected_code_sha256 or expected_revision):
        raise ValueError("check_current needs at least one expectation to compare against")

    ident = runtime_identity()
    me = ident["redactcam"]
    problems: list[str] = []
    if expected_version and me["version"] != expected_version:
        problems.append(f"version is {me['version']}, expected {expected_version}")
    if expected_code_sha256:
        if me["code_sha256"] is None:
            problems.append("code_sha256 cannot be determined (no package sources on disk)")
        elif me["code_sha256"] != expected_code_sha256:
            problems.append(
                f"code_sha256 is {me['code_sha256'][:12]}…, expected {expected_code_sha256[:12]}…"
            )
    if expected_revision:
        commit = me["revision"]["commit"]
        if commit is None:
            problems.append(
                "installed revision cannot be determined (not a VCS install or a git checkout)"
            )
        elif not commit.startswith(expected_revision.lower()):
            problems.append(f"installed revision is {commit[:12]}, expected {expected_revision}")
    if problems:
        raise StaleInstallError(
            "installed redactcam is not the expected build: " + "; ".join(problems)
        )
    return ident


def write_receipt(path: str | Path, receipt: dict) -> Path:
    """Write ``receipt`` as JSON via a temp file and rename, so a crash never
    leaves a half-written receipt that looks like a real one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return path
