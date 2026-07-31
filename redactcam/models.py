"""Download-on-demand cache for the detector ONNX models.

No model weights ship with this package. Weights are multi-megabyte binaries with
their own licences and their own update cadence; vendoring them into a source
repository is how you end up shipping a stale model under the wrong licence. So a
model is described by a ``ModelSpec`` — a URL and a SHA256 — and fetched once,
verified, and cached on disk.

Resolution order for every model:

1. ``ModelSpec.path`` — an explicit local file. Used verbatim; if it does not
   exist you get a warning and ``None``, never a surprise download.
2. ``ModelSpec.url`` + ``ModelSpec.sha256`` — fetched once over HTTPS, checksum
   verified before the file is published into the cache, then reused forever.
3. Neither configured — ``None``.

A failed fetch is never fatal here. It logs a warning and returns ``None``, and
the caller decides what that means: ``build_plate_detector`` disables plates,
``build_face_detector`` falls back to CenterFace or raises.

The download is a plain HTTPS GET of a public file. Nothing is uploaded — no
frames, no telemetry. Point ``url`` at your own mirror if you would rather not
reach the public host at all, or pre-populate ``path`` and stay fully offline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("~/.cache/redactcam/models").expanduser()
DOWNLOAD_TIMEOUT = 180


@dataclass(frozen=True)
class ModelSpec:
    """Where one ONNX model comes from.

    ``path`` (a local file) wins over ``url``/``sha256`` (fetch and cache). Both
    empty means "this model is not configured" — the caller degrades. ``name`` is
    used to build the cache filename and to label log lines.
    """

    name: str
    url: str = ""
    sha256: str = ""
    path: str = ""


# Public models the defaults in this library were validated against. These are
# third-party releases under their own licences — check each one before
# redistributing anything derived from it. Override any of them with your own
# ModelSpec; nothing in the library hard-codes a URL at the point of use.
#
# The COCO model serves both the vehicle and the person detector (person is
# class 0 there), so one cached file covers both.
_COCO_URL = "https://huggingface.co/aaurelions/yolo11n.onnx/resolve/main/yolo11n.onnx"
_COCO_SHA = "5ad2aa77f7f678c06880f024444cb02f5eb53f991b4a64c819805c6c701fc9b8"

FACE_MODEL = ModelSpec(
    name="face",
    url="https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.onnx",
    sha256="61d8cf127ff377939cf009565457aaddf216f4d9fdd9ac24f5aab3cb10ee67b5",
)
PLATE_MODEL = ModelSpec(
    name="plate",
    url=(
        "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
        "/resolve/main/license-plate-finetune-v1m.onnx"
    ),
    sha256="53e6a1c535141a483e2fe5b802d4f8179e820561d5ed892b0bdeb5fbd200f010",
)
VEHICLE_MODEL = ModelSpec(name="vehicle", url=_COCO_URL, sha256=_COCO_SHA)
PERSON_MODEL = ModelSpec(name="person", url=_COCO_URL, sha256=_COCO_SHA)

DEFAULT_MODELS = {
    "face": FACE_MODEL,
    "plate": PLATE_MODEL,
    "vehicle": VEHICLE_MODEL,
    "person": PERSON_MODEL,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_model(spec: ModelSpec, cache_dir: Path | None = None) -> Path | None:
    """Local path to ``spec``'s ONNX file, or ``None`` if it cannot be resolved.

    Never raises: every failure path — missing override, unconfigured spec,
    non-HTTPS URL, network error, checksum mismatch — logs a warning and returns
    ``None`` so the caller can degrade on its own terms.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    override = (spec.path or "").strip()
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        logger.warning("%s model path %s does not exist", spec.name, p)
        return None

    url, sha = (spec.url or "").strip(), (spec.sha256 or "").strip()
    if not url or not sha:
        return None  # not configured → the caller degrades

    dest = cache / f"{spec.name}_{sha[:12]}.onnx"
    if dest.exists() and _sha256(dest) == sha:
        return dest

    if not url.lower().startswith("https://"):
        logger.warning("%s model url is not https — refusing to fetch", spec.name)
        return None
    try:
        return _download(url, sha, dest)
    except Exception as exc:  # network / IO — never fatal, the caller degrades
        logger.warning("%s model fetch failed: %s", spec.name, exc)
        return None


def _download(url: str, sha: str, dest: Path) -> Path | None:
    """Fetch ``url`` to ``dest`` atomically; verify SHA256 before publishing it.

    The download lands in a temp file beside the destination and is renamed into
    place only once the checksum matches, so a killed process or a truncated
    transfer can never leave a half-written model in the cache for the next run
    to load.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching model (one time only): %s", url)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "redactcam"})
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp,
        ):
            shutil.copyfileobj(resp, out)
        got = _sha256(tmp)
        if got != sha:
            logger.warning("model checksum mismatch — discarding %s", url)
            return None
        tmp.replace(dest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)
