"""Clean-machine qualification — can a fresh install actually redact a video?

A successful download proves nothing: the wheel installs fine on a machine with
no ffmpeg, no usable onnxruntime, or a model file that is not ONNX at all, and
each of those surfaces only on the first real detection. ``redactcam
--check-deps --receipt FILE --synthetic-render --model KIND=PATH …`` runs the
checks a first real job would hit, in order, and records the outcome:

1. the native stack loads (OpenCV, onnxruntime, the inference-provider decision);
2. ``ffmpeg`` and ``ffprobe`` run;
3. every **explicitly supplied** model file exists, hashes, and builds a detector
   that survives one real inference — nothing is downloaded, ever, because the
   point is to qualify *those* artifacts, not whatever a fetch returned;
4. a synthetic clip goes through the whole pipeline to an encoded output.

The result is a versioned JSON receipt whose ``qualified`` flag is true only if
all four passed. Judging *which* OS and *which* model files count as supported is
the maintainer's call and is not encoded here: the receipt records what ran.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .models import DEFAULT_MODELS, ModelSpec
from .provenance import runtime_identity, sha256_file

RECEIPT_SCHEMA = "redactcam.clean-machine-receipt/v1"

_INFER_SIZE = 640
_IOU = 0.45


def _load_detector(kind: str, path: Path) -> str:
    """Build the detector for ``kind`` from ``path`` and run it on one blank
    frame; return the execution provider its session landed on.

    Construction proves onnxruntime can read the file; the blank-frame inference
    proves the graph runs and its output decodes, which a truncated or wrong-shape
    model can fail even when it loads."""
    import numpy as np

    from .detect import PERSON_CLASSES, VEHICLE_CLASSES, YoloDetector
    from .presets import ROAD_FOOTAGE

    min_aspect = max_aspect = 0.0
    class_filter = None
    if kind == "plate":
        min_aspect = float(ROAD_FOOTAGE["plate_min_aspect"])
        max_aspect = float(ROAD_FOOTAGE["plate_max_aspect"])
    elif kind == "vehicle":
        class_filter = VEHICLE_CLASSES
    elif kind == "person":
        class_filter = PERSON_CLASSES
    det = YoloDetector(
        path, _INFER_SIZE, _IOU, min_aspect, 1, max_aspect=max_aspect, class_filter=class_filter
    )
    det.detect(np.zeros((360, 640, 3), dtype=np.uint8), 0.9)
    providers = det._sess.get_providers()
    return providers[0] if providers else "unknown"


def check_models(
    specs: dict[str, ModelSpec],
    *,
    require_all: bool = False,
    loader=_load_detector,
) -> dict[str, dict]:
    """Per-kind result for the models the caller supplied by ``path``.

    ``status`` is ``ok``, ``missing`` (file not there), ``failed`` (the runtime
    rejected it), or ``not_supplied`` (no local path; only reported with
    ``require_all``). A ``url``-only spec is not supplied: qualification judges
    files a person handed over, and never fetches.
    """
    out: dict[str, dict] = {}
    for kind in DEFAULT_MODELS:
        spec = specs.get(kind)
        path_str = (spec.path or "").strip() if spec is not None else ""
        if not path_str:
            if require_all:
                out[kind] = {
                    "status": "not_supplied",
                    "error": (
                        f"no local {kind} model given. Supply the qualified file with "
                        f"--model {kind}=PATH; qualification never downloads weights."
                    ),
                }
            continue
        path = Path(path_str).expanduser()
        if not path.is_file():
            out[kind] = {
                "status": "missing",
                "path": str(path),
                "error": f"{kind} model file not found: {path}. Fix --model {kind}=PATH.",
            }
            continue
        entry = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        try:
            entry["provider"] = loader(kind, path)
            entry["status"] = "ok"
        except Exception as exc:  # any load failure is a finding, not a crash
            entry["status"] = "failed"
            entry["error"] = (
                f"{kind} model {path.name} did not load or run: {type(exc).__name__}: {exc}. "
                "It must be a YOLO-format ONNX file the installed onnxruntime can execute."
            )
        out[kind] = entry
    return out


def _make_synthetic_input(dest: Path) -> None:
    """A 2 s, 640x360, 10 fps synthetic clip via ffmpeg's ``testsrc`` (no real
    footage, ever). H.264 on purpose: the render encodes H.264, so a build
    without libx264 fails here, early and legibly."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-nostats", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not create the synthetic clip: {proc.stderr[-300:]}")


def synthetic_render(specs: dict[str, ModelSpec], work_dir: Path) -> dict:
    """Run one synthetic clip through ``redact_video`` and describe the outcome.

    Never raises. Attempted only when all four models are supplied locally, since
    ``redact_video`` would otherwise resolve the missing ones by downloading.
    """
    missing = [k for k in DEFAULT_MODELS if not (k in specs and (specs[k].path or "").strip())]
    if missing:
        return {
            "attempted": False,
            "ok": False,
            "error": (
                "the synthetic render needs all four models supplied locally "
                f"(missing: {', '.join(missing)}); it never downloads weights."
            ),
        }
    work_dir = Path(work_dir)
    src = work_dir / "synthetic_input.mp4"
    try:
        _make_synthetic_input(src)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"attempted": True, "ok": False, "stage": "synthetic_input", "error": str(exc)}
    try:
        from .pipeline import redact_video

        result = redact_video(
            src, work_dir=work_dir, models=dict(specs), reuse_sidecar=False, render=True
        )
    except Exception as exc:  # the receipt must record a failed render, not die on it
        return {
            "attempted": True,
            "ok": False,
            "stage": "redact_video",
            "error": f"{type(exc).__name__}: {exc}",
        }
    out = result.output
    if out is None or not Path(out).is_file() or Path(out).stat().st_size == 0:
        return {
            "attempted": True,
            "ok": False,
            "stage": "redact_video",
            "error": "the pipeline returned but produced no output video",
        }
    return {
        "attempted": True,
        "ok": True,
        "frames": result.frame_count,
        "output_sha256": sha256_file(out),
        "output_size_bytes": Path(out).stat().st_size,
    }


def build_clean_machine_receipt(*, diagnostics: dict, models: dict, render: dict) -> dict:
    """Assemble the versioned receipt. ``qualified`` is true only when the
    diagnostics carry no errors, all four detectors are ``ok``, and the render
    completed — a gap anywhere is a not-qualified receipt, never a softer pass."""
    detectors_ok = set(models) >= set(DEFAULT_MODELS) and all(
        models[k].get("status") == "ok" for k in DEFAULT_MODELS
    )
    qualified = bool(
        not diagnostics.get("errors") and detectors_ok and render.get("ok") is True
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "launch": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "identity": runtime_identity(),
        "diagnostics": diagnostics,
        "detectors": models,
        "render": render,
        "qualified": qualified,
    }
