"""The five stages wired together: detect → timeline → mask → verify → apply.

Everything here is ordinary calls into the other modules. Wire your own if you
need something different — but note the ordering constraint, because it is the
one thing that is easy to get wrong:

**Verification runs before the render, not after.** Once you have encoded a
multi-gigabyte file and uploaded it, discovering that the blur missed a driver
costs you a re-encode and a takedown. Checking the mask takes seconds and costs
nothing. So ``redact_video`` refuses to render when coverage fails, rather than
rendering and reporting.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from . import apply as _apply
from . import mask as _mask
from . import timeline as _timeline
from .coverage import CoverageReport, verify_cabin_coverage
from .detect import detect_and_track
from .models import (
    DEFAULT_MODELS,
    ModelSpec,
    resolve_model,
)
from .presets import (
    ROAD_FOOTAGE,
    ROAD_FOOTAGE_BOX_DILATION_PX,
    ROAD_FOOTAGE_FEATHER_PX,
)

logger = logging.getLogger(__name__)


class CoverageError(RuntimeError):
    """The mask does not cover every corroborated vehicle cabin.

    Raised instead of rendering. The report is on ``.report`` if you want to
    inspect the individual leaks and decide for yourself.
    """

    def __init__(self, message: str, report: CoverageReport):
        super().__init__(message)
        self.report = report


@dataclass
class RedactionResult:
    """What a run produced. ``output`` is ``None`` for a ``render=False`` run."""

    sidecar: Path
    mask: Path
    output: Path | None
    frame_count: int
    face_detections: int
    plate_detections: int
    coverage: CoverageReport | None


def file_hash(path: str | Path, block: int = 1 << 20) -> str:
    """SHA256 of a file — the sidecar's cache key.

    Keyed on content rather than filename so a re-encoded or re-exported clip
    never silently reuses a timeline built from different pixels.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_video(
    source_video: str | Path,
    output_path: str | Path | None = None,
    *,
    work_dir: str | Path | None = None,
    models: dict[str, ModelSpec] | None = None,
    cache_dir: Path | None = None,
    detect_kwargs: dict | None = None,
    box_dilation_px: int = ROAD_FOOTAGE_BOX_DILATION_PX,
    feather_px: int = ROAD_FOOTAGE_FEATHER_PX,
    blur_strength: int = _apply.REFERENCE_BLUR_STRENGTH,
    verify: bool = True,
    min_coverage: float = 0.5,
    reuse_sidecar: bool = True,
    render: bool = True,
) -> RedactionResult:
    """Detect, track, mask, verify and (optionally) render one video.

    ``detect_kwargs`` overrides individual ``ROAD_FOOTAGE`` preset values; pass
    ``{}`` to keep the preset as is. ``models`` overrides individual entries in
    ``DEFAULT_MODELS`` by key (``face``, ``plate``, ``vehicle``, ``person``).

    With ``reuse_sidecar`` the timeline is loaded from disk when its stored hash
    matches the source's, skipping detection — which is nearly all of the run
    time. ``render=False`` stops after verification, leaving you the mask and the
    sidecar to feed into your own encode.

    Raises ``CoverageError`` when verification finds an uncovered vehicle cabin.
    """
    source_video = Path(source_video)
    work = Path(work_dir) if work_dir else source_video.parent
    work.mkdir(parents=True, exist_ok=True)
    stem = source_video.stem

    specs = {**DEFAULT_MODELS, **(models or {})}
    paths = {k: resolve_model(v, cache_dir) for k, v in specs.items()}

    opts = {**ROAD_FOOTAGE, **(detect_kwargs or {})}
    opts["face_model_path"] = paths.get("face")
    opts["plate_model_path"] = paths.get("plate")
    opts["vehicle_model_path"] = paths.get("vehicle")
    opts["person_model_path"] = paths.get("person")

    fps, probe_count, width, height = _timeline.probe_video(source_video)
    source_hash = file_hash(source_video)
    sidecar_path = work / f"{stem}_redactcam.json"

    tl = _timeline.load_sidecar(sidecar_path) if reuse_sidecar else None
    if tl is not None and tl.source_hash == source_hash:
        logger.info("reusing timeline %s (hash match)", sidecar_path.name)
        n_face = n_plate = -1  # not re-counted on a reuse
    else:
        frame_boxes, decoded, n_face, n_plate = detect_and_track(source_video, **opts)
        # The decoded count is authoritative: container metadata routinely lies,
        # and a mask one frame short blurs the wrong pixels for the whole tail.
        tl = _timeline.build_timeline(
            frame_boxes,
            source_hash=source_hash,
            source_fps=fps,
            source_frame_count=decoded or probe_count,
            frame_width=width,
            frame_height=height,
            box_dilation_px=box_dilation_px,
            sample_fps=float(opts.get("sample_fps", 8.0)),
            conf=float(opts.get("conf", 0.4)),
        )
        _timeline.write_sidecar(sidecar_path, tl)

    mask_path = _mask.materialize_mask_video(
        tl, work / f"{stem}_redactcam_mask.mkv", feather_px=feather_px
    )

    report = None
    if verify and paths.get("vehicle"):
        report = verify_cabin_coverage(
            source_video,
            mask_path,
            paths["vehicle"],
            plate_model=paths.get("plate"),
            cabin_frac=float(opts.get("track_cabin_frac", 0.6)),
            conf=float(opts.get("vehicle_conf", 0.6)),
            min_coverage=min_coverage,
            min_height_frac=float(opts.get("vehicle_min_height_frac", 0.07)),
            ignore_bottom_frac=float(opts.get("vehicle_ignore_bottom_frac", 0.12)),
            plate_max_area_frac=float(opts.get("plate_max_area_frac", 0.006)),
        )
        if not report.ok:
            worst = sorted(report.leaks, key=lambda leak: leak.coverage)[:5]
            detail = ", ".join(f"frame {v.frame} ({v.coverage:.0%})" for v in worst)
            raise CoverageError(
                f"{len(report.leaks)} of {report.vehicle_frames} corroborated vehicle "
                f"sightings have an uncovered cabin — e.g. {detail}. Refusing to "
                "render: a driver would ship unblurred.",
                report,
            )

    output = None
    if render:
        if output_path is None:
            output_path = work / f"{stem}_redacted.mp4"
        output = _apply.apply_blur(
            source_video,
            mask_path,
            output_path,
            video_width=width,
            strength=blur_strength,
        )

    return RedactionResult(
        sidecar=sidecar_path,
        mask=mask_path,
        output=output,
        frame_count=tl.source_frame_count,
        face_detections=n_face,
        plate_detections=n_plate,
        coverage=report,
    )
