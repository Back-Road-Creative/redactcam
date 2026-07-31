"""Blur timeline plus its JSON sidecar — stage 2 of the redaction chain.

``detect.detect_and_track`` yields DENSE per-frame face ∪ plate boxes (sparse
detections optical-flow-propagated to every frame). This module grows each box
by a safety margin (``box_dilation_px``) and stores the result keyed by source
frame number, ready for ``mask`` to rasterise.

Each frame's boxes come straight from its own tracked boxes — there is no
interval or union gap-fill, because that approach freezes a fast object's box
for a second at a time, blurring empty road ahead of it and exposing it as it
passes.

The timeline serialises to a JSON sidecar keyed by the hash of the exact file it
was detected on, so re-running against the same input can skip detection (the
expensive part) on a hash match. Native-pixel ``(x, y, w, h)`` boxes throughout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .detect import Box

SCHEMA = "redactcam.timeline/v1"


@dataclass
class BlurTimeline:
    """Dense per-frame blur timeline + the detection params that produced it.
    ``frame_boxes`` maps a source frame index → its dilated boxes; a frame absent
    from the map carries no blur."""

    source_hash: str
    sample_fps: float
    conf: float
    box_dilation_px: int
    source_fps: float
    source_frame_count: int
    frame_width: int
    frame_height: int
    frame_boxes: dict[int, list[Box]] = field(default_factory=dict)


def probe_video(path: str | Path) -> tuple[float, int, int, int]:
    """Return ``(fps, frame_count, width, height)`` for a clip. Best-effort —
    some containers report 0 for one or more of these, and callers are expected
    to backfill from what they already know. ``mask.materialize_mask_video``
    owns the strict "mask length equals source length" assertion."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return fps, frame_count, w, h


def dilate_box(box: Box, px: int, w: int, h: int) -> Box:
    """Grow ``box`` by ``px`` on every side, clamped to the ``w``×``h`` frame."""
    x, y, bw, bh = box
    nx, ny = max(0, x - px), max(0, y - px)
    nx2, ny2 = min(w, x + bw + px), min(h, y + bh + px)
    return (nx, ny, max(0, nx2 - nx), max(0, ny2 - ny))


def build_timeline(
    frame_boxes: dict[int, list[Box]],
    *,
    source_hash: str,
    source_fps: float,
    source_frame_count: int,
    frame_width: int,
    frame_height: int,
    box_dilation_px: int,
    sample_fps: float,
    conf: float,
) -> BlurTimeline:
    """Dilate every per-frame box by the safety margin. ``frame_boxes`` is the
    DENSE per-frame result of ``detect_and_track`` (one entry per source frame the
    tracker had an active box on), so each frame's mask comes straight from its
    own tracked boxes — no interval/union gap-fill. Frames with no tracked object
    are simply absent (no blur)."""
    dilated = {
        f: [dilate_box(b, box_dilation_px, frame_width, frame_height) for b in boxes]
        for f, boxes in frame_boxes.items()
        if boxes  # the tracker emits an entry per frame; only persist active ones
    }
    return BlurTimeline(
        source_hash=source_hash,
        sample_fps=sample_fps,
        conf=conf,
        box_dilation_px=box_dilation_px,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
        frame_boxes=dilated,
    )


def write_sidecar(path: Path, tl: BlurTimeline) -> None:
    """Serialize ``tl`` to ``path`` as deterministic JSON (frame keys sorted)."""
    payload = {
        "schema": SCHEMA,
        "source_hash": tl.source_hash,
        "sample_fps": tl.sample_fps,
        "conf": tl.conf,
        "box_dilation_px": tl.box_dilation_px,
        "source_fps": tl.source_fps,
        "source_frame_count": tl.source_frame_count,
        "frame_width": tl.frame_width,
        "frame_height": tl.frame_height,
        "frame_boxes": {
            str(f): [list(b) for b in tl.frame_boxes[f]] for f in sorted(tl.frame_boxes)
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def load_sidecar(path: Path) -> BlurTimeline | None:
    """Load a sidecar, or ``None`` if absent / unreadable / wrong schema (a v1
    sidecar mismatches → recompute, so a resume never reuses a stale interval
    model)."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("schema") != SCHEMA:
        return None
    frame_boxes = {
        int(f): [tuple(b) for b in boxes] for f, boxes in d.get("frame_boxes", {}).items()
    }
    return BlurTimeline(
        source_hash=d["source_hash"],
        sample_fps=d["sample_fps"],
        conf=d["conf"],
        box_dilation_px=d["box_dilation_px"],
        source_fps=d["source_fps"],
        source_frame_count=d["source_frame_count"],
        frame_width=d["frame_width"],
        frame_height=d["frame_height"],
        frame_boxes=frame_boxes,
    )
