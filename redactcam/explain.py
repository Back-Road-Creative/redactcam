"""``redactcam explain``: why does the blur cover what it covers?

A read-only diagnostic. It runs the SAME detection and tracking the mask uses
(``detect.detect_and_track`` with the ``ROAD_FOOTAGE`` preset, then
``timeline.build_timeline``'s dilation) and reports every box the mask would
rasterise, per frame, with its kind, its cause and, for vehicles and people, the
detector's COCO class id. It never renders a mask and never touches the video.

Outputs, all under ``out_dir``:

* ``boxes.jsonl``  one record per box per frame, in the order the mask draws them.
* ``summary.json`` blurred-area share by kind x source, and a vehicle aspect
  histogram split by COCO class.
* ``sheet.png``    (with ``sheet=N``) N evenly spaced frames with the boxes drawn.

Boxes are the DILATED boxes the mask uses (``x, y, w, h``); ``raw`` is the same box
before the flat dilation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import timeline as _timeline
from .models import ModelSpec
from .pipeline import ROAD_FOOTAGE_BOX_DILATION_PX, detect_and_track, detect_options

#: COCO ids the vehicle detector keeps, in report order.
VEHICLE_CLASS_NAMES = {7: "truck", 5: "bus", 2: "car", 3: "motorcycle"}
#: Lower bin edges of the aspect (w/h) histogram; the last bin is open-ended.
ASPECT_EDGES = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

# BGR colours for the contact sheet.
KIND_COLOURS = {
    "face": (0, 200, 0),
    "plate": (0, 200, 255),
    "person": (255, 120, 0),
    "vehicle_cabin": (0, 0, 255),
    "vehicle": (200, 0, 200),
}


def _kind(cls: str, cabin_frac: float) -> str:
    if cls == "vehicle":
        return "vehicle_cabin" if cabin_frac > 0 else "vehicle"
    return cls


def explain_video(
    video: str | Path,
    out_dir: str | Path,
    *,
    models: dict[str, ModelSpec] | None = None,
    cache_dir: Path | None = None,
    detect_kwargs: dict | None = None,
    box_dilation_px: int = ROAD_FOOTAGE_BOX_DILATION_PX,
    sheet: int = 0,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict:
    """Explain the blur on ``video`` into ``out_dir``; returns the summary dict."""
    video = Path(video)
    out = Path(out_dir)
    opts, _paths = detect_options(models, cache_dir, detect_kwargs)
    fps, _count, width, height = _timeline.probe_video(video)
    if width <= 0 or height <= 0:
        raise ValueError(f"could not read the frame size of {video}")
    start = max(0, round(start_s * fps)) if fps > 0 else 0
    limit = max(1, round(duration_s * fps)) if duration_s is not None and fps > 0 else None

    trace: dict = {}
    frame_boxes, end, _nf, _np = detect_and_track(
        video, start_frame=start, max_frames=limit, trace=trace, **opts
    )
    entries = trace.get("entries", {})
    # The mask's own dilation step, so these are the boxes materialize_mask_video draws.
    tl = _timeline.build_timeline(
        frame_boxes,
        source_hash="explain",
        source_fps=fps,
        source_frame_count=end,
        frame_width=width,
        frame_height=height,
        box_dilation_px=box_dilation_px,
        sample_fps=float(opts.get("sample_fps", 8.0)),
        conf=float(opts.get("conf", 0.4)),
    )
    cabin_frac = float(opts.get("track_cabin_frac", 0.0))
    records: list[dict] = []
    for f in sorted(tl.frame_boxes):
        es, boxes = entries[f], tl.frame_boxes[f]
        if len(es) != len(boxes):  # fail closed: never explain boxes the mask does not use
            raise RuntimeError(f"frame {f}: {len(es)} explained boxes vs {len(boxes)} mask boxes")
        for e, (x, y, w, h) in zip(es, boxes, strict=True):
            records.append(
                {
                    "frame": f,
                    "kind": _kind(e.cls, cabin_frac),
                    "source": e.source,
                    "coco_class": e.coco_cls,
                    "track_id": e.track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "raw": list(e.box),
                    "vehicle_box": list(e.vehicle_box) if e.vehicle_box else None,
                    "aspect": w / h if h else 0.0,
                    "area_frac": w * h / (width * height),
                }
            )
    n_frames = max(0, end - start)
    summary = summarize(records, frames=n_frames, width=width, height=height)
    summary.update(video=str(video), start_frame=start, end_frame=end)

    out.mkdir(parents=True, exist_ok=True)
    (out / "boxes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if sheet > 0:
        write_sheet(video, out / "sheet.png", records, start, end, sheet)
    return summary


def summarize(records: list[dict], *, frames: int, width: int, height: int) -> dict:
    """Aggregate ``records``. ``area_share[kind][source]`` is the mean, over the
    ``frames`` analysed, of the summed box area as a fraction of the frame (overlaps
    count twice); ``share_of_blurred`` is each cell's fraction of all box area.
    ``union_share`` is the mean fraction of pixels actually covered. The vehicle aspect
    histogram counts direct DETECTIONS only (source ``detected``), using the whole
    vehicle box, not the cabin band cut from it."""
    area: dict[str, dict[str, float]] = {}
    count: dict[str, dict[str, int]] = {}
    for r in records:
        area.setdefault(r["kind"], {}).setdefault(r["source"], 0.0)
        area[r["kind"]][r["source"]] += r["area_frac"]
        count.setdefault(r["kind"], {}).setdefault(r["source"], 0)
        count[r["kind"]][r["source"]] += 1
    total = sum(v for k in area.values() for v in k.values())
    n = max(frames, 1)
    hist = {
        f"{cid}_{name}": {"edges": ASPECT_EDGES, "counts": [0] * len(ASPECT_EDGES)}
        for cid, name in VEHICLE_CLASS_NAMES.items()
    }
    for r in records:
        if r["source"] != "detected" or r["coco_class"] not in VEHICLE_CLASS_NAMES:
            continue
        vb = r.get("vehicle_box") or (r["x"], r["y"], r["w"], r["h"])
        if vb[3] <= 0:
            continue
        aspect = vb[2] / vb[3]
        i = max(j for j, e in enumerate(ASPECT_EDGES) if aspect >= e)
        hist[f"{r['coco_class']}_{VEHICLE_CLASS_NAMES[r['coco_class']]}"]["counts"][i] += 1
    return {
        "frames_analysed": frames,
        "boxes": len(records),
        "frame_size": [width, height],
        "area_share": {k: {s: a / n for s, a in v.items()} for k, v in area.items()},
        "share_of_blurred": {
            k: {s: (a / total if total else 0.0) for s, a in v.items()} for k, v in area.items()
        },
        "box_count": count,
        "union_share": _union_share(records, frames, width, height),
        "vehicle_aspect_histogram": hist,
    }


def _union_share(records: list[dict], frames: int, width: int, height: int) -> float:
    """Mean per-frame fraction of pixels covered by at least one box (1/8-scale grid)."""
    if frames <= 0:
        return 0.0
    k = 8
    gw, gh = max(1, width // k), max(1, height // k)
    by_frame: dict[int, np.ndarray] = {}
    for r in records:
        g = by_frame.setdefault(r["frame"], np.zeros((gh, gw), dtype=bool))
        g[
            r["y"] // k : math.ceil((r["y"] + r["h"]) / k),
            r["x"] // k : math.ceil((r["x"] + r["w"]) / k),
        ] = True
    return float(sum(g.mean() for g in by_frame.values()) / frames)


def write_sheet(video: Path, path: Path, records: list[dict], start: int, end: int, n: int) -> None:
    """PNG contact sheet of ``n`` evenly spaced frames in [start, end) with boxes drawn
    (colour by kind, label by source). Reads the video; never writes it."""
    if end <= start:
        raise ValueError("no frames were analysed, so there is nothing to put on a sheet")
    n = min(n, end - start)
    picks = sorted({start + round(i * (end - start - 1) / max(n - 1, 1)) for i in range(n)})
    by_frame: dict[int, list[dict]] = {}
    for r in records:
        by_frame.setdefault(r["frame"], []).append(r)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    tiles = []
    try:
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            scale = 640 / img.shape[1]
            tile = cv2.resize(img, (640, max(1, round(img.shape[0] * scale))))
            for r in by_frame.get(f, []):
                col = KIND_COLOURS.get(r["kind"], (255, 255, 255))
                p1 = (round(r["x"] * scale), round(r["y"] * scale))
                p2 = (round((r["x"] + r["w"]) * scale), round((r["y"] + r["h"]) * scale))
                cv2.rectangle(tile, p1, p2, col, 2)
                label = r["source"] + (f" {r['coco_class']}" if r["coco_class"] is not None else "")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tile, label, (p1[0] + 2, max(12, p1[1] + 12)), font, 0.45, col, 1)
            cv2.putText(
                tile, f"frame {f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            tiles.append(tile)
    finally:
        cap.release()
    if not tiles:
        raise ValueError("could not read any frame for the contact sheet")
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r_, c_ = divmod(i, cols)
        canvas[r_ * th : r_ * th + t.shape[0], c_ * tw : c_ * tw + t.shape[1]] = t
    cv2.imwrite(str(path), canvas)
