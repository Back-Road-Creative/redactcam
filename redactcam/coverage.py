"""Independent coverage verification — the last stage, and the point of the library.

Every stage upstream of this one can be wrong. The detector can miss a car, the
tracker can drift, the mask can be a frame out of step — and all of those fail
*silently*, producing a plausible-looking blurred video with someone's face still
readable in it. This module is the check that does not trust any of them.

It re-runs detection from scratch on the source frames, plate-corroborates each
vehicle it finds (a real road car shows a licence plate; a roadside billboard,
sign or parked caravan that the COCO model misreads as a vehicle does not — see
``TrackManager._corroborate_vehicles``), and then confirms the MASK actually
covers each corroborated car's cabin region, the top ``cabin_frac`` of the box.
A cabin the mask leaves uncovered is a driver about to ship unblurred.

Verified against the MASK rather than the rendered output on purpose. A finished
render usually carries other changes too — colour grading, overlays, a different
codec — so differencing it against the source cannot isolate the blur. The mask
is exactly "what the blur will cover", so mask coverage equals output coverage
without the confound, and the check runs BEFORE the expensive, irreversible
encode rather than after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .detect import (
    Box,
    build_plate_detector,
    build_vehicle_detector,
    detect_plates,
    filter_vehicles,
    frame_tiles,
)


@dataclass
class Leak:
    """One exposed cabin: ``frame`` index, mask ``coverage`` fraction, vehicle ``box``."""

    frame: int
    coverage: float
    box: Box


@dataclass
class CoverageReport:
    """``vehicle_frames`` corroborated vehicle sightings were checked; each entry
    in ``leaks`` is one the mask did not adequately cover. ``ok`` is the single
    boolean a caller gates on before rendering."""

    vehicle_frames: int = 0
    leaks: list[Leak] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaks


def cabin_region(box: Box, frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    """The cabin sub-rectangle (top ``frac`` of the vehicle ``box``), clamped to frame."""
    x, y, bw, bh = box
    ch = max(1, int(round(bh * frac)))
    return max(0, x), max(0, y), min(w, x + bw), min(h, y + ch)


def plate_inside(vehicle: Box, plates: list[Box]) -> bool:
    """True if any plate box center falls inside the vehicle box (real-car corroboration)."""
    vx, vy, vw, vh = vehicle
    for px, py, pw, ph in plates:
        cx, cy = px + pw / 2, py + ph / 2
        if vx <= cx <= vx + vw and vy <= cy <= vy + vh:
            return True
    return False


def verify_cabin_coverage(
    source_video: str | Path,
    mask_video: str | Path,
    vehicle_model: str | Path,
    *,
    plate_model: str | Path | None = None,
    cabin_frac: float = 0.6,
    conf: float = 0.6,
    sample_fps: float = 4.0,
    min_coverage: float = 0.5,
    min_height_frac: float = 0.07,
    ignore_bottom_frac: float = 0.12,
    plate_conf: float = 0.4,
    plate_max_area_frac: float = 0.006,
    infer_size: int = 640,
) -> CoverageReport:
    """Per-sampled-frame check that the MASK covers every plate-corroborated
    vehicle's cabin.

    A cabin is a LEAK when the mask is white over fewer than ``min_coverage`` of
    its pixels — the blur missed the driver. ``min_coverage`` defaults to 0.5
    rather than 1.0 because this verifier detects the vehicle independently of
    whatever produced the blur, so a box that lands a few percent off trims edge
    pixels while the driver, who sits near the cabin centre, stays covered.

    Reads the source and the mask in frame-locked lockstep — no seeking, so the
    two can never silently drift apart. The confidence, size filter and plate
    corroboration should mirror whatever produced the mask, so the check covers
    exactly the vehicles the blur is responsible for and not the roadside signage
    it deliberately leaves readable."""
    det = build_vehicle_detector(str(vehicle_model), infer_size, 0.45, 0)
    if det is None:
        raise ValueError(f"vehicle model not resolvable: {vehicle_model}")
    plate_det = build_plate_detector(plate_model, infer_size, 0.45, 2.0, 6.0)
    cap_in = cv2.VideoCapture(str(source_video))
    cap_m = cv2.VideoCapture(str(mask_video))
    if not (cap_in.isOpened() and cap_m.isOpened()):
        raise ValueError("could not open the source video or the mask video")
    fps = cap_in.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(fps / sample_fps)) if sample_fps > 0 else 1
    tiles = None
    rep = CoverageReport()
    idx = 0
    while True:
        ok_i, frame = cap_in.read()
        ok_m, mask = cap_m.read()
        if not (ok_i and ok_m):
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            if tiles is None:
                tiles = frame_tiles(w, h, 1280, 0.15)
            mgray = mask if mask.ndim == 2 else mask[:, :, 0]
            plates = (
                detect_plates(frame, plate_det, plate_conf, tiles, plate_max_area_frac)
                if plate_det is not None
                else None
            )
            for bx in filter_vehicles(
                det.detect(frame, conf), h, min_height_frac, ignore_bottom_frac
            ):
                if plates is not None and not plate_inside(bx, plates):
                    continue  # billboard/sign/caravan — the blur skips it, so do we
                rep.vehicle_frames += 1
                x0, y0, x1, y1 = cabin_region(bx, cabin_frac, w, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                cov = float((mgray[y0:y1, x0:x1] > 0).mean())
                if cov < min_coverage:
                    rep.leaks.append(Leak(idx, cov, bx))
        idx += 1
    cap_in.release()
    cap_m.release()
    return rep
