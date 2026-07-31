"""The independent coverage verifier.

Checks that the MASK covers every plate-corroborated vehicle's cabin: a black
mask (the blur missed the car) leaks, a covering mask passes, and a plate-less
vehicle — a roadside billboard or sign — is never checked at all. Detectors are
stubbed and tiny generated FFV1 clips stand in for the source and the mask, so
nothing here touches a model file or real footage.
"""

import cv2
import numpy as np

from redactcam import coverage as bc

W, H = 200, 120
VEH = (40, 20, 100, 60)  # x, y, w, h — cabin (top 60%) = rows 20:56, cols 40:140
PLATE_IN = (70, 60, 30, 12)  # center (85, 66) inside VEH → corroborates


class _FakeVeh:
    def detect(self, frame, conf):
        return [VEH]


def _write(path, frames, fps=3.0):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for f in frames:
        wr.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    wr.release()


def _stub(monkeypatch, plates):
    monkeypatch.setattr(bc, "build_vehicle_detector", lambda *a, **k: _FakeVeh())
    monkeypatch.setattr(bc, "build_plate_detector", lambda *a, **k: object())
    monkeypatch.setattr(bc, "detect_plates", lambda *a, **k: list(plates))


def _run(tmp_path, mask_frame, plates=(PLATE_IN,)):
    inp = tmp_path / "in.mkv"
    _write(inp, [np.full((H, W, 3), 50, np.uint8)] * 3)
    mvid = tmp_path / "mask.mkv"
    _write(mvid, [mask_frame] * 3)
    return bc.verify_cabin_coverage(
        inp,
        mvid,
        "v.onnx",
        plate_model="p.onnx",
        cabin_frac=0.6,
        conf=0.6,
        sample_fps=3,
        min_coverage=0.5,
        min_height_frac=0.0,
        ignore_bottom_frac=0.0,
    )


def test_cabin_region_and_plate_inside():
    assert bc.cabin_region(VEH, 0.6, W, H) == (40, 20, 140, 56)
    assert bc.plate_inside(VEH, [PLATE_IN])
    assert not bc.plate_inside(VEH, [(0, 0, 10, 10)])


def test_mask_covering_cabin_passes(tmp_path, monkeypatch):
    _stub(monkeypatch, [PLATE_IN])
    mask = np.zeros((H, W), np.uint8)
    mask[20:56, 40:140] = 255  # white over the cabin region
    rep = _run(tmp_path, mask)
    assert rep.vehicle_frames == 3 and rep.ok


def test_black_mask_leaks(tmp_path, monkeypatch):
    _stub(monkeypatch, [PLATE_IN])
    rep = _run(tmp_path, np.zeros((H, W), np.uint8))  # nothing blurred
    assert rep.vehicle_frames == 3 and not rep.ok and len(rep.leaks) == 3
    assert all(leak.coverage == 0.0 for leak in rep.leaks)


def test_plateless_vehicle_not_checked(tmp_path, monkeypatch):
    """A vehicle with no plate inside (a billboard/sign) is skipped — not a leak even
    against a black mask."""
    _stub(monkeypatch, [(0, 0, 5, 5)])  # plate detected but OUTSIDE the vehicle box
    rep = _run(tmp_path, np.zeros((H, W), np.uint8))
    assert rep.vehicle_frames == 0 and rep.ok
