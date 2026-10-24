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


# --- per-class verification state (RC-1) -------------------------------------


def _report(frames=0, leaks=(), plate=True):
    return bc.CoverageReport(frames, list(leaks), plate)


def test_cabin_only_pass_is_not_an_all_classes_pass():
    rep = _report(frames=5)
    assert rep.ok  # no leak found
    cls = rep.classes
    assert cls["cabin"].status is bc.Status.VERIFIED and cls["cabin"].checked == 5
    for name in ("face", "plate", "person"):
        assert cls[name].status is bc.Status.UNCHECKED
        assert cls[name].limitation  # says what it cannot see
    assert not rep.all_verified


def test_zero_sightings_is_unchecked_not_verified():
    rep = _report(frames=0)
    assert rep.ok
    assert rep.classes["cabin"].status is bc.Status.UNCHECKED
    assert not rep.all_verified


def test_leaks_fail_the_cabin_class():
    rep = _report(frames=4, leaks=[bc.Leak(1, 0.0, VEH)])
    cabin = rep.classes["cabin"]
    assert cabin.status is bc.Status.FAILED and cabin.misses == 1 and cabin.checked == 4


def test_no_plate_model_is_disclosed_in_the_basis():
    assert "no plate model" in _report(frames=2, plate=False).classes["cabin"].basis


def test_require_verified_refuses_unchecked_and_failed():
    rep = _report(frames=5)
    rep.require_verified("cabin")  # verified: fine
    try:
        rep.require_verified("cabin", "face")
    except bc.RequiredClassError as exc:
        assert set(exc.classes) == {"face"} and exc.report is rep
    else:
        raise AssertionError("an unchecked required class must refuse")
    failed = _report(frames=1, leaks=[bc.Leak(0, 0.0, VEH)])
    try:
        failed.require_verified("cabin")
    except bc.RequiredClassError:
        pass
    else:
        raise AssertionError("a failed required class must refuse")


def test_require_verified_rejects_unknown_class_names():
    try:
        _report(frames=1).require_verified("vehicle")
    except ValueError:
        pass
    else:
        raise AssertionError("a typo must not become a silent pass")


def test_to_dict_is_a_stable_json_contract():
    import json

    d = json.loads(json.dumps(_report(3, [bc.Leak(2, 0.25, VEH)]).to_dict()))
    assert d["schema"] == bc.SCHEMA_VERSION == 1
    assert d["ok"] is False and d["all_verified"] is False
    assert set(d["classes"]) == {"cabin", "face", "plate", "person"}
    assert d["classes"]["cabin"]["status"] == "failed"
    assert d["classes"]["face"]["status"] == "unchecked"
    assert d["leaks"] == [{"frame": 2, "coverage": 0.25, "box": list(VEH)}]


def test_verifier_records_whether_plates_corroborated(tmp_path, monkeypatch):
    _stub(monkeypatch, [PLATE_IN])
    mask = np.zeros((H, W), np.uint8)
    mask[20:56, 40:140] = 255
    assert _run(tmp_path, mask).plate_corroborated is True
    monkeypatch.setattr(bc, "build_plate_detector", lambda *a, **k: None)
    assert _run(tmp_path, mask).plate_corroborated is False


# --- only sampled frames are decoded to pixels --------------------------------


def _lockstep_clips(tmp_path, n=12, fps=6.0):
    """Source frame i is gray level 5*i; the mask covers the cabin on even frames
    only. A verifier that drifts a frame between the two reads the wrong level or
    the wrong mask."""
    inp, mvid = tmp_path / "in.mkv", tmp_path / "mask.mkv"
    _write(inp, [np.full((H, W, 3), 5 * i, np.uint8) for i in range(n)], fps)
    masks = []
    for i in range(n):
        m = np.zeros((H, W), np.uint8)
        if i % 2 == 0:
            m[20:56, 40:140] = 255
        masks.append(m)
    _write(mvid, masks, fps)
    return inp, mvid


def _verify(inp, mvid):
    return bc.verify_cabin_coverage(
        inp, mvid, "v.onnx", plate_model="p.onnx", sample_fps=2,
        min_height_frac=0.0, ignore_bottom_frac=0.0,
    )


def test_sampled_frames_stay_in_lockstep(tmp_path, monkeypatch):
    """Equivalence: fps 6 / sample 2 = every 3rd frame. The detector sees source
    frames 0, 3, 6, 9 and the leaks are exactly the odd ones, 3 and 9."""
    levels = []

    class _Rec:
        def detect(self, frame, conf):
            levels.append(int(frame[0, 0, 0]))
            return [VEH]

    monkeypatch.setattr(bc, "build_vehicle_detector", lambda *a, **k: _Rec())
    monkeypatch.setattr(bc, "build_plate_detector", lambda *a, **k: object())
    monkeypatch.setattr(bc, "detect_plates", lambda *a, **k: [PLATE_IN])
    rep = _verify(*_lockstep_clips(tmp_path))
    assert levels == [0, 15, 30, 45]
    assert rep.vehicle_frames == 4 and [lk.frame for lk in rep.leaks] == [3, 9]


def test_unsampled_frames_are_grabbed_not_decoded_to_pixels(tmp_path, monkeypatch):
    """A full day is 352,934 4K frames and the verifier samples 1 in 15 at 60 fps.
    Every frame is still grabbed (lockstep, no seeking); only sampled ones are
    retrieved, which skips the BGR conversion and copy of the rest."""
    _stub(monkeypatch, [PLATE_IN])
    inp, mvid = _lockstep_clips(tmp_path)
    real = cv2.VideoCapture
    counts = []

    class _Counting:
        def __init__(self, path):
            self._cap = real(path)
            self.n = {"read": 0, "grab": 0, "retrieve": 0}
            counts.append(self.n)

        def __getattr__(self, name):
            return getattr(self._cap, name)

        def read(self):
            self.n["read"] += 1
            return self._cap.read()

        def grab(self):
            self.n["grab"] += 1
            return self._cap.grab()

        def retrieve(self):
            self.n["retrieve"] += 1
            return self._cap.retrieve()

    monkeypatch.setattr(bc.cv2, "VideoCapture", _Counting)
    rep = _verify(inp, mvid)
    assert rep.vehicle_frames == 4
    for n in counts:  # source, mask
        assert n["read"] + n["retrieve"] == 4
