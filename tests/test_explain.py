"""``redactcam explain``: a read-only report of every box the mask blurs and why.

Detectors are stubbed with deterministic stand-ins on a drawn clip; nothing here
needs a model file or footage of a real person.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from redactcam import cli, detect, explain
from redactcam import mask as mask_mod
from redactcam import pipeline as pl
from redactcam import timeline as tl_mod
from redactcam.detect import ClassedBox, YoloDetector
from redactcam.models import ModelSpec

_YY, _XX = np.indices((64, 64))
_PATCH = np.where(((_XX // 8) + (_YY // 8)) % 2 == 0, 255, 40).astype(np.uint8)

KW = dict(
    sample_fps=3.0,
    face_sample_fps=3.0,
    tile_px=9999,
    track_downscale=1,
    track_lk_win=15,
    track_lk_levels=2,
    track_max_horizon_frames=60,
    pre_roll_frames=3,
)
VEHICLE = ClassedBox((40, 30, 120, 80), 7)  # a truck


class _PatchFinder:
    def __init__(self, *_a, **_k):
        pass

    def detect(self, frame_bgr, conf):
        gray = frame_bgr[:, :, 0]
        ys, xs = np.where(gray > 200)
        if not len(xs):
            return []
        return [
            (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
        ]


class _FakeVehicle:
    def detect(self, frame, conf):
        return [VEHICLE]


def _video(tmp_path, n_frames=24, fps=12.0, w=480, h=270):
    out = tmp_path / "clip.mkv"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w), 50, np.uint8)
        cx = 120 + i * 8
        frame[135 - 32 : 135 + 32, cx - 32 : cx + 32] = _PATCH
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return out


@pytest.fixture(autouse=True)
def stubbed(monkeypatch):
    monkeypatch.setattr(detect, "build_face_detector", lambda *a, **k: _PatchFinder())
    monkeypatch.setattr(detect, "build_vehicle_detector", lambda *a, **k: _FakeVehicle())
    # a plate inside the vehicle box corroborates it as a real road car
    monkeypatch.setattr(detect, "detect_plates", lambda *a, **k: [(70, 90, 40, 15)])
    monkeypatch.setattr(
        pl, "resolve_model", lambda spec, cache=None: "x" if spec.name == "vehicle" else None
    )


def _read(out: Path):
    return [json.loads(line) for line in (out / "boxes.jsonl").read_text().splitlines()]


def _mask_boxes(video, **opts):
    """What the real mask rasterises: build_timeline over detect_and_track."""
    fps, n, w, h = tl_mod.probe_video(video)
    o = {**pl.ROAD_FOOTAGE, **KW, **opts}
    o.update(face_model_path=None, plate_model_path=None, person_model_path=None)
    o["vehicle_model_path"] = "x"
    fb, decoded, *_ = pl.detect_and_track(video, **o)
    tl = tl_mod.build_timeline(
        fb,
        source_hash="h",
        source_fps=fps,
        source_frame_count=decoded,
        frame_width=w,
        frame_height=h,
        box_dilation_px=pl.ROAD_FOOTAGE_BOX_DILATION_PX,
        sample_fps=3.0,
        conf=0.4,
    )
    return {f: mask_mod.active_boxes_at(tl, f) for f in range(decoded)}


def test_explained_boxes_equal_the_mask_boxes_every_frame(tmp_path):
    video = _video(tmp_path)
    explain.explain_video(video, tmp_path / "out", detect_kwargs=KW)
    got: dict[int, list] = {}
    for r in _read(tmp_path / "out"):
        got.setdefault(r["frame"], []).append((r["x"], r["y"], r["w"], r["h"]))
    expected = {f: b for f, b in _mask_boxes(video).items() if b}
    assert got == expected
    assert expected  # the stubs really produced boxes


def test_every_kind_and_cause_is_reported(tmp_path):
    video = _video(tmp_path)
    explain.explain_video(video, tmp_path / "out", detect_kwargs=KW)
    recs = _read(tmp_path / "out")
    assert {r["kind"] for r in recs} >= {"face", "plate", "vehicle_cabin"}
    sources = {r["source"] for r in recs if r["kind"] == "face"}
    assert {"detected", "interpolated", "flow"} <= sources
    # sample frames (every 4th) are direct detections
    assert {r["source"] for r in recs if r["kind"] == "face" and r["frame"] % 4 == 0} == {
        "detected"
    }
    for r in recs:
        assert r["aspect"] == pytest.approx(r["w"] / r["h"])
        assert r["area_frac"] == pytest.approx(r["w"] * r["h"] / (480 * 270))


def test_coco_class_survives_to_the_record_without_changing_the_mask(tmp_path):
    video = _video(tmp_path)
    explain.explain_video(video, tmp_path / "out", detect_kwargs=KW)
    recs = _read(tmp_path / "out")
    cabins = [r for r in recs if r["kind"] == "vehicle_cabin"]
    assert cabins and {r["coco_class"] for r in cabins} == {7}
    assert {r["coco_class"] for r in recs if r["kind"] in ("face", "plate")} == {None}
    # the full (pre-cabin) vehicle box rides along for the aspect question
    assert cabins[0]["vehicle_box"] is not None
    # a ClassedBox is a plain tuple to everything else
    assert VEHICLE == (40, 30, 120, 80) and hash(VEHICLE) == hash((40, 30, 120, 80))


def test_yolo_decoder_attaches_the_argmax_class():
    arr = np.zeros((1, 84, 100), dtype=np.float32)
    arr[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]
    arr[0, 4 + 5, 0] = 0.9  # class 5 = bus

    class _Sess:
        def run(self, *_a):
            return [arr]

    det = YoloDetector.__new__(YoloDetector)
    det._sess, det._input_name, det._infer_size = _Sess(), "images", 640
    det._iou, det._min_aspect, det._max_aspect = 0.45, 0.0, 0.0
    det._class_filter = frozenset({2, 3, 5, 7})
    (box,) = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8), 0.4)
    assert box.coco_cls == 5


def test_summary_shares_and_vehicle_aspect_histogram():
    def rec(frame, kind, source, w, h, cls=None, vbox=None):
        return {
            "frame": frame,
            "kind": kind,
            "source": source,
            "coco_class": cls,
            "x": 0,
            "y": 0,
            "w": w,
            "h": h,
            "aspect": w / h,
            "area_frac": w * h / (100 * 100),
            "vehicle_box": vbox,
        }

    recs = [
        rec(0, "face", "detected", 10, 10),  # 0.01
        rec(0, "vehicle_cabin", "detected", 50, 20, 7, (0, 0, 50, 40)),  # 0.10, aspect 1.25
        rec(1, "face", "flow", 10, 10),  # 0.01
        rec(1, "vehicle_cabin", "detected", 60, 20, 7, (0, 0, 60, 20)),  # aspect 3.0
        rec(1, "vehicle_cabin", "coast", 50, 20, 2, (0, 0, 50, 40)),  # not a detection
        rec(1, "vehicle_cabin", "detected", 40, 20, 2, (0, 0, 40, 40)),  # car, aspect 1.0
    ]
    s = explain.summarize(recs, frames=4, width=100, height=100)
    assert s["frames_analysed"] == 4 and s["boxes"] == 6
    # mean over the 4 frames of summed box area fraction
    assert s["area_share"]["face"]["detected"] == pytest.approx(0.01 / 4)
    assert s["area_share"]["vehicle_cabin"]["detected"] == pytest.approx((0.10 + 0.12 + 0.08) / 4)
    assert s["area_share"]["vehicle_cabin"]["coast"] == pytest.approx(0.10 / 4)
    assert sum(
        v for kinds in s["share_of_blurred"].values() for v in kinds.values()
    ) == pytest.approx(1.0)
    hist = s["vehicle_aspect_histogram"]
    assert set(hist) == {"7_truck", "5_bus", "2_car", "3_motorcycle"}
    assert sum(hist["7_truck"]["counts"]) == 2 and sum(hist["2_car"]["counts"]) == 1
    assert sum(hist["5_bus"]["counts"]) == 0
    edges = hist["7_truck"]["edges"]
    counts = dict(zip(edges, hist["7_truck"]["counts"], strict=False))
    assert counts[1.25] == 1 and counts[3.0] == 1  # bins are [edge, next edge)


def test_summary_json_and_nothing_else_is_written(tmp_path):
    video = _video(tmp_path)
    before = video.read_bytes()
    out = tmp_path / "out"
    explain.explain_video(video, out, detect_kwargs=KW, sheet=3)
    assert video.read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["boxes.jsonl", "sheet.png", "summary.json"]
    s = json.loads((out / "summary.json").read_text())
    assert s["frames_analysed"] == 24 and "area_share" in s
    img = cv2.imread(str(out / "sheet.png"))
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0
    assert not list(tmp_path.rglob("*mask*"))


def test_start_and_duration_window_the_analysis(tmp_path):
    video = _video(tmp_path)  # 12 fps
    explain.explain_video(video, tmp_path / "out", detect_kwargs=KW, start_s=0.5, duration_s=0.5)
    frames = {r["frame"] for r in _read(tmp_path / "out")}
    assert frames and min(frames) >= 6 and max(frames) < 12


def test_cli_routes_explain(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(video, out, **kw):
        seen.update(video=video, out=out, **kw)
        return {"frames_analysed": 1, "boxes": 0}

    monkeypatch.setattr(explain, "explain_video", fake)
    rc = cli.main(
        ["explain", "a.mp4", "--out", str(tmp_path), "--sheet", "4", "--start", "1.5",
         "--duration", "2", "--model", "vehicle=/m/v.onnx"]
    )  # fmt: skip
    assert rc == 0
    assert seen["sheet"] == 4 and seen["start_s"] == 1.5 and seen["duration_s"] == 2.0
    assert seen["models"]["vehicle"] == ModelSpec(name="vehicle", path="/m/v.onnx")
    assert Path(seen["out"]) == tmp_path


def test_frames_before_start_from_pre_roll_are_neither_written_nor_counted(tmp_path, monkeypatch):
    from redactcam.track import BoxEntry

    video = _video(tmp_path)  # 12 fps; --start 1.0 s is frame 12, --duration 0.5 s ends at 18
    box = (50, 50, 40, 40)

    def fake(path, start_frame=0, max_frames=None, trace=None, **_o):
        frames = range(start_frame - 3, start_frame + max_frames)  # pre-roll reaches back
        trace["entries"] = {f: [BoxEntry(box, 1, "face", "pre_roll")] for f in frames}
        return {f: [box] for f in frames}, start_frame + max_frames, 0, 0

    monkeypatch.setattr(explain, "detect_and_track", fake)
    s = explain.explain_video(video, tmp_path / "out", start_s=1.0, duration_s=0.5)
    frames = [r["frame"] for r in _read(tmp_path / "out")]
    assert sorted(set(frames)) == list(range(12, 18))
    assert s["boxes"] == len(frames) == 6 and s["frames_analysed"] == 6


def test_checkpoint_with_a_window_is_refused(tmp_path):
    video = _video(tmp_path)
    for kw in ({"start_frame": 4}, {"max_frames": 5}):
        with pytest.raises(ValueError, match="checkpoint_path"):
            pl.detect_and_track(video, checkpoint_path=tmp_path / "ck.json", **kw)


def test_cli_dilation_defaults_to_the_mask_default_and_is_passed_through(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        explain,
        "explain_video",
        lambda *a, **kw: seen.append(kw) or {"frames_analysed": 0, "boxes": 0},
    )
    assert cli.main(["explain", "a.mp4", "--out", str(tmp_path)]) == 0
    assert cli.main(["explain", "a.mp4", "--out", str(tmp_path), "--dilation", "20"]) == 0
    assert seen[0]["box_dilation_px"] == pl.ROAD_FOOTAGE_BOX_DILATION_PX
    assert seen[1]["box_dilation_px"] == 20
