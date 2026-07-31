"""Tests for the detector — faces ∪ plates ∪ vehicles ∪ people.

Every fixture here is generated: synthetic frames drawn from numpy primitives,
and detectors stubbed with deterministic stand-ins. Nothing in this suite needs
a model file, a network connection, or a photograph of a real person.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from redactcam import detect  # noqa: E402
from redactcam.detect import (  # noqa: E402
    FrameRegions,
    ModelUnavailableError,
    YoloDetector,
    build_face_detector,
    dedupe,
    detect_and_track,
    detect_frame,
    detect_regions,
    detect_tiled,
    filter_vehicles,
    frame_tiles,
    iou,
    letterbox,
    next_boost_run,
)

CONF = 0.4
INFER_SHAPE = (256, 256)

# A synthetic "object": bright (255) squares give an easy >200 threshold bbox,
# dark (40) squares give Lucas-Kanade strong internal corners to track.
_YY, _XX = np.indices((64, 64))
_PATCH = np.where(((_XX // 8) + (_YY // 8)) % 2 == 0, 255, 40).astype(np.uint8)


class _PatchFinder:
    """Stand-in face detector: the bounding box of the bright patch.

    Deterministic, model-free, and — importantly — it lets the end-to-end video
    paths be exercised on drawn shapes rather than on footage of real people.
    """

    def __init__(self, *_a, **_k):
        pass

    def detect(self, frame_bgr, conf):
        gray = frame_bgr[:, :, 0] if frame_bgr.ndim == 3 else frame_bgr
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


@pytest.fixture
def patch_frame():
    """A 256x256 BGR frame with the synthetic patch in the upper-centre."""
    frame = np.full((256, 256), 50, np.uint8)
    frame[30:94, 96:160] = _PATCH
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


@pytest.fixture(autouse=True)
def _stub_face_detector(monkeypatch):
    """No real face model anywhere in this suite."""
    monkeypatch.setattr(detect, "build_face_detector", lambda *a, **k: _PatchFinder())


def _make_video(tmp_path, img, n_frames=8, fps=4.0):
    """Write a tiny lossless video that repeats one frame; returns its path."""
    h, w = img.shape[:2]
    out = tmp_path / "clip.mkv"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()
    return out


class TestBuildFaceDetector:
    """A face detector that silently finds nothing is the one failure this
    library must not have, so the no-model-no-fallback case raises."""

    def test_resolvable_model_builds_yolo(self, tmp_path, monkeypatch):
        seen = {}

        def _fake_yolo(path, infer_size, iou_thresh, min_aspect, threads, **kw):
            seen["path"] = path
            return "yolo"

        monkeypatch.setattr(detect, "YoloDetector", _fake_yolo)
        model = tmp_path / "face.onnx"
        model.write_bytes(b"x")
        assert build_face_detector(model, INFER_SHAPE, 640, 0.45, 0) == "yolo"
        assert seen["path"] == model  # the resolved model, not the fallback

    def test_missing_model_falls_back_to_centerface(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(detect, "CenterFaceDetector", lambda shape: f"centerface{shape}")
        with caplog.at_level("WARNING"):
            got = build_face_detector(None, INFER_SHAPE, 640, 0.45, 0)
        assert got == f"centerface{INFER_SHAPE}"
        assert "falling back to CenterFace" in caplog.text

    def test_no_model_and_no_fallback_raises(self, monkeypatch):
        def _unavailable(_shape):
            raise ModelUnavailableError("extra not installed")

        monkeypatch.setattr(detect, "CenterFaceDetector", _unavailable)
        with pytest.raises(ModelUnavailableError):
            build_face_detector(None, INFER_SHAPE, 640, 0.45, 0)


class TestDetectRegions:
    def test_e2e_object_found_missing_plate_model_not_fatal(self, patch_frame, tmp_path, caplog):
        """Sampled frames come back as FrameRegions; the synthetic object is
        found; a non-existent plate model disables plates with a warning rather
        than raising — the caller decides whether that is fatal."""
        video = _make_video(tmp_path, patch_frame)
        with caplog.at_level("WARNING"):
            result = detect_regions(video, plate_model_path=tmp_path / "nope.onnx")

        assert result and all(isinstance(fr, FrameRegions) for fr in result)
        assert len(result) == 4  # fps 4 / sample_fps 2 default → 8 frames / 2
        assert any(fr.faces for fr in result)
        for fr in result:
            assert fr.boxes == fr.faces + fr.plates
            assert fr.plates == []  # no plate model
        assert "plate detection disabled" in caplog.text

    def test_deterministic_across_runs(self, patch_frame, tmp_path):
        video = _make_video(tmp_path, patch_frame)
        a = detect_regions(video)
        b = detect_regions(video)
        assert [(f.frame_idx, f.faces) for f in a] == [(f.frame_idx, f.faces) for f in b]

    def test_unopenable_video_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not open"):
            detect_regions(tmp_path / "missing.mp4")


class _FakeSession:
    """Stand-in onnxruntime session returning a crafted YOLO output tensor."""

    def __init__(self, output):
        self._output = np.asarray(output, dtype=np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _names, _feed):
        return [self._output]


def _plate_detector(
    output, infer_size=640, iou=0.45, min_aspect=0.0, max_aspect=0.0, class_filter=None
):
    det = YoloDetector.__new__(YoloDetector)  # bypass __init__ (no real model)
    det._sess = _FakeSession(output)
    det._input_name = "images"
    det._infer_size = infer_size
    det._iou = iou
    det._min_aspect = min_aspect
    det._max_aspect = max_aspect
    det._class_filter = class_filter
    return det


class TestSessionOptions:
    """The session thread cap — 0 = all cores (the default, since the YOLO
    forward is the bottleneck), 1 = bit-reproducible single-thread."""

    def test_thread_count_passthrough(self):
        assert detect._session_options(0).intra_op_num_threads == 0
        assert detect._session_options(1).intra_op_num_threads == 1
        assert detect._session_options(4).intra_op_num_threads == 4

    def test_inter_op_stays_sequential(self):
        import onnxruntime as ort

        opts = detect._session_options(0)
        assert opts.inter_op_num_threads == 1
        assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


class TestProviders:
    """CUDA is used when the GPU build exposes it; CPU stays the always-present
    fallback, so a CPU-only host is unaffected until onnxruntime-gpu is added."""

    def test_cuda_preferred_when_available(self):
        avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        assert detect._providers(avail) == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_cpu_only_when_no_cuda(self):
        # The stock CPU build → CPU only.
        assert detect._providers(
            ["AzureExecutionProvider", "CPUExecutionProvider"]
        ) == ["CPUExecutionProvider"]


class TestPlateDecoder:
    """YOLOv11 ``(1,5,N)`` decode — letterbox un-projection + NMS, no real model.

    Geometry fixture: native 1280×720 letterboxed to 640 → scale 0.5, pad (0,140).
    A plate at native ``(300,400,100,30)`` lands at letterbox center ``(175,347.5)``
    size ``(50,15)``; the decode must invert that exactly.
    """

    FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
    PLATE = [175.0, 347.5, 50.0, 15.0, 0.9]  # cx, cy, w, h, score (letterbox space)

    def _yolo(self, n=8, anchor0=None, anchor1=None):
        arr = np.zeros((1, 5, n), dtype=np.float32)
        arr[0, 4, :] = 0.01  # baseline: every anchor below conf
        if anchor0 is not None:
            arr[0, :, 0] = anchor0
        if anchor1 is not None:
            arr[0, :, 1] = anchor1
        return arr

    def test_decodes_and_unletterboxes(self):
        det = _plate_detector(self._yolo(anchor0=self.PLATE))
        assert det.detect(self.FRAME, conf=0.4) == [(300, 400, 100, 30)]

    def test_conf_filters_low_score(self):
        det = _plate_detector(self._yolo(anchor0=[175.0, 347.5, 50.0, 15.0, 0.2]))
        assert det.detect(self.FRAME, conf=0.4) == []

    def test_nms_dedups_overlapping(self):
        det = _plate_detector(
            self._yolo(anchor0=self.PLATE, anchor1=[176.0, 348.0, 50.0, 15.0, 0.85])
        )
        assert len(det.detect(self.FRAME, conf=0.4)) == 1

    def test_accepts_transposed_layout(self):
        arr = np.zeros((1, 8, 5), dtype=np.float32)  # (1, N, 5) rows-are-anchors
        arr[0, 0] = self.PLATE
        assert _plate_detector(arr).detect(self.FRAME, conf=0.4) == [(300, 400, 100, 30)]

    def test_letterbox_geometry(self):
        canvas, scale, px, py = letterbox(self.FRAME, 640)
        assert canvas.shape == (640, 640, 3) and scale == 0.5 and px == 0 and py == 140


class TestPlateAspectFilter:
    """The aspect band drops squarish road-sign FPs (below min) and over-elongated
    shoreline/railing structures (above max), keeping wide real plates between.
    Same 1280×720→640 letterbox geometry as the decoder tests (scale 0.5,
    pad (0,140)); aspect is computed on the model box, pre-clamp."""

    FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _one(self, cx, cy, w, h, score=0.9):
        arr = np.zeros((1, 5, 8), dtype=np.float32)
        arr[0, 4, :] = 0.01  # every anchor below conf...
        arr[0, :, 0] = [cx, cy, w, h, score]  # ...except anchor 0
        return arr

    def test_wide_plate_survives_gate(self):
        # native (300,400,100,30) → aspect 3.33 ≥ 2.0 (a real plate)
        det = _plate_detector(self._one(175.0, 347.5, 50.0, 15.0), min_aspect=2.0)
        assert det.detect(self.FRAME, conf=0.4) == [(300, 400, 100, 30)]

    def test_squarish_sign_dropped(self):
        # native (300,400,80,50) → aspect 1.6 < 2.0 (a road sign the model misfires on)
        det = _plate_detector(self._one(170.0, 352.5, 40.0, 25.0), min_aspect=2.0)
        assert det.detect(self.FRAME, conf=0.4) == []

    def test_gate_off_keeps_squarish(self):
        # min_aspect 0 → the same box survives, proving the gate (not conf) drops it
        det = _plate_detector(self._one(170.0, 352.5, 40.0, 25.0), min_aspect=0.0)
        assert det.detect(self.FRAME, conf=0.4) == [(300, 400, 80, 50)]

    def test_wide_headon_plate_survives_max_gate(self):
        # native (300,400,200,50) → aspect 4.0 (a head-on plate) ≤ 6.0 max
        det = _plate_detector(self._one(200.0, 352.5, 100.0, 25.0), min_aspect=2.0, max_aspect=6.0)
        assert det.detect(self.FRAME, conf=0.4) == [(300, 400, 200, 50)]

    def test_elongated_structure_dropped_by_max_gate(self):
        # native (0,400,1200,100) → aspect 12.0 (a far-bank shoreline/railing band
        # outscoring conf, like the real ~13:1 FP) > 6.0 max → dropped
        det = _plate_detector(self._one(300.0, 365.0, 600.0, 50.0), min_aspect=2.0, max_aspect=6.0)
        assert det.detect(self.FRAME, conf=0.4) == []

    def test_max_gate_off_keeps_elongated(self):
        # max_aspect 0 → the same band survives, proving the gate (not conf) drops it
        det = _plate_detector(self._one(300.0, 365.0, 600.0, 50.0), min_aspect=2.0)
        assert det.detect(self.FRAME, conf=0.4) == [(0, 400, 1200, 100)]


def _make_moving_video(tmp_path, n_frames=24, fps=12.0, x0=120, dx=8, cy=135, w=480, h=270):
    """Lossless (FFV1) clip of ``_PATCH`` translating right ``dx`` px/frame, so a
    sparse detector + dense flow can be checked against a known trajectory."""
    out = tmp_path / "moving.mkv"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    ph, pw = _PATCH.shape
    for i in range(n_frames):
        frame = np.full((h, w), 50, np.uint8)
        cx = x0 + i * dx
        px, py = cx - pw // 2, cy - ph // 2
        frame[py : py + ph, px : px + pw] = _PATCH
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return out


class TestFilterVehicles:
    """Worth-blurring filter: keep cars with a discernible cabin, drop tiny
    distant cars (no identifiable occupant) and the camera vehicle's own bonnet
    in the bottom strip."""

    H = 2160

    def test_drops_tiny_and_bonnet_keeps_real(self):
        real = (1800, 1300, 600, 400)  # mid-frame, 400 px tall → keep
        tiny = (1600, 1390, 100, 65)  # 65 px tall < 0.06*2160 → drop (no occupant)
        bonnet = (15, 1950, 1766, 200)  # top 1950 > 0.88*2160 → drop (camera vehicle)
        assert filter_vehicles([real, tiny, bonnet], self.H, 0.06, 0.12) == [real]

    def test_zero_thresholds_keep_all(self):
        boxes = [(0, 2000, 100, 50), (10, 10, 20, 20)]
        assert filter_vehicles(boxes, self.H, 0.0, 0.0) == boxes


class TestMultiClassDecoder:
    """COCO ``(1, 84, N)`` decode (vehicle model): score = max class prob, class =
    argmax, kept only if in ``class_filter``. Same letterbox geometry as the plate
    decoder (1280×720 → 640, scale 0.5, pad (0, 140))."""

    FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _coco_output(self):
        # (1, 84, N) with N > 84 so the (4+nc, N) → rows transpose triggers (the real
        # model is 8400 anchors); all but two anchors score 0 (dropped by conf).
        arr = np.zeros((1, 84, 100), dtype=np.float32)  # 4 box + 80 class scores
        arr[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]  # anchor 0
        arr[0, 4 + 2, 0] = 0.9  # class 2 = car
        arr[0, :4, 1] = [200.0, 200.0, 60.0, 120.0]  # anchor 1
        arr[0, 4 + 0, 1] = 0.9  # class 0 = person — must be filtered out
        return arr

    def test_keeps_only_vehicle_classes(self):
        det = _plate_detector(self._coco_output(), class_filter=frozenset({2, 3, 5, 7}))
        boxes = det.detect(self.FRAME, 0.4)
        assert len(boxes) == 1  # the car; the person is dropped by the class filter
        _, _, w, h = boxes[0]
        assert abs(w - 200) <= 2 and abs(h - 160) <= 2  # un-letterboxed (÷ scale 0.5)

    def test_no_class_filter_keeps_all_classes(self):
        det = _plate_detector(self._coco_output(), class_filter=None)
        assert len(det.detect(self.FRAME, 0.4)) == 2  # no filter → car + person both kept


class TestDetectAndTrack:
    """The dense flow path: sparse detections propagated to per-frame boxes."""

    def test_densifies_and_follows_motion(self, tmp_path, monkeypatch):
        video = _make_moving_video(tmp_path)
        frame_boxes, n_frames, n_face, n_plate = detect_and_track(
            video,
            sample_fps=3.0,  # fps 12 / 3 → detect every 4th frame
            plate_model_path=None,
            tile_px=9999,  # single tile (small synthetic frame)
            track_downscale=1,
            track_lk_win=15,
            track_lk_levels=2,
            track_max_horizon_frames=60,
        )
        assert n_frames == 24 and len(frame_boxes) == 24  # one entry per decoded frame
        assert n_face == 6 and n_plate == 0  # detected only on the 6 sample frames
        # A NON-sample frame still carries a box → flow propagated it, not frozen.
        assert frame_boxes[6]
        centers = [b[0][0] + b[0][2] / 2 for f, b in sorted(frame_boxes.items()) if b]
        assert centers == sorted(centers)  # box tracks the rightward motion
        assert max(centers) - min(centers) > 100  # moved with the object, not frozen
        assert len({round(c) for c in centers}) > 6  # many positions, not one held box

    def test_plate_boost_densifies_plates_not_faces(self, tmp_path, monkeypatch):
        """Occupant-face regression guard: the boost re-detects only PLATES every
        frame, never faces — face detection through a moving windscreen is
        unreliable at any confidence, so occupants are covered by the geometric
        cabin blur instead of chased per-face. With a plate arming the boost,
        plate detection densifies while faces stay on their sparse cadence."""
        video = _make_moving_video(tmp_path)  # 24 frames @ 12 fps
        kw = dict(
            sample_fps=3.0,
            face_sample_fps=3.0,
            tile_px=9999,
            track_downscale=1,
            track_lk_win=15,
            track_lk_levels=2,
            track_max_horizon_frames=60,
            plate_boost_speed_px=8,
        )
        # Baseline: no plate → faces only on the sparse cadence (24 frames / step 4).
        _, _, nf0, np0 = detect_and_track(video, plate_model_path=None, **kw)
        assert nf0 == 6 and np0 == 0
        # A plate present every detection arms + holds the boost → PLATES densify.
        monkeypatch.setattr(detect, "detect_plates", lambda *a, **k: [(5, 5, 40, 16)])
        _, _, nf1, np1 = detect_and_track(video, plate_model_path=None, **kw)
        assert np1 > nf0  # plate detection densified through the boost window
        assert nf1 == nf0  # faces stayed sparse — NO densification

    def test_vehicle_cabin_needs_plate_corroboration(self, tmp_path, monkeypatch):
        """A detected vehicle emits the TOP cabin_frac of its box ONLY when a
        licence plate inside it corroborates a real car; a plate-less vehicle (a
        roadside sign, billboard or parked caravan that COCO misreads) is left
        readable. cabin_frac 0 builds no vehicle detector at all."""

        class _FakeVehicle:
            def detect(self, frame, conf):
                return [(40, 30, 120, 80)]

        monkeypatch.setattr(
            detect, "build_vehicle_detector", lambda *a, **k: _FakeVehicle()
        )
        video = _make_moving_video(tmp_path)
        kw = dict(
            sample_fps=3.0,
            plate_model_path=None,
            tile_px=9999,
            track_downscale=1,
            track_lk_win=15,
            track_lk_levels=2,
            track_max_horizon_frames=60,
        )
        cabin = (40, 30, 120, 40)  # top 50% of the (40,30,120,80) vehicle box
        # No plate inside the vehicle → NOT corroborated → no cabin emitted (sign/billboard).
        monkeypatch.setattr(detect, "detect_plates", lambda *a, **k: [])
        fb_np, *_ = detect_and_track(video, track_cabin_frac=0.5, vehicle_model_path="x", **kw)
        assert not any(b == cabin for boxes in fb_np.values() for b in boxes)
        # A plate inside the vehicle box → corroborated → cabin emitted (real car).
        monkeypatch.setattr(detect, "detect_plates", lambda *a, **k: [(70, 90, 40, 15)])
        fb, *_ = detect_and_track(video, track_cabin_frac=0.5, vehicle_model_path="x", **kw)
        assert cabin in fb[0]
        # cabin_frac 0 → no vehicle detector built → no cabin box anywhere
        fb0, *_ = detect_and_track(video, track_cabin_frac=0.0, vehicle_model_path="x", **kw)
        assert not any(b == cabin for boxes in fb0.values() for b in boxes)

    def test_person_blurred_as_whole_box_no_corroboration(self, tmp_path, monkeypatch):
        """A COCO person — someone standing beside the camera, a pedestrian — is
        blurred as a WHOLE box, not just a face, and needs no plate corroboration
        (unlike the vehicle cabin). A face blur alone leaves body, clothing and
        pose, which identify a person nearly as well as their face does."""
        person = (300, 40, 60, 180)  # a tall person box

        class _FakePerson:
            def detect(self, frame, conf):
                return [person]

        monkeypatch.setattr(detect, "build_person_detector", lambda *a, **k: _FakePerson())
        video = _make_moving_video(tmp_path)
        kw = dict(
            sample_fps=3.0,
            plate_model_path=None,
            tile_px=9999,
            track_downscale=1,
            track_lk_win=15,
            track_lk_levels=2,
            track_max_horizon_frames=60,
            # zero the margins so the emitted box equals the tight detector box exactly
            track_motion_gain=0.0,
            track_face_margin_frac=0.0,
            track_face_motion_gain=0.0,
        )
        # person blur OFF → no person detector built → the person is never emitted
        fb_off, *_ = detect_and_track(video, track_person=False, **kw)
        assert not any(b == person for boxes in fb_off.values() for b in boxes)
        # person blur ON → the whole person box enters the dense mask on its detect frame
        fb, *_ = detect_and_track(
            video, track_person=True, person_model_path="x", person_min_height_frac=0.0, **kw
        )
        assert person in fb[0]

    def test_unopenable_video_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Could not open"):
            detect_and_track(tmp_path / "missing.mp4")


class _BoxDetector:
    """Fake detector returning one fixed tile-local box per crop."""

    def __init__(self, box):
        self._box = box

    def detect(self, crop, conf):
        return [self._box]


class TestTiling:
    def test_no_tiling_when_tile_ge_frame(self):
        assert frame_tiles(1080, 720, 1280, 0.15) == [(0, 0, 1080, 720)]

    def test_4k_splits_into_3x2_overlapping_and_covers(self):
        tiles = frame_tiles(3840, 2160, 1280, 0.15)
        assert len(tiles) == 6
        # full coverage and in-bounds
        assert min(t[0] for t in tiles) == 0 and min(t[1] for t in tiles) == 0
        assert max(t[2] for t in tiles) == 3840 and max(t[3] for t in tiles) == 2160
        # adjacent columns overlap (tile 1 starts before tile 0 ends)
        assert tiles[1][0] < tiles[0][2]

    def test_iou(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
        assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(5 / 15)

    def test_dedupe_drops_overlap_keeps_distinct_order_stable(self):
        a, near_a, b = (0, 0, 10, 10), (1, 0, 10, 10), (100, 100, 10, 10)
        assert dedupe([a, near_a, b]) == [a, b]

    def test_detect_tiled_offsets_into_full_frame(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tiles = frame_tiles(3840, 2160, 1280, 0.15)
        boxes = detect_tiled(frame, _BoxDetector((10, 20, 5, 5)), 0.4, tiles)
        # one box per tile, each offset by its tile origin
        assert (10, 20, 5, 5) in boxes  # tile (0,0)
        assert (tiles[1][0] + 10, 20, 5, 5) in boxes  # tile to its right
        assert len(boxes) == len(tiles)

    def test_detect_tiled_single_tile_passthrough(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        boxes = detect_tiled(frame, _BoxDetector((1, 2, 3, 4)), 0.4, [(0, 0, 256, 256)])
        assert boxes == [(1, 2, 3, 4)]


class TestPlateAreaGate:
    """A plate box over ``plate_max_area_frac`` of the FULL frame is dropped (a
    stone wall / verge / abutment FP the aspect band can't catch); a real-plate-
    size box survives, and faces are never area-gated."""

    FRAME = np.zeros((1000, 1000, 3), dtype=np.uint8)  # 1e6 px
    TILES = [(0, 0, 1000, 1000)]  # single tile → detector box passes straight through

    def test_oversized_plate_dropped_face_kept(self):
        face = _BoxDetector((5, 5, 20, 20))
        big_plate = _BoxDetector((10, 10, 300, 300))  # 9% of frame → a structure FP
        faces, plates = detect_frame(self.FRAME, face, big_plate, 0.4, 0.4, self.TILES, 0.01)
        assert plates == []  # 9% > 1% → dropped
        assert faces == [(5, 5, 20, 20)]  # faces never area-gated

    def test_real_size_plate_survives(self):
        small_plate = _BoxDetector((10, 10, 50, 50))  # 0.25% → a real plate
        _, plates = detect_frame(
            self.FRAME, _BoxDetector((0, 0, 1, 1)), small_plate, 0.4, 0.4, self.TILES, 0.01
        )
        assert plates == [(10, 10, 50, 50)]

    def test_gate_off_keeps_oversized(self):
        big_plate = _BoxDetector((10, 10, 300, 300))
        _, plates = detect_frame(
            self.FRAME, _BoxDetector((0, 0, 1, 1)), big_plate, 0.4, 0.4, self.TILES, 0.0
        )
        assert plates == [(10, 10, 300, 300)]  # frac 0 → no area gate


class _ScoredDetector:
    """Fake detector that RESPECTS the conf threshold (unlike _BoxDetector) — so a
    box whose score sits between the face and plate confs is kept for faces, dropped
    for plates."""

    def __init__(self, box, score):
        self._box, self._score = box, score

    def detect(self, crop, conf):
        return [self._box] if self._score >= conf else []


class TestDecoupledConf:
    """Faces and plates take independent confidence thresholds: a small clear face
    scores below a plate's usable conf, so face recall is pushed without lowering the
    plate conf (which would resurface squarish road-sign FPs)."""

    FRAME = np.zeros((256, 256, 3), dtype=np.uint8)
    TILES = [(0, 0, 256, 256)]

    def test_low_score_object_kept_as_face_dropped_as_plate(self):
        # an object scoring 0.20: above face_conf 0.15, below plate_conf 0.40
        face = _ScoredDetector((5, 5, 20, 20), 0.20)
        plate = _ScoredDetector((50, 50, 40, 15), 0.20)
        faces, plates = detect_frame(self.FRAME, face, plate, 0.15, 0.40, self.TILES, 0.0)
        assert faces == [(5, 5, 20, 20)]  # face recall pushed to 0.15
        assert plates == []  # plate threshold unchanged → low-score box stays out

    def test_face_conf_defaults_do_not_drop_strong_face(self):
        face = _ScoredDetector((5, 5, 20, 20), 0.50)
        faces, _ = detect_frame(self.FRAME, face, None, 0.15, 0.40, self.TILES, 0.0)
        assert faces == [(5, 5, 20, 20)]


class TestPlateBoostCadence:
    """``next_boost_run`` arms per-frame plate re-detection while a plate is fast and
    keeps it on ``max_run`` frames past the last fast reading, then self-clears."""

    def test_arms_to_max_run_when_fast(self):
        assert next_boost_run(0, 30.0, 25, 6) == 6  # cold start arms
        assert next_boost_run(2, 30.0, 25, 6) == 6  # re-arm overrides the decay

    def test_decays_when_slow(self):
        assert next_boost_run(6, 10.0, 25, 6) == 5
        assert next_boost_run(1, 0.0, 25, 6) == 0
        assert next_boost_run(0, 0.0, 25, 6) == 0  # never goes negative

    def test_disabled_threshold_only_decays(self):
        assert next_boost_run(6, 999.0, 0, 6) == 5  # threshold 0 → boost never arms

    def test_threshold_boundary_inclusive(self):
        assert next_boost_run(0, 25.0, 25, 6) == 6  # speed == threshold arms

    def test_recently_seen_arms_even_when_slow(self):
        # a small plate retires with no fast track, so DETECTION itself re-arms the boost
        assert next_boost_run(0, 0.0, 8, 6, plate_seen=True) == 6
        assert next_boost_run(0, 0.0, 8, 6, plate_seen=False) == 0  # neither trigger
        assert next_boost_run(3, 0.0, 0, 6, plate_seen=True) == 2  # disabled → only decays
