"""The cheaper mask render must decode to exactly the pixels the old one produced.

``_ref_render_mask_frame`` and ``_ref_materialize`` are verbatim copies of the
pre-optimisation code (full-frame blur on every frame, default FFV1), so every
comparison below is against the real previous behaviour, not a restatement of it.
"""

import random
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from redactcam import mask as bm
from redactcam.timeline import BlurTimeline

W, H = 48, 40
N = 36
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg required"
)


@pytest.fixture(autouse=True)
def _single_threaded_cv2():
    """OpenCV 4.13's multithreaded GaussianBlur is NONDETERMINISTIC on tiny frames:
    blurring an all-zero 54x14 image with a 5x5 kernel returned non-zero pixels in ~30%
    of calls at 12 threads (identical input, different output; 4K frames were stable).
    A byte-identity test needs a deterministic reference, so pin one thread here."""
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(old)


def _ref_render_mask_frame(width, height, boxes, feather_px=0):
    frame = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, thickness=-1)
    if feather_px > 0:
        k = feather_px * 2 + 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return frame


def _fixture_boxes():
    """Empty frames, interior, every edge and corner, overlapping, off-frame, tiny."""
    rng = random.Random(7)
    boxes = {}
    for i in range(N):
        if i % 3 == 0:
            continue  # empty frame
        kind = i % 6
        if kind == 1:
            boxes[i] = [(10, 8, 12, 10)]
        elif kind == 2:
            boxes[i] = [(0, 0, 9, 9), (W - 10, H - 10, 9, 9)]  # touches corners
        elif kind == 4:
            boxes[i] = [(5, 5, 20, 20), (15, 12, 20, 20), (18, 14, 4, 4)]  # overlapping
        else:
            boxes[i] = [
                (
                    rng.randint(-6, W - 3),
                    rng.randint(-6, H - 3),
                    rng.randint(0, 25),
                    rng.randint(0, 25),
                )
                for _ in range(rng.randint(1, 3))
            ]
    boxes[N - 1] = [(W - 1, H - 1, 5, 5), (-4, -4, 5, 5)]  # single-pixel edge, off-frame
    return boxes


def _timeline(boxes):
    return BlurTimeline(
        source_hash="h",
        sample_fps=8.0,
        conf=0.4,
        box_dilation_px=4,
        source_fps=10.0,
        source_frame_count=N,
        frame_width=W,
        frame_height=H,
        frame_boxes=boxes,
    )


def _decode(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=True,
    ).stdout
    assert len(raw) == W * H * N
    return raw


def _ref_materialize(timeline, out, feather_px):
    """The old encoder call: whole-frame blur on every frame, no slice flags."""
    cmd = [
        "ffmpeg", "-y", "-nostats", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", "10.0", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray", str(out),
    ]  # fmt: skip
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        boxes = timeline.frame_boxes.get(i, [])
        p.stdin.write(_ref_render_mask_frame(W, H, boxes, feather_px).tobytes())
    p.stdin.close()
    assert p.wait() == 0


@pytest.mark.parametrize("feather", [0, 1, 2, 3, 8, 25, 60])
def test_render_mask_frame_equals_reference_on_every_fixture_frame(feather):
    for i, boxes in _fixture_boxes().items():
        new = bm.render_mask_frame(W, H, boxes, feather)
        ref = _ref_render_mask_frame(W, H, boxes, feather)
        assert new.dtype == ref.dtype and new.shape == ref.shape
        assert np.array_equal(new, ref), f"frame {i} feather {feather} boxes {boxes}"


def test_render_mask_frame_equals_reference_fuzz():
    rng = random.Random(11)
    for _ in range(600):
        w, h = rng.randint(1, 70), rng.randint(1, 60)
        boxes = [
            (
                rng.randint(-20, w + 10),
                rng.randint(-20, h + 10),
                rng.randint(0, 40),
                rng.randint(0, 40),
            )
            for _ in range(rng.randint(0, 4))
        ]
        f = rng.choice([0, 1, 2, 4, 8, 13])
        assert np.array_equal(
            bm.render_mask_frame(w, h, boxes, f), _ref_render_mask_frame(w, h, boxes, f)
        ), (w, h, boxes, f)


def test_empty_frame_runs_no_blur(monkeypatch):
    calls = []
    real = cv2.GaussianBlur
    monkeypatch.setattr(
        cv2, "GaussianBlur", lambda *a, **k: calls.append(a[0].shape) or real(*a, **k)
    )
    frame = bm.render_mask_frame(W, H, [], 8)
    assert not frame.any() and frame.shape == (H, W)
    assert calls == []


def test_blur_runs_only_on_the_box_region(monkeypatch):
    calls = []
    real = cv2.GaussianBlur
    monkeypatch.setattr(
        cv2, "GaussianBlur", lambda *a, **k: calls.append(a[0].shape) or real(*a, **k)
    )
    big_w, big_h = 400, 300
    bm.render_mask_frame(big_w, big_h, [(100, 100, 10, 10)], 8)
    assert len(calls) == 1
    ch, cw = calls[0]
    assert ch < big_h // 2 and cw < big_w // 2


def test_every_box_pixel_stays_white_and_nothing_shrinks():
    """Privacy invariant: the core of each box is full white, as before."""
    for boxes in _fixture_boxes().values():
        frame = bm.render_mask_frame(W, H, boxes, 0)
        for x, y, w, h in boxes:
            region = frame[max(0, y) : max(0, y + h + 1), max(0, x) : max(0, x + w + 1)]
            assert (region == 255).all()


@needs_ffmpeg
@pytest.mark.parametrize("feather", [0, 2, 8])
def test_decoded_video_is_byte_identical_to_the_old_render(tmp_path, feather):
    tl = _timeline(_fixture_boxes())
    new = bm.materialize_mask_video(tl, tmp_path / "new.mkv", feather_px=feather)
    _ref_materialize(tl, tmp_path / "ref.mkv", feather)
    assert _decode(new) == _decode(tmp_path / "ref.mkv")
    assert bm._count_frames(new) == N


@needs_ffmpeg
def test_all_empty_timeline_is_all_black_and_exact_length(tmp_path):
    tl = _timeline({})
    out = bm.materialize_mask_video(tl, tmp_path / "m.mkv", feather_px=8)
    assert _decode(out) == bytes(W * H * N)


def test_encoder_call_uses_ffv1_slices(monkeypatch, tmp_path):
    seen = {}

    class _P:
        stdin = type("S", (), {"write": lambda self, b: 0, "close": lambda self: None})()
        stderr = None

        def wait(self):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        bm.subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd) and _P()
    )
    with pytest.raises(RuntimeError):
        bm.materialize_mask_video(_timeline(_fixture_boxes()), tmp_path / "m.mkv")
    cmd = seen["cmd"]
    assert cmd[cmd.index("-slices") + 1] == str(bm.FFV1_SLICES)
    assert cmd.index("-slices") > cmd.index("ffv1")
