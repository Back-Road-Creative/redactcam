"""Mask-video tests — rasterising a BlurTimeline into a gray mask clip."""

import shutil

import numpy as np
import pytest

from redactcam import mask as bm
from redactcam.timeline import BlurTimeline


def _timeline(**over):
    base = dict(
        source_hash="h",
        sample_fps=8.0,
        conf=0.4,
        box_dilation_px=4,
        source_fps=10.0,
        source_frame_count=5,
        frame_width=32,
        frame_height=32,
        frame_boxes={0: [(4, 4, 8, 8)], 1: [(4, 4, 8, 8)], 2: [(4, 4, 8, 8)]},
    )
    base.update(over)
    return BlurTimeline(**base)


def test_active_boxes_at_is_o1_dict_lookup():
    tl = _timeline()
    assert bm.active_boxes_at(tl, 0) == [(4, 4, 8, 8)]
    assert bm.active_boxes_at(tl, 2) == [(4, 4, 8, 8)]
    assert bm.active_boxes_at(tl, 3) == []  # frame absent → no blur
    assert bm.active_boxes_at(tl, 99) == []  # past all frames


def test_render_mask_frame_white_box_on_black():
    frame = bm.render_mask_frame(32, 32, [(4, 4, 8, 8)], feather_px=0)
    assert frame.shape == (32, 32) and frame.dtype == np.uint8
    assert frame[8, 8] == 255  # inside the box
    assert frame[0, 0] == 0  # outside
    assert set(np.unique(frame)) <= {0, 255}  # no feather → binary


def test_render_mask_frame_feather_creates_gradient():
    hard = bm.render_mask_frame(32, 32, [(8, 8, 16, 16)], feather_px=0)
    soft = bm.render_mask_frame(32, 32, [(8, 8, 16, 16)], feather_px=3)
    assert ((soft > 0) & (soft < 255)).any()  # feather → intermediate values
    assert not ((hard > 0) & (hard < 255)).any()


def test_materialize_rejects_bad_geometry(tmp_path):
    with pytest.raises(ValueError):
        bm.materialize_mask_video(_timeline(frame_width=0), tmp_path / "m.mkv")
    with pytest.raises(ValueError):
        bm.materialize_mask_video(_timeline(source_frame_count=0), tmp_path / "m.mkv")


def test_materialize_suppresses_ffmpeg_stats(monkeypatch, tmp_path):
    """The mask-render ffmpeg invocation suppresses periodic stats
    (``-nostats -loglevel error``) so its un-drained stderr PIPE can't fill on a
    long render and deadlock the parent writing raw frames to stdin."""
    import io

    captured = {}

    class _FakeProc:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return 0  # already exited → the finally-block kill is skipped

        def wait(self):
            return 0

        def kill(self):
            pass

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(bm.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(bm, "_count_frames", lambda p: 5)  # == source_frame_count
    bm.materialize_mask_video(_timeline(), tmp_path / "m.mkv")
    cmd = captured["cmd"]
    assert "-nostats" in cmd  # no periodic stats → stderr PIPE can't fill
    assert cmd[cmd.index("-loglevel") + 1] == "error"


def test_frame_count_mismatch_is_fatal(monkeypatch, tmp_path):
    """A mask one frame out of step blurs the wrong pixels for the rest of the
    clip, and looks perfectly fine until someone watches it. So a written length
    that disagrees with the source raises rather than warns."""
    import io

    class _FakeProc:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return 0

        def wait(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(bm.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    monkeypatch.setattr(bm, "_count_frames", lambda p: 4)  # source_frame_count is 5
    with pytest.raises(RuntimeError, match="frame count"):
        bm.materialize_mask_video(_timeline(), tmp_path / "m.mkv")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_materialize_writes_exact_frame_count(tmp_path):
    out = bm.materialize_mask_video(_timeline(), tmp_path / "mask.mkv", feather_px=2)
    assert out.exists()
    assert bm._count_frames(out) == 5  # == source_frame_count, no drops
