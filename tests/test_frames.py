"""Frame source: cv2 vs NVDEC parity, the pipe reader, and backend selection.

Clips are generated with ffmpeg in ``tmp_path`` (no fixtures on disk). The NVDEC
parity tests SKIP, with the reason, on a box whose ffmpeg lacks cuda/cuvid or whose
1-frame probe decode fails; they never fail for lack of a GPU.
"""

import io
import logging
import shutil
import subprocess
from contextlib import closing

import numpy as np
import pytest

from redactcam import detect, frames
from redactcam.frames import FrameSourceError, iter_frames, read_full


@pytest.fixture(autouse=True)
def _no_ambient_decode_override(monkeypatch):
    """The suite may run with REDACTCAM_DECODE=cv2; these tests choose their own."""
    monkeypatch.delenv("REDACTCAM_DECODE", raising=False)


needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe missing"
)


def _encode(path, codec, extra=()):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30:duration=1", "-pix_fmt", "yuv420p",
         "-c:v", codec, *extra, str(path)],
        check=True,
    )  # fmt: skip


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    p = tmp_path_factory.mktemp("clip") / "a.mp4"
    _encode(p, "libx264")
    return p


@pytest.fixture(scope="module")
def hevc_clip(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    p = tmp_path_factory.mktemp("clip") / "b.mp4"
    try:
        _encode(p, "libx265", ("-x265-params", "log-level=error"))
    except subprocess.CalledProcessError:
        pytest.skip("libx265 unavailable")
    return p


def _nvdec_or_skip(clip):
    meta, it = iter_frames(clip, decode="nvdec")
    it.close()
    if meta.backend != "nvdec":
        pytest.skip("nvdec unavailable on this box (cuda/cuvid/probe)")


def _all(clip, mode):
    meta, it = iter_frames(clip, decode=mode)
    with closing(it):
        return meta, list(it)


class _Trickle:
    """Fake pipe returning one byte per ``readinto``."""

    def __init__(self, data):
        self._b = io.BytesIO(data)

    def readinto(self, view):
        return self._b.readinto(view[:1])


class TestReadFull:
    def test_short_reads_are_not_end_of_stream(self):
        data = bytes(range(12))
        stream = _Trickle(data * 2)
        assert bytes(read_full(stream, 12)) == data
        assert bytes(read_full(stream, 12)) == data
        assert read_full(stream, 12) is None

    def test_partial_final_frame_raises(self):
        with pytest.raises(FrameSourceError, match="partial frame"):
            read_full(_Trickle(b"x" * 7), 12)


@needs_ffmpeg
class TestParity:
    @pytest.mark.parametrize("clipname", ["h264_clip", "hevc_clip"])
    def test_nvdec_matches_cv2(self, clipname, request):
        clip = request.getfixturevalue(clipname)
        _nvdec_or_skip(clip)
        m_cv, f_cv = _all(clip, "cv2")
        m_nv, f_nv = _all(clip, "nvdec")
        assert m_nv.backend == "nvdec" and m_cv.backend == "cv2"
        assert (m_nv.fps, m_nv.width, m_nv.height, m_nv.frame_count) == (
            m_cv.fps, m_cv.width, m_cv.height, m_cv.frame_count,
        )  # fmt: skip
        assert len(f_nv) == len(f_cv) == m_cv.frame_count
        for a, b in zip(f_cv, f_nv, strict=True):
            assert a.shape == b.shape == (m_cv.height, m_cv.width, 3)
            assert a.dtype == b.dtype == np.uint8
            assert np.abs(a.astype(int) - b.astype(int)).mean() < 2

    def test_cv2_backend_metadata(self, h264_clip):
        meta, fr = _all(h264_clip, "cv2")
        assert (meta.width, meta.height, meta.frame_count) == (320, 240, 30)
        assert len(fr) == 30


@needs_ffmpeg
class TestSelection:
    def test_auto_falls_back_when_ffmpeg_missing(self, h264_clip, monkeypatch, caplog):
        monkeypatch.setattr(frames.shutil, "which", lambda name: None)
        with caplog.at_level(logging.INFO, logger="redactcam.frames"):
            meta, it = iter_frames(h264_clip, decode="auto")
        it.close()
        assert meta.backend == "cv2"
        assert any("using cv2" in r.getMessage() for r in caplog.records)

    def test_auto_falls_back_when_probe_fails(self, h264_clip, monkeypatch):
        monkeypatch.setattr(frames, "_nvdec_plan", lambda *a: "probe decode of 1 frame failed")
        meta, it = iter_frames(h264_clip, decode="auto")
        it.close()
        assert meta.backend == "cv2"

    def test_dead_pipe_start_falls_back(self, h264_clip, monkeypatch):
        monkeypatch.setattr(frames, "_nvdec_plan", lambda *a: ("/bin/false", "h264_cuvid"))
        monkeypatch.setattr(frames, "_nvdec_command", lambda ff, dec, p: ["/bin/false"])
        meta, it = iter_frames(h264_clip, decode="nvdec")
        with closing(it):
            assert meta.backend == "cv2" and len(list(it)) == 30

    def test_env_override_cv2_beats_argument(self, h264_clip, monkeypatch):
        monkeypatch.setenv("REDACTCAM_DECODE", "cv2")
        monkeypatch.setattr(
            frames, "_nvdec_plan", lambda *a: pytest.fail("nvdec planned despite env=cv2")
        )
        meta, it = iter_frames(h264_clip, decode="nvdec")
        it.close()
        assert meta.backend == "cv2"

    def test_env_override_nvdec_is_honoured(self, h264_clip, monkeypatch):
        monkeypatch.setenv("REDACTCAM_DECODE", "nvdec")
        seen = []
        monkeypatch.setattr(frames, "_nvdec_plan", lambda *a: seen.append(1) or "stub")
        meta, it = iter_frames(h264_clip, decode="cv2")
        it.close()
        assert seen and meta.backend == "cv2"

    def test_bad_mode_rejected(self, h264_clip):
        with pytest.raises(ValueError, match="decode must be"):
            iter_frames(h264_clip, decode="gpu")

    def test_rotation_metadata_stays_on_cv2(self):
        assert frames.is_rotated({"side_data_list": [{"side_data_type": "Display Matrix"}]})
        assert frames.is_rotated({"tags": {"rotate": "90"}})
        assert not frames.is_rotated({"tags": {}, "side_data_list": []})

    def test_resume_start_uses_cv2(self, h264_clip):
        meta, it = iter_frames(h264_clip, decode="auto", start_frame=10)
        with closing(it):
            assert meta.backend == "cv2" and len(list(it)) == 20


class _FakeProc:
    """Pipe that yields N whole frames then dies with rc=1."""

    def __init__(self, nframes, size):
        self.stdout = io.BytesIO(b"\x01" * (nframes * size))
        self.returncode = 1

    def wait(self, timeout=None):
        return 1

    def terminate(self):
        pass

    kill = terminate


class TestMidStreamFailure:
    def test_dead_pipe_raises_instead_of_truncating(self, monkeypatch):
        meta = frames.FrameMeta(30.0, 4, 3, 10, "nvdec")
        monkeypatch.setattr(
            frames.subprocess, "Popen", lambda *a, **k: _FakeProc(3, 4 * 3 * 3)
        )
        it = frames._open_nvdec(["ffmpeg"], meta)
        got = []
        with pytest.raises(FrameSourceError, match="rc=1 after 3 frames"):
            for f in it:
                got.append(f.shape)
        assert got == [(3, 4, 3)] * 3


@needs_ffmpeg
class TestDetectParity:
    def test_sampled_indices_identical_across_backends(self, hevc_clip, monkeypatch):
        _nvdec_or_skip(hevc_clip)
        from redactcam import track

        calls = []
        real = track.TrackManager.step

        def spy(self, idx, gray, dets):
            calls.append((idx, dets is not None))
            return real(self, idx, gray, dets)

        monkeypatch.setattr(track.TrackManager, "step", spy)
        monkeypatch.setattr(detect, "build_detectors", lambda *a, **k: (None, None))
        monkeypatch.setattr(detect, "detect_tiled", lambda *a, **k: [])
        kw = dict(sample_fps=5.0, plate_model_path=None, tile_px=9999, track_downscale=1,
                  track_lk_win=15, track_lk_levels=2)  # fmt: skip
        out = {}
        for mode in ("cv2", "nvdec"):
            calls.clear()
            _, n, _, _ = detect.detect_and_track(hevc_clip, decode=mode, **kw)
            out[mode] = (n, list(calls))
        assert out["cv2"] == out["nvdec"]
        assert out["cv2"][0] == 30
        assert any(sampled for _, sampled in out["cv2"][1])

    def test_decode_backend_is_logged(self, h264_clip, monkeypatch, caplog):
        # No real model: CI has neither the face ONNX nor the centerface extra.
        monkeypatch.setattr(detect, "build_detectors", lambda *a, **k: (None, None))
        monkeypatch.setattr(detect, "detect_tiled", lambda *a, **k: [])
        with caplog.at_level(logging.INFO, logger="redactcam.detect"):
            detect.detect_and_track(
                h264_clip, decode="cv2", sample_fps=5.0, plate_model_path=None,
                tile_px=9999, track_downscale=1, track_lk_win=15, track_lk_levels=2,
            )  # fmt: skip
        msgs = [r.getMessage() for r in caplog.records]
        assert "detect: decode=cv2" in msgs
        assert any(m.startswith("detect: done") and "decode=cv2" in m for m in msgs)
