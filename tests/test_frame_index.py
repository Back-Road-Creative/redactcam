"""One frame-index contract: every stage counts the container's frames, and any
disagreement fails closed.

A clip with a timestamp gap is built with ffmpeg. A rawvideo muxer defaults to
constant-frame-rate output and duplicates frames across the gap, which put detection
and the mask in a longer index space than the picture. These tests run a CPU ffmpeg
with the NVDEC command's own timing arguments, so no GPU is touched.
"""

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from redactcam import coverage as bc
from redactcam import frames
from redactcam import mask as bm
from redactcam import pipeline as pl
from redactcam.frames import FrameSourceError, iter_frames
from redactcam.timeline import BlurTimeline

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe missing"
)

N = 30
W, H = 64, 48


@pytest.fixture(autouse=True)
def _no_ambient_decode_override(monkeypatch):
    monkeypatch.delenv("REDACTCAM_DECODE", raising=False)


@pytest.fixture(scope="module")
def gap_clip(tmp_path_factory):
    """30 frames at 30 fps whose pts jump by 0.5 s after frame 14."""
    p = tmp_path_factory.mktemp("gap") / "gap.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size={W}x{H}:rate=30:duration=1",
         "-vf", "setpts='if(gte(N,15),PTS+0.5/TB,PTS)'",
         "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(p)],
        check=True,
    )  # fmt: skip
    return p


def _cpu_cmd(cmd):
    """The NVDEC command with only the GPU decode arguments removed."""
    out = list(cmd)
    i = out.index("-hwaccel")
    del out[i : i + 4]  # -hwaccel cuda -c:v <decoder>
    return out


def _count_raw(cmd) -> int:
    r = subprocess.run(cmd, capture_output=True, check=True)
    return len(r.stdout) // (W * H * 3)


def _nb_frames(path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return int(r.stdout.strip())


def test_fixture_has_a_gap(gap_clip):
    assert _nb_frames(gap_clip) == N


def test_nvdec_command_emits_exactly_the_container_frames(gap_clip):
    cmd = _cpu_cmd(frames._nvdec_command("ffmpeg", "hevc_cuvid", gap_clip))
    assert _count_raw(cmd) == N


def test_without_passthrough_the_gap_is_filled_with_duplicates(gap_clip):
    """The defect: same command minus the timing flag emits extra frames."""
    cmd = _cpu_cmd(frames._nvdec_command("ffmpeg", "hevc_cuvid", gap_clip))
    while "-fps_mode" in cmd:
        i = cmd.index("-fps_mode")
        del cmd[i : i + 2]
    assert _count_raw(cmd) > N


def _stub_nvdec(monkeypatch, *, strip_flag=False, extra=()):
    """Route the nvdec backend through a CPU ffmpeg."""
    monkeypatch.setattr(frames, "_nvdec_plan", lambda path, meta: ("ffmpeg", "hevc_cuvid"))
    real = frames._nvdec_command

    def cmd(ffmpeg, decoder, path):
        c = _cpu_cmd(real(ffmpeg, decoder, path))
        if strip_flag and "-fps_mode" in c:
            i = c.index("-fps_mode")
            del c[i : i + 2]
        return c

    monkeypatch.setattr(frames, "_nvdec_command", cmd)


def test_iter_frames_delivers_container_count(gap_clip, monkeypatch):
    _stub_nvdec(monkeypatch)
    meta, it = iter_frames(gap_clip, decode="nvdec")
    assert meta.backend == "nvdec"
    assert len(list(it)) == N


def test_iter_frames_raises_when_delivery_exceeds_container_count(gap_clip, monkeypatch):
    _stub_nvdec(monkeypatch, strip_flag=True)
    meta, it = iter_frames(gap_clip, decode="nvdec")
    with pytest.raises(FrameSourceError, match="frames"):
        list(it)


def test_iter_frames_raises_when_delivery_is_short(gap_clip, monkeypatch):
    """Strict equality: one frame short is a mismatch, not a tolerated estimate."""
    _stub_nvdec(monkeypatch)
    real = frames._nvdec_command
    monkeypatch.setattr(
        frames, "_nvdec_command",
        lambda f, d, p: [*real(f, d, p)[:-1], "-frames:v", str(N - 1), "-"],
    )  # fmt: skip
    meta, it = iter_frames(gap_clip, decode="nvdec")
    with pytest.raises(FrameSourceError, match="frames"):
        list(it)


def test_resume_stays_in_the_nvdec_index_space(gap_clip, monkeypatch):
    """A resume must not switch decoders: it decodes from 0 and skips, so frame
    ``start`` is the same frame a fresh pass labels ``start``."""
    _stub_nvdec(monkeypatch)
    _, full = iter_frames(gap_clip, decode="nvdec")
    full = [f.copy() for f in full]
    meta, it = iter_frames(gap_clip, decode="nvdec", start_frame=20)
    got = [f.copy() for f in it]
    assert meta.backend == "nvdec"
    assert len(got) == N - 20
    assert all(np.array_equal(a, b) for a, b in zip(got, full[20:], strict=True))


def test_resume_in_cv2_mode_still_seeks(gap_clip):
    meta, it = iter_frames(gap_clip, decode="cv2", start_frame=20)
    assert meta.backend == "cv2"
    assert len(list(it)) == N - 20


def _timeline(n=N):
    return BlurTimeline(
        source_hash="h", sample_fps=8.0, conf=0.4, box_dilation_px=4, source_fps=30.0,
        source_frame_count=n, frame_width=W, frame_height=H, frame_boxes={0: [(4, 4, 8, 8)]},
    )  # fmt: skip


def test_mask_refuses_a_container_count_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="container"):
        bm.materialize_mask_video(_timeline(N + 5), tmp_path / "m.mkv", container_frame_count=N)
    assert not (tmp_path / "m.mkv").exists()


def test_mask_accepts_matching_container_count(tmp_path):
    out = bm.materialize_mask_video(_timeline(), tmp_path / "m.mkv", container_frame_count=N)
    assert _nb_frames(out) == N


def test_pipeline_passes_the_container_count_to_the_mask(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pl, "require_ffmpeg", lambda: {})
    monkeypatch.setattr(pl, "resolve_model", lambda spec, cache=None: tmp_path / "m.onnx")
    monkeypatch.setattr(pl._timeline, "probe_video", lambda p: (30.0, 90, 1920, 1080))
    monkeypatch.setattr(pl, "detect_and_track", lambda v, **kw: ({0: [(1, 1, 5, 5)]}, 90, 1, 0))

    def fake_mask(tl, path, **kw):
        seen.update(kw)
        path.write_bytes(b"m")
        return path

    monkeypatch.setattr(pl._mask, "materialize_mask_video", fake_mask)
    src = tmp_path / "c.mp4"
    src.write_bytes(b"x" * 20)
    pl.redact_video(src, work_dir=tmp_path / "w", render=False, verify=False)
    assert seen.get("container_frame_count") == 90


def _clip(path, n, fps=30.0):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H))
    for i in range(n):
        wr.write(np.full((H, W, 3), 10 + i, np.uint8))
    wr.release()


@pytest.mark.parametrize("src_n, mask_n", [(6, 4), (4, 6)])
def test_coverage_raises_when_streams_differ_in_length(tmp_path, monkeypatch, src_n, mask_n):
    class _Veh:
        def detect(self, frame, conf):
            return []

    monkeypatch.setattr(bc, "build_vehicle_detector", lambda *a, **k: _Veh())
    monkeypatch.setattr(bc, "build_plate_detector", lambda *a, **k: object())
    monkeypatch.setattr(bc, "detect_plates", lambda *a, **k: [])
    inp, mvid = tmp_path / "in.mkv", tmp_path / "m.mkv"
    _clip(inp, src_n)
    _clip(mvid, mask_n)
    with pytest.raises(FrameSourceError, match="mask"):
        bc.verify_cabin_coverage(inp, mvid, "v.onnx", plate_model="p.onnx", sample_fps=3)
