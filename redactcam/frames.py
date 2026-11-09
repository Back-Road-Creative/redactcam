"""Frame source for the per-frame loops: OpenCV (``cv2``) or NVIDIA NVDEC (``nvdec``).

Stage 1 (``detect_and_track``) reads every frame of a 4K clip and is decode-bound:
on a measured 4K60 HEVC master OpenCV and a CPU ``ffmpeg`` pipe both top out near
22 fps, while ``ffmpeg -hwaccel cuda -c:v hevc_cuvid`` piped as ``bgr24`` reaches
~40 fps. ``iter_frames`` hides which one runs; both yield identical BGR ``uint8``
``HxWx3`` arrays in the same order, so frame indices, sampling and tracking do not
depend on the backend.

``REDACTCAM_DECODE=cv2|nvdec|auto`` overrides the ``decode`` argument. ``auto`` uses
NVDEC only when ffmpeg reports ``-hwaccel cuda``, has the matching ``*_cuvid``
decoder, and decodes one frame of the input; anything else uses ``cv2``. A rotated
clip (display-matrix / rotate tag) always uses ``cv2``, which applies the rotation
the raw pipe would not.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ENV_DECODE = "REDACTCAM_DECODE"
DECODE_MODES = ("auto", "cv2", "nvdec")
_CUVID = {"hevc": "hevc_cuvid", "h264": "h264_cuvid", "av1": "av1_cuvid"}
_PROBE_TIMEOUT_S = 60
_WAIT_TIMEOUT_S = 10
# A clean ffmpeg exit with fewer frames than the container promised is tolerated up
# to this fraction (container frame counts are estimates); more is a truncated read.
_SHORTFALL_FRAC = 0.01


@dataclass(frozen=True)
class FrameMeta:
    """Stream facts derived exactly as ``cv2.VideoCapture`` reports them."""

    fps: float
    width: int
    height: int
    frame_count: int
    backend: str = "cv2"


class FrameSourceError(RuntimeError):
    """The decoder died or truncated the stream before the video ended."""


def resolve_mode(decode: str) -> str:
    mode = (os.environ.get(ENV_DECODE) or decode or "auto").strip().lower()
    if mode not in DECODE_MODES:
        raise ValueError(f"decode must be one of {DECODE_MODES}, got {mode!r}")
    return mode


def read_full(stream, size: int) -> bytearray | None:
    """Read exactly ``size`` bytes from ``stream`` (looping ``readinto``).

    ``None`` is a clean end of stream (zero bytes). A pipe may return any number of
    bytes per read, so a short read is NOT end of stream; ending partway through a
    frame raises ``FrameSourceError`` rather than yielding or dropping a torn frame.
    """
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = stream.readinto(view[got:])
        if not n:
            break
        got += n
    if got == 0:
        return None
    if got < size:
        raise FrameSourceError(f"partial frame at end of stream: {got} of {size} bytes")
    return buf


@functools.lru_cache(maxsize=4)
def _ffmpeg_caps(ffmpeg: str) -> tuple[bool, frozenset[str]]:
    """(``-hwaccel cuda`` reported, the ``*_cuvid`` decoders this ffmpeg has)."""

    def run(*args: str) -> str:
        try:
            r = subprocess.run(
                [ffmpeg, "-hide_banner", *args],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return r.stdout

    hw = any(line.strip() == "cuda" for line in run("-hwaccels").splitlines())
    decoders = frozenset(
        parts[1]
        for line in run("-decoders").splitlines()
        if len(parts := line.split()) >= 2 and parts[1].endswith("_cuvid")
    )
    return hw, decoders


def probe_stream(path: Path) -> dict | None:
    """First video stream via ``ffprobe -show_streams`` (None if unavailable)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-of", "json", str(path)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return streams[0] if streams else None


def is_rotated(stream: dict) -> bool:
    """True if the stream carries rotation metadata (rotate tag or display matrix).

    cv2 applies it; the raw ffmpeg pipe is not proven to under ``-hwaccel cuda``,
    so such clips stay on cv2."""
    if (stream.get("tags") or {}).get("rotate") not in (None, "", "0"):
        return True
    for sd in stream.get("side_data_list") or []:
        kind = str(sd.get("side_data_type", "")).lower()
        if "display" in kind or sd.get("rotation") not in (None, 0, "0"):
            return True
    return False


def _nvdec_command(ffmpeg: str, decoder: str, path: Path) -> list[str]:
    return [
        ffmpeg, "-v", "error", "-hwaccel", "cuda", "-c:v", decoder,
        "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip


def _nvdec_plan(path: Path, meta: FrameMeta) -> tuple[str, str] | str:
    """(ffmpeg, decoder) if NVDEC can serve ``path``, else the reason it cannot."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "ffmpeg not found"
    stream = probe_stream(path)
    if stream is None:
        return "ffprobe could not read the stream"
    decoder = _CUVID.get(str(stream.get("codec_name")))
    if decoder is None:
        return f"no cuvid decoder for codec {stream.get('codec_name')!r}"
    if is_rotated(stream):
        return "stream has rotation metadata (cv2 applies it)"
    if (int(stream.get("width") or 0), int(stream.get("height") or 0)) != (meta.width, meta.height):
        return "ffprobe and cv2 disagree on frame size"
    hw, decoders = _ffmpeg_caps(ffmpeg)
    if not hw:
        return "ffmpeg lacks -hwaccel cuda"
    if decoder not in decoders:
        return f"ffmpeg lacks the {decoder} decoder"
    try:
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-hwaccel", "cuda", "-c:v", decoder, "-i", str(path),
             "-frames:v", "1", "-f", "null", "-"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"probe decode failed to run: {exc}"
    if r.returncode != 0:
        return "probe decode of 1 frame failed"
    return ffmpeg, decoder


def _open_nvdec(cmd: list[str], meta: FrameMeta) -> Iterator[np.ndarray]:
    """Start the pipe and read frame 0 eagerly so a dead start falls back to cv2.

    Raises ``FrameSourceError`` if it cannot start or yields no first frame."""
    size = meta.width * meta.height * 3
    err = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0)
    except OSError as exc:
        err.close()
        raise FrameSourceError(f"could not start ffmpeg: {exc}") from exc
    try:
        first = read_full(proc.stdout, size)
    except FrameSourceError:
        first = None
    if first is None:
        _reap(proc)
        msg = _stderr(err)
        err.close()
        raise FrameSourceError(f"nvdec pipe produced no frame: {msg}")

    def gen() -> Iterator[np.ndarray]:
        delivered = 0
        shape = (meta.height, meta.width, 3)
        buf: bytearray | None = first
        try:
            while buf is not None:
                yield np.frombuffer(buf, dtype=np.uint8).reshape(shape)
                delivered += 1
                buf = read_full(proc.stdout, size)
            rc = _reap(proc)
            if rc != 0:
                raise FrameSourceError(
                    f"ffmpeg nvdec exited rc={rc} after {delivered} frames: {_stderr(err)}"
                )
            if meta.frame_count and delivered < meta.frame_count * (1 - _SHORTFALL_FRAC):
                raise FrameSourceError(
                    f"nvdec stream ended after {delivered} of {meta.frame_count} frames"
                )
        finally:
            _reap(proc)
            err.close()

    return gen()


def _reap(proc: subprocess.Popen) -> int | None:
    """Close the pipe, wait briefly, terminate (then kill) on timeout."""
    try:
        if proc.stdout:
            proc.stdout.close()
    except OSError:
        pass
    try:
        return proc.wait(timeout=_WAIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return proc.wait(timeout=_WAIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def _stderr(err) -> str:
    try:
        err.seek(0)
        return err.read().decode("utf-8", "replace").strip()[-500:]
    except (OSError, ValueError):
        return ""


def _cv2_frames(cap) -> Iterator[np.ndarray]:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def iter_frames(
    path: str | Path,
    *,
    decode: str = "auto",
    start_frame: int = 0,
    capture=None,
) -> tuple[FrameMeta, Iterator[np.ndarray]]:
    """Open ``path`` and return ``(meta, frames)``; ``meta.backend`` names the decoder.

    Metadata always comes from ``cv2.VideoCapture`` (``capture`` lets the caller pass
    one it already opened). ``start_frame`` > 0 (checkpoint resume) is served by cv2,
    which can seek; NVDEC starts at frame 0. Close the iterator (``.close()`` /
    ``contextlib.closing``) to stop ffmpeg and release the file."""
    path = Path(path)
    mode = resolve_mode(decode)
    cap = capture if capture is not None else cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    meta = FrameMeta(
        fps=cap.get(cv2.CAP_PROP_FPS) or 0.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    if mode != "cv2":
        reason = "resume from a checkpoint seeks with cv2" if start_frame > 0 else None
        plan = reason or _nvdec_plan(path, meta)
        if isinstance(plan, tuple):
            ffmpeg, decoder = plan
            try:
                frames = _open_nvdec(
                    _nvdec_command(ffmpeg, decoder, path),
                    FrameMeta(meta.fps, meta.width, meta.height, meta.frame_count, "nvdec"),
                )
            except FrameSourceError as exc:
                plan = str(exc)
            else:
                cap.release()
                return FrameMeta(
                    meta.fps, meta.width, meta.height, meta.frame_count, "nvdec"
                ), frames
        logger.info("decode: nvdec unavailable (%s); using cv2", plan)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return meta, _cv2_frames(cap)
