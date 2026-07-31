"""Frame-accurate blur mask video — stage 3 of the redaction chain.

Rasterises a ``BlurTimeline`` into a lossless single-channel (gray) mask video:
black (0) background, white (255) filled and feathered boxes wherever a face or
plate is active, at the source's exact resolution, frame rate and FRAME COUNT.
``apply.apply_blur`` then feeds this mask as a second input to ffmpeg, blurring
the source only where the mask is white.

The written frame count is hard-asserted to equal ``source_frame_count``. A mask
one frame out of step blurs the wrong pixels for the rest of the clip, so a
mismatch raises — it is never logged and shrugged off.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .timeline import BlurTimeline, Box

logger = logging.getLogger(__name__)


def active_boxes_at(timeline: BlurTimeline, frame_idx: int) -> list[Box]:
    """Boxes active at source ``frame_idx`` — an O(1) lookup into the dense
    per-frame timeline (empty for a frame with no tracked object). Keeping this a
    dict lookup is what makes ``materialize_mask_video`` O(frames), not the O(n²)
    a per-frame interval scan would have cost."""
    return timeline.frame_boxes.get(frame_idx, [])


def render_mask_frame(width: int, height: int, boxes: list[Box], feather_px: int = 0) -> np.ndarray:
    """One gray mask frame: white (255) filled boxes on black (0), optionally
    feathered so the composite blends the blur edge instead of hard-cutting it."""
    frame = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, thickness=-1)
    if feather_px > 0:
        k = feather_px * 2 + 1  # odd kernel
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return frame


def _count_frames(path: Path) -> int:
    """Exact decoded frame count via ``ffprobe -count_frames`` (-1 on failure)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def materialize_mask_video(timeline: BlurTimeline, output_path: Path, feather_px: int = 8) -> Path:
    """Render ``timeline`` to a lossless gray FFV1 mask at the source's exact
    resolution / fps / frame count. Hard-fails if the written frame count does
    not equal ``source_frame_count`` (a misaligned mask blurs the wrong pixels)."""
    w, h, n = timeline.frame_width, timeline.frame_height, timeline.source_frame_count
    fps = timeline.source_fps or 30.0
    if w <= 0 or h <= 0 or n <= 0:
        raise ValueError(f"mask needs positive geometry/length; got {w}x{h}, {n} frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        # No periodic encode stats on the stderr PIPE: it is never drained while the
        # parent writes raw frames to stdin, so stats would fill the 64 KB pipe buffer
        # on a long render and deadlock both sides (real errors still reach stderr).
        "-nostats",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        for idx in range(n):
            frame = render_mask_frame(w, h, active_boxes_at(timeline, idx), feather_px)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        err = proc.stderr.read().decode(errors="replace")[-500:] if proc.stderr else ""
        raise RuntimeError(f"mask ffmpeg failed (rc={rc}): {err}")
    written = _count_frames(output_path)
    if written != n:
        raise RuntimeError(
            f"blur mask frame count {written} != source {n} — would blur the wrong "
            "pixels; refusing to proceed"
        )
    logger.info("  BLUR: mask video %s (%d frames @ %.3f fps)", output_path.name, n, fps)
    return output_path
