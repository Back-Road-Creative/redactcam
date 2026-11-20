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

#: FFV1 slice count (and encoder threads) for the mask render. Must be a layout the
#: encoder accepts for the frame size; ffmpeg rejects an invalid count at open.
FFV1_SLICES = 4


def active_boxes_at(timeline: BlurTimeline, frame_idx: int) -> list[Box]:
    """Boxes active at source ``frame_idx`` — an O(1) lookup into the dense
    per-frame timeline (empty for a frame with no tracked object). Keeping this a
    dict lookup is what makes ``materialize_mask_video`` O(frames), not the O(n²)
    a per-frame interval scan would have cost."""
    return timeline.frame_boxes.get(frame_idx, [])


def render_mask_frame(width: int, height: int, boxes: list[Box], feather_px: int = 0) -> np.ndarray:
    """One gray mask frame: white (255) filled boxes on black (0), optionally
    feathered so the composite blends the blur edge instead of hard-cutting it.

    Pixel-identical to blurring the whole frame, at a fraction of the cost: a frame
    with no boxes is returned as zeros without any blur, and otherwise the Gaussian
    runs only on the union bounding box of the boxes grown by the kernel radius.
    A blurred pixel depends only on source pixels within ``feather_px`` of it, so
    everything farther than that from every box is exactly 0, and the crop keeps
    ``feather_px`` of real context on each side that is not the frame edge. A crop
    side that IS the frame edge reflects exactly as the full frame does. No mask
    region is ever shrunk or shifted (``test_mask_perf.py`` pins this byte for byte)."""
    frame = np.zeros((height, width), dtype=np.uint8)
    if not boxes:
        return frame
    # cv2.rectangle fills x..x+w and y..y+h INCLUSIVE, so the covered span is w+1.
    bx0 = min(x for x, _, _, _ in boxes)
    by0 = min(y for _, y, _, _ in boxes)
    bx1 = max(x + w for x, _, w, _ in boxes)
    by1 = max(y + h for _, y, _, h in boxes)
    if feather_px <= 0:
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, thickness=-1)
        return frame
    # Output region: box bbox grown by the kernel radius, clipped to the frame.
    ox0, oy0 = max(0, bx0 - feather_px), max(0, by0 - feather_px)
    ox1, oy1 = min(width, bx1 + feather_px + 1), min(height, by1 + feather_px + 1)
    if ox0 >= ox1 or oy0 >= oy1:
        return frame  # every box lies wholly off-frame and the blur cannot reach in
    # Input crop: the output region plus one more kernel radius of context.
    cx0, cy0 = max(0, ox0 - feather_px), max(0, oy0 - feather_px)
    cx1, cy1 = min(width, ox1 + feather_px), min(height, oy1 + feather_px)
    crop = np.zeros((cy1 - cy0, cx1 - cx0), dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(crop, (x - cx0, y - cy0), (x + w - cx0, y + h - cy0), 255, thickness=-1)
    k = feather_px * 2 + 1  # odd kernel
    blurred = cv2.GaussianBlur(crop, (k, k), 0)
    frame[oy0:oy1, ox0:ox1] = blurred[oy0 - cy0 : oy1 - cy0, ox0 - cx0 : ox1 - cx0]
    return frame


def _count_frames(path: Path) -> int:
    """Exact frame count of an FFV1 mask via ``ffprobe -count_packets`` (-1 on
    failure). FFV1 is intra-only, one whole frame per packet, so a demux gives
    the decoded count without decoding: 352,934 4K frames took 52 min to decode
    on one core and take seconds to count this way."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
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


def materialize_mask_video(
    timeline: BlurTimeline,
    output_path: Path,
    feather_px: int = 8,
    container_frame_count: int | None = None,
) -> Path:
    """Render ``timeline`` to a lossless gray FFV1 mask at the source's exact
    resolution / fps / frame count. Hard-fails if the written frame count does
    not equal ``source_frame_count`` (a misaligned mask blurs the wrong pixels).

    The mask stamps frame ``j`` at ``j / fps``, so it lines up with the picture only
    when the source has exactly ``n`` frames. ``container_frame_count`` (the source
    container's own count) is asserted equal to ``n`` before anything is rendered."""
    w, h, n = timeline.frame_width, timeline.frame_height, timeline.source_frame_count
    fps = timeline.source_fps or 30.0
    if w <= 0 or h <= 0 or n <= 0:
        raise ValueError(f"mask needs positive geometry/length; got {w}x{h}, {n} frames")
    if (
        container_frame_count is not None
        and container_frame_count > 0
        and container_frame_count != n
    ):
        raise RuntimeError(
            f"timeline has {n} frames but the source container has {container_frame_count}; "
            "a mask in a different index space blurs the wrong pixels; refusing to proceed"
        )
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
        # Slice-threaded encode. FFV1 is lossless, so the slice layout changes the
        # bitstream but never a decoded pixel (test_mask_perf.py compares decoded frames).
        "-slices",
        str(FFV1_SLICES),
        "-threads",
        str(FFV1_SLICES),
        "-pix_fmt",
        "gray",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    sent = 0
    empty = bytes(w * h)  # an all-black frame, shared by every frame with no boxes
    try:
        try:
            for idx in range(n):
                boxes = active_boxes_at(timeline, idx)
                if boxes:
                    proc.stdin.write(render_mask_frame(w, h, boxes, feather_px).tobytes())
                else:
                    proc.stdin.write(empty)
                sent += 1
            proc.stdin.close()
        except BrokenPipeError:
            # ffmpeg exited mid-stream; its exit code and stderr say why, the
            # broken pipe on our side does not.
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        err = proc.stderr.read().decode(errors="replace")[-500:] if proc.stderr else ""
        raise RuntimeError(f"mask ffmpeg failed (rc={rc}) after {sent} of {n} frames: {err}")
    written = _count_frames(output_path)
    if written != n:
        raise RuntimeError(
            f"blur mask frame count {written} != source {n} — would blur the wrong "
            "pixels; refusing to proceed"
        )
    logger.info("  BLUR: mask video %s (%d frames @ %.3f fps)", output_path.name, n, fps)
    return output_path
