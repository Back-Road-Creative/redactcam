"""Apply a mask video to a source video with ffmpeg — stage 4 of the chain.

Blurs the source wherever the mask is white, leaves it untouched where the mask
is black, and blends across the mask's feathered edge.

The composite is ``alphamerge`` + ``overlay``, deliberately NOT ``maskedmerge``.
``maskedmerge`` promotes a gray mask to the video's pixel format, which pins
chroma at 128 and then half-blends chroma across the WHOLE frame regardless of
what the mask actually says — a subtle full-frame colour shift, measurable by
PSNR against the source and easy to miss by eye. Splitting the frame, blurring
one copy, and using the mask as that copy's alpha channel keeps every unmasked
pixel bit-identical.

``boxblur=<r>:2`` is the blur itself: radius ``r``, two passes, which approximates
a Gaussian closely enough for anonymisation at a fraction of the cost. The radius
scales with frame width so a 4K frame is blurred as heavily, in relative terms,
as a 1080p one.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# boxblur luma radius at 1920 px wide. Scaled by actual width in blur_radius().
# 12 at 1080p reduces a face to an unrecognisable smear while leaving the scene
# legible; well below this and features start to survive at full-screen playback.
REFERENCE_BLUR_STRENGTH = 12
REFERENCE_WIDTH = 1920


def blur_radius(video_width: int, strength: int = REFERENCE_BLUR_STRENGTH) -> int:
    """Boxblur radius for a frame ``video_width`` px wide.

    ``strength`` is the radius at 1920 px; it scales linearly with width so the
    blur covers the same fraction of a face at any resolution. Floored at 2,
    below which boxblur is close to a no-op.
    """
    return max(2, round(strength * video_width / REFERENCE_WIDTH))


def blur_filter_complex(blur: str) -> str:
    """The ffmpeg ``filter_complex`` compositing input 1 (mask) over input 0.

    ``blur`` is the filter applied to the masked copy, e.g. ``boxblur=24:2``.
    Returns a graph whose single output pad is ``[vout]``.
    """
    return ";".join(
        [
            "[0:v]split[keep][tomask]",
            f"[tomask]{blur}[blurred]",
            # Force gray so a mask stored as yuv/rgb still yields a single plane.
            "[1:v]format=gray[m]",
            "[blurred][m]alphamerge[masked]",
            "[keep][masked]overlay[vout]",
        ]
    )


def apply_blur(
    source_video: str | Path,
    mask_video: str | Path,
    output_path: str | Path,
    *,
    video_width: int,
    strength: int = REFERENCE_BLUR_STRENGTH,
    crf: int = 18,
    preset: str = "medium",
    copy_audio: bool = True,
    extra_output_args: list[str] | None = None,
) -> Path:
    """Render ``source_video`` with every masked region blurred.

    ``video_width`` sizes the blur radius — pass the source's real width, not the
    output's. ``crf``/``preset`` are handed to libx264; ``extra_output_args`` is
    appended verbatim if you need a different codec, a pixel format, or hardware
    encoding. Raises ``RuntimeError`` if ffmpeg exits non-zero.

    This is a convenience wrapper. If you already run your own encode, you do not
    need it — take ``blur_filter_complex`` and splice the graph into that encode
    instead, so the video is compressed exactly once.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fc = blur_filter_complex(f"boxblur={blur_radius(video_width, strength)}:2")
    cmd = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-i",
        str(mask_video),
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
    ]
    if copy_audio:
        # '?' makes the audio map optional, so a silent source is not an error.
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]
    cmd += extra_output_args or []
    cmd.append(str(output_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg blur composite failed (rc={proc.returncode}): {proc.stderr[-800:]}")
    logger.info("wrote blurred video %s", output_path)
    return output_path
