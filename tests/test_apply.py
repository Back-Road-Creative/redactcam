"""Blur composite: radius scaling and the ffmpeg filter graph."""

import numpy as np
import pytest

from redactcam import apply as ap


class TestBlurRadius:
    def test_scales_with_width(self):
        assert ap.blur_radius(1920) == 12
        assert ap.blur_radius(3840) == 24  # 4K gets twice the radius, same relative blur

    def test_floored_at_two(self):
        # Below the floor boxblur is close to a no-op, which would ship a face.
        assert ap.blur_radius(320, strength=1) == 2

    def test_strength_is_configurable(self):
        assert ap.blur_radius(1920, strength=30) == 30


class TestFilterComplex:
    def test_uses_alphamerge_not_maskedmerge(self):
        """``maskedmerge`` pins chroma at 128 and half-blends colour across the
        WHOLE frame regardless of the mask — a full-frame tint that is easy to
        miss by eye. Splitting and using the mask as alpha keeps every unmasked
        pixel bit-identical."""
        fc = ap.blur_filter_complex("boxblur=24:2")
        assert "alphamerge" in fc and "overlay" in fc
        assert "maskedmerge" not in fc

    def test_mask_is_input_one_and_forced_gray(self):
        fc = ap.blur_filter_complex("boxblur=24:2")
        assert "[1:v]format=gray[m]" in fc
        assert "[0:v]" in fc

    def test_single_named_output_pad(self):
        assert ap.blur_filter_complex("boxblur=8:2").rstrip().endswith("[vout]")

    def test_blur_filter_is_applied_to_the_split_copy(self):
        fc = ap.blur_filter_complex("gblur=sigma=9")
        assert "[tomask]gblur=sigma=9[blurred]" in fc


class TestApplyBlur:
    def _cmd(self, monkeypatch):
        captured = {}

        class _Proc:
            returncode = 0
            stderr = ""

        def _run(cmd, **kw):
            captured["cmd"] = cmd
            return _Proc()

        monkeypatch.setattr(ap.subprocess, "run", _run)
        return captured

    def test_maps_the_graph_output_and_optional_audio(self, tmp_path, monkeypatch):
        captured = self._cmd(monkeypatch)
        ap.apply_blur("in.mp4", "mask.mkv", tmp_path / "out.mp4", video_width=3840)
        cmd = captured["cmd"]
        assert cmd[cmd.index("-map") + 1] == "[vout]"
        # '0:a?' is optional, so a silent source is not an ffmpeg error.
        assert "0:a?" in cmd
        assert "boxblur=24:2" in cmd[cmd.index("-filter_complex") + 1]

    def test_audio_can_be_dropped(self, tmp_path, monkeypatch):
        captured = self._cmd(monkeypatch)
        ap.apply_blur(
            "in.mp4", "mask.mkv", tmp_path / "out.mp4", video_width=1920, copy_audio=False
        )
        assert "0:a?" not in captured["cmd"]

    def test_extra_args_are_appended_before_the_output(self, tmp_path, monkeypatch):
        captured = self._cmd(monkeypatch)
        out = tmp_path / "out.mp4"
        ap.apply_blur(
            "in.mp4", "mask.mkv", out, video_width=1920, extra_output_args=["-pix_fmt", "yuv420p"]
        )
        cmd = captured["cmd"]
        assert cmd[-1] == str(out)
        assert cmd[-3:-1] == ["-pix_fmt", "yuv420p"]

    def test_nonzero_exit_raises(self, tmp_path, monkeypatch):
        class _Proc:
            returncode = 1
            stderr = "boom"

        monkeypatch.setattr(ap.subprocess, "run", lambda cmd, **kw: _Proc())
        with pytest.raises(RuntimeError, match="ffmpeg blur composite failed"):
            ap.apply_blur("in.mp4", "mask.mkv", tmp_path / "out.mp4", video_width=1920)


@pytest.mark.parametrize("width", [640, 1920, 3840])
def test_radius_never_zero_for_real_widths(width):
    """Regression guard on the whole point of the module: no supported frame
    width may resolve to a radius that leaves a face legible."""
    assert ap.blur_radius(width) >= 2
    assert isinstance(ap.blur_radius(width), (int, np.integer))
