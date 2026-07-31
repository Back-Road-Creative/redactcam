"""End-to-end wiring, with every expensive stage stubbed.

What matters here is the ordering contract, not the pixels: verification runs
before the render, and a failed verification stops the render happening at all.
"""

import pytest

from redactcam import pipeline as pl
from redactcam.coverage import CoverageReport, Leak
from redactcam.models import ModelSpec


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Stub detection, mask rendering, model resolution and ffmpeg; record calls."""
    calls = []

    monkeypatch.setattr(pl, "resolve_model", lambda spec, cache=None: tmp_path / f"{spec.name}.onnx")
    monkeypatch.setattr(pl._timeline, "probe_video", lambda p: (30.0, 90, 1920, 1080))
    monkeypatch.setattr(
        pl,
        "detect_and_track",
        lambda video, **kw: (calls.append("detect") or ({0: [(10, 10, 20, 20)]}, 90, 1, 0)),
    )
    monkeypatch.setattr(
        pl._mask,
        "materialize_mask_video",
        lambda tl, path, **kw: (calls.append("mask"), path.write_bytes(b"m"), path)[2],
    )
    monkeypatch.setattr(
        pl._apply,
        "apply_blur",
        lambda src, mask, out, **kw: (calls.append("render"), out)[1],
    )
    monkeypatch.setattr(
        pl,
        "verify_cabin_coverage",
        lambda *a, **k: (calls.append("verify"), CoverageReport(vehicle_frames=3))[1],
    )
    return calls


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"pretend this is a video" * 10)
    return p


def test_verifies_before_rendering(wired, source, tmp_path):
    result = pl.redact_video(source, work_dir=tmp_path / "work")
    assert wired == ["detect", "mask", "verify", "render"]
    assert wired.index("verify") < wired.index("render")
    assert result.output is not None
    assert result.coverage.vehicle_frames == 3


def test_failed_coverage_refuses_to_render(wired, source, tmp_path, monkeypatch):
    """The whole point: a mask that misses a driver must stop the pipeline, not
    produce a plausible-looking file that ships someone's face."""
    leaky = CoverageReport(vehicle_frames=4, leaks=[Leak(7, 0.1, (0, 0, 10, 10))])
    monkeypatch.setattr(
        pl, "verify_cabin_coverage", lambda *a, **k: (wired.append("verify"), leaky)[1]
    )
    with pytest.raises(pl.CoverageError) as exc:
        pl.redact_video(source, work_dir=tmp_path / "work")
    assert "render" not in wired
    assert exc.value.report is leaky
    assert "frame 7" in str(exc.value)


def test_no_render_stops_after_verification(wired, source, tmp_path):
    result = pl.redact_video(source, work_dir=tmp_path / "work", render=False)
    assert "render" not in wired
    assert result.output is None
    assert result.mask.exists()


def test_sidecar_is_reused_on_a_hash_match(wired, source, tmp_path):
    work = tmp_path / "work"
    pl.redact_video(source, work_dir=work, render=False)
    assert wired.count("detect") == 1
    pl.redact_video(source, work_dir=work, render=False)
    assert wired.count("detect") == 1  # second run skipped the expensive stage


def test_changed_source_invalidates_the_sidecar(wired, source, tmp_path):
    """Keyed on content, not filename: a re-encoded clip must not reuse boxes
    computed from different pixels."""
    work = tmp_path / "work"
    pl.redact_video(source, work_dir=work, render=False)
    source.write_bytes(b"different pixels entirely" * 10)
    pl.redact_video(source, work_dir=work, render=False)
    assert wired.count("detect") == 2


def test_fresh_run_ignores_the_sidecar(wired, source, tmp_path):
    work = tmp_path / "work"
    pl.redact_video(source, work_dir=work, render=False)
    pl.redact_video(source, work_dir=work, render=False, reuse_sidecar=False)
    assert wired.count("detect") == 2


def test_verification_can_be_skipped(wired, source, tmp_path):
    pl.redact_video(source, work_dir=tmp_path / "work", verify=False)
    assert "verify" not in wired


def test_model_overrides_reach_resolution(monkeypatch, wired, source, tmp_path):
    seen = {}
    monkeypatch.setattr(
        pl,
        "resolve_model",
        lambda spec, cache=None: seen.setdefault(spec.name, spec) and None or tmp_path / "m.onnx",
    )
    pl.redact_video(
        source,
        work_dir=tmp_path / "work",
        render=False,
        models={"plate": ModelSpec(name="plate", path="/custom/plate.onnx")},
    )
    assert seen["plate"].path == "/custom/plate.onnx"
    assert seen["face"].url.startswith("https://")  # untouched default


def test_file_hash_is_content_addressed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert pl.file_hash(a) == pl.file_hash(b)
    b.write_bytes(b"different")
    assert pl.file_hash(a) != pl.file_hash(b)
