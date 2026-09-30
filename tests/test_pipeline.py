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


class TestRenderReceipt:
    """A render must say exactly what produced it: the code, the model files, the
    native runtime -- not a version string that did not change across the
    tracking fix."""

    @pytest.fixture
    def real_models(self, monkeypatch, tmp_path):
        files = {}

        def _resolve(spec, cache=None):
            p = tmp_path / f"{spec.name}.onnx"
            p.write_bytes(f"weights-of-{spec.name}".encode())
            files[spec.name] = p
            return p

        monkeypatch.setattr(pl, "resolve_model", _resolve)
        return files

    def _receipt(self, result):
        import json

        return json.loads(result.receipt.read_text())

    def test_a_receipt_is_written_beside_the_outputs(self, wired, real_models, source, tmp_path):
        result = pl.redact_video(source, work_dir=tmp_path / "work")
        assert result.receipt == tmp_path / "work" / "clip_redactcam_receipt.json"
        r = self._receipt(result)
        assert r["schema"] == "redactcam.render-receipt/v1"

    def test_it_carries_code_and_model_hashes(self, wired, real_models, source, tmp_path):
        from redactcam.provenance import code_sha256, sha256_file

        r = self._receipt(pl.redact_video(source, work_dir=tmp_path / "work"))
        assert r["identity"]["redactcam"]["code_sha256"] == code_sha256()[0]
        for kind, path in real_models.items():
            assert r["models"][kind]["sha256"] == sha256_file(path)
        assert r["identity"]["native"]["opencv"]

    def test_it_carries_source_and_output_hashes(self, wired, real_models, source, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pl._apply,
            "apply_blur",
            lambda src, mask, out, **kw: (out.write_bytes(b"rendered"), out)[1],
        )
        result = pl.redact_video(source, work_dir=tmp_path / "work")
        r = self._receipt(result)
        assert r["source_sha256"] == pl.file_hash(source)
        assert r["output_sha256"] == pl.file_hash(result.output)

    def test_no_render_has_no_output_hash(self, wired, real_models, source, tmp_path):
        r = self._receipt(pl.redact_video(source, work_dir=tmp_path / "work", render=False))
        assert r["output_sha256"] is None
        assert r["settings"]["render"] is False

    def test_a_reused_timeline_is_labelled_as_not_produced_by_this_run(
        self, wired, real_models, source, tmp_path
    ):
        """The sidecar does not record what produced it, so a reused timeline's
        code and models are unknown -- the receipt must not imply otherwise."""
        work = tmp_path / "work"
        first = self._receipt(pl.redact_video(source, work_dir=work, render=False))
        again = self._receipt(pl.redact_video(source, work_dir=work, render=False))
        assert first["detection"] == "fresh"
        assert again["detection"] == "reused_sidecar"

    def test_it_records_coverage_and_settings(self, wired, real_models, source, tmp_path):
        r = self._receipt(pl.redact_video(source, work_dir=tmp_path / "work", blur_strength=20))
        assert r["coverage"] == {"checked": True, "ok": True, "vehicle_frames": 3, "leaks": 0}
        assert r["settings"]["blur_strength"] == 20
        assert "face_model_path" not in r["settings"]["detect"]  # paths live under models

    def test_skipped_verification_is_recorded_as_unchecked(self, wired, real_models, source, tmp_path):
        r = self._receipt(pl.redact_video(source, work_dir=tmp_path / "work", verify=False))
        assert r["coverage"]["checked"] is False and r["coverage"]["ok"] is None

    def test_an_unresolved_model_is_recorded_as_missing(self, wired, source, tmp_path, monkeypatch):
        monkeypatch.setattr(pl, "resolve_model", lambda spec, cache=None: None)
        r = self._receipt(pl.redact_video(source, work_dir=tmp_path / "work", verify=False))
        assert r["models"]["plate"] is None

