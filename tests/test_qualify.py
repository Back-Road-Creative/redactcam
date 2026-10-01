"""Clean-machine qualification: the pieces that decide whether a fresh install
can actually redact a video, not just whether the download finished.

Everything real-machine (a disposable supported OS, the supplied model files, a
GPU) is out of reach for a hermetic suite, so the loaders and the render are
seams here; what is proven is the decision logic, the error text a person acts
on, and that the synthetic clip generator's ffmpeg command works.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from redactcam import apply, qualify
from redactcam.models import ModelSpec


def _fake_clip(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"clip")


class TestFfmpegTools:
    def test_a_missing_binary_is_named_with_the_install_line(self, monkeypatch):
        monkeypatch.setattr(apply.shutil, "which", lambda name: None)
        with pytest.raises(apply.FfmpegUnavailableError) as exc:
            apply.require_ffmpeg()
        msg = str(exc.value)
        assert "ffmpeg was not found on PATH" in msg
        assert "ffprobe was not found on PATH" in msg
        assert "winget install" in msg

    def test_a_binary_that_will_not_run_is_not_a_pass(self, monkeypatch):
        monkeypatch.setattr(apply.shutil, "which", lambda name: f"/x/{name}")

        def _boom(*a, **k):
            raise OSError("exec format error")

        monkeypatch.setattr(apply.subprocess, "run", _boom)
        status = apply.tool_status("ffmpeg")
        assert status["version"] is None and "exec format error" in status["error"]

    def test_a_healthy_tool_reports_its_version_line(self, monkeypatch):
        monkeypatch.setattr(apply.shutil, "which", lambda name: f"/x/{name}")
        monkeypatch.setattr(
            apply.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="ffmpeg version 7.1\nmore\n"),
        )
        assert apply.tool_status("ffmpeg")["version"] == "ffmpeg version 7.1"
        assert set(apply.require_ffmpeg()) == {"ffmpeg", "ffprobe"}


def _specs(tmp_path, kinds=("face", "plate", "vehicle", "person")):
    out = {}
    for k in kinds:
        p = tmp_path / f"{k}.onnx"
        p.write_bytes(f"w-{k}".encode())
        out[k] = ModelSpec(name=k, path=str(p))
    return out


class TestCheckModels:
    def test_a_loadable_model_is_ok_with_its_hash_and_provider(self, tmp_path):
        specs = _specs(tmp_path, ("face",))
        out = qualify.check_models(specs, loader=lambda kind, path: "CPUExecutionProvider")
        assert out["face"]["status"] == "ok"
        assert out["face"]["provider"] == "CPUExecutionProvider"
        assert len(out["face"]["sha256"]) == 64
        assert "plate" not in out  # not asked about, not required

    def test_a_missing_file_says_which_flag_to_fix(self, tmp_path):
        specs = {"plate": ModelSpec(name="plate", path=str(tmp_path / "nope.onnx"))}
        out = qualify.check_models(specs, loader=lambda k, p: "x")
        assert out["plate"]["status"] == "missing"
        assert "--model plate=" in out["plate"]["error"]

    def test_a_file_the_runtime_rejects_is_failed_with_the_reason(self, tmp_path):
        def _bad(kind, path):
            raise RuntimeError("INVALID_PROTOBUF: not an ONNX file")

        out = qualify.check_models(_specs(tmp_path, ("face",)), loader=_bad)
        assert out["face"]["status"] == "failed"
        assert "INVALID_PROTOBUF" in out["face"]["error"]

    def test_require_all_flags_every_unsupplied_kind_and_never_downloads(self, tmp_path):
        out = qualify.check_models(
            _specs(tmp_path, ("face",)), require_all=True, loader=lambda k, p: "CPUExecutionProvider"
        )
        for kind in ("plate", "vehicle", "person"):
            assert out[kind]["status"] == "not_supplied"
            assert "download" in out[kind]["error"]

    def test_a_url_only_spec_counts_as_not_supplied(self):
        """The qualification must judge supplied artifacts, not whatever a
        network fetch happened to return."""
        specs = {"face": ModelSpec(name="face", url="https://x/f.onnx", sha256="ab" * 32)}
        out = qualify.check_models(specs, require_all=True, loader=lambda k, p: "x")
        assert out["face"]["status"] == "not_supplied"


class TestSyntheticRender:
    def test_needs_all_four_supplied_and_downloads_nothing(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(qualify, "_make_synthetic_input", lambda p: called.append(p))
        out = qualify.synthetic_render(_specs(tmp_path, ("face",)), tmp_path / "w")
        assert out["attempted"] is False and out["ok"] is False
        assert "all four" in out["error"]
        assert called == []

    def test_a_completed_render_is_recorded_with_its_hash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qualify, "_make_synthetic_input", _fake_clip)

        def _fake(src, out=None, **kw):
            o = Path(kw["work_dir"]) / "o.mp4"
            o.write_bytes(b"rendered")
            return SimpleNamespace(output=o, frame_count=20, receipt=None, coverage=None)

        monkeypatch.setattr("redactcam.pipeline.redact_video", _fake)
        out = qualify.synthetic_render(_specs(tmp_path), tmp_path / "w")
        assert out["ok"] is True and out["frames"] == 20
        assert len(out["output_sha256"]) == 64

    def test_a_pipeline_failure_is_recorded_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qualify, "_make_synthetic_input", _fake_clip)

        def _boom(*a, **k):
            raise RuntimeError("mask ffmpeg failed (rc=1): no libx264")

        monkeypatch.setattr("redactcam.pipeline.redact_video", _boom)
        out = qualify.synthetic_render(_specs(tmp_path), tmp_path / "w")
        assert out["ok"] is False and "no libx264" in out["error"]

    def test_an_empty_output_is_not_a_render(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qualify, "_make_synthetic_input", _fake_clip)

        def _empty(src, out=None, **kw):
            o = Path(kw["work_dir"]) / "o.mp4"
            o.write_bytes(b"")
            return SimpleNamespace(output=o, frame_count=20, receipt=None, coverage=None)

        monkeypatch.setattr("redactcam.pipeline.redact_video", _empty)
        assert qualify.synthetic_render(_specs(tmp_path), tmp_path / "w")["ok"] is False

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    def test_the_generator_command_makes_a_real_clip(self, tmp_path):
        src = tmp_path / "in.mp4"
        qualify._make_synthetic_input(src)
        assert src.stat().st_size > 0
        if shutil.which("ffprobe"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0", str(src)],
                capture_output=True, text=True,
            )
            assert probe.stdout.strip() == "640,360"


class TestReceipt:
    def _ok_inputs(self):
        return dict(
            diagnostics={"errors": [], "onnxruntime": "1.0"},
            models={k: {"status": "ok"} for k in ("face", "plate", "vehicle", "person")},
            render={"attempted": True, "ok": True},
        )

    def test_qualified_needs_diagnostics_detectors_and_render(self):
        r = qualify.build_clean_machine_receipt(**self._ok_inputs())
        assert r["schema"] == "redactcam.clean-machine-receipt/v1"
        assert r["qualified"] is True
        assert r["identity"]["redactcam"]["code_sha256"]
        assert r["launch"]["python"]

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda i: i["diagnostics"].update(errors=["ffmpeg was not found on PATH"]),
            lambda i: i["models"].update(plate={"status": "failed"}),
            lambda i: i["models"].pop("person"),
            lambda i: i.update(render={"attempted": False, "ok": False}),
            lambda i: i.update(render={"attempted": True, "ok": False}),
        ],
    )
    def test_any_gap_is_not_qualified(self, mutate):
        inputs = self._ok_inputs()
        mutate(inputs)
        assert qualify.build_clean_machine_receipt(**inputs)["qualified"] is False
