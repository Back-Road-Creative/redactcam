"""CLI argument handling and exit codes."""

import pytest

from redactcam import cli
from redactcam.coverage import CoverageReport, Leak
from redactcam.pipeline import CoverageError, RedactionResult


def _result(tmp_path, coverage=None, output=None):
    return RedactionResult(
        sidecar=tmp_path / "clip.json",
        mask=tmp_path / "clip_mask.mkv",
        output=output,
        frame_count=90,
        face_detections=3,
        plate_detections=1,
        coverage=coverage,
    )


def test_success_prints_artefacts_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "redact_video",
        lambda *a, **k: _result(tmp_path, CoverageReport(vehicle_frames=5), tmp_path / "out.mp4"),
    )
    assert cli.main(["clip.mp4"]) == 0
    out = capsys.readouterr().out
    assert "clip_mask.mkv" in out and "out.mp4" in out
    assert "5 vehicle sightings" in out


def test_coverage_failure_exits_two(tmp_path, monkeypatch, capsys):
    """A distinct exit code, so a calling script can tell "the blur is unsafe"
    apart from "the file was missing"."""

    def _boom(*a, **k):
        raise CoverageError("cabin exposed", CoverageReport(1, [Leak(2, 0.0, (0, 0, 1, 1))]))

    monkeypatch.setattr(cli, "redact_video", _boom)
    assert cli.main(["clip.mp4"]) == 2
    assert "coverage check FAILED" in capsys.readouterr().err


def test_ordinary_failure_exits_one(monkeypatch, capsys):
    def _boom(*a, **k):
        raise ValueError("Could not open video")

    monkeypatch.setattr(cli, "redact_video", _boom)
    assert cli.main(["missing.mp4"]) == 1
    assert "Could not open video" in capsys.readouterr().err


class TestModelFlag:
    def test_parses_kind_equals_path(self):
        specs = cli._models(["plate=/tmp/p.onnx", "face=/tmp/f.onnx"])
        assert specs["plate"].path == "/tmp/p.onnx"
        assert specs["face"].name == "face"

    def test_rejects_unknown_kind(self):
        with pytest.raises(SystemExit, match="unknown model kind"):
            cli._models(["banana=/tmp/b.onnx"])

    def test_rejects_missing_path(self):
        with pytest.raises(SystemExit, match="KIND=PATH"):
            cli._models(["plate"])


def test_flags_reach_the_pipeline(tmp_path, monkeypatch):
    seen = {}

    def _capture(*a, **k):
        seen.update(k)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "redact_video", _capture)
    cli.main(["clip.mp4", "--no-verify", "--no-render", "--fresh", "--blur-strength", "20"])
    assert seen["verify"] is False
    assert seen["render"] is False
    assert seen["reuse_sidecar"] is False
    assert seen["blur_strength"] == 20


_HEALTHY_FFMPEG = {
    "ffmpeg": {"name": "ffmpeg", "path": "/x/ffmpeg", "version": "ffmpeg version 7.1", "error": None},
    "ffprobe": {"name": "ffprobe", "path": "/x/ffprobe", "version": "ffprobe version 7.1", "error": None},
}
_NO_FFMPEG = {
    n: {"name": n, "path": None, "version": None, "error": f"{n} was not found on PATH"}
    for n in ("ffmpeg", "ffprobe")
}


class TestCheckDeps:
    """`--check-deps` exists so the frozen Windows build can prove its native
    extensions resolved. `--help` never touched onnxruntime -- detect.py imports
    it inside a function -- so the release smoke test passed against an
    executable that might not have been able to detect anything."""

    @pytest.fixture(autouse=True)
    def _host_independent(self, monkeypatch):
        """These exercise the real import path, so they must not depend on which
        onnxruntime wheel the HOST venv has: a GPU build that lost CUDA makes
        --check-deps exit 1 by design (tested below with a stub). Accept the
        CPU path here; the assertion itself is unit-tested in test_detect."""
        monkeypatch.setenv("REDACTCAM_ALLOW_CPU", "1")
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _HEALTHY_FFMPEG)

    def test_reports_both_extensions_and_exits_zero(self, capsys):
        assert cli.main(["--check-deps"]) == 0
        out = capsys.readouterr().out
        assert "opencv" in out
        assert "onnxruntime" in out

    def test_actually_builds_a_session_options(self, capsys):
        """Importing onnxruntime is not the same as it working. This goes through
        detect._session_options, the real call the detectors make."""
        cli.main(["--check-deps"])
        assert "session options" in capsys.readouterr().out

    def test_needs_no_input_file(self):
        """The point is to run on a machine with no video on it."""
        assert cli.main(["--check-deps"]) == 0

    def test_input_is_still_required_without_it(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_reports_the_inference_provider_decision(self, monkeypatch, capsys):
        """The 2026-09-25 incident was invisible to ``--check-deps`` as it stood:
        it printed the provider list and exited 0 whether or not a detector
        would actually use CUDA. It now prints the decision the detectors make."""
        monkeypatch.setattr(
            cli,
            "_check_inference_providers_for_check",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert cli.main(["--check-deps"]) == 0
        out = capsys.readouterr().out
        assert "inference       CUDAExecutionProvider" in out

    def test_a_gpu_build_that_lost_cuda_is_a_nonzero_exit(self, monkeypatch, capsys):
        from redactcam.detect import GpuProviderUnavailableError

        def _lost():
            raise GpuProviderUnavailableError("onnxruntime-gpu installed, CUDA missing")

        monkeypatch.setattr(cli, "_check_inference_providers_for_check", _lost)
        assert cli.main(["--check-deps"]) == 1
        assert "CUDA missing" in capsys.readouterr().err

    def test_a_missing_extension_is_a_nonzero_exit(self, monkeypatch, capsys):
        def _boom(_):
            raise ImportError("no onnxruntime here")

        monkeypatch.setattr(cli, "_session_options_for_check", _boom)
        assert cli.main(["--check-deps"]) == 1
        assert "no onnxruntime here" in capsys.readouterr().err


class TestCheckDepsFfmpegAndModels:
    """The clean-machine gaps: a first run on a fresh box fails on a missing
    ffmpeg or an unusable model file, not on anything --check-deps used to test."""

    @pytest.fixture(autouse=True)
    def _host_independent(self, monkeypatch):
        monkeypatch.setenv("REDACTCAM_ALLOW_CPU", "1")
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _HEALTHY_FFMPEG)

    def test_reports_the_ffmpeg_and_ffprobe_builds(self, capsys):
        assert cli.main(["--check-deps"]) == 0
        out = capsys.readouterr().out
        assert "ffmpeg          ffmpeg version 7.1" in out
        assert "ffprobe         ffprobe version 7.1" in out

    def test_a_missing_ffmpeg_is_a_nonzero_exit_with_the_install_line(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _NO_FFMPEG)
        assert cli.main(["--check-deps"]) == 1
        err = capsys.readouterr().err
        assert "ffmpeg was not found on PATH" in err
        assert "winget install" in err

    def test_skip_ffmpeg_is_for_a_build_smoke_test_that_has_none(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _NO_FFMPEG)
        assert cli.main(["--check-deps", "--skip-ffmpeg"]) == 0
        assert "skipped" in capsys.readouterr().out

    def test_supplied_models_are_checked_and_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.qualify,
            "check_models",
            lambda specs, **kw: {
                "face": {"status": "ok", "sha256": "ab" * 32, "provider": "CPUExecutionProvider"}
            },
        )
        assert cli.main(["--check-deps", "--model", "face=/m/f.onnx"]) == 0
        assert "model face      ok" in capsys.readouterr().out

    def test_a_model_the_runtime_rejects_is_a_nonzero_exit(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.qualify,
            "check_models",
            lambda specs, **kw: {"plate": {"status": "failed", "error": "plate model x did not load"}},
        )
        assert cli.main(["--check-deps", "--model", "plate=/m/p.onnx"]) == 1
        assert "plate model x did not load" in capsys.readouterr().err

    def test_no_model_flag_means_no_model_check_and_no_download(self, monkeypatch):
        def _never(*a, **k):
            raise AssertionError("must not check models when none were supplied")

        monkeypatch.setattr(cli.qualify, "check_models", _never)
        assert cli.main(["--check-deps"]) == 0


class TestCleanMachineReceipt:
    @pytest.fixture(autouse=True)
    def _host_independent(self, monkeypatch):
        monkeypatch.setenv("REDACTCAM_ALLOW_CPU", "1")
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _HEALTHY_FFMPEG)

    ALL = ["--model", "face=/m/f", "--model", "plate=/m/p", "--model", "vehicle=/m/v",
           "--model", "person=/m/n"]

    def _fakes(self, monkeypatch, *, models_ok=True, render_ok=True):
        seen = {"render_called": False, "require_all": None}

        def _models(specs, *, require_all=False, **kw):
            seen["require_all"] = require_all
            st = "ok" if models_ok else "failed"
            return {
                k: {"status": st, "error": "bad", "sha256": "cd" * 32, "provider": "CPUExecutionProvider"}
                for k in ("face", "plate", "vehicle", "person")
            }

        def _render(specs, work_dir):
            seen["render_called"] = True
            return {
                "attempted": True,
                "ok": render_ok,
                "error": None if render_ok else "boom",
                "frames": 20,
                "output_sha256": "ef" * 32,
            }

        monkeypatch.setattr(cli.qualify, "check_models", _models)
        monkeypatch.setattr(cli.qualify, "synthetic_render", _render)
        return seen

    def test_a_qualified_run_writes_the_receipt_and_exits_zero(self, tmp_path, monkeypatch):
        import json

        seen = self._fakes(monkeypatch)
        out = tmp_path / "r.json"
        assert cli.main(["--check-deps", "--receipt", str(out), *self.ALL]) == 0
        r = json.loads(out.read_text())
        assert r["schema"] == "redactcam.clean-machine-receipt/v1"
        assert r["qualified"] is True
        assert r["diagnostics"]["onnxruntime"]
        assert r["diagnostics"]["ffmpeg"]["version"] == "ffmpeg version 7.1"
        assert seen["require_all"] is True and seen["render_called"] is True

    def test_a_failed_render_is_a_written_receipt_and_a_nonzero_exit(self, tmp_path, monkeypatch, capsys):
        import json

        self._fakes(monkeypatch, render_ok=False)
        out = tmp_path / "r.json"
        assert cli.main(["--check-deps", "--receipt", str(out), *self.ALL]) == 1
        assert json.loads(out.read_text())["qualified"] is False
        assert "boom" in capsys.readouterr().err

    def test_a_failed_detector_skips_the_render_but_still_writes_the_receipt(
        self, tmp_path, monkeypatch
    ):
        import json

        seen = self._fakes(monkeypatch, models_ok=False)
        out = tmp_path / "r.json"
        assert cli.main(["--check-deps", "--receipt", str(out), *self.ALL]) == 1
        r = json.loads(out.read_text())
        assert seen["render_called"] is False
        assert r["render"]["attempted"] is False and r["qualified"] is False

    def test_missing_ffmpeg_is_recorded_and_fails_the_receipt(self, tmp_path, monkeypatch):
        import json

        self._fakes(monkeypatch)
        monkeypatch.setattr(cli, "_ffmpeg_tools_for_check", lambda: _NO_FFMPEG)
        out = tmp_path / "r.json"
        assert cli.main(["--check-deps", "--receipt", str(out), *self.ALL]) == 1
        r = json.loads(out.read_text())
        assert r["qualified"] is False
        assert any("ffmpeg" in e for e in r["diagnostics"]["errors"])

    def test_a_broken_native_stack_still_produces_a_receipt(self, tmp_path, monkeypatch):
        import json

        self._fakes(monkeypatch)

        def _boom(_):
            raise ImportError("no onnxruntime here")

        monkeypatch.setattr(cli, "_session_options_for_check", _boom)
        out = tmp_path / "r.json"
        assert cli.main(["--check-deps", "--receipt", str(out), *self.ALL]) == 1
        r = json.loads(out.read_text())
        assert r["qualified"] is False
        assert "no onnxruntime here" in " ".join(r["diagnostics"]["errors"])

    def test_receipt_needs_check_deps(self):
        with pytest.raises(SystemExit):
            cli.main(["--receipt", "r.json"])

    def test_receipt_cannot_skip_ffmpeg(self):
        with pytest.raises(SystemExit):
            cli.main(["--check-deps", "--skip-ffmpeg", "--receipt", "r.json"])


class TestIdentityFlags:
    @pytest.fixture(autouse=True)
    def _fixed(self, monkeypatch):
        from redactcam import provenance

        monkeypatch.setattr(
            provenance,
            "runtime_identity",
            lambda: {
                "redactcam": {
                    "version": "0.2.0",
                    "code_sha256": "ab" * 32,
                    "revision": {"commit": "c34854e" + "0" * 33, "source": "git"},
                }
            },
        )

    def test_identity_prints_json_and_needs_no_input(self, capsys):
        import json

        assert cli.main(["--identity"]) == 0
        assert json.loads(capsys.readouterr().out)["redactcam"]["version"] == "0.2.0"

    def test_a_matching_expectation_exits_zero(self):
        assert cli.main(["--expect-revision", "c34854e"]) == 0

    def test_a_stale_install_is_a_nonzero_exit_that_says_so(self, capsys):
        assert cli.main(["--expect-revision", "6d731a5fb"]) == 1
        assert "not the expected build" in capsys.readouterr().err

    def test_a_too_short_revision_is_refused(self, capsys):
        assert cli.main(["--expect-revision", "c34"]) == 1
        assert "at least 7" in capsys.readouterr().err
