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
