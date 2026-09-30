"""Command-line entry point: ``redactcam <input> [-o output]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from . import provenance, qualify
from .apply import FFMPEG_INSTALL_HINT, FFMPEG_TOOLS, tool_status
from .models import DEFAULT_MODELS, ModelSpec
from .pipeline import CoverageError, redact_video


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="redactcam",
        description=(
            "Blur faces, licence plates, vehicle occupants and people in a video, "
            "then verify the blur actually landed before rendering."
        ),
    )
    p.add_argument("input", type=Path, nargs="?", help="source video")
    p.add_argument("-o", "--output", type=Path, help="output video (default: <input>_redacted.mp4)")
    p.add_argument(
        "--work-dir",
        type=Path,
        help="where the timeline sidecar and mask are written (default: alongside the input)",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help=(
            "use a local ONNX file instead of downloading, e.g. --model plate=./plate.onnx. "
            f"KIND is one of: {', '.join(sorted(DEFAULT_MODELS))}. Repeatable."
        ),
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the independent coverage check (not recommended)",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="stop after the mask and the coverage check; do not encode a video",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any existing timeline sidecar and re-detect from scratch",
    )
    p.add_argument(
        "--blur-strength",
        type=int,
        default=12,
        help="boxblur radius at 1920 px wide, scaled to the real width (default: 12)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="log every stage")
    p.add_argument(
        "--check-deps",
        action="store_true",
        help=(
            "load the native extensions, check ffmpeg/ffprobe, verify any --model files "
            "load, report versions, and exit; takes no input"
        ),
    )
    p.add_argument(
        "--skip-ffmpeg",
        action="store_true",
        help="with --check-deps: do not require ffmpeg/ffprobe (for a build smoke test "
        "on a runner that has none; not valid with --receipt)",
    )
    p.add_argument(
        "--receipt",
        type=Path,
        metavar="FILE",
        help=(
            "with --check-deps: qualify this machine and write a versioned clean-machine "
            "receipt. Needs all four --model KIND=PATH files (nothing is downloaded); runs "
            "one synthetic clip through the whole pipeline; exits 0 only if qualified"
        ),
    )
    p.add_argument(
        "--identity",
        action="store_true",
        help="print the installed code/native-runtime identity as JSON and exit",
    )
    p.add_argument("--expect-version", metavar="X.Y.Z", help="exit 1 unless the installed version matches")
    p.add_argument(
        "--expect-code-sha256",
        metavar="HEX",
        help="exit 1 unless the installed sources hash to this (see --identity)",
    )
    p.add_argument(
        "--expect-revision",
        metavar="COMMIT",
        help="exit 1 unless the installed git commit starts with this (>= 7 chars)",
    )
    return p


def _session_options_for_check(intra_threads: int):
    """Indirection so the dependency check can be driven in a test on a machine
    where onnxruntime is deliberately absent."""
    from .detect import _session_options

    return _session_options(intra_threads)


def _check_inference_providers_for_check() -> list[str]:
    """Indirection for the same reason as ``_session_options_for_check``."""
    from .detect import check_inference_providers

    return check_inference_providers()


def _ffmpeg_tools_for_check() -> dict[str, dict]:
    """Indirection so the ffmpeg check can be driven in a test on any machine."""
    return {name: tool_status(name) for name in FFMPEG_TOOLS}


def _check_deps(
    models: dict[str, ModelSpec] | None = None,
    *,
    receipt: Path | None = None,
    skip_ffmpeg: bool = False,
    work_dir: Path | None = None,
) -> int:
    """Load the native extensions, use them, and say what loaded.

    This is for the frozen Windows build and for a first run on a clean machine.
    PyInstaller resolves OpenCV and onnxruntime through hooks, and a miss surfaces
    only when someone runs a real detection -- while the release smoke test was
    ``--help``, which never touches onnxruntime, because detect.py imports it
    inside a function. The installer could therefore ship an executable that
    could not detect anything with every gate green.

    Importing alone would not prove much, so this builds a real SessionOptions
    through the same helper the detectors call. With no ``--model`` it needs no
    video and no model file: the weights are a ~110 MB download and a release
    runner has neither. ffmpeg and ffprobe are checked because the mask render
    and the blur composite shell out to them; ``--model`` files are loaded and run
    once, never downloaded.

    With ``receipt`` it is a qualification: every check runs even after one
    fails (so the receipt says everything that is wrong, not the first thing),
    all four models are required, a synthetic clip is rendered, and the receipt
    is written whatever happened. Exit 0 only if it qualified.
    """
    from .detect import GpuProviderUnavailableError, _import_onnxruntime

    models = models or {}
    diag: dict = {"errors": []}
    errors: list[str] = diag["errors"]

    def fail(msg: str) -> None:
        print(f"redactcam: {msg}", file=sys.stderr)
        errors.append(msg)

    try:
        import cv2

        opts = _session_options_for_check(1)
        ort = _import_onnxruntime()
    except ImportError as exc:
        fail(f"a native dependency did not load: {exc}")
    else:
        providers = list(ort.get_available_providers())
        print(f"opencv          {cv2.__version__}")
        print(f"onnxruntime     {ort.__version__}")
        print(f"providers       {', '.join(providers)}")
        print(f"session options intra={opts.intra_op_num_threads} inter={opts.inter_op_num_threads}")
        diag.update(opencv=cv2.__version__, onnxruntime=ort.__version__, providers=providers)
        # The provider LIST above is what the wheel advertises; this line is the
        # decision a detector makes from it — and it is where a GPU build that lost
        # CUDA fails, instead of exiting 0 and running on CPU for a day.
        try:
            inference = _check_inference_providers_for_check()
        except GpuProviderUnavailableError as exc:
            fail(str(exc))
        else:
            print(f"inference       {inference[0]}")
            diag["inference"] = inference[0]

    if skip_ffmpeg:
        print("ffmpeg          skipped (--skip-ffmpeg)")
    else:
        tools = _ffmpeg_tools_for_check()
        broken = False
        for name, status in tools.items():
            diag[name] = status
            if status["error"]:
                fail(status["error"])
                broken = True
            else:
                print(f"{name:<15} {status['version']}")
        if broken:
            print(f"redactcam: {FFMPEG_INSTALL_HINT}", file=sys.stderr)

    qualifying = receipt is not None
    model_results: dict[str, dict] = {}
    if models or qualifying:
        model_results = qualify.check_models(models, require_all=qualifying)
        for kind, m in model_results.items():
            if m["status"] == "ok":
                print(f"model {kind:<9} ok  {m['sha256'][:12]}  {m['provider']}")
            else:
                fail(m["error"])

    if not qualifying:
        return 1 if errors else 0

    render = {"attempted": False, "ok": False, "error": "skipped: an earlier check failed"}
    if not errors:
        with tempfile.TemporaryDirectory(prefix="redactcam-qualify-") as tmp:
            render = qualify.synthetic_render(models, work_dir or Path(tmp))
        if render.get("ok"):
            print(f"render          ok  {render['frames']} frames  {render['output_sha256'][:12]}")
        else:
            fail(f"synthetic render failed: {render.get('error')}")
    result = qualify.build_clean_machine_receipt(
        diagnostics=diag, models=model_results, render=render
    )
    provenance.write_receipt(receipt, result)
    print(f"receipt         {receipt} qualified={str(result['qualified']).lower()}")
    return 0 if result["qualified"] else 1


def _check_expectations(args: argparse.Namespace) -> int:
    """Exit 1 when the installed package is not the build the caller expected."""
    try:
        provenance.check_current(
            expected_version=args.expect_version,
            expected_code_sha256=args.expect_code_sha256,
            expected_revision=args.expect_revision,
        )
    except (provenance.StaleInstallError, ValueError) as exc:
        print(f"redactcam: {exc}", file=sys.stderr)
        return 1
    print("identity        matches the expected build")
    return 0


def _models(pairs: list[str]) -> dict[str, ModelSpec]:
    out: dict[str, ModelSpec] = {}
    for pair in pairs:
        kind, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--model expects KIND=PATH, got {pair!r}")
        if kind not in DEFAULT_MODELS:
            raise SystemExit(f"unknown model kind {kind!r}; expected one of {sorted(DEFAULT_MODELS)}")
        out[kind] = ModelSpec(name=kind, path=path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if (args.receipt or args.skip_ffmpeg) and not args.check_deps:
        parser.error("--receipt and --skip-ffmpeg only apply together with --check-deps")
    if args.receipt and args.skip_ffmpeg:
        parser.error("--receipt qualifies a machine to render, so it cannot skip ffmpeg")
    expecting = args.expect_version or args.expect_code_sha256 or args.expect_revision
    if args.check_deps or expecting or args.identity:
        rc = 0
        if args.check_deps:
            rc = _check_deps(
                _models(args.model),
                receipt=args.receipt,
                skip_ffmpeg=args.skip_ffmpeg,
                work_dir=args.work_dir,
            )
        if expecting:
            rc = max(rc, _check_expectations(args))
        if args.identity:
            print(json.dumps(provenance.runtime_identity(), indent=2, sort_keys=True))
        return rc
    if args.input is None:
        parser.error("input is required")
    try:
        result = redact_video(
            args.input,
            args.output,
            work_dir=args.work_dir,
            models=_models(args.model),
            verify=not args.no_verify,
            render=not args.no_render,
            reuse_sidecar=not args.fresh,
            blur_strength=args.blur_strength,
        )
    except CoverageError as exc:
        print(f"coverage check FAILED: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"redactcam: {exc}", file=sys.stderr)
        return 1

    print(f"timeline  {result.sidecar}")
    print(f"mask      {result.mask} ({result.frame_count} frames)")
    if result.coverage is not None:
        print(f"coverage  OK across {result.coverage.vehicle_frames} vehicle sightings")
    if result.output is not None:
        print(f"output    {result.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
