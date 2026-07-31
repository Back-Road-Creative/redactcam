"""Command-line entry point: ``redactcam <input> [-o output]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    p.add_argument("input", type=Path, help="source video")
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
    return p


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
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
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
