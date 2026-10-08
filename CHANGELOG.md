# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Detection progress heartbeat.** `detect_and_track(progress_every_s=60.0)` logs
  frame, percentage, detection count and fps about once a minute (wall clock) plus a
  final `detect: done` line; `0` disables it. A 54 197-frame pass was silent for 4.5 h.
  Logging only, results unchanged.

### Changed

- **`onnxruntime` is no longer a base dependency — install `redactcam[cpu]` or
  `redactcam[gpu]`.** The CPU and CUDA wheels both own the `onnxruntime/` package
  directory and the last one installed wins, so a base dependency on the CPU wheel
  let any later `pip install` of this package (or of a project depending on it)
  silently replace `onnxruntime-gpu`. That is what happened to a GPU pipeline venv on
  2026-09-25: the blur pass ran on CPU for a day with no line in the log saying why.
  `dev` now carries the CPU runtime so CI still runs; the frozen Windows build
  installs `.[cpu]`. A bare import without either extra raises a
  `ModuleNotFoundError` that names both extras.
- **A GPU build that cannot use CUDA now fails instead of falling back to CPU.**
  When the `onnxruntime-gpu` distribution is installed, `YoloDetector` requires
  `CUDAExecutionProvider` both in the advertised providers and in the constructed
  session (the GPU wheel advertises CUDA even when `libcudart` is missing, then
  builds the session on CPU with a warning). Either miss raises
  `GpuProviderUnavailableError`, whose message names the cause and the exact
  `--force-reinstall --no-deps` repair. `REDACTCAM_ALLOW_CPU=1` accepts the CPU path
  knowingly. New `check_inference_providers()` runs the same check as a preflight so a
  pipeline can fail at minute 0; `redactcam --check-deps` calls it and prints an
  `inference` line with the provider a detector would use, exiting 1 on the miss.
- `__version__` is 0.2.0 and now checked against `pyproject.toml` (they had drifted).

### Fixed

- Two objects visible at once no longer collapse onto one track, leaving one of
  them unblurred. Association had no per-frame exclusivity — nothing marked a
  track as already matched — and its center-distance fallback gate was sized by
  the LARGER of detection/track box, which for a near 4K car reaches 840-1210 px
  across the frame. So a second car's detection matched the first car's track and
  moved it there; one track emits one box, and which car it landed on was decided
  by detection order, alternating frame to frame. Worse, plate corroboration
  rides the track, so a stolen track carried its licence to the wrong car and
  blurred that one instead. Measured on a 353,861-frame dashcam day master: 13 of
  552 plate-corroborated vehicle-frames had an exposed occupant cabin. Now one
  detection claims one track — IoU matches settle first, then the distance
  fallback over unclaimed tracks only, sized by the SMALLER box — and a detection
  with no track left to claim spawns its own. The flat `assoc_min_gate` floor,
  which is what actually carries the tiny-distant-face case, is unchanged, as is
  a healthy single-object pass. This adds no gap-fill: every new box is a real
  track snapped to its own detection and moved by its own optical flow.
  Brief passes that the detector never sees at all are a separate defect.

### Added

- `detect_and_track(checkpoint_path=…, checkpoint_every_frames=…)` makes a long
  detection pass crash-resumable. Previously the pass was all-or-nothing across
  one decode: a GPU fault 2h55m into a 3h17m video (onnxruntime raising CUDA 999
  from a C++ destructor, so `std::terminate` aborted the process before Python
  could react) cost the entire pass, and the restart began again at frame 0.
  A checkpoint is a periodic *snapshot* — the tracker is never reset — so an
  uninterrupted run returns byte-for-byte what it always did. Only a resume
  starts a fresh tracker, re-decoding `checkpoint_overlap_frames` (default 90,
  3s at 30fps) before the resume point so a face already mid-track when the
  process died is re-confirmed; those boxes are UNIONed over the checkpointed
  ones, never substituted, because over-blur is safe and under-blur is a leak.
  A checkpoint written for a different video, or a corrupt one, is ignored and
  costs a full pass rather than a wrong mask. `n_face`/`n_plate` count the
  detections performed by that invocation, so a resumed pass reports its own
  work rather than a total it cannot verify.

## 0.1.1 - 2026-08-03

### Added

- Windows installer. Pushing a `v*` tag builds a PyInstaller one-file executable
  on `windows-latest`, wraps it with Inno Setup, and attaches
  `redactcam-setup-<version>.exe` to that tag's release. It installs into Program
  Files, appends that directory to the machine `PATH` and registers an
  uninstaller. Neither ffmpeg nor any model weight is bundled — the installer's
  finish page and the README both say so, since a frozen redactcam with no ffmpeg
  on `PATH` and no network on first run cannot do anything.

## 0.1.0

First release.

### Added

- `redactcam.detect` — tiled YOLO/ONNX detection of faces, licence plates, vehicles
  and people. Overlapping native-scale tiles (`frame_tiles`, `detect_tiled`) so a
  small object in a 4K frame is not destroyed by a whole-frame downscale to a
  640 px network input; letterboxing, NMS and IoU dedupe of seam-straddling
  duplicates. Faces and plates take independent confidence thresholds. Plate false
  positives are removed geometrically — an aspect band (`plate_min_aspect` /
  `plate_max_aspect`) that drops near-square road signs and over-elongated
  shorelines and railings, plus an area gate (`plate_max_area_frac`) for walls and
  verges that box at a plausible aspect. `detect_and_track()` is the dense
  per-frame entry point, `detect_regions()` the sparse one, `detect_image()` the
  single-still sibling that runs the same detector without a `VideoCapture`.
- `redactcam.track` — `TrackManager`, Lucas-Kanade optical-flow propagation of each
  detection to every frame with RANSAC partial-affine scale, IoU-plus-distance
  association, snap-correction at each detection, class-aware retirement, and
  velocity-predictive coasting for faces. Temporal confirmation
  (`face_confirm_min_detections`) drops one-shot foliage and shadow false positives
  retroactively, which is what makes a very low face confidence usable. Between-
  detection gaps are linearly interpolated between bracketing detections, and
  `pre_roll_frames` extrapolates a confirmed track backwards before its first
  detection.
- `redactcam.timeline` — dilated dense per-frame boxes plus a deterministic JSON
  sidecar (`redactcam.timeline/v1`) keyed by the SHA256 of the source file, so a
  re-run against identical pixels skips detection and a re-encoded file does not.
- `redactcam.mask` — the timeline rasterised to a lossless gray FFV1 mask clip at
  the source's exact resolution, frame rate and frame count. A written frame count
  that does not equal the source's raises rather than being logged.
- `redactcam.coverage` — `verify_cabin_coverage()`, an independent re-detection
  pass that plate-corroborates each vehicle and confirms the mask covers its cabin
  band, read in frame-locked lockstep with no seeking.
- `redactcam.apply` — the ffmpeg composite. `alphamerge` + `overlay` rather than
  `maskedmerge`, which promotes a gray mask to the video's pixel format and
  half-blends chroma across the whole frame. Blur radius scales with frame width.
- `redactcam.pipeline` — `redact_video()` wires the six stages together and
  **verifies before it renders**, raising `CoverageError` instead of producing a
  file whose blur missed a driver.
- `redactcam.models` — download-on-demand `ModelSpec` cache. Every model is a URL
  plus a SHA256, fetched over HTTPS into a temp file and renamed into the cache
  only after the checksum matches. A local `path` overrides the URL; a non-HTTPS
  URL is refused; every failure warns and returns `None` so the caller degrades on
  its own terms.
- `redactcam.presets` — `ROAD_FOOTAGE`, the parameter set tuned for
  forward-facing, vehicle-mounted 4K video, with the reasoning for each group of
  values in the module docstring.
- A `redactcam` command-line entry point, and `python -m redactcam`.

### Notes

- **No model weights ship with this package.** The four defaults reference public
  third-party releases by URL and checksum; those carry their own licences, which
  are not this project's MIT licence.
- Requires an `ffmpeg`/`ffprobe` binary on PATH for the mask render and the blur
  composite.
- Nothing is ever uploaded. The only network access in the library is the one-time
  HTTPS GET of a model file, which you can avoid entirely by supplying local paths.
- The test suite uses no models, no network and no recorded footage: frames are
  drawn with numpy, detectors are stubbed, and the few clips involved are generated
  into a temp directory at run time.
- Automated redaction is not a guarantee of anonymity, and the coverage verifier
  checks vehicle cabins specifically — not every object in frame. The README's
  *Limits* section spells out what it does and does not prove.
