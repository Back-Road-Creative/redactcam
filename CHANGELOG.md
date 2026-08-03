# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
