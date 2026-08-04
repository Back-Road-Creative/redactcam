# redactcam

Blur faces, licence plates, vehicle occupants and people in video — and then
**prove the blur landed** before you encode anything.

The blur is tracked, not stamped: sparse detection is propagated to every frame
with optical flow, so a blur follows a moving face instead of freezing where the
detector last saw it. The proof is an independent re-detection pass that checks
the mask against the source and refuses to render when it finds an uncovered
face behind the wheel.

```
detect ──► track ──► timeline ──► mask ──► verify ──► apply
 YOLO      optical    JSON        FFV1     re-detect   ffmpeg
 tiled      flow      sidecar     gray     the source   blur
```

Each of those six is a plain function you can call on its own. `redact_video()`
wires them together for the common case.

## Why

Three things go wrong with the usual approach, and all three fail *silently* —
you get a plausible-looking blurred video with somebody's face readable in it.

**A blur stamped at the detection position is not on the object.** Detection is
expensive, so it runs on a sample of frames — every 8th, say. The naive fix is to
hold each detection's box until the next one, or to union the two endpoints.
Either way a plate caught in one sample has its box frozen for a fraction of a
second while the car keeps moving: the blur covers empty road ahead of the plate,
and the plate itself drives out from under it in plain view. redactcam propagates
each box with Lucas-Kanade optical flow on every frame, plus RANSAC scale so an
approaching object's box grows with it, and snaps the box back to ground truth at
each new detection. Detection stays sparse; propagation costs milliseconds.

**Confidence cannot separate a licence plate from a road sign.** Plate models
fire hard on rectangular signage — a place-name sign scoring 0.6, above any
threshold you can actually use without losing real plates. Raise the confidence
and you lose plates; lower it and you blur every sign on the road. The separable
axes turn out to be geometric: a real plate is wide but *bounded* (roughly 2.4–3:1
at an angle, up to about 4.7:1 head-on) while signs box near-square and shorelines
and bridge railings box at 13:1, and a real plate is a tiny fraction of a 4K frame
(≤0.4%) while a stone wall the model misreads covers 1.4–7.6%. The same problem
appears for faces — foliage and shadows score 0.5–0.7, comfortably above a genuine
small face at 0.19 — and there the separable axis is *time*: a real face
re-detects across consecutive samples, a texture artefact fires once and stops. So
the face threshold can be set very low as long as a track must accumulate several
detections before it is emitted.

**Verifying after the render is verifying too late.** Once you have encoded a
multi-gigabyte file and shipped it, discovering that the blur missed a driver
costs a re-encode and a takedown. redactcam re-detects on the source, corroborates
each vehicle with a licence plate (so it does not chase billboards), and checks
that the *mask* covers each real car's cabin — before a single frame is encoded. A
failure raises `CoverageError` instead of producing a file.

## Install

```bash
pip install redactcam
```

Requires Python 3.11 or newer, and an **`ffmpeg`/`ffprobe` binary on your PATH**
(the mask render and the blur composite shell out to it).

Runtime dependencies are `opencv-python-headless`, `numpy` and `onnxruntime`.

Optional extras:

| Extra | What it adds |
| --- | --- |
| `redactcam[gpu]` | The CUDA onnxruntime build. Uninstall `onnxruntime` first — the two distributions conflict. Roughly 19× faster on the YOLO forward pass. |
| `redactcam[centerface]` | A CenterFace fallback used when no YOLO face model is configured. Noticeably weaker outdoors; a safety net, not a default. |
| `redactcam[dev]` | pytest and ruff. |

### Model weights

**No weights ship with this package.** A model is described by a `ModelSpec` — a
URL and a SHA256 — and fetched once on first use, checksum-verified, and cached
under `~/.cache/redactcam/models`. Four are used: `face`, `plate`, `vehicle` and
`person` (the last two share one COCO model).

The URLs are ordinary parameters. Point them at your own mirror, or skip the
network entirely by supplying local files:

```python
from redactcam import ModelSpec, redact_video

redact_video(
    "clip.mp4",
    models={
        # a local file — nothing is downloaded
        "face": ModelSpec(name="face", path="/models/yolov11n-face.onnx"),
        # or your own mirror, still checksum-verified
        "plate": ModelSpec(name="plate", url="https://mirror.example/plate.onnx", sha256="…"),
    },
)
```

Or on the command line, `--model face=/models/yolov11n-face.onnx` (repeatable).

### The default weights are copyleft — read this before you deploy

redactcam's own code is MIT. The weights it downloads by default are not, and
none of them are this project's to relicense:

| Default | Source | Licence |
| --- | --- | --- |
| `face` | `akanametov/yolo-face` | **GPL-3.0** |
| `plate` | `morsetechlab/yolov11-license-plate-detection` | **AGPL-3.0** |
| `vehicle`, `person` | `aaurelions/yolo11n.onnx` | **none declared** |

Two consequences worth stating plainly:

- **AGPL-3.0 carries a network clause.** Run a service over the plate model and
  let users interact with it across a network, and that licence asks you to offer
  them the corresponding source. Private self-hosting is unaffected; offering
  redaction as a product is not.
- **The COCO weights declare no licence at all.** That is a reupload with no
  stated terms, which is not a grant — there is nothing there to rely on. It is
  most likely derived from an AGPL-3.0 upstream. If licensing matters to you,
  point `vehicle` and `person` at weights whose terms you have read.

Every default is replaceable and nothing hard-codes a URL at the point of use:
pass your own `ModelSpec` (or `--model <kind>=<path>`) and redactcam uses it,
still checksum-verified. Nothing is ever uploaded — the fetch is a plain HTTPS
GET of a public file, a non-HTTPS URL is refused, and a download is only moved
into the cache once its SHA-256 matches.

All three default checksums in `models.py` were fetched and verified against the
live URLs on 2026-07-31.

### Windows installer

Each release also carries `redactcam-setup-<version>.exe` on its
[releases page](https://github.com/Back-Road-Creative/redactcam/releases) —
redactcam frozen into one executable, so it needs no Python. It installs into
Program Files, appends that directory to the system `PATH`, and registers an
uninstaller in Add/Remove Programs. Open a *new* terminal afterwards; one that
was already open still holds the old `PATH`. It is a large download: opencv and
onnxruntime are frozen into it.

Three things to know before downloading it.

**ffmpeg is not bundled, and redactcam cannot render anything without it.** The
mask render and the blur composite shell out to `ffmpeg` and `ffprobe`. An
ffmpeg build dwarfs redactcam, and which licence a given build falls under
depends on how it was configured, so shipping one inside this installer would
be both large and a claim this project is not in a position to make. Install it
yourself — this puts both on `PATH`:

```powershell
winget install -e --id Gyan.FFmpeg
```

`choco install ffmpeg` works too. Confirm with `ffmpeg -version`.

**No model weights are bundled either, so the first run needs a network
connection.** The frozen executable resolves models exactly the way the pip
install does: roughly 110 MB of ONNX files fetched once over HTTPS,
checksum-verified, and cached under `C:\Users\<you>\.cache\redactcam\models`. A
machine that will never have internet access needs the files supplied instead —
`--model face=C:\models\yolov11n-face.onnx`, repeatable, as above. Everything in
[the copyleft section](#the-default-weights-are-copyleft--read-this-before-you-deploy)
applies to an installer user too: the defaults are GPL-3.0, AGPL-3.0 and
undeclared, and none of them are this project's to relicense.

**The build is not code-signed.** There is no code-signing certificate for this
project, so Windows cannot show you a publisher. Expect the blue *"Windows
protected your PC"* box — "Microsoft Defender SmartScreen prevented an
unrecognized app from starting" — which runs the installer only after **More
info** → **Run anyway**, and expect your browser to warn during the download.
That is simply what an unsigned binary looks like; it is not evidence the file
is safe. If you would rather not make that call, `pip install redactcam` needs
no installer.

## Use it

### Command line

```bash
redactcam dashcam.mp4 -o dashcam_redacted.mp4 -v
```

```
timeline  dashcam_redactcam.json
mask      dashcam_redactcam_mask.mkv (5412 frames)
coverage  OK across 63 vehicle sightings
output    dashcam_redacted.mp4
```

Exit code `2` means the coverage check failed and **nothing was rendered**.

Useful flags: `--no-render` stops after the mask and the check (bring your own
encode), `--fresh` ignores the cached timeline sidecar and re-detects,
`--work-dir` puts the sidecar and mask somewhere other than beside the input, and
`--blur-strength` sets the boxblur radius at 1920 px wide (scaled to the real
width).

`--check-deps` takes no input and answers one question: did the native
extensions load? It imports OpenCV and onnxruntime and builds a real ONNX
session options object, then prints their versions and the available execution
providers. It is most useful after installing the Windows build, where those two
are bundled by PyInstaller rather than by pip, and where a packaging miss would
otherwise surface on your first real video:

```console
$ redactcam --check-deps
opencv          5.0.0
onnxruntime     1.28.0
providers       AzureExecutionProvider, CPUExecutionProvider
session options intra=1 inter=1
```

It exits non-zero and names the module if one fails to load, and needs no model
weights, so it will not trigger the first-run download.

### Python

```python
from redactcam import CoverageError, redact_video

try:
    result = redact_video("dashcam.mp4")
except CoverageError as exc:
    for leak in exc.report.leaks[:5]:
        print(f"frame {leak.frame}: cabin only {leak.coverage:.0%} covered")
    raise

print(result.output)                       # the blurred video
print(result.mask)                         # the gray mask clip
print(result.sidecar)                      # the JSON blur timeline
print(result.coverage.vehicle_frames)      # sightings the verifier checked
```

Detection is nearly all the run time, so the timeline is cached in a JSON sidecar
keyed by the SHA256 of the source file. Re-running against the same input reuses
it; re-running against a re-encoded copy does not, because the hash changed.

### Splice the blur into your own encode

`redact_video(render=False)` stops after the mask and the verification. If you
already run an ffmpeg encode, take the filter graph and composite once rather than
encoding twice:

```python
from redactcam import blur_filter_complex, blur_radius, redact_video

result = redact_video("dashcam.mp4", render=False)
graph = blur_filter_complex(f"boxblur={blur_radius(3840)}:2")   # output pad: [vout]
# ffmpeg -i dashcam.mp4 -i {result.mask} -filter_complex "{graph}" -map "[vout]" ...
```

### One still image

```python
import cv2
from redactcam import build_image_detectors, detect_image

detectors = build_image_detectors(face_model_path="/models/yolov11n-face.onnx")
rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

regions = detect_image(rgb, detectors=detectors)
print(regions.faces, regions.plates)       # (x, y, w, h) in native pixels
```

Build the detectors once and reuse them across a batch — constructing an ONNX
session costs far more than one still's inference.

## Tuning

Every default in the function signatures is a general-purpose starting point.
`redactcam.presets.ROAD_FOOTAGE` is a second, more aggressive set tuned for
**forward-facing, vehicle-mounted 4K video**, and it is what `redact_video()`
uses. It is an ordinary dict — print it, diff it, override any key:

```python
from redactcam import ROAD_FOOTAGE, redact_video

redact_video("clip.mp4", detect_kwargs={"face_conf": 0.25, "track_person": False})
redact_video("clip.mp4", detect_kwargs={})          # the preset, unmodified
```

The module docstring in `presets.py` explains what each group of values is for.
The short version:

| Knob | Default | What it is doing |
| --- | --- | --- |
| `tile_px` | 1280 | Detect on overlapping native-scale tiles. A 4K frame crushed to a 640 px network input loses a distant plate entirely. |
| `sample_fps` / `face_sample_fps` | 8 / 16 | Detection cadence. The gap must stay inside the optical-flow horizon (~0.13 s). Faces get a denser one — they are cheap, and small faces track badly. |
| `face_conf` + `face_confirm_min_detections` | 0.15 + 2 | Chase genuinely small faces at low confidence; require persistence so foliage does not survive. **Never lower the first without the second.** |
| `plate_min_aspect` / `plate_max_aspect` / `plate_max_area_frac` | 2.0 / 6.0 / 0.006 | Geometry gates that drop road signs (too square), shorelines and railings (too long), and walls and verges (too big). |
| `plate_boost_speed_px` | 8 | While a plate track is moving, re-detect plates on *every* frame so a close pass snaps to ground truth instead of trailing flow. |
| `track_cabin_frac` | 0.6 | Blur the top 60% of a vehicle's box. Occupants sit there at every viewing angle; face detection through a moving windscreen does not work. |
| `vehicle_min_height_frac` / `vehicle_ignore_bottom_frac` | 0.07 / 0.12 | Skip cars too distant to have an identifiable occupant, and the bottom strip where a vehicle-mounted camera sees its own bonnet. |
| `pre_roll_frames` | 5 | Extrapolate a confirmed track backwards: an approaching plate is readable a few frames before the detector first scores it. |

**If your footage is not road footage, most of these are wrong for you.** A fixed
CCTV camera, a handheld phone and a drone all break different assumptions — the
aspect and area gates assume vehicle plates seen from a vehicle, and the
bottom-strip exclusion assumes a bonnet is there. Start from the bare function
defaults and add gates one at a time as you find your own false positives.

## Limits

Read this part before you rely on it.

- **This is not a guarantee of anonymity, and no automated redaction is.** It
  raises the cost of identification; it does not reduce it to zero. Gait, build,
  clothing, vehicle, location and timestamp all survive a face blur. Treat the
  output as pseudonymised, not anonymous, and keep whatever legal basis you were
  relying on.
- **The coverage verifier checks vehicle cabins, not everything.** It is the
  strongest check in the library and it is deliberately narrow: it re-detects
  vehicles, corroborates them with a plate, and confirms the mask covers each
  cabin. It does *not* verify that every pedestrian face was found — an
  independent verifier can only check what an independent detector can find, and
  a face the detector misses twice is missed by both passes. Coverage OK means "no
  uncovered driver", not "nothing was missed".
- **`min_coverage` defaults to 0.5, not 1.0.** The verifier detects the vehicle
  independently of whatever produced the blur, so a box landing a few percent off
  trims edge pixels while the driver, who sits near the cabin centre, stays
  covered. Raise it if you would rather have false alarms than misses.
- **Recall depends entirely on the models you point it at.** The tuning here
  compensates for known failure modes of the default weights; a different model
  has different failure modes and the gates may fight it.
- **The blur is irreversible, and it is applied to a re-encode.** `apply_blur()`
  produces a new H.264 file (CRF 18 by default) — generational loss on everything,
  not just the blurred regions. Keep your original.
- **Sidecars are cached by content hash.** If you edit the video and keep the
  filename you get a new hash and a fresh detection pass, which is the point. If
  you keep the pixels and rename the file you correctly reuse the timeline.
- **Determinism is close but not bit-exact by default.** Feature selection,
  association and the RNG seed are pinned, but multi-threaded and GPU float
  reductions reorder, so boxes can shift sub-pixel between runs. Irrelevant for a
  privacy mask — the box still covers the object. Set `plate_threads=1` and stay
  on CPU if you need byte-identical output.
- **Speed is dominated by the YOLO forward pass.** Tiled 4K detection measured
  around 1.1 s per detected frame across all CPU cores, versus roughly 3.8 s
  single-threaded and 17–21 ms per tile on a mid-range consumer GPU. Sample
  cadence, not clip length, is the cost driver.
- **The mask must match the source frame count exactly.** It is asserted, not
  logged: a mask one frame out of step blurs the wrong pixels for the rest of the
  clip, so a mismatch raises rather than shipping something subtly wrong.

## API

Everything below is importable straight from `redactcam`.

**Pipeline** — `redact_video()`, `RedactionResult`, `CoverageError`,
`file_hash()`.

**Detection** — `detect_and_track()` (the dense per-frame workhorse),
`detect_regions()` (sparse, no tracking), `detect_image()` (one still),
`build_detectors()`, `build_image_detectors()`, and the four single-model
builders. `YoloDetector`, `CenterFaceDetector`, `FrameRegions`, `Box`, `iou()`,
`ModelUnavailableError`.

**Tracking** — `TrackManager`, `Track`.

**Timeline** — `build_timeline()`, `BlurTimeline`, `dilate_box()`,
`probe_video()`, `write_sidecar()`, `load_sidecar()`.

**Mask** — `materialize_mask_video()`, `render_mask_frame()`,
`active_boxes_at()`.

**Apply** — `apply_blur()`, `blur_filter_complex()`, `blur_radius()`.

**Coverage** — `verify_cabin_coverage()`, `CoverageReport`, `Leak`,
`cabin_region()`, `plate_inside()`.

**Models** — `ModelSpec`, `resolve_model()`, `DEFAULT_MODELS`, `FACE_MODEL`,
`PLATE_MODEL`, `VEHICLE_MODEL`, `PERSON_MODEL`.

**Presets** — `ROAD_FOOTAGE`, `ROAD_FOOTAGE_BOX_DILATION_PX`,
`ROAD_FOOTAGE_FEATHER_PX`.

The docstrings are the reference documentation and they are unusually detailed —
each one records what failed on real footage and why the parameter exists.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check .
```

The suite needs no models, no network and no video files: every frame is drawn
with numpy, every detector is stubbed, and the few clips involved are generated
into a temp directory at run time. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence

MIT — see [`LICENSE`](LICENSE). The model weights it downloads are third-party
and carry their own licences.
