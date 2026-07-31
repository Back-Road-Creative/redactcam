# Contributing to redactcam

Thanks for taking a look. This library exists to stop a face shipping unblurred,
so the bar for anything that could reduce coverage is high.

## Reporting

- **Bugs and feature requests:** open an issue. For a missed or spurious blur,
  describe the footage (resolution, frame rate, camera mounting, roughly what is in
  shot), the parameters you used, and what you expected. **Do not attach the
  footage** — see *Test fixtures* below. A synthetic reproduction is far more
  useful and far safer.
- **Security vulnerabilities:** do not open a public issue. Follow
  [`SECURITY.md`](SECURITY.md).

## Development setup

Python 3.11 or newer, plus an `ffmpeg`/`ffprobe` binary on your PATH.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## The gates (run before you push)

CI runs these on every push and pull request:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

The suite needs no models and no network. If a change you are making makes that
untrue, it is the change that is wrong.

## Things this project cares about

- **A missed blur is the only bug that matters.** Everything else is a
  performance or ergonomics question. When a trade-off is genuinely unclear,
  prefer over-covering: a slightly too-large blur box costs a few pixels of
  scenery, a slightly too-small one costs somebody their privacy.
- **Verify before the irreversible step, never after.** The coverage check runs
  against the mask, before the encode, because that is when a failure is still
  cheap. A check that runs after the render is a bug even when it passes. If you
  add a new stage, put its validation upstream of the first thing that cannot be
  undone.
- **Better still, make the failure impossible.** Before adding a gate, ask whether
  the generator can be changed so the invalid output cannot be produced at all.
  That is the better fix and it costs nothing at run time.
- **Never weaken a test to make a suite pass.** If a test is wrong, fix the test
  and say why in the commit message.
- **No model weights in the repository, ever.** A model is a URL plus a SHA256 in
  a `ModelSpec`. Weights are large, licensed separately, and go stale.
- **Say what you measured.** Most of the constants in `presets.py` and the
  detector signatures exist because something specific failed on real footage —
  a place-name sign scoring 0.6, a face at 0.19, a shoreline boxing at 13:1. If
  you change one, put the observation that justifies it in the docstring or the
  comment, not just the number.
- **Keep it deterministic where it is cheap to be.** Feature selection is
  RNG-free, association is greedy and order-stable, and cv2's global RNG is pinned
  at construction. Do not introduce an unseeded random path.
- **Docs land with the code.** A behaviour change updates the README, the
  docstring, and `CHANGELOG.md` in the same commit.

## Test fixtures

**Every fixture in this suite is generated, and it must stay that way.** No
recorded video, no photographs, no frames containing a real face or a real licence
plate, and no cached detection output from real footage — not even cropped,
blurred or "just for a regression test". Frames are drawn with numpy, detectors
are stubbed with deterministic stand-ins, model downloads are stubbed with a few
bytes of fake payload, and the handful of clips the mask and coverage tests need
are encoded into a temp directory at run time and deleted with it.

This is not a formality. A privacy library whose test data is somebody's face has
already failed.

## Commits and pull requests

- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, …).
- One logical change per pull request, with the gate output in the description.
- Pin GitHub Actions to a version, never a moving branch.

## Conduct

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
