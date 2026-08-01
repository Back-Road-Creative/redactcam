# Security Policy

## Supported versions

Only the latest released tag receives fixes. Pin a released `v*` tag; `main` is
unstable.

## Attack surface, honestly

redactcam decodes untrusted video, runs untrusted model files, and shells out to
ffmpeg. All three deserve care.

- **Model downloads.** `redactcam.models` fetches each model over HTTPS and
  verifies a SHA256 before the file is renamed into the cache, so a tampered or
  truncated download is discarded rather than loaded. A non-HTTPS URL is refused.
  **The checksum is the whole of the trust model** — if you point a `ModelSpec` at
  a URL without pinning `sha256`, nothing is verified. An ONNX file is executable
  input to onnxruntime; treat one from an untrusted source the way you would treat
  a shared library.
- **Video decoding.** Source clips are decoded by OpenCV, and therefore by FFmpeg's
  demuxers and decoders. Those parse hostile input for a living and have a long
  CVE history. Decoding a video you did not produce runs that code on attacker-
  controlled bytes. Keep your FFmpeg and OpenCV current, and sandbox the process if
  the input is arbitrary.
- **Subprocess invocation.** `mask.materialize_mask_video()` and
  `apply.apply_blur()` build argument lists and call `subprocess` **without a
  shell**, so a path containing shell metacharacters is not interpreted. But
  `apply_blur(extra_output_args=...)` is appended to the ffmpeg command verbatim —
  never build it from untrusted input.
- **Filesystem writes.** The sidecar, the mask and the output are written to paths
  you supply (defaulting to beside the input), with parent directories created.
  Validate any path derived from untrusted input before passing it in.
- **The JSON sidecar is trusted input.** `timeline.load_sidecar()` checks the
  schema string and then reads the geometry it finds. A sidecar an attacker can
  write is a sidecar that can move or remove your blur boxes. Treat it as part of
  your working data, not as an untrusted upload.

The library holds no credentials and uploads nothing. Its only outbound request is
the one-time model fetch, which you can eliminate entirely by supplying local
paths.

## What this library does not promise

Automated redaction is not anonymisation. A coverage check that passes means no
*plate-corroborated vehicle cabin* was left uncovered — it does not mean every
person in the frame was found. Do not treat a green run as a compliance
attestation. The README's *Limits* section is the full statement.

## Reporting a vulnerability

Please report privately. Do **not** open a public issue for a security report, and
do not attach footage containing real faces or plates to any report.

- Preferred: GitHub's **Security → Report a vulnerability** tab on this repository
  (Private Vulnerability Reporting).
- Fallback: email **backroadcreativeco@gmail.com** with `redactcam security` in the subject.

Please include the affected version or commit, a description of the impact, and
reproduction steps or a proof of concept — synthetic, wherever possible.

## What to expect

- Acknowledgement within 5 business days.
- An initial assessment and severity triage within 10 business days.
- Coordinated disclosure: we will agree a timeline with you before any public
  write-up, and credit reporters who want it.
