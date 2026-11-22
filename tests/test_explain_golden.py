"""Golden check: the explain-capable detect/track path reproduces the pre-explain mask.

``tests/fixtures/explain_golden.json`` holds the per-frame boxes (tracker output and the
dilated boxes the mask draws) that commit 143b3be -- the code before ``ClassedBox`` and
the entry provenance existed -- produced on four drawn clips with stub detectors. The
test runs the CURRENT code on the same clips and demands the same boxes, so it cannot
pass just because both sides changed together.

Regenerate ONLY from a checkout of 143b3be (it must not be the code under test)::

    PYTHONPATH=<143b3be checkout> python tests/test_explain_golden.py tests/fixtures/explain_golden.json
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from redactcam import detect, presets
from redactcam import mask as mask_mod
from redactcam import timeline as tl_mod

FIXTURE = Path(__file__).parent / "fixtures" / "explain_golden.json"
W, H, FPS, N = 1280, 720, 24.0, 48
SCENES = ("mixed", "edge_vehicle", "long_truck", "truck_flip")

_TEX = (np.indices((64, 64)).sum(0) * 37 % 256).astype(np.uint8)
_TEX[::4, :] = 255
_TEX[:, ::4] = 0


def _clamp(x, y, w, h):
    x0, y0 = max(0, round(x)), max(0, round(y))
    x1, y1 = min(W, round(x + w)), min(H, round(y + h))
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def _scene(name):
    """gt[f] = [(kind, (x, y, w, h) float, coco class or None, detectable)]"""
    gt = {f: [] for f in range(N)}
    for f in range(N):
        if name == "mixed":
            gt[f].append(("face", (-40 + 9 * f, 200, 44, 48), None, True))  # enters left
            gt[f].append(("face", (1180 + 6 * f, 420, 40, 44), None, True))  # exits right
            cx = 100 + 12 * f
            gt[f].append(("vehicle", (cx, 300, 320, 160), 2, f >= 4))
            gt[f].append(("plate", (cx + 130, 420, 60, 18), None, True))
            gt[f].append(("person", (700, 250, 90, 230), 0, True))
            gt[f].append(("face", (900 - 5 * f, 260, 40, 46), None, f % 7 != 3))
        elif name == "edge_vehicle":
            cx = -60 + 18 * f  # detected late near the left edge: pre-roll runs off it
            gt[f].append(("vehicle", (cx, 280, 300, 150), 2, f >= 6))
            gt[f].append(("plate", (cx + 120, 390, 60, 18), None, f >= 6))
            gt[f].append(("vehicle", (500 - 14 * f, 120 - 6 * f, 280, 140), 2, f < 40))
            gt[f].append(("plate", (500 - 14 * f + 110, 120 - 6 * f + 100, 60, 18), None, f < 40))
            gt[f].append(("face", (300 - 8 * f, 560, 40, 44), None, f >= 3))
        else:  # long_truck / truck_flip: a class-7 box wider than 5x its height
            x = 100 + 4 * f
            coco = 5 if name == "truck_flip" and f % 6 == 0 else 7
            gt[f].append(("vehicle", (x, 260, 1000, 170), coco, True))
            gt[f].append(("plate", (x + 900, 380, 60, 18), None, True))
            gt[f].append(("face", (x + 480, 290, 40, 44), None, True))
            gt[f].append(("person", (x + 300, 270, 60, 150), 0, f % 3 == 0))
    return gt


def _render(path, gt):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W), 50, np.uint8)
        img[6::12, :] = 80
        for _k, (x, y, w, h), _c, _d in gt[f]:
            b = _clamp(x, y, w, h)
            if b is None:
                continue
            bx, by, bw, bh = b
            tex = cv2.resize(_TEX, (max(1, round(w)), max(1, round(h))))
            img[by : by + bh, bx : bx + bw] = tex[
                by - round(y) : by - round(y) + bh, bx - round(x) : bx - round(x) + bw
            ]
        img[H - 1, W - 1] = f % 256  # frame id, read back by the stub detectors
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    wr.release()


def run_scene(name, tmp: Path, mp) -> dict:
    """Run detect_and_track + build_timeline on one drawn scene; return the boxes."""
    gt, vid = _scene(name), tmp / f"{name}.mkv"
    _render(vid, gt)
    classed = getattr(detect, "ClassedBox", None)  # absent in 143b3be

    def dets(frame, kind):
        out = []
        for k, box, coco, ok in gt[int(frame[H - 1, W - 1, 0])]:
            b = _clamp(*box) if k == kind and ok else None
            if b is not None:
                out.append(classed(b, coco) if classed and coco is not None else b)
        return out

    def stub(kind):
        return type("Stub", (), {"detect": lambda self, frame, conf: dets(frame, kind)})()

    mp.setattr(detect, "build_detectors", lambda *a, **k: (stub("face"), stub("plate")))
    mp.setattr(detect, "build_vehicle_detector", lambda *a, **k: stub("vehicle"))
    mp.setattr(detect, "build_person_detector", lambda *a, **k: stub("person"))
    mp.setattr(detect, "detect_plates", lambda frame, det, *a, **k: dets(frame, "plate"))
    opts = {**presets.ROAD_FOOTAGE, "tile_px": 99999, "progress_every_s": 0}
    for key in ("face", "plate", "vehicle", "person"):
        opts[f"{key}_model_path"] = "x"
    fb, decoded, *_ = detect.detect_and_track(vid, **opts)
    tl = tl_mod.build_timeline(
        fb,
        source_hash="h",
        source_fps=FPS,
        source_frame_count=decoded,
        frame_width=W,
        frame_height=H,
        box_dilation_px=presets.ROAD_FOOTAGE_BOX_DILATION_PX,
        sample_fps=8.0,
        conf=0.4,
    )
    mask = {f: mask_mod.active_boxes_at(tl, f) for f in range(decoded)}
    return {
        "frames": decoded,
        "tracked": {str(f): [list(map(int, b)) for b in fb[f]] for f in sorted(fb)},
        "mask": {str(f): [list(map(int, b)) for b in bs] for f, bs in mask.items() if bs},
    }


@pytest.mark.parametrize("name", SCENES)
def test_boxes_match_the_pre_explain_golden(name, tmp_path, monkeypatch):
    golden = json.loads(FIXTURE.read_text())[name]
    got = run_scene(name, tmp_path, monkeypatch)
    assert got == golden
    assert golden["tracked"]  # the scene really produced boxes


def test_the_golden_scenes_cover_the_cases_that_matter():
    golden = json.loads(FIXTURE.read_text())
    assert set(golden) == set(SCENES)
    # an edge box (touching a frame side), and a box wider than 5x its height
    edge, wide = golden["edge_vehicle"]["tracked"], golden["long_truck"]["tracked"]
    assert any(b[0] == 0 or b[0] + b[2] == W for bs in edge.values() for b in bs)
    assert any(b[2] > 5 * b[3] for bs in wide.values() for b in bs)


if __name__ == "__main__":  # regenerate the fixture from a 143b3be checkout
    out = {}
    with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as mp:
        for scene in SCENES:
            out[scene] = run_scene(scene, Path(td), mp)
    Path(sys.argv[1]).write_text(json.dumps(out, separators=(",", ":")))
