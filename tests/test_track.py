"""Synthetic-motion tests for the optical-flow tracker.

A fixed-texture square is drawn moving across a generated frame, and detections
are supplied only on sparse "sample" frames. The tracker must propagate the box
so it FOLLOWS the object between samples — the regression guard against the box
freezing at the detection position for a second at a time, which blurs empty
road ahead of a moving object and exposes the object itself.

No video files, no models, no photographs: every frame here is drawn by numpy.
"""

import cv2
import numpy as np

from redactcam.detect import iou as _iou
from redactcam.track import Track, TrackManager

W, H = 480, 270
# Fixed object texture (RNG-free determinism): same content, only its position
# changes frame to frame, so LK has stable features to track.
_PATCH = (np.indices((48, 48)).sum(0) * 37 % 256).astype(np.uint8)
_PATCH[::4, :] = 255  # strong corners for goodFeaturesToTrack
_PATCH[:, ::4] = 0


def _frame(cx, cy):
    img = np.full((H, W), 50, np.uint8)
    img[6::12, :] = 80  # mild static background grid
    h, w = _PATCH.shape
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    if 0 <= x0 <= W - w and 0 <= y0 <= H - h:
        img[y0 : y0 + h, x0 : x0 + w] = _PATCH
    return img


def _box(cx, cy):
    h, w = _PATCH.shape
    return (int(cx - w // 2), int(cy - h // 2), w, h)


def _cx(box):
    return box[0] + box[2] / 2


def _scaled(scale):
    sz = max(8, int(48 * scale))
    return cv2.resize(_PATCH, (sz, sz), interpolation=cv2.INTER_NEAREST), sz


def _frame_scaled(cx, cy, scale):
    """Frame with the object grown by ``scale`` — exercises affine scale tracking."""
    img = np.full((H, W), 50, np.uint8)
    img[6::12, :] = 80
    patch, sz = _scaled(scale)
    x0, y0 = int(cx - sz // 2), int(cy - sz // 2)
    if 0 <= x0 <= W - sz and 0 <= y0 <= H - sz:
        img[y0 : y0 + sz, x0 : x0 + sz] = patch
    return img


def _box_scaled(cx, cy, scale):
    _, sz = _scaled(scale)
    return (int(cx - sz // 2), int(cy - sz // 2), sz, sz)


def test_forward_propagation_follows_motion():
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    for i in range(30):
        cx = 120 + i * 6  # 6 px/frame to the right
        det = [("plate", _box(cx, 135))] if i == 5 else None
        tm.step(i, _frame(cx, 135), det)
    assert tm.frame_boxes[0] == []  # nothing before first detection (no back-prop yet)
    b5, b20 = tm.frame_boxes[5][0], tm.frame_boxes[20][0]
    assert _cx(b20) - _cx(b5) > 60  # box moved ~90 px, not frozen
    assert abs(_cx(b20) - (120 + 20 * 6)) < 14  # tracks the true center


def test_not_frozen_regression():
    """The exact defect: the box must move between samples, never stay put."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    for i in range(20):
        cx = 100 + i * 8
        det = [("plate", _box(cx, 135))] if i == 2 else None
        tm.step(i, _frame(cx, 135), det)
    centers = [_cx(v[0]) for f, v in sorted(tm.frame_boxes.items()) if v]
    assert centers == sorted(centers)  # monotonic with the object
    assert max(centers) - min(centers) > 50
    assert len({round(c) for c in centers}) > 5  # many positions, not 1 frozen box


def test_snap_correct_resets_drift():
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    for i in range(12):
        cx = 100 + i * 5
        det = [("plate", _box(cx, 135))] if i in (0, 10) else None
        tm.step(i, _frame(cx, 135), det)
    b10 = tm.frame_boxes[10][0]
    assert abs(_cx(b10) - (100 + 10 * 5)) < 2  # snapped exactly onto the redetection


def test_association_reuses_one_track():
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    for i in range(12):
        cx = 100 + i * 5
        det = [("plate", _box(cx, 135))] if i in (0, 5, 10) else None
        tm.step(i, _frame(cx, 135), det)
    assert tm._nid == 1  # three detections of one object → a single track
    assert len(tm.frame_boxes[11]) == 1


def test_retire_after_horizon_no_stale_smear():
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=5)
    for i in range(20):
        cx = 100 + i * 5
        det = [("plate", _box(cx, 135))] if i == 2 else None
        tm.step(i, _frame(cx, 135), det)
    assert tm.frame_boxes[5]  # within horizon
    assert tm.frame_boxes[15] == []  # past horizon → retired, never a frozen smear


def test_clean_frames_stay_empty():
    tm = TrackManager(W, H, downscale=1)
    for i in range(5):
        tm.step(i, _frame(-999, -999), None)
    assert all(v == [] for v in tm.frame_boxes.values())


def test_determinism_bit_identical():
    def run():
        tm = TrackManager(W, H, downscale=2, max_horizon_frames=60)
        for i in range(20):
            cx = 100 + i * 6
            det = [("plate", _box(cx, 135))] if i == 4 else None
            tm.step(i, _frame(cx, 135), det)
        return {f: list(v) for f, v in tm.frame_boxes.items()}

    assert run() == run()  # also exercises the downscale=2 coord conversion


def test_iou_helper():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_box_grows_with_object():
    """Affine scale: a growing (approaching) object's box must grow too — pure
    translation kept the box too small as the real plate enlarged."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    for i in range(13):
        scale = 1.0 + i * 0.06  # ~6 %/frame growth (within the per-step clamp)
        det = [("plate", _box_scaled(240, 135, scale))] if i == 0 else None
        tm.step(i, _frame_scaled(240, 135, scale), det)
    w0 = tm.frame_boxes[0][0][2]
    w12 = tm.frame_boxes[12][0][2]
    assert w12 > w0 * 1.3  # tracked the growth (true ~1.72x); not frozen at seed size


def test_per_step_scale_clamped_no_explosion():
    """Even under explosive apparent growth, the per-step scale clamp (and the
    translation fallback when LK loses the object) bounds box growth per frame."""
    tm = TrackManager(W, H, downscale=1, scale_max=1.2, max_horizon_frames=60)
    for i in range(6):
        det = [("plate", _box_scaled(240, 135, 1.0))] if i == 0 else None
        tm.step(i, _frame_scaled(240, 135, 1.0 + i * 0.5), det)  # 50 %/frame
    sizes = [tm.frame_boxes[i][0][2] for i in range(6) if tm.frame_boxes[i]]
    for a, b in zip(sizes, sizes[1:], strict=False):
        assert b <= a * 1.2 + 1  # never an unclamped balloon (+1 for rounding)


# A featureless frame: flow finds nothing to track, so points die cleanly (vs
# _frame's background grid, which flow would latch onto and drift along).
_BLANK = np.full((H, W), 50, np.uint8)


def test_face_coasts_through_flow_dropout():
    """A face HOLDS its last box through a flow dropout (no blink), surviving past
    the plate ``max_horizon`` and up to ``face_coast_frames``, then retires."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=5, face_coast_frames=15)
    tm.step(0, _frame(240, 135), [("face", _box(240, 135))])
    for i in range(1, 20):
        tm.step(i, _BLANK, None)  # no features anywhere → the face's flow dies
    # Survives the dropout past the plate horizon (5) up to face_coast (15) — the
    # anti-blink hold — and HOLDS one fixed box rather than blinking or drifting.
    assert tm.frame_boxes[5] and tm.frame_boxes[10] and tm.frame_boxes[14]
    assert tm.frame_boxes[2] == tm.frame_boxes[10] == tm.frame_boxes[14]
    assert tm.frame_boxes[16] == []  # but bounded: retired once past face_coast (15)


def test_motion_pad_grows_fast_box_not_static():
    """A moving box is padded by its own flow speed (covering plate lag); a static
    one is not — so a large near-static FP is never ballooned."""
    tight_w = _PATCH.shape[1]  # the tracked box width (~48)

    # Fast object (8 px/frame): gain 2.0 → ~16 px margin/side → a wider emitted box.
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60, motion_gain=2.0, motion_max=100)
    for i in range(10):
        cx = 100 + i * 8
        det = [("plate", _box(cx, 135))] if i == 0 else None
        tm.step(i, _frame(cx, 135), det)
    moving = tm.frame_boxes[6][0]
    assert moving[2] > tight_w + 8  # padded wider than the bare tracked box
    assert abs((moving[0] + moving[2] / 2) - (100 + 6 * 8)) < 14  # symmetric: center preserved

    # Static object: speed ~0 → ~no motion margin (a structure FP stays its raw size).
    tm2 = TrackManager(W, H, downscale=1, max_horizon_frames=60, motion_gain=2.0, motion_max=100)
    for i in range(10):
        det = [("plate", _box(240, 135))] if i == 0 else None
        tm2.step(i, _frame(240, 135), det)
    assert tm2.frame_boxes[6][0][2] <= tight_w + 4


def test_plate_retires_immediately_on_flow_loss():
    """The contrast that keeps fast plates honest: a plate gets NO coast — losing
    flow retires it at once, so its blur never freezes on bare road."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=5, face_coast_frames=15)
    tm.step(0, _frame(240, 135), [("plate", _box(240, 135))])
    for i in range(1, 8):
        tm.step(i, _BLANK, None)
    assert tm.frame_boxes[3] == []  # flow lost → retired, no face-style coast


def test_face_velocity_coast_follows_heading():
    """A MOVING face whose flow drops out must coast ALONG its last
    heading, not freeze as a ghost on bare road behind the live face. The face
    moves right with texture (flow learns a +6 px/frame heading), then the frame
    goes featureless — the held box must keep advancing, not sit still."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=3, face_coast_frames=15)
    for i in range(6):
        cx = 120 + i * 6
        tm.step(i, _frame(cx, 135), [("face", _box(cx, 135))] if i == 0 else None)
    assert tm._tracks[0].vx > 4  # heading established from live flow
    for i in range(6, 11):
        tm.step(i, _BLANK, None)  # flow dies → coast on the heading
    b7, b10 = tm.frame_boxes[7][0], tm.frame_boxes[10][0]
    assert b10[0] - b7[0] >= 12  # advanced ~3 frames of +6 px heading (not frozen)


def test_implausible_flow_jump_does_not_poison_coast():
    """Robustness for the coast: a single garbage flow sample (LK latching onto
    background on a textureless frame) must NOT become the coast heading, or the
    held box would fly off the frame. A static face holds put across the dropout."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=2, face_coast_frames=15)
    tm.step(0, _frame(240, 135), [("face", _box(240, 135))])
    for i in range(1, 14):
        tm.step(i, _BLANK, None)  # featureless → spurious flow, then points die
    # Held (anti-blink) and bounded — never coasts itself off-frame and retires early.
    assert tm.frame_boxes[10] and tm.frame_boxes[10] == tm.frame_boxes[5]


def test_associate_distance_gate_reuses_one_track():
    """A fast small face whose detections do not OVERLAP frame to frame
    (IoU = 0) still re-associates by the speed/size motion gate, so one face stays
    one track instead of spawning the duplicate that then coasts as a ghost."""
    tm = TrackManager(W, H, downscale=1, assoc_motion_gain=1.2)
    tm._tracks = [Track(0, "face", (76, 111, 48, 48), None, 0, speed=4.0)]
    tm._nid = 1
    far = (136, 111, 48, 48)  # 60 px away — no overlap, but inside the motion gate
    assert _iou(far, tm._tracks[0].box) == 0.0
    assert tm._associate(far, "face") is tm._tracks[0]
    assert tm._associate((420, 111, 48, 48), "face") is None  # beyond the gate → new track
    tm.assoc_gain = 0.0  # gate off → IoU-only, the near miss spawns a duplicate
    assert tm._associate(far, "face") is None


def test_associate_min_gate_floors_tiny_box():
    """A TINY distant face (17 px tight box) moving ~45 px
    between 8 fps detections has a size-scaled gate (~31 px) UNDER its motion, so it
    would spawn a duplicate. The flat ``assoc_min_gate`` floor re-associates it."""
    tm = TrackManager(W, H, downscale=1, assoc_motion_gain=1.2, assoc_min_gate=72)
    tm._tracks = [Track(0, "face", (100, 100, 17, 20), None, 0, speed=0.0)]
    moved = (145, 100, 17, 20)  # 45 px away, no overlap; size term alone misses it
    assert _iou(moved, tm._tracks[0].box) == 0.0
    assert tm._associate(moved, "face") is tm._tracks[0]  # floor (72) catches it
    tm.assoc_min_gate = 0  # floor off → the tiny size gate misses the fast box
    assert tm._associate(moved, "face") is None


def test_retire_superseded_ghost():
    """A coasting ghost (a face holding a stale box, no live flow points)
    is retired when a freshly-detected track of the same class overlaps it — the
    live detection supersedes the stale hold instead of both blurring at once."""
    tm = TrackManager(W, H, downscale=1)
    ghost = Track(0, "face", (76, 111, 48, 48), None, 0)  # stale: detected at frame 0
    fresh = Track(1, "face", (80, 113, 48, 48), np.zeros((1, 1, 2), np.float32), 10)
    tm._tracks = [ghost, fresh]
    tm._retire_superseded(10)
    assert tm._tracks == [fresh]  # the superseded ghost is dropped


def test_plate_margin_floor_covers_slow_plate():
    """A slow or static plate's speed margin is ≈0, so a flat floor keeps it
    covered. A face gets no floor (it relies on the coast, not a wider box)."""
    raw_w = _PATCH.shape[1]  # 48
    tmp = TrackManager(
        W,
        H,
        downscale=1,
        max_horizon_frames=60,
        motion_gain=1.0,
        motion_max=100,
        plate_margin_floor=16,
    )
    tmf = TrackManager(
        W,
        H,
        downscale=1,
        max_horizon_frames=60,
        motion_gain=1.0,
        motion_max=100,
        plate_margin_floor=16,
    )
    for i in range(8):
        tmp.step(i, _frame(240, 135), [("plate", _box(240, 135))] if i == 0 else None)
        tmf.step(i, _frame(240, 135), [("face", _box(240, 135))] if i == 0 else None)
    assert tmp.frame_boxes[6][0][2] >= raw_w + 2 * 16 - 2  # plate floored on both sides
    assert tmf.frame_boxes[6][0][2] <= raw_w + 4  # face un-floored (static → no margin)


def test_face_margin_frac_floors_slow_face_proportionally():
    """A face's motion margin is speed-scaled, so it collapses to
    the bare detector box when flow speed drops to ≈0 (a face exiting frame),
    exposing the cheek/profile the frontal box misses. ``face_margin_frac`` holds a
    SIZE-PROPORTIONAL floor; plates are unaffected."""
    raw_w = _PATCH.shape[1]  # 48
    tm = TrackManager(
        W,
        H,
        downscale=1,
        max_horizon_frames=60,
        motion_gain=1.0,
        motion_max=100,
        face_margin_frac=0.3,
    )
    for i in range(8):  # static face → speed ≈0 → only the proportional floor applies
        tm.step(i, _frame(240, 135), [("face", _box(240, 135))] if i == 0 else None)
    floor = round(0.3 * raw_w)  # ~14 px each side
    assert tm.frame_boxes[6][0][2] >= raw_w + 2 * floor - 2  # floored despite zero speed
    # Proportional, not flat: a tiny FP face gets a tiny floor (precision held),
    # a large near face a large one; plates ignore it.
    assert tm._floor("face", (0, 0, 20, 20)) == round(0.3 * 20)
    assert tm._floor("face", (0, 0, 120, 90)) == round(0.3 * 120)
    assert tm._floor("plate", (0, 0, 120, 90)) == tm.plate_floor
    assert TrackManager(W, H)._floor("face", (0, 0, 80, 80)) == 0  # default off → un-floored


def test_face_motion_gain_widens_fast_face_not_plate():
    """A fast face gets a larger speed-scaled margin than the shared gain (covering
    its flow lag / trailing exit); a plate keeps the lower shared gain."""
    kw = dict(downscale=1, max_horizon_frames=60, motion_gain=1.0, motion_max=300)
    base = TrackManager(W, H, **kw)
    hi = TrackManager(W, H, face_motion_gain=2.0, **kw)
    for i in range(8):
        cx = 100 + i * 10
        det = [("face", _box(cx, 135))] if i == 0 else None
        base.step(i, _frame(cx, 135), det)
        hi.step(i, _frame(cx, 135), det)
    assert hi.frame_boxes[6][0][2] > base.frame_boxes[6][0][2]  # face gain widened it
    # A plate ignores face_motion_gain (keeps the shared gain → identical box).
    bp = TrackManager(W, H, **kw)
    hp = TrackManager(W, H, face_motion_gain=2.0, **kw)
    for i in range(8):
        cx = 100 + i * 10
        det = [("plate", _box(cx, 135))] if i == 0 else None
        bp.step(i, _frame(cx, 135), det)
        hp.step(i, _frame(cx, 135), det)
    assert hp.frame_boxes[6][0][2] == bp.frame_boxes[6][0][2]


def test_vehicle_cabin_blur_top_fraction():
    """Vehicle cabin blur: a vehicle-class track emits the TOP cabin_frac of its box
    (the windshield/side-window band where occupants sit at every angle), NOT the whole
    car. cabin_frac 0 falls back to the whole box; face/plate tracks are unaffected."""
    veh = (100, 80, 300, 160)  # x, y, w, h
    tm = TrackManager(W, H, downscale=1, cabin_frac=0.6)
    # speed 0 → no motion margin → cabin = (x, y, w, round(h*0.6)) = top 96 px
    assert tm._emit_box(Track(0, "vehicle", veh, None, 5)) == (100, 80, 300, 96)

    # cabin_frac 0 → the whole vehicle box (no cabin crop)
    off = TrackManager(W, H, downscale=1, cabin_frac=0.0)
    assert off._emit_box(Track(0, "vehicle", veh, None, 5)) == veh

    # a face track keeps its top edge (the cabin crop only applies to vehicles)
    face = Track(1, "face", (120, 120, 48, 48), None, 5)
    assert tm._emit_box(face)[1] == 120


def test_vehicle_corroboration_by_plate_inside_box():
    """A vehicle is plate-corroborated only when a plate box center sits inside it (a
    real car), and confirmed_frame_boxes emits a cabin only for corroborated vehicles —
    a plate-less vehicle (a roadside sign/billboard/caravan) is dropped."""
    veh = (1000, 800, 600, 400)
    # plate inside the vehicle box → corroborated
    tm = TrackManager(W, H, downscale=1, cabin_frac=0.6)
    tm._tracks = [
        Track(0, "vehicle", veh, None, 5),
        Track(1, "plate", (1250, 1100, 200, 80), None, 5),
    ]
    tm._emitted = {
        0: {"cls": "vehicle", "detections": 1, "anchors": [5], "frames": []},
        1: {"cls": "plate", "detections": 1, "anchors": [5], "frames": []},
    }
    tm._corroborate_vehicles()
    assert tm._emitted[0]["has_plate"] is True

    # plate OUTSIDE the vehicle box → not corroborated (a billboard with no plate)
    out = TrackManager(W, H, downscale=1, cabin_frac=0.6)
    out._tracks = [
        Track(0, "vehicle", veh, None, 5),
        Track(1, "plate", (100, 100, 200, 80), None, 5),
    ]
    out._emitted = {
        0: {"cls": "vehicle", "detections": 1, "anchors": [5], "frames": []},
        1: {"cls": "plate", "detections": 1, "anchors": [5], "frames": []},
    }
    out._corroborate_vehicles()
    assert not out._emitted[0].get("has_plate")

    # confirmed_frame_boxes emits a corroborated vehicle's cabin, drops an uncorroborated one
    cabin = (100, 80, 300, 96)
    keep = TrackManager(W, H, downscale=1, cabin_frac=0.6)
    keep._emitted = {
        0: {
            "cls": "vehicle",
            "detections": 1,
            "anchors": [5],
            "frames": [(5, cabin)],
            "has_plate": True,
        }
    }
    assert keep.confirmed_frame_boxes()[5] == [cabin]
    drop = TrackManager(W, H, downscale=1, cabin_frac=0.6)
    drop._emitted = {0: {"cls": "vehicle", "detections": 1, "anchors": [5], "frames": [(5, cabin)]}}
    assert drop.confirmed_frame_boxes() == {}


def test_moving_face_coast_retires_sooner_than_static_hold():
    """Exit-ghost guard: a MOVING coast (likely ghosting past an object that left
    frame) retires at the short ``face_coast_moving``; a STATIC hold (present face
    momentarily missed) keeps the long ``face_coast``."""
    kw = dict(
        downscale=1,
        max_horizon_frames=3,
        face_coast_frames=20,
        face_coast_moving_frames=6,
        coast_move_speed=4.0,
    )
    # Moving: a heading is learned from live flow, then flow dies → short coast.
    mov = TrackManager(W, H, **kw)
    for i in range(6):
        mov.step(i, _frame(120 + i * 10, 135), [("face", _box(120, 135))] if i == 0 else None)
    assert (mov._tracks[0].vx ** 2 + mov._tracks[0].vy ** 2) ** 0.5 >= 4.0  # moving heading
    for i in range(6, 20):
        mov.step(i, _BLANK, None)
    assert mov.frame_boxes[6]  # covered through the short moving coast (last det f0)
    assert mov.frame_boxes[9] == []  # retired well before face_coast (20) — no exit ghost

    # Static: no heading → holds the full anti-blink horizon.
    st = TrackManager(W, H, **kw)
    st.step(0, _frame(240, 135), [("face", _box(240, 135))])
    for i in range(1, 20):
        st.step(i, _BLANK, None)
    assert st.frame_boxes[15]  # static hold survives past the moving horizon (anti-blink kept)


def test_edge_face_survives_sparse_sample_cadence():
    """An edge-pinned face re-detected on its sparse face cadence (a detection every N
    frames) survives the gap between samples and accrues the confirm minimum — the
    edge-retirement staleness bound tracks the sample interval, not a single frame —
    instead of retiring each cycle and respawning at one detection (then dropped)."""
    n = 3  # detections every 3 frames (blur_face_sample_fps @ a typical source fps)
    tm = TrackManager(
        W,
        H,
        downscale=1,
        max_horizon_frames=3,
        face_coast_frames=15,
        retire_edge_coast_margin=3,
        face_sample_interval_frames=n,
    )
    for i in range(3 * n + 1):  # several full sample cycles
        det = [("face", _box(24, 135))] if i % n == 0 else None  # face at the LEFT edge
        tm.step(i, _frame(24, 135), det)
    # Re-detected every n frames the edge track reaches >= 2 detections and stays
    # confirmed (pre-fix it retired at staleness 2 and never got past one detection).
    assert tm.confirmed_frame_boxes(face_min_detections=2)


def test_edge_coast_face_retires_by_detection_staleness():
    """Exit-ghost guard: a face pinned at a frame edge retires once it is stale by MORE
    than one face-sample interval (so a face re-detected on its sparse cadence survives
    the gap between samples), gauged by detection staleness, not flow points — a
    still-present mid-frame face is untouched, and margin 0 disables it."""
    pts = np.zeros((10, 1, 2), np.float32)  # "has flow points" (re-seeded on background)
    tm = TrackManager(
        W,
        H,
        downscale=1,
        max_horizon_frames=3,
        face_coast_frames=30,
        retire_edge_coast_margin=3,
        face_sample_interval_frames=3,
    )
    edge = Track(0, "face", (0, 120, 40, 40), pts, 0)  # parked at x=0 (left edge), last det f0
    assert tm._alive(edge, 3) is True  # stale by one sample interval → grace, still held
    assert tm._alive(edge, 4) is False  # stale beyond the interval + at edge → exit-ghost retired
    mid = Track(1, "face", (200, 120, 40, 40), pts, 0)  # mid-frame, not at any edge
    assert tm._alive(mid, 8) is True  # uses the normal face_coast (30) — kept
    off = TrackManager(W, H, downscale=1, max_horizon_frames=3, face_coast_frames=30)
    assert off._alive(Track(0, "face", (0, 120, 40, 40), pts, 0), 8) is True  # disabled → kept


def test_dedup_drops_weaker_overlapping_track():
    """A duplicate (overlapping a stronger same-class track) is retired so the object
    isn't double-blurred; distinct or cross-class tracks both survive."""
    tm = TrackManager(W, H, downscale=1, dedup_iou=0.5)
    tm._tracks = [
        Track(0, "face", (100, 100, 48, 48), None, 5),
        Track(1, "face", (104, 102, 48, 48), None, 2),
    ]  # overlaps (IoU>0.5)
    tm._emitted = {0: {"detections": 4}, 1: {"detections": 1}}
    tm._dedup_overlapping()
    assert [t.track_id for t in tm._tracks] == [0]  # weaker duplicate dropped

    tm._tracks = [
        Track(0, "face", (100, 100, 48, 48), None, 5),
        Track(1, "face", (400, 100, 48, 48), None, 5),
    ]  # no overlap
    tm._emitted = {0: {"detections": 2}, 1: {"detections": 2}}
    tm._dedup_overlapping()
    assert len(tm._tracks) == 2  # distinct faces both kept

    tm._tracks = [
        Track(0, "face", (100, 100, 48, 48), None, 5),
        Track(1, "plate", (100, 100, 48, 48), None, 5),
    ]  # same spot, diff class
    tm._emitted = {0: {"detections": 2}, 1: {"detections": 2}}
    tm._dedup_overlapping()
    assert len(tm._tracks) == 2  # a face over its own plate — both kept

    # dedup_iou=0 (default) disables it entirely.
    off = TrackManager(W, H, downscale=1)
    off._tracks = [
        Track(0, "face", (100, 100, 48, 48), None, 5),
        Track(1, "face", (104, 102, 48, 48), None, 2),
    ]
    off._emitted = {0: {"detections": 4}, 1: {"detections": 1}}
    off._dedup_overlapping()
    assert len(off._tracks) == 2  # disabled → no merge


def test_confirmed_frame_boxes_posthoc_dedup():
    """Two near-identical RECONSTRUCTED boxes for one object (live tracks that never
    co-existed for the streaming dedup) are merged per-frame in the output, keeping
    the larger so coverage isn't shrunk; distinct boxes both survive."""

    def recs(b1):
        return {
            0: {
                "cls": "face",
                "detections": 3,
                "anchors": [0],
                "frames": [(5, (100, 100, 50, 50))],
            },
            1: {"cls": "face", "detections": 3, "anchors": [1], "frames": [(5, b1)]},
        }

    tm = TrackManager(W, H, downscale=1, dedup_iou=0.55)
    tm._emitted = recs((102, 101, 48, 48))  # overlaps box0 (IoU>0.55)
    assert tm.confirmed_frame_boxes(face_min_detections=2)[5] == [(100, 100, 50, 50)]
    tm._emitted = recs((400, 100, 50, 50))  # distinct
    assert len(tm.confirmed_frame_boxes(face_min_detections=2)[5]) == 2
    off = TrackManager(W, H, downscale=1)  # dedup_iou=0 → disabled
    off._emitted = recs((102, 101, 48, 48))
    assert len(off.confirmed_frame_boxes(face_min_detections=2)[5]) == 2


def test_max_speed_reports_active_plate_and_zero_for_absent_class():
    """max_speed drives adaptive plate cadence: a moving plate track reports its
    per-frame flow speed; a class with no track reports 0."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    step = 12
    for i in range(4):
        cx = 80 + i * step
        det = [("plate", _box(cx, 135))] if i == 0 else None
        tm.step(i, _frame(cx, 135), det)
    assert tm.max_speed("plate") > 5  # ~12 px/frame flow, well above the floor
    assert tm.max_speed("face") == 0.0  # no face track of this class


def test_max_speed_zero_when_no_tracks():
    tm = TrackManager(W, H, downscale=1)
    assert tm.max_speed("plate") == 0.0
    assert tm.max_speed("face") == 0.0


def test_confirm_drops_single_detection_face():
    """A face seen ONCE then coasting (the foliage/shadow-flicker FP signature) is
    dropped at confirm>=2; confirm=1 reproduces frame_boxes exactly."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=5, face_coast_frames=15)
    tm.step(0, _frame(240, 135), [("face", _box(240, 135))])
    for i in range(1, 6):
        tm.step(i, _BLANK, None)  # no further detections → 1-detection track
    assert tm.confirmed_frame_boxes(1) == tm.frame_boxes  # no-op at 1
    assert tm.confirmed_frame_boxes(2) == {}  # 1 < 2 → unconfirmed face dropped


def test_confirm_keeps_multi_detection_face_retroactively():
    """A face re-detected across samples (a real persistent face) is kept — and its
    EARLY frames (before the 2nd detection confirmed it) are emitted retroactively."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60)
    tm.step(0, _frame(240, 135), [("face", _box(240, 135))])  # discovery (det 1)
    for i in range(1, 5):
        tm.step(i, _frame(240 + i, 135), None)  # coast — unconfirmed so far
    tm.step(5, _frame(245, 135), [("face", _box(245, 135))])  # 2nd detection → confirmed
    fb = tm.confirmed_frame_boxes(2)
    assert 0 in fb and 5 in fb  # the pre-confirmation frame 0 is emitted retroactively


def test_confirm_never_drops_plates():
    """Confirmation is face-only — a one-detection plate is always emitted."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=5)
    tm.step(0, _frame(240, 135), [("plate", _box(240, 135))])
    for i in range(1, 4):
        tm.step(i, _frame(240 + i * 5, 135), None)
    assert tm.confirmed_frame_boxes(2)  # plate kept despite a single detection


def test_interpolation_fills_between_detection_anchors():
    """A track held/coasting between two detections gets its gap frames replaced by
    the straight line between the bracketing boxes — covering a small object that
    optical flow froze in place between sparse samples."""
    tm = TrackManager(
        W, H, downscale=1, max_horizon_frames=60, face_coast_frames=30, assoc_min_gate=150
    )
    tm.step(0, _frame(240, 135), [("face", _box(240, 135))])
    for i in range(1, 5):
        tm.step(i, _BLANK, None)  # flow dies → the box holds static at x≈216
    tm.step(5, _frame(270, 135), [("face", _box(270, 135))])  # associates → anchors {0, 5}
    plain = tm.confirmed_frame_boxes(1, interpolate=False)
    interp = tm.confirmed_frame_boxes(1, interpolate=True)
    assert 2 in plain and 2 in interp
    # the held flow box doesn't move; interpolation slides the gap box toward the 2nd anchor
    assert interp[2][0][0] > plain[2][0][0]


def test_pre_roll_covers_head_before_first_detection():
    """Backward extrapolation covers the frames a confirmed moving object is present
    before its first detection (an approaching plate readable before its first sample)."""
    tm = TrackManager(W, H, downscale=1, max_horizon_frames=60, assoc_min_gate=200)
    tm.step(5, _frame(200, 135), [("plate", _box(200, 135))])  # first detection at f5
    for i in range(6, 11):
        cx = 200 + (i - 5) * 8
        det = [("plate", _box(cx, 135))] if i == 10 else None
        tm.step(i, _frame(cx, 135), det)  # moves +8px/frame; 2nd detection at f10
    no_roll = tm.confirmed_frame_boxes(1, pre_roll=0)
    rolled = tm.confirmed_frame_boxes(1, pre_roll=5)
    assert all(f not in no_roll for f in range(5))  # head absent without pre-roll
    head = [f for f in range(5) if f in rolled]
    assert head  # head covered with pre-roll
    assert rolled[head[-1]][0][0] < tm.frame_boxes[5][0][0]  # pre-rolled box is left of f5
