"""Cross-frame motion tracking: densify sparse detections into per-frame boxes
via Lucas-Kanade optical flow, so a blur FOLLOWS a moving face or plate instead
of freezing at the detection position between samples.

The naive alternative — hold each detection's box until the next one, or union
the two endpoints — fails badly on fast objects: a plate caught in a single
2 fps sample gets its box frozen for about a second, blurring empty road ahead
of the plate and then leaving the plate itself exposed as it passes. Detection
stays sparse (it is the expensive part); flow propagation is cheap enough to run
on every frame.

Per-frame LK propagation (translation plus RANSAC partial-affine SCALE, so an
approaching object's box grows with it) + IoU association + snap-correct at each
detection + bounded-horizon retirement. Propagation is forward-only: measurement
on real footage showed backward flow latches onto static background once an
approaching object shrinks below the trackable-feature floor. The default
``downscale`` / ``max_horizon_frames`` come from that measurement — flow drift
sets in roughly 0.15 s out on a worst-case fast pass.

Deterministic: fixed LK parameters, RNG-free feature selection, greedy
order-stable association, and cv2's global RNG pinned at construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Box
from .detect import iou as _iou

# Fixed LK / feature params → deterministic flow across runs and hosts. Window
# size and pyramid depth are per-instance (a 4K frame at downscale 2 needs a
# bigger window and deeper pyramid than a small frame).
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
_FEAT = dict(maxCorners=64, qualityLevel=0.01, minDistance=4, blockSize=7)
# A coasting face extrapolates its last flow heading; an object rarely moves more
# than ~this multiple of its own size between frames (at 60 fps). A larger "flow"
# is LK latching onto background (a textureless face → static grid lines), so we
# keep the prior heading rather than coast on that noise.
_COAST_MAX_STEP = 1.5


@dataclass
class Track:
    """One tracked object. ``box`` is native pixels; ``points`` are LK feature
    points in DOWNSCALED-frame coords (None once the track loses its features).
    ``speed`` is the last per-frame flow displacement of the box center (native
    px), driving the emitted box's motion margin; ``(vx, vy)`` is that
    displacement as a signed vector, refreshed only from live flow so a coasting
    face can EXTRAPOLATE along its last heading instead of freezing."""

    track_id: int
    cls: str
    box: Box
    points: np.ndarray | None
    last_detect_frame: int
    speed: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


class TrackManager:
    """Streaming tracker. Feed full-resolution gray per frame (and the detection
    list on sample frames); read the dense per-frame result from ``frame_boxes``
    after the pass. Emitted boxes are tight + a per-box MOTION margin
    (``motion_gain × the box's own flow speed``, capped at ``motion_max``); the
    flat privacy dilation is applied downstream when the mask timeline is built.
    The margin scales with SPEED, not size, so a fast plate whose flow box trails
    the real plate stays covered while a large but near-static structure FP is not
    ballooned.

    Retirement is class-aware: a plate retires the instant flow points are lost
    (a held fast plate would freeze blur on bare road) and after the tight
    ``max_horizon``; a face instead COASTS through a flow dropout for up to
    ``face_coast_frames`` (> ``max_horizon``), so a small/low-texture pedestrian
    face blurs continuously instead of blinking between detections. The coast is
    velocity-PREDICTIVE: it extrapolates the held box along the track's last
    flow heading, so a fast face's hold follows it instead of ghosting on bare
    road behind it (a static hold left 2-3 stale boxes while the live face moved
    on). Association is widened by ``assoc_motion_gain`` + ``assoc_min_gate``: when
    IoU fails across fast/small motion the detection re-associates to its track by
    a center-distance gate (size-scaled, but floored so a tiny distant face that
    still moves tens of px between sparse detections re-associates instead of
    spawning a duplicate), and a freshly-detected box retires any coasting ghost
    it supersedes.
    ``plate_margin_floor`` gives plates a small motion-independent margin so a
    slow/distant plate (whose speed margin is ≈0) stays covered without
    ballooning a large FP. ``face_margin_frac`` is the analogue for faces but
    SIZE-PROPORTIONAL: the speed margin collapses to the bare frontal-detector box
    when a face's flow speed drops to ≈0 (a pedestrian exiting frame), exposing the
    turned cheek/profile the tight box misses — a proportional floor keeps the blur
    on the whole head while a tiny distant FP's floor scales down with it.
    ``cabin_frac`` drives the occupant blur: a ``vehicle``-class track emits the TOP
    ``cabin_frac`` of its (motion-padded) box (``_emit_box``) — the windshield/
    side-window band where occupants sit at every angle — covering the driver via the
    reliable vehicle box instead of localizing a through-glass face."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        *,
        downscale: int = 2,
        min_points: int = 6,
        max_horizon_frames: int = 9,
        iou_match: float = 0.3,
        lk_win: int = 15,
        lk_levels: int = 2,
        scale_min: float = 0.8,
        scale_max: float = 1.25,
        min_inliers: int = 4,
        face_coast_frames: int = 0,
        face_coast_moving_frames: int = 0,
        coast_move_speed: float = 0.0,
        retire_edge_coast_margin: int = 0,
        face_sample_interval_frames: int = 1,
        motion_gain: float = 0.0,
        motion_max: int = 0,
        face_motion_gain: float = 0.0,
        plate_margin_floor: int = 0,
        face_margin_frac: float = 0.0,
        assoc_motion_gain: float = 0.0,
        assoc_min_gate: int = 0,
        dedup_iou: float = 0.0,
        cabin_frac: float = 0.0,
        rng_seed: int = 0,
    ):
        self.w, self.h = frame_w, frame_h
        self.ds = max(1, downscale)
        self.min_points = min_points
        self.max_horizon = max_horizon_frames
        self.face_coast = face_coast_frames
        self.face_coast_moving = max(0, face_coast_moving_frames)
        self.coast_move_speed = max(0.0, coast_move_speed)
        self.retire_edge_coast_margin = max(0, retire_edge_coast_margin)
        self.face_sample_interval = max(1, face_sample_interval_frames)
        self.dedup_iou = dedup_iou
        self.motion_gain = motion_gain
        self.motion_max = motion_max
        self.face_motion_gain = max(0.0, face_motion_gain)
        self.plate_floor = max(0, plate_margin_floor)
        self.face_margin_frac = max(0.0, face_margin_frac)
        self.assoc_gain = max(0.0, assoc_motion_gain)
        self.assoc_min_gate = max(0, assoc_min_gate)
        self.cabin_frac = max(0.0, cabin_frac)
        self.iou_match = iou_match
        self.scale_min, self.scale_max = scale_min, scale_max
        self.min_inliers = min_inliers
        self._lk = dict(winSize=(lk_win, lk_win), maxLevel=lk_levels, criteria=_LK_CRITERIA)
        # RANSAC in estimateAffinePartial2D draws from cv2's GLOBAL RNG; pin it
        # so two passes over the same input produce bit-identical trajectories.
        cv2.setRNGSeed(rng_seed)
        self._tracks: list[Track] = []
        self._prev: np.ndarray | None = None
        self._nid = 0
        self.frame_boxes: dict[int, list[Box]] = {}
        # Per-track emission history for temporal confirmation: track_id → {cls,
        # detections, frames:[(idx, padded_box)]}. Survives a track's retirement so
        # confirmed_frame_boxes can emit a real face's WHOLE life — including the
        # low-conf discovery frames before it was confirmed — while dropping a
        # one-detection foliage/shadow flicker entirely.
        self._emitted: dict[int, dict] = {}

    def _small(self, gray: np.ndarray) -> np.ndarray:
        if self.ds == 1:
            return gray
        return cv2.resize(gray, (self.w // self.ds, self.h // self.ds))

    def _clamp(self, box: Box) -> Box:
        x, y, bw, bh = box
        x, y = max(0, x), max(0, y)
        return (x, y, max(0, min(bw, self.w - x)), max(0, min(bh, self.h - y)))

    @staticmethod
    def _center(box: Box) -> tuple[float, float]:
        return (box[0] + box[2] / 2, box[1] + box[3] / 2)

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _floor(self, cls: str, box: Box) -> int:
        """Minimum motion margin for a class. Plates get a flat ``plate_floor`` — a
        slow/distant plate's speed margin is ≈0, so a tight box exposes the plate
        between detections. Faces get a SIZE-PROPORTIONAL floor (``face_margin_frac``
        × the box's larger side): the per-box motion margin is speed-scaled, so it
        COLLAPSES to the bare detector box when flow speed drops to ≈0 — a face
        exiting frame loses its flow features, and the frontal detector box is
        tighter than the visible head — leaving the leading cheek/profile poking out
        at the soft blur edge. The proportional floor keeps coverage from shrinking
        to the tight box; proportional (not a flat constant) so a tiny distant FP
        face's floor scales down with it and isn't ballooned (precision held)."""
        if cls == "plate":
            return self.plate_floor
        if self.face_margin_frac > 0:
            return round(self.face_margin_frac * max(box[2], box[3]))
        return 0

    def _motion_pad(
        self, box: Box, speed: float, floor: int = 0, gain: float | None = None
    ) -> Box:
        """Grow a tight ``box`` on every side by ``gain × speed`` px (capped at
        ``motion_max``, floored at ``floor``) to cover the flow lag of a fast object —
        the box trails because LK underestimates large motion. The margin scales with
        SPEED, not size, so a large near-static box (a structure FP) is never
        ballooned by its size; ``floor`` is a flat per-class minimum (a constant, so
        it likewise never scales a big FP) that keeps a slow/distant plate covered.
        ``gain`` overrides ``motion_gain`` per class (faces use a higher
        ``face_motion_gain`` so a fast face's blob grows enough to cover the leading
        edge it lags on / its trailing collapse as it exits frame). Returns the box
        unchanged when both are zero."""
        g = self.motion_gain if gain is None else gain
        m = round(g * speed) if (g > 0 and speed > 0) else 0
        if self.motion_max > 0:
            m = min(m, self.motion_max)
        m = max(m, floor)
        if m <= 0:
            return box
        x, y, bw, bh = box
        return self._clamp((x - m, y - m, bw + 2 * m, bh + 2 * m))

    def _seed(self, box: Box, small: np.ndarray) -> np.ndarray | None:
        """Feature points inside ``box`` (downscaled coords), or None if barren."""
        x, y, bw, bh = (v // self.ds for v in box)
        x0, y0 = max(0, x), max(0, y)
        roi = small[y0 : y0 + max(1, bh), x0 : x0 + max(1, bw)]
        if roi.size == 0:
            return None
        pts = cv2.goodFeaturesToTrack(roi, **_FEAT)
        if pts is None:
            return None
        pts = pts.astype(np.float32)
        pts[:, 0, 0] += x0
        pts[:, 0, 1] += y0
        return pts

    def _move_box(self, box: Box, src: np.ndarray, dst: np.ndarray) -> Box:
        """New box from old→new point pairs (downscaled Nx2). A RANSAC
        partial-affine gives translation + uniform scale (so a growing oncoming
        plate's box grows with it); the per-step scale is clamped to keep one bad
        estimate from ballooning/collapsing the box. Falls back to median
        translation (scale 1.0) when too few inliers make the affine unstable."""
        cx, cy = (box[0] + box[2] / 2) / self.ds, (box[1] + box[3] / 2) / self.ds
        s = 1.0
        m = None
        if len(src) >= self.min_inliers:
            m, inl = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3
            )
            if m is None or inl is None or int(inl.sum()) < self.min_inliers:
                m = None
        if m is not None:
            ncx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
            ncy = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
            s = min(self.scale_max, max(self.scale_min, float(np.hypot(m[0, 0], m[0, 1]))))
        else:
            ncx = cx + float(np.median(dst[:, 0] - src[:, 0]))
            ncy = cy + float(np.median(dst[:, 1] - src[:, 1]))
        nw, nh = box[2] * s, box[3] * s
        ncx, ncy = ncx * self.ds, ncy * self.ds
        return self._clamp((round(ncx - nw / 2), round(ncy - nh / 2), round(nw), round(nh)))

    def _coast(self, t: Track) -> None:
        """A face that lost its flow points but is still within ``face_coast``
        EXTRAPOLATES its box along the track's last flow heading (``vx, vy``)
        instead of freezing — so a moving face's hold follows it across the
        detection gap rather than ghosting on bare road behind it. Plates don't
        coast (``_alive`` retires them on flow loss), so this is a no-op for them;
        a static face (heading 0) holds in place, the old anti-blink behaviour."""
        if t.cls != "face":
            return
        x, y, bw, bh = t.box
        t.box = self._clamp((round(x + t.vx), round(y + t.vy), bw, bh))

    def _propagate(self, t: Track, small: np.ndarray) -> bool:
        """Move + scale ``t.box`` to follow its feature points across one frame.
        Sets ``t.points`` to None when fewer than 3 survive — then a face coasts
        (velocity-extrapolated) and a plate is left for ``_alive`` to retire.
        Returns True when LIVE flow moved the box (the caller refreshes velocity
        only then; a coast must reuse the last live heading, not its own output)."""
        if t.points is None or len(t.points) < 3:
            t.points = self._seed(t.box, self._prev)
        if t.points is None or len(t.points) == 0:
            t.points = None
            self._coast(t)
            return False
        new, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, small, t.points, None, **self._lk)
        if new is None or st is None:
            t.points = None
            self._coast(t)
            return False
        ok = st.reshape(-1) == 1
        gn, go = new[ok], t.points[ok]
        if len(gn) < 3:
            t.points = None
            self._coast(t)
            return False
        t.box = self._move_box(t.box, go.reshape(-1, 2), gn.reshape(-1, 2))
        if len(gn) >= self.min_points:
            t.points = gn
        else:  # top up depleted points from the moved box; keep gn if barren
            reseed = self._seed(t.box, small)
            t.points = reseed if reseed is not None else gn
        return True

    def _coast_horizon(self, t: Track) -> int:
        """Frames a track may outlive its last detection. A face coasts longer
        (a slow pedestrian whose box is held across a detection gap beats a
        blinking blur); a plate keeps the tight ``max_horizon`` — a held fast
        plate would freeze blur on bare road (the static-union defect). BUT a
        MOVING face coast — velocity-predictive, extrapolating onto NEW area — is
        far more likely ghosting past an object that has left frame (a passing car's
        occupant, a pedestrian walking out) than a STATIC hold (a present face the
        detector momentarily missed), so a moving face retires after the shorter
        ``face_coast_moving`` instead of the full hold. A mid-traverse detection gap
        is filled by interpolation, not the coast, so this only trims the dead tail
        after the LAST detection — no blink of a still-present face."""
        if t.cls == "face" and self.face_coast > self.max_horizon:
            # Gauge "moving" by the box's actual per-frame DISPLACEMENT (``t.speed``),
            # not the flow heading (``vx, vy``): a coast whose flow latched onto a
            # fast-passing object (a car body the face's box rode out of frame) moves
            # the box hundreds of px/frame, but that step is "implausible" for the
            # heading filter so ``vx, vy`` stay ~0 — it would read as a static hold and
            # ghost for the full horizon. ``t.speed`` captures the real motion.
            if self.face_coast_moving > 0 and t.speed >= self.coast_move_speed:
                return self.face_coast_moving
            return self.face_coast
        return self.max_horizon

    def _at_frame_edge(self, box: Box) -> bool:
        """The box touches a frame boundary (within ``retire_edge_coast_margin``)."""
        m = self.retire_edge_coast_margin
        x, y, bw, bh = box
        return x <= m or y <= m or x + bw >= self.w - m or y + bh >= self.h - m

    def _alive(self, t: Track, frame_idx: int) -> bool:
        # A face survives a flow dropout (points went None on a small/low-texture
        # face) by HOLDING its last box, so the blur doesn't blink off between
        # detections; a plate must keep live flow points or it retires at once.
        has_track = t.points is not None or t.cls == "face"
        # A face pinned against a frame edge that hasn't been DETECTED for more than one
        # FACE-SAMPLE interval has walked/driven OUT of frame — its hold would otherwise
        # freeze a blur on the bare edge for the whole coast horizon (the exit-ghost: a
        # pedestrian off frame-left, a car's occupant off frame-right). Gauged by
        # detection staleness, NOT live flow points: a coasting box re-seeds points on
        # the static background it's parked over (a wall at x=0), so it "has flow" yet
        # tracks nothing — the points test would miss it. The bound is the sample
        # interval, not one frame: faces detect sparsely (``face_sample_interval``), so a
        # still-present edge face's staleness naturally grows to a full interval between
        # samples — a 1-frame bound retired it every cycle before it could accumulate the
        # confirm minimum. It exceeds the interval only when an expected sample is missed.
        if (
            self.retire_edge_coast_margin > 0
            and t.cls == "face"
            and frame_idx - t.last_detect_frame > self.face_sample_interval
            and self._at_frame_edge(t.box)
        ):
            return False
        return (
            has_track
            and t.box[2] > 0
            and t.box[3] > 0
            and frame_idx - t.last_detect_frame <= self._coast_horizon(t)
        )

    def _associate(self, box: Box, cls: str) -> Track | None:
        """Best existing track for a detection: IoU first (tight), then — when IoU
        fails across fast/small motion — a center-distance gate, so the SAME fast
        face re-associates to its track instead of spawning a duplicate that then
        coasts as a ghost. The gate is ``max(assoc_gain × box diagonal,
        assoc_min_gate) + track speed``: the size term widens it for big (near)
        faces, while the flat floor covers a SMALL distant face whose tight box is
        tiny yet still moves tens of px between sparse detections (its size term
        alone would be under the inter-detection motion). Greedy / order-stable →
        deterministic. ``assoc_gain``/``assoc_min_gate`` both 0 → IoU-only."""
        best, best_iou = None, self.iou_match
        for t in self._tracks:
            if t.cls != cls:
                continue
            score = _iou(box, t.box)
            if score >= best_iou:
                best, best_iou = t, score
        if best is not None or (self.assoc_gain <= 0 and self.assoc_min_gate <= 0):
            return best
        bc = self._center(box)
        bdiag = (box[2] ** 2 + box[3] ** 2) ** 0.5
        best, best_d = None, None
        for t in self._tracks:
            if t.cls != cls:
                continue
            tdiag = (t.box[2] ** 2 + t.box[3] ** 2) ** 0.5
            gate = max(self.assoc_gain * max(bdiag, tdiag), self.assoc_min_gate) + t.speed
            d = self._dist(bc, self._center(t.box))
            if d <= gate and (best_d is None or d < best_d):
                best, best_d = t, d
        return best

    def _retire_superseded(self, frame_idx: int) -> None:
        """Drop a coasting ghost — a face holding a box with no live flow points —
        when a freshly-detected track of the same class overlaps it. The live
        detection supersedes the stale hold; keeping both would blur the object's
        old position on bare road. Fires only when association failed to merge the
        two (a duplicate spawned atop a ghost); order-stable."""
        fresh = [t for t in self._tracks if t.last_detect_frame == frame_idx]
        if not fresh:
            return
        self._tracks = [
            t
            for t in self._tracks
            if not (
                t.points is None
                and t.last_detect_frame != frame_idx
                and any(
                    f is not t and f.cls == t.cls and _iou(f.box, t.box) > self.iou_match
                    for f in fresh
                )
            )
        ]

    def _dedup_overlapping(self) -> None:
        """Drop a duplicate track: when two same-class tracks overlap by more than
        ``dedup_iou`` keep the STRONGER (more associated detections, then the more
        recently detected) and retire the other. A fast/sparse object can spawn a
        second track the association gate didn't merge — doubling its blur — and a
        coasting remnant can re-overlap a re-acquired track; this is the catch-all
        ``_retire_superseded`` (coast-vs-fresh only) misses. A high threshold so two
        genuinely distinct faces crossing aren't merged. Order-stable / deterministic."""
        if self.dedup_iou <= 0 or len(self._tracks) < 2:
            return
        order = sorted(
            self._tracks,
            key=lambda t: (self._emitted[t.track_id]["detections"], t.last_detect_frame),
            reverse=True,
        )
        kept_ids: set[int] = set()
        kept: list[Track] = []
        for t in order:
            if any(k.cls == t.cls and _iou(k.box, t.box) > self.dedup_iou for k in kept):
                continue
            kept.append(t)
            kept_ids.add(t.track_id)
        if len(kept_ids) != len(self._tracks):
            self._tracks = [t for t in self._tracks if t.track_id in kept_ids]

    def _corroborate_vehicles(self) -> None:
        """Mark a vehicle track plate-corroborated once a plate box's center falls
        inside it. A real road car shows a license plate; a roadside billboard / sign /
        parked caravan that the COCO model misreads as a vehicle (at the same confidence
        as a real car — conf can't separate them) does NOT. Sticky: once seen, the car's
        cabin blurs for its whole track (the plate isn't re-detected every frame). A
        vehicle never corroborated is never cabin-blurred (``confirmed_frame_boxes``), so
        roadside signage is left readable. A car close enough for identifiable occupants
        is close enough for its plate to detect, so this scopes the blur where it matters."""
        plates = [t.box for t in self._tracks if t.cls == "plate"]
        if not plates:
            return
        for t in self._tracks:
            if t.cls != "vehicle" or self._emitted[t.track_id].get("has_plate"):
                continue
            vx, vy, vw, vh = t.box
            for px, py, pw, ph in plates:
                cx, cy = px + pw / 2, py + ph / 2
                if vx <= cx <= vx + vw and vy <= cy <= vy + vh:
                    self._emitted[t.track_id]["has_plate"] = True
                    break

    def _assimilate(self, frame_idx: int, small: np.ndarray, detections) -> None:
        """Snap a matching track to each detection (resets flow drift) or spawn
        a new track, then retire any ghost a fresh detection supersedes. Greedy in
        detection order → deterministic."""
        for cls, raw in detections:
            box = self._clamp(raw)
            best = self._associate(box, cls)
            if best is not None:
                best.box = box
                best.points = self._seed(box, small)
                best.last_detect_frame = frame_idx
                rec = self._emitted[best.track_id]
                rec["detections"] += 1
                rec["anchors"].append(frame_idx)
            else:
                self._tracks.append(Track(self._nid, cls, box, self._seed(box, small), frame_idx))
                self._emitted[self._nid] = {
                    "cls": cls,
                    "detections": 1,
                    "frames": [],
                    "anchors": [frame_idx],
                }
                self._nid += 1
        self._retire_superseded(frame_idx)

    def max_speed(self, cls: str) -> float:
        """Largest per-frame flow speed (native px) among active tracks of ``cls``,
        0.0 if none. Drives adaptive plate cadence: a plate accelerating through a
        close pass moves farther per frame than the sparse sample cadence can snap
        to, so the caller boosts plate detection while this exceeds a threshold."""
        return max((t.speed for t in self._tracks if t.cls == cls), default=0.0)

    def step(self, frame_idx: int, gray: np.ndarray, detections=None) -> list[Box]:
        """Advance one frame. ``detections`` is ``[(cls, box), ...]`` on sample
        frames (``None``/empty otherwise). Returns this frame's active boxes — each
        the tight tracked box plus its motion margin."""
        small = self._small(gray)
        if self._prev is not None:
            for t in self._tracks:
                pre = self._center(t.box)
                moved = self._propagate(t, small)
                post = self._center(t.box)
                t.speed = self._dist(pre, post)  # flow displacement, pre-snap
                if moved:  # refresh heading only from LIVE, PLAUSIBLE flow; an
                    dx, dy = post[0] - pre[0], post[1] - pre[1]  # implausible jump
                    bound = _COAST_MAX_STEP * max(t.box[2], t.box[3])  # is LK noise
                    if dx * dx + dy * dy <= bound * bound:  # → keep the prior heading
                        t.vx, t.vy = dx, dy
        self._tracks = [t for t in self._tracks if self._alive(t, frame_idx)]
        if detections:
            self._assimilate(frame_idx, small, detections)
        self._dedup_overlapping()
        self._corroborate_vehicles()
        boxes = [self._emit_box(t) for t in self._tracks]
        self.frame_boxes[frame_idx] = boxes
        for t, box in zip(self._tracks, boxes, strict=True):
            self._emitted[t.track_id]["frames"].append((frame_idx, box))
        self._prev = small
        return boxes

    @staticmethod
    def _interpolate_gaps(frames: dict[int, Box], anchors: list[int]) -> dict[int, Box]:
        """Override a track's between-anchor gap frames with the LINEAR interpolation
        of the bracketing detection boxes, instead of the optical-flow box. Flow fails
        on small low-texture objects (a distant plate, a small face) — the box freezes
        or drifts between sparse detections — but a detection at each end pins the true
        endpoints, so a straight line covers the gap far better. Frames before the first
        / after the last anchor keep their flow box (no bracket to interpolate)."""
        out = dict(frames)
        uniq = sorted(set(anchors))
        for a, b in zip(uniq, uniq[1:], strict=False):
            ba, bb = frames.get(a), frames.get(b)
            if b - a <= 1 or ba is None or bb is None:
                continue
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                out[f] = tuple(round(ba[i] + t * (bb[i] - ba[i])) for i in range(4))
        return out

    def _pre_roll_boxes(self, frames: dict[int, Box], anchors: list[int], k: int) -> dict[int, Box]:
        """Up to ``k`` backward-extrapolated boxes BEFORE a track's first detection,
        from the velocity of its first two detections — covers the head frames a
        confirmed object is present but not yet detected (a small/distant oncoming
        plate readable before its first sample scores it; a pedestrian walking in
        before her first hit). Size extrapolates too (a growing approaching box
        shrinks going back; once it collapses to ≤0 the object wasn't there, so it
        stops). Empty with < 2 anchors (no velocity). Bare-road risk is bounded by a
        small ``k`` and by only ever pre-rolling a CONFIRMED, moving track."""
        uniq = sorted(set(anchors))
        if k <= 0 or len(uniq) < 2:
            return {}
        a0, a1 = uniq[0], uniq[1]
        b0, b1 = frames.get(a0), frames.get(a1)
        if b0 is None or b1 is None:
            return {}
        span = a1 - a0
        vel = [(b1[i] - b0[i]) / span for i in range(4)]
        out: dict[int, Box] = {}
        for f in range(max(0, a0 - k), a0):
            d = f - a0  # negative → step backward along the heading
            x, y, bw, bh = (b0[i] + vel[i] * d for i in range(4))
            if bw <= 0 or bh <= 0:  # box collapsed → object wasn't there yet
                continue
            out[f] = self._clamp((round(x), round(y), round(bw), round(bh)))
        return out

    def _emit_box(self, t: Track) -> Box:
        """The blur box a track contributes this frame. Face/plate: the tight box plus
        a speed-scaled motion margin (size-proportional face floor / flat plate floor;
        faces take the higher face gain). VEHICLE: the motion-padded box's TOP
        ``cabin_frac`` — the windshield/side-window band where occupants sit at EVERY
        viewing angle (head-on or passing) — covering the driver without localizing a
        through-glass face. ``cabin_frac`` ≤ 0 falls back to the whole vehicle box."""
        if t.cls == "vehicle":
            # The tight box, snapped to ground truth every frame on the close pass (the
            # boost). NO speed-scaled motion pad — it ballooned a drifted box; the cabin
            # fraction already over-covers and the box is accurate per-frame.
            x, y, bw, bh = t.box
            if self.cabin_frac <= 0:
                return (x, y, bw, bh)
            return self._clamp((x, y, bw, max(1, round(bh * self.cabin_frac))))
        gain = self.face_motion_gain if (t.cls == "face" and self.face_motion_gain > 0) else None
        return self._motion_pad(t.box, t.speed, self._floor(t.cls, t.box), gain)

    def confirmed_frame_boxes(
        self,
        face_min_detections: int = 1,
        interpolate: bool = False,
        pre_roll: int = 0,
    ) -> dict[int, list[Box]]:
        """Dense per-frame map rebuilt from track histories, emitting a FACE track's
        boxes only once it accumulated ``>= face_min_detections`` associated detections.
        A small clear face (the recall target) re-detects across consecutive samples as
        it persists → many detections → kept, INCLUDING the low-conf frames before the
        first high-conf hit (retroactive, because the whole track life is recorded). A
        foliage/shadow/wall false positive fires once and then coasts → one detection →
        dropped entirely, so a low face conf can chase real small faces without blurring
        the static-texture FPs that share its score range. Plates are always emitted
        (recall-critical and already precision-gated by aspect/area). ``interpolate``
        replaces each track's between-detection gap boxes with the straight-line
        interpolation of the bracketing detections (covers a small object flow can't
        track between sparse samples). ``pre_roll`` adds up to that many backward-
        extrapolated boxes before the first detection (covers the head a confirmed
        object is present-but-undetected). ``=1`` + no interpolate + no pre_roll
        reproduces ``frame_boxes`` exactly, the default. Vehicle tracks (whose emitted
        box is already the cabin band, ``_emit_box``) are always emitted, like plates.
        When ``dedup_iou`` is set, a final per-frame pass drops a box overlapping a
        larger kept box by more than it — the post-hoc catch for two near-identical
        reconstructed boxes of one object whose live tracks never co-existed for the
        streaming dedup."""
        from collections import defaultdict

        fb: dict[int, list[Box]] = defaultdict(list)
        for rec in self._emitted.values():
            if rec["cls"] == "face" and rec["detections"] < face_min_detections:
                continue
            # A vehicle is cabin-blurred only once a license plate corroborated it as a
            # real road car (see _corroborate_vehicles) — a plate-less roadside billboard
            # /sign/caravan COCO misreads as a vehicle is left unblurred (readable).
            if rec["cls"] == "vehicle" and not rec.get("has_plate"):
                continue
            base = dict(rec["frames"])
            frames = (
                self._interpolate_gaps(base, rec["anchors"])
                if interpolate and len(rec["anchors"]) >= 2
                else dict(base)
            )
            if pre_roll > 0:
                for f, box in self._pre_roll_boxes(base, rec["anchors"], pre_roll).items():
                    frames.setdefault(f, box)  # only ADD head frames, never overwrite
            for f, box in frames.items():
                fb[f].append(box)
        if self.dedup_iou <= 0:
            return dict(fb)
        # Post-hoc per-frame dedup: interpolation / pre-roll can reconstruct two
        # near-identical boxes for ONE object whose live tracks never co-existed at a
        # single step for the streaming ``_dedup_overlapping`` to catch them (e.g. two
        # tail tracks of the same exiting face). Drop a box overlapping a LARGER kept
        # box by more than ``dedup_iou`` — keep the larger so coverage never shrinks;
        # the high threshold leaves two genuinely distinct (crossing) faces.
        out: dict[int, list[Box]] = {}
        for f, boxes in fb.items():
            kept: list[Box] = []
            for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
                if not any(_iou(b, k) > self.dedup_iou for k in kept):
                    kept.append(b)
            out[f] = kept
        return out
