"""Tuned parameter sets.

Every number below was arrived at by watching a detector fail on real footage and
then finding the axis that separated the failure from the thing it was mistaken
for. They are exposed as an ordinary dict so you can start from them, print them,
diff them, or ignore them entirely.

``ROAD_FOOTAGE`` is tuned for **forward-facing, vehicle-mounted 4K video**: a
camera moving at road speed, with oncoming traffic, roadside signage and
pedestrians. The characteristic problems of that footage, and what each group of
values is doing about them:

* **Objects are small and far away.** A licence plate 40 m out is a handful of
  pixels. Downscaling a 4K frame to a 640 px network input destroys it, so
  detection is tiled (``tile_px``) and each tile runs at close to native scale.

* **Objects move fast.** An oncoming car closes at 100+ km/h combined. Optical
  flow cannot follow that between sparse samples, so a fast plate re-detects on
  every frame (``plate_boost_speed_px``) instead of on the base cadence.

* **Roadside signage looks like a licence plate to a plate model.** A place-name
  sign scores above any usable confidence. Confidence cannot separate the two;
  aspect ratio and area can (``plate_min_aspect`` / ``plate_max_aspect`` /
  ``plate_max_area_frac``).

* **Foliage, walls and shadows look like faces to a face model.** They score
  0.5-0.7, comfortably above a genuine small face at 0.19. Again confidence
  cannot separate them — but persistence can: a real face re-detects across
  consecutive samples as it stays in frame, a texture artefact fires once. Hence
  a low ``face_conf`` paired with ``face_confirm_min_detections``. Lowering the
  confidence without the confirmation gives you a video full of blurred hedges.

* **Occupants sit behind moving glass.** Face detection through a windscreen is
  unreliable at any confidence — the model latches onto reflections and roof
  lines. So the cabin is covered geometrically: detect the whole vehicle, blur
  the top ``track_cabin_frac`` of its box, which is where occupants sit at every
  viewing angle.

* **The camera vehicle's own bonnet is in frame.** The COCO model reports it as a
  car every single frame. ``vehicle_ignore_bottom_frac`` excludes that strip;
  ``vehicle_min_height_frac`` drops cars too distant to have an identifiable
  occupant.

If your footage is not road footage — a fixed CCTV camera, a handheld phone, a
drone — most of these will be wrong for you. In particular the aspect and area
gates assume vehicle plates seen from a vehicle, and the bottom-strip exclusion
assumes a bonnet is there. Start from the function defaults instead and add gates
one at a time as you find your own false positives.
"""

from __future__ import annotations

# Passed straight to ``detect.detect_and_track(video, **ROAD_FOOTAGE)``.
ROAD_FOOTAGE: dict[str, object] = {
    # --- detection cadence -------------------------------------------------
    # 8 fps keeps the inter-detection gap (~0.125 s) inside the measured optical
    # flow horizon (~0.13 s), so flow never has to bridge more than it can.
    "sample_fps": 8.0,
    # Faces get a denser cadence of their own: a small low-texture face cannot be
    # flow-tracked between sparse samples (its box freezes while the person walks
    # out of shot), and faces are cheap relative to the plate-detector bottleneck.
    "face_sample_fps": 16.0,
    # --- confidence --------------------------------------------------------
    "conf": 0.4,  # plate detector: below 0.4 squarish road signs reappear
    "face_conf": 0.15,  # deliberately low; see face_confirm_min_detections
    # The precision mechanism that makes face_conf=0.15 safe. A real face
    # re-detects across samples; a foliage/shadow/wall artefact fires once and
    # then coasts. Never lower face_conf without this.
    "face_confirm_min_detections": 2,
    # --- plate false-positive gates ---------------------------------------
    # Real plates measured ≈2.4-3:1 at an angle and ≤4.7:1 head-on. Road signs
    # box ≈1.4-1.65:1; shorelines and bridge railings ≈13:1. 2.0/6.0 brackets the
    # gap on both sides.
    "plate_min_aspect": 2.0,
    "plate_max_aspect": 6.0,
    # Stone walls, verges and bridge abutments box at a plausible 2.0-2.7:1 that
    # the aspect band cannot catch, but they cover 1.4-7.6% of the frame; a large
    # roadside place-name sign covers 0.84-1.45%. A real plate is ≤0.4% of a 4K
    # frame, so 0.6% keeps every real plate and drops all of the above.
    "plate_max_area_frac": 0.006,
    # --- tiling ------------------------------------------------------------
    "tile_px": 1280,  # 4K → a 3x2 grid; 1080p mostly stays single-tile
    "tile_overlap": 0.15,  # a seam-straddling object stays whole in one tile
    # --- adaptive plate cadence -------------------------------------------
    # Any moving plate re-detects every frame. Set low on purpose: a static
    # parked plate does not leak (flow holds it fine), a moving one does.
    "plate_boost_speed_px": 8,
    "plate_boost_max_run": 6,
    # --- tracking ----------------------------------------------------------
    # 4K at downscale 2 needs a bigger LK window and a deeper pyramid than a
    # small frame does.
    "track_downscale": 2,
    "track_lk_win": 31,
    "track_lk_levels": 4,
    "track_max_horizon_frames": 9,
    "track_face_coast_frames": 20,
    "track_face_coast_moving_frames": 6,
    "track_coast_move_speed": 6.0,
    "track_retire_edge_coast_margin": 3,
    "track_motion_gain": 1.0,
    "track_motion_max_px": 160,
    "track_face_motion_gain": 1.6,
    "track_plate_margin_floor_px": 16,
    "track_face_margin_frac": 0.35,
    "track_assoc_motion_gain": 1.2,
    "track_assoc_min_gate_px": 72,
    "track_dedup_iou": 0.55,
    # --- occupants and people ---------------------------------------------
    "track_cabin_frac": 0.6,  # top 60% of a vehicle box = the cabin band
    # High on purpose: the COCO model calls rectangular roadside boards "car" at
    # 0.35-0.55, while real oncoming cars score 0.63-0.95. 0.6 keeps the cars and
    # drops the signage, which must stay readable.
    "vehicle_conf": 0.6,
    "vehicle_min_height_frac": 0.07,
    "vehicle_ignore_bottom_frac": 0.12,  # the camera vehicle's own bonnet
    "track_person": True,
    "person_conf": 0.35,
    "person_min_height_frac": 0.05,  # drop distant pedestrian dots
    # --- head coverage -----------------------------------------------------
    # Extrapolate a confirmed track backwards: an approaching plate is readable,
    # and a pedestrian present, for a few frames before the detector first scores
    # them.
    "pre_roll_frames": 5,
}

# Flat safety margin added to every box when the timeline is built. Separate from
# the tracker's speed-scaled margin, which covers flow lag rather than detector
# tightness.
ROAD_FOOTAGE_BOX_DILATION_PX = 8

# Mask feather radius. A hard mask edge reads as an obvious rectangle; a feather
# blends the blur into the surrounding pixels.
ROAD_FOOTAGE_FEATHER_PX = 8
