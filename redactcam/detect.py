"""Deterministic privacy detector — faces ∪ licence plates per sampled frame.

Stage 1 of the redaction chain: locate the regions. Everything downstream
(``timeline`` → ``mask`` → ``apply``) turns those regions into blurred pixels.

Faces and plates both run single-class YOLO ONNX models behind one
``YoloDetector`` (the face model adds no aspect gate; the plate model drops
squarish road-sign false positives). If no face model resolves, the optional
``redactcam[centerface]`` extra supplies a CenterFace fallback; with neither, a
face detector cannot be built and construction raises rather than silently
detecting nothing. A missing PLATE model is a warning, not an error — plates are
simply disabled, and the caller decides whether that is fatal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """No face detector could be built — neither a face ONNX model nor the
    optional CenterFace fallback is available. Raised rather than returning a
    detector that finds nothing, because a silent no-op here ships faces."""


# Box is (x, y, w, h) in NATIVE frame pixels. One box shape everywhere, so the
# timeline, mask and coverage verifier never have to convert.
Box = tuple[int, int, int, int]

# COCO class ids whose occupants need the cabin blur: car, motorcycle, bus, truck.
VEHICLE_CLASSES = frozenset({2, 3, 5, 7})
# COCO person class — the WHOLE-person blur (someone standing beside the camera,
# a pedestrian): the full box is blurred, not just the face. A face blur alone
# leaves body, clothing and pose, which identify a person nearly as well.
PERSON_CLASSES = frozenset({0})


def filter_vehicles(
    boxes: list[Box], frame_h: int, min_height_frac: float, ignore_bottom_frac: float
) -> list[Box]:
    """Keep only vehicle boxes worth a cabin blur: tall enough that occupants are
    discernible (``min_height_frac`` × frame height — a distant 80 px car has no
    identifiable occupant) and not glued to the bottom strip (box top above
    ``1 - ignore_bottom_frac`` of the frame — on a vehicle-mounted camera that strip
    is the camera car's own bonnet, which the COCO model reports as a 'car' but which
    carries no visible occupants). Both gates default off at 0."""
    min_h = min_height_frac * frame_h
    top_limit = (1.0 - ignore_bottom_frac) * frame_h if ignore_bottom_frac > 0 else frame_h
    return [b for b in boxes if b[3] >= min_h and b[1] <= top_limit]


def _xyxy_to_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Box | None:
    """Round + clamp an xyxy detection to an in-bounds ``(x, y, w, h)`` box."""
    bx, by = max(0, round(min(x1, x2))), max(0, round(min(y1, y2)))
    bx2, by2 = min(w, round(max(x1, x2))), min(h, round(max(y1, y2)))
    return (bx, by, bx2 - bx, by2 - by) if bx2 > bx and by2 > by else None


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize into a square ``size``x``size`` canvas (YOLO
    convention). Returns the canvas plus ``(scale, pad_x, pad_y)`` so a box in
    canvas space maps back to native via ``(coord - pad) / scale``."""
    h, w = img.shape[:2]
    scale = size / max(h, w) if max(h, w) else 1.0
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA
    )
    return canvas, scale, pad_x, pad_y


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedupe(boxes: list[Box], iou_thresh: float = 0.5) -> list[Box]:
    """Drop boxes that overlap an earlier kept box. Overlapping tiles emit
    near-duplicates of a seam-straddling object; the mask unions boxes anyway,
    so this only keeps detection counts honest. Order-stable → deterministic."""
    kept: list[Box] = []
    for b in boxes:
        if not any(iou(b, k) > iou_thresh for k in kept):
            kept.append(b)
    return kept


def frame_tiles(w: int, h: int, tile_px: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping ``(x0, y0, x1, y1)`` crops covering a ``w``×``h`` frame. The
    grid is derived from ``tile_px`` (≈ target tile edge) so a 4K frame splits
    while a 1080p one mostly doesn't — running each detector at native scale
    instead of crushing small faces/plates to a few pixels in a whole-frame
    downscale. ``overlap`` pads each tile so a seam-straddling object stays fully
    visible in at least one tile. ``tile_px`` ≥ the frame → one tile (no tiling)."""
    nx, ny = max(1, round(w / tile_px)), max(1, round(h / tile_px))
    if nx == 1 and ny == 1:
        return [(0, 0, w, h)]
    tw, th = w / nx, h / ny
    ox, oy = tw * overlap, th * overlap
    return [
        (
            max(0, int(ix * tw - ox)),
            max(0, int(iy * th - oy)),
            min(w, int((ix + 1) * tw + ox)),
            min(h, int((iy + 1) * th + oy)),
        )
        for iy in range(ny)
        for ix in range(nx)
    ]


def detect_tiled(frame: np.ndarray, detector, conf: float, tiles) -> list[Box]:
    """Run ``detector`` over each tile, merge boxes into full-frame coords, dedupe."""
    if len(tiles) == 1:
        return detector.detect(frame, conf)
    merged: list[Box] = []
    for x0, y0, x1, y1 in tiles:
        for bx, by, bw, bh in detector.detect(frame[y0:y1, x0:x1], conf):
            merged.append((x0 + bx, y0 + by, bw, bh))
    return dedupe(merged)


@dataclass
class FrameRegions:
    """Privacy regions in one sampled frame. ``boxes`` is ``faces`` ∪ ``plates``;
    the split is kept so callers can treat classes differently. Native pixels."""

    frame_idx: int
    timestamp: float
    boxes: list[Box] = field(default_factory=list)
    faces: list[Box] = field(default_factory=list)
    plates: list[Box] = field(default_factory=list)


class CenterFaceDetector:
    """CenterFace face detector at a fixed reduced inference resolution.

    The fallback when no face ONNX model is configured. Requires the optional
    ``redactcam[centerface]`` extra, which pulls in ``deface`` (whose wheel ships
    the CenterFace weights). Measurably weaker than the YOLO face model on
    outdoor scenes — it scores foliage above real faces at usable thresholds —
    so treat it as a safety net, not a default.
    """

    def __init__(self, infer_shape: tuple[int, int]):
        # Lazy import: the extra is only needed when this fallback is actually
        # used. A FIXED in_shape (width, height) → every frame goes through the
        # same network input size → deterministic boxes.
        try:
            from deface.centerface import CenterFace
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ModelUnavailableError(
                "CenterFace fallback needs the optional extra: "
                "pip install 'redactcam[centerface]'"
            ) from exc

        self._net = CenterFace(in_shape=infer_shape)

    def detect(self, frame_bgr: np.ndarray, conf: float) -> list[Box]:
        """Face boxes (x,y,w,h) in native pixels (CenterFace rescales for us)."""
        dets, _ = self._net(frame_bgr, threshold=conf)
        h, w = frame_bgr.shape[:2]
        boxes = (_xyxy_to_box(x1, y1, x2, y2, w, h) for x1, y1, x2, y2, _s in dets)
        return [b for b in boxes if b is not None]


def _session_options(intra_threads: int):
    """ONNX Runtime options for a detector session. ``intra_threads`` caps the
    intra-op pool: ``0`` lets ORT use every core (the default — the YOLO forward
    is the whole pipeline's bottleneck, measured at ~3.8 s/frame single-threaded
    against ~1.1 s on all cores for 4K tiled inference), ``1`` restores
    bit-reproducible single-thread mode. Multi-threaded float reductions reorder,
    so boxes can shift sub-pixel run to run — irrelevant for a privacy mask,
    where the box still covers the object, but set 1 if you need byte-identical
    output across runs."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def _providers(available: list[str]) -> list[str]:
    """Prefer CUDA when the ``onnxruntime-gpu`` build exposes it (the YOLO forward
    measured ~19× faster on GPU — 17-21 ms against ~400 ms per tile on a mid-range
    consumer card), always keeping CPU as a fallback so a CPU-only host still
    works. The stock CPU build never lists ``CUDAExecutionProvider``, so this is a
    no-op until you install ``onnxruntime-gpu``. GPU float reductions are
    non-deterministic — fine for a privacy mask, where boxes shift sub-pixel."""
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class YoloDetector:
    """Single-class YOLO detector on an onnxruntime session — CUDA when the GPU
    build is installed, else multi-thread CPU (see ``_providers`` /
    ``_session_options``). Serves both the face and plate models:
    letterboxes to a fixed square ``infer_size``; decodes the raw YOLO output,
    applies NMS, then a ``min_aspect``/``max_aspect`` width/height band; boxes
    rescaled to native pixels. Faces pass ``min_aspect=max_aspect=0`` (no gate).

    For plates the band drops both squarish road signs (the model fires hard on
    rectangular signage — a place-name sign scored 0.6, above any usable
    confidence) AND over-elongated structures (a distant shoreline or bridge
    railing boxed at roughly 13:1 also outscores confidence), while a real plate
    is long but bounded (measured ≈2.4-3:1 at an angle, ≤4.7:1 head-on). Aspect
    ratio — not confidence — is what separates them, on both sides."""

    def __init__(
        self,
        model_path: Path,
        infer_size: int,
        iou: float,
        min_aspect: float,
        intra_threads: int = 0,
        max_aspect: float = 0.0,
        class_filter: frozenset[int] | None = None,
    ):
        import onnxruntime as ort

        # The GPU build ships CUDA/cuDNN as pip wheels; preload them so ORT finds
        # the shared objects without a manual LD_LIBRARY_PATH (no-op on CPU build).
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        self._sess = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(intra_threads),
            providers=_providers(ort.get_available_providers()),
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._infer_size = infer_size
        self._iou = iou
        self._min_aspect = min_aspect
        self._max_aspect = max_aspect
        # None → single-class model (face/plate, score = column 4). A set → COCO
        # multi-class model: keep only these class ids (the vehicle classes), score
        # = max class prob, no aspect gate.
        self._class_filter = class_filter

    def detect(self, frame_bgr: np.ndarray, conf: float) -> list[Box]:
        """Plate boxes (x,y,w,h) in native pixels. Decodes a YOLOv11 single-class
        output ``(1, 5, N)`` (or its ``(1, N, 5)`` transpose): each anchor is
        ``cx, cy, w, h, score`` in letterbox space; filter by ``conf``, convert to
        xyxy, un-letterbox to native, NMS, then drop boxes outside the
        ``[min_aspect, max_aspect]`` width/height band (squarish road signs below,
        elongated shoreline/railing structures above). Deliberately forgiving — an
        unrecognised output tensor yields no boxes rather than a crash, so one
        malformed model does not take down the whole run."""
        h0, w0 = frame_bgr.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(frame_bgr, self._infer_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]  # NCHW

        try:
            outputs = self._sess.run(None, {self._input_name: blob})
        except Exception as exc:  # pragma: no cover - model-shape dependent
            logger.warning("YOLO inference failed: %s — skipping this frame", exc)
            return []

        preds = np.asarray(outputs[0])
        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            return []
        # YOLOv11 emits (4+nc, N); transpose the channels-first layout to rows.
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        if preds.shape[1] < 5:
            return []

        # Single-class (face/plate): score is column 4. Multi-class (COCO): score is
        # the max class prob, the class is its argmax — kept only if in class_filter.
        nc = preds.shape[1] - 4
        if nc == 1:
            row_score = preds[:, 4]
            row_cls = None
        else:
            cls_block = preds[:, 4:]
            row_cls = cls_block.argmax(1)
            row_score = cls_block[np.arange(cls_block.shape[0]), row_cls]

        boxes: list[list[float]] = []
        scores: list[float] = []
        for i in np.where(row_score >= conf)[0]:
            if (
                row_cls is not None
                and self._class_filter is not None
                and int(row_cls[i]) not in self._class_filter
            ):
                continue
            cx, cy, bw, bh = (float(v) for v in preds[i, :4])
            boxes.append(
                [
                    (cx - bw / 2 - pad_x) / scale,
                    (cy - bh / 2 - pad_y) / scale,
                    bw / scale,
                    bh / scale,
                ]
            )
            scores.append(float(row_score[i]))
        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, scores, conf, self._iou)
        result: list[Box] = []
        for i in np.asarray(keep).reshape(-1):
            x, y, bw, bh = boxes[int(i)]
            # Aspect band on the model's box (pre-clamp): a real plate is wide
            # and thin but bounded. A squarish box (below min) is a road sign the
            # model misfires on; a very-elongated box (above max) is a shoreline,
            # bridge railing or road edge — one such band measured ≈13:1 and
            # outscored the confidence threshold, so aspect is what removes it.
            if bh <= 0 or bw < self._min_aspect * bh:
                continue
            if self._max_aspect and bw > self._max_aspect * bh:
                continue
            box = _xyxy_to_box(x, y, x + bw, y + bh, w0, h0)
            if box is not None:
                result.append(box)
        return result


def build_detectors(
    plate_model_path: str | Path | None,
    face_model_path: str | Path | None,
    face_infer_shape: tuple[int, int],
    infer_size: int,
    iou: float,
    plate_min_aspect: float,
    plate_max_aspect: float = 0.0,
    threads: int = 0,
) -> tuple[object, YoloDetector | None]:
    """Construct the face detector (always) and the plate detector (only if its
    model exists).

    The face detector is the YOLO face model when resolvable, else the CenterFace
    fallback; if neither is available this raises ``ModelUnavailableError`` rather
    than returning something that finds nothing. A missing PLATE model only warns
    and disables plates — the caller decides whether that is fatal for its use
    case. ``threads`` caps each ONNX session's intra-op pool (0 = all cores)."""
    face_detector = build_face_detector(
        face_model_path, face_infer_shape, infer_size, iou, threads
    )
    plate_detector = build_plate_detector(
        plate_model_path, infer_size, iou, plate_min_aspect, plate_max_aspect, threads
    )
    return face_detector, plate_detector


def build_plate_detector(
    plate_model_path: str | Path | None,
    infer_size: int,
    iou: float,
    min_aspect: float,
    max_aspect: float = 0.0,
    threads: int = 0,
) -> YoloDetector | None:
    """Single-class YOLO plate detector, or None when the model is unresolvable.

    A missing plate model warns and disables plates rather than raising — unlike
    faces, where a silent no-op is unacceptable. Callers that consider plates
    mandatory should check for None themselves."""
    plate_path = Path(plate_model_path).expanduser() if plate_model_path else None
    if plate_path and plate_path.exists():
        logger.info("plate model loaded from %s", plate_path)
        return YoloDetector(
            plate_path, infer_size, iou, min_aspect, threads, max_aspect=max_aspect
        )
    logger.warning(
        "no plate model at %r — plate detection disabled",
        str(plate_path) if plate_path else None,
    )
    return None


def build_vehicle_detector(
    vehicle_model_path: str | Path | None,
    infer_size: int,
    iou: float,
    threads: int,
) -> YoloDetector | None:
    """COCO vehicle detector (car/bus/truck/motorcycle) for the occupant cabin blur,
    or None when the model is unresolvable (cabin blur disabled — WARN, never fatal).
    Shares the YOLO/onnxruntime path; no aspect gate, class-filtered to the vehicle
    ids. Cars are large and high-confidence, so it runs whole-frame (no tiling)."""
    vp = Path(vehicle_model_path).expanduser() if vehicle_model_path else None
    if vp and vp.exists():
        logger.info("vehicle model loaded from %s", vp)
        return YoloDetector(vp, infer_size, iou, 0.0, threads, class_filter=VEHICLE_CLASSES)
    logger.warning(
        "no vehicle model at %r — occupant cabin blur disabled",
        str(vp) if vp else None,
    )
    return None


def build_person_detector(
    person_model_path: str | Path | None,
    infer_size: int,
    iou: float,
    threads: int,
) -> YoloDetector | None:
    """COCO person detector (class 0) for the WHOLE-person blur, or None when the model
    is unresolvable (person blur disabled — WARN, never fatal). Shares the YOLO path;
    no aspect gate, class-filtered to the person id. People are large/high-confidence, so
    it runs whole-frame (no tiling), like the vehicle detector."""
    pp = Path(person_model_path).expanduser() if person_model_path else None
    if pp and pp.exists():
        logger.info("person model loaded from %s", pp)
        return YoloDetector(pp, infer_size, iou, 0.0, threads, class_filter=PERSON_CLASSES)
    logger.warning(
        "no person model at %r — whole-person blur disabled",
        str(pp) if pp else None,
    )
    return None


def build_face_detector(
    face_model_path: str | Path | None,
    face_infer_shape: tuple[int, int],
    infer_size: int,
    iou: float,
    threads: int,
):
    """YOLO face detector (single class, no aspect gate) when its model resolves,
    else the optional CenterFace fallback.

    The fallback is measurably worse on outdoor footage — CenterFace scored
    foliage above real faces at usable thresholds — so a loud warning marks the
    degraded path instead of letting it pass unnoticed. If the fallback is not
    installed either, ``ModelUnavailableError`` is raised: a face detector that
    silently finds nothing is the one failure mode this library must not have."""
    fp = Path(face_model_path).expanduser() if face_model_path else None
    if fp and fp.exists():
        logger.info("face model loaded from %s", fp)
        return YoloDetector(fp, infer_size, iou, 0.0, threads)
    logger.warning(
        "no face model at %r — falling back to CenterFace (degraded recall and "
        "precision; point face_model_path at a YOLO face ONNX to fix)",
        str(fp) if fp else None,
    )
    return CenterFaceDetector(face_infer_shape)


def detect_plates(
    frame: np.ndarray,
    plate_detector,
    conf: float,
    tiles,
    plate_max_area_frac: float = 0.0,
) -> list[Box]:
    """Tiled plate detection on one frame + the oversized-plate area gate. A plate
    box covering more than ``plate_max_area_frac`` of the FULL frame is dropped:
    a real plate is a small fraction of a 4K frame (≤0.4% on real footage) while
    the model's structure FPs (a stone wall / verge / bridge abutment) span
    1.4-7.6% — size is the separable axis the aspect band can't see (those FPs box
    at a plausible 2.0-2.7:1). 0 disables. Split out so a boost frame (adaptive
    plate cadence) can re-detect plates without re-running face detection."""
    if not plate_detector:
        return []
    plates = detect_tiled(frame, plate_detector, conf, tiles)
    if plate_max_area_frac > 0 and plates:
        limit = plate_max_area_frac * frame.shape[0] * frame.shape[1]
        plates = [b for b in plates if b[2] * b[3] <= limit]
    return plates


def detect_frame(
    frame: np.ndarray,
    face_detector,
    plate_detector,
    face_conf: float,
    plate_conf: float,
    tiles,
    plate_max_area_frac: float = 0.0,
) -> tuple[list[Box], list[Box]]:
    """Tiled face ∪ plate detection on one native-resolution BGR frame. Faces and
    plates take INDEPENDENT confidence thresholds: a small clear pedestrian face
    can score well below a plate's usable conf (a 16 px face scored 0.19 where the
    plate band needs 0.4), so face recall is pushed without lowering the plate
    threshold (which would resurface squarish road-sign FPs). Faces are never
    area-gated."""
    faces = detect_tiled(frame, face_detector, face_conf, tiles)
    plates = detect_plates(frame, plate_detector, plate_conf, tiles, plate_max_area_frac)
    return faces, plates


def build_image_detectors(**overrides):
    """Reusable ``(face_detector, plate_detector)`` handle for ``detect_image``. ONNX
    construction costs far more than one still's inference, so a caller blurring a BATCH
    builds once here and reuses it; ``overrides`` take ``build_detectors`` keyword names."""
    opts = {"plate_model_path": None, "face_model_path": None, "face_infer_shape": (640, 360)}
    opts |= {"infer_size": 640, "iou": 0.45, "plate_min_aspect": 2.0, "plate_max_aspect": 6.0}
    return build_detectors(**{**opts, "threads": 0, **overrides})


def detect_image(
    image: np.ndarray,
    *,
    detectors=None,
    conf: float = 0.4,
    tile_px: int = 1280,
    tile_overlap: float = 0.15,
    plate_max_area_frac: float = 0.01,
    **detector_kwargs,
) -> FrameRegions:
    """Detect faces ∪ plates in ONE in-memory ``HxWx3`` **RGB** frame — the
    single-image sibling of ``detect_regions``: same tiled detection and the same
    detectors, but no ``VideoCapture``, so a photo goes through exactly the same
    detector the video path uses. ``detectors`` is a ``build_image_detectors``
    handle (build once, reuse across a batch); omit it and one is built per call,
    with any extra keyword arguments forwarded to it."""
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        raise ValueError("detect_image expects an HxWx3 RGB frame")
    if detectors is None:
        detectors = build_image_detectors(**detector_kwargs)
    # Detectors run on OpenCV's BGR order, so an RGB still must be converted.
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    tiles = frame_tiles(frame.shape[1], frame.shape[0], tile_px, tile_overlap)
    faces, plates = detect_frame(
        frame, detectors[0], detectors[1], conf, conf, tiles, plate_max_area_frac
    )
    return FrameRegions(0, 0.0, boxes=faces + plates, faces=faces, plates=plates)


def detect_regions(
    video_path: str | Path,
    sample_fps: float = 2.0,
    conf: float = 0.4,
    plate_model_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    face_infer_shape: tuple[int, int] = (640, 360),
    plate_infer_size: int = 640,
    plate_iou: float = 0.45,
    tile_px: int = 1280,
    tile_overlap: float = 0.15,
    plate_min_aspect: float = 2.0,
    plate_max_aspect: float = 6.0,
    plate_max_area_frac: float = 0.01,
    plate_threads: int = 0,
) -> list[FrameRegions]:
    """Detect face ∪ plate regions across a video, one ``FrameRegions`` per
    sampled frame (empty ``boxes`` is normal for clean frames).

    ``sample_fps`` is a fixed cadence; ``conf`` is shared by both detectors;
    ``plate_model_path`` None/missing → faces only + WARN; ``face_model_path``
    None/missing → CenterFace fallback. ``face_infer_shape`` (w, h) fixes the
    CenterFace fallback's inference resolution; ``plate_infer_size`` is the square
    YOLO letterbox edge and ``plate_iou`` the plate NMS threshold. ``tile_px`` /
    ``tile_overlap`` drive tiled inference so small/distant faces and plates in a
    4K frame are detected at native scale instead of vanishing in a whole-frame
    downscale (``tile_px`` ≥ the frame → no tiling). ``plate_min_aspect`` /
    ``plate_max_aspect`` bound the plate width/height ratio, dropping squarish
    road signs below and elongated shoreline/railing structures above (both
    outscore conf; aspect separates them); ``plate_max_area_frac`` additionally
    drops oversized plate boxes (large stone walls/verges the model misfires on at
    a plausible aspect — a real plate is ≤0.4% of a 4K frame). ``plate_threads`` caps the plate
    session's intra-op pool (0 = all cores;
    1 = bit-reproducible single-thread — see ``_session_options``).
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    face_detector, plate_detector = build_detectors(
        plate_model_path,
        face_model_path,
        face_infer_shape,
        plate_infer_size,
        plate_iou,
        plate_min_aspect,
        plate_max_aspect,
        plate_threads,
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # Native frames to advance between samples; fall back to every frame when
    # fps is zero/garbage (some containers report 0).
    step = max(1, round(fps / sample_fps)) if fps > 0 and sample_fps > 0 else 1

    results: list[FrameRegions] = []
    frame_idx = 0
    tiles: list[tuple[int, int, int, int]] | None = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                if tiles is None:  # frame size is constant — derive the grid once
                    tiles = frame_tiles(frame.shape[1], frame.shape[0], tile_px, tile_overlap)
                faces, plates = detect_frame(
                    frame, face_detector, plate_detector, conf, conf, tiles, plate_max_area_frac
                )
                results.append(
                    FrameRegions(
                        frame_idx=frame_idx,
                        timestamp=(frame_idx / fps) if fps > 0 else float(frame_idx),
                        boxes=faces + plates,
                        faces=faces,
                        plates=plates,
                    )
                )
            frame_idx += 1
    finally:
        cap.release()

    return results


def next_boost_run(
    boost_run: int,
    plate_speed: float,
    threshold: int,
    max_run: int,
    plate_seen: bool = False,
) -> int:
    """Adaptive-plate-cadence counter for the next frame: decrement the current run,
    then re-arm to ``max_run`` on either trigger (when the boost is enabled,
    ``threshold`` > 0): a plate track is fast (``plate_speed`` ≥ ``threshold``), OR a
    plate was just DETECTED (``plate_seen``). The speed trigger covers a big plate
    accelerating through a close pass; the recently-seen trigger covers a SMALL plate
    that won't hold optical-flow features between samples — it retires with no surviving
    track for the speed gate to read, so detection itself re-arms per-frame re-detection
    through the rest of its pass. Self-clears ``max_run`` frames after the last trigger.
    ``threshold`` ≤ 0 disables the boost (counter only decays)."""
    nxt = max(0, boost_run - 1)
    if threshold > 0 and (plate_speed >= threshold or plate_seen):
        nxt = max_run
    return nxt


def detect_and_track(
    video_path: str | Path,
    *,
    sample_fps: float = 8.0,
    conf: float = 0.4,
    face_conf: float | None = None,
    face_sample_fps: float | None = None,
    face_confirm_min_detections: int = 1,
    pre_roll_frames: int = 0,
    plate_model_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    face_infer_shape: tuple[int, int] = (640, 360),
    plate_infer_size: int = 640,
    plate_iou: float = 0.45,
    tile_px: int = 1280,
    tile_overlap: float = 0.15,
    plate_min_aspect: float = 2.0,
    plate_max_aspect: float = 6.0,
    plate_max_area_frac: float = 0.01,
    plate_boost_speed_px: int = 0,
    plate_boost_max_run: int = 6,
    track_downscale: int = 2,
    track_lk_win: int = 31,
    track_lk_levels: int = 4,
    track_max_horizon_frames: int = 9,
    track_face_coast_frames: int = 20,
    track_face_coast_moving_frames: int = 6,
    track_coast_move_speed: float = 6.0,
    track_retire_edge_coast_margin: int = 3,
    track_min_points: int = 6,
    track_scale_min: float = 0.8,
    track_scale_max: float = 1.25,
    track_min_inliers: int = 4,
    track_motion_gain: float = 1.0,
    track_motion_max_px: int = 160,
    track_face_motion_gain: float = 1.6,
    track_plate_margin_floor_px: int = 16,
    track_face_margin_frac: float = 0.35,
    track_assoc_motion_gain: float = 1.2,
    track_assoc_min_gate_px: int = 72,
    track_dedup_iou: float = 0.55,
    track_cabin_frac: float = 0.0,
    vehicle_model_path: str | Path | None = None,
    vehicle_conf: float = 0.35,
    vehicle_min_height_frac: float = 0.0,
    vehicle_ignore_bottom_frac: float = 0.0,
    track_person: bool = False,
    person_model_path: str | Path | None = None,
    person_conf: float = 0.35,
    person_min_height_frac: float = 0.0,
    plate_threads: int = 0,
) -> tuple[dict[int, list[Box]], int, int, int]:
    """One decode pass producing DENSE per-frame privacy boxes.

    Reads every frame once. On sample frames (cadence ``sample_fps``) it runs the
    same tiled face ∪ plate detection as ``detect_regions``; every frame it feeds
    a ``TrackManager`` that optical-flow-propagates each box. So a blur FOLLOWS a
    moving face/plate instead of freezing at the detection position between
    samples (the static-union defect). Detection stays sparse (the expensive
    part); flow is ms/frame. ``sample_fps`` defaults to 8 so the inter-detection
    gap (≈1/8 s) stays within the measured flow horizon (~0.13 s).

    Returns ``(frame_boxes, frame_count, n_face_det, n_plate_det)``:
    ``frame_boxes`` maps every decoded frame index → its active boxes (native px,
    tight; dilation is applied downstream by the timeline). ``frame_count`` is the
    exact decoded length (drives the mask geometry). The detection counts are for
    logging. ``TrackManager`` pins cv2's RNG; the detector sessions run on all cores
    by default (``plate_threads``) — the YOLO forward dominates the run time, and
    bit-exact reproducibility isn't needed for a privacy mask. ``track_*`` mirror
    ``TrackManager`` params; the defaults here are the values validated on real
    forward-facing road footage (4K at downscale 2 needs a bigger LK window and a
    deeper pyramid than ``TrackManager``'s own smaller-frame defaults — see
    ``redactcam.presets``). ``track_face_coast_frames`` (> ``max_horizon``) lets
    a face hold its box across a detection gap so the blur doesn't blink, while
    plates still retire fast — the hold is velocity-PREDICTIVE, extrapolating
    along the face's last flow heading so a fast face's blur follows it instead
    of leaving stale ghosts on bare road. ``track_face_coast_moving_frames`` (<
    ``track_face_coast_frames``) trims that hold for a MOVING coast (speed ≥
    ``track_coast_move_speed``): a velocity-predictive coast extrapolating onto new
    area is far more likely ghosting past an object that left frame (a passing car's
    occupant, a pedestrian walking out) than a static hold of a present-but-missed
    face, so it retires sooner — a mid-traverse gap is interpolation's job, so only
    the dead tail after the last detection is trimmed (gauged by the box's actual
    per-frame displacement, since a coast whose flow latched onto a fast-passing car
    body has a near-zero heading yet moves hundreds of px/frame).
    ``track_retire_edge_coast_margin`` retires a face pinned against a frame edge once
    its last detection is >1 frame stale — the exit-ghost a pedestrian/occupant leaves
    when it walks/drives off-frame and its box re-seeds flow points on the static
    background it parks over (so a points-based test would miss it). ``track_dedup_iou``
    retires the weaker of two same-class tracks overlapping by more than it (a
    fast/sparse object that spawned a second track the association gate didn't merge —
    doubled blur), a high threshold so two distinct crossing faces aren't merged. ``track_motion_gain``/
    ``track_motion_max_px`` grow each emitted box by a margin scaled to its own
    per-frame flow speed (capped), covering a fast plate's flow lag without
    ballooning a large near-static FP; ``track_plate_margin_floor_px`` is a flat
    minimum margin for plates so a slow/distant plate (speed margin ≈0) stays
    covered. ``track_face_margin_frac`` is the face analogue but SIZE-PROPORTIONAL:
    the speed margin collapses to the bare frontal-detector box when a face's flow
    speed drops to ≈0 (a pedestrian exiting frame loses their flow features), exposing
    the turned cheek/profile the tight box misses, so a proportional floor keeps the
    blur on the whole head without ballooning a tiny distant FP. ``track_face_motion_gain``
    overrides ``track_motion_gain`` for faces (default higher): a fast face's flow box
    lags its true position between samples and its blur trails to the right of a fast
    leftward subject / collapses as it exits frame, so a larger speed-scaled margin
    keeps the blur over the leading edge and the trailing exit. ``track_assoc_motion_gain``/``track_assoc_min_gate_px`` widen
    detection→track association by a center-distance gate when IoU fails across
    fast small motion (the gain scales it with face size; the flat floor covers a
    tiny distant face that still moves tens of px between sparse detections), so
    one fast face stays one track instead of spawning duplicate ghosts.

    ``face_conf`` (default = ``conf``) thresholds the FACE detector independently of
    the plate ``conf``: a small clear pedestrian face can score well below a plate's
    usable threshold, so face recall is pushed (catching a 16 px face at 0.19) without
    lowering the plate conf and resurfacing squarish road-sign FPs.

    ``plate_boost_speed_px`` enables adaptive plate cadence (0 disables): while a plate
    track's per-frame flow speed is ≥ this, plates are re-detected EVERY frame (not just
    the sparse ``sample_fps`` cadence), so a plate accelerating through a close pass —
    which moves farther per frame than flow can follow, leaving the box lagging and the
    plate readable — gets its box snapped to ground truth each frame. ``plate_boost_max_run``
    keeps the boost on that many frames past the last fast reading (bridging a momentary
    dip); the boost self-clears when the plate retires. The cost is bounded to the few
    close-pass frames, not the whole clip (plate detection is the pipeline bottleneck).
    The dense, snapped plate track is also what the cabin blur rides.

    ``track_cabin_frac`` (0 = off) drives the occupant CABIN blur off a COCO
    ``vehicle_model_path`` (car/bus/truck/motorcycle, conf ``vehicle_conf``). A vehicle's
    occupants sit behind a moving windshield where per-face detection is unreliable (the
    model latches onto the wall/roof/glass reflections, not the face) and a plate-anchored
    box mis-locates the cabin as the car passes (the windshield leans back off the front
    plate). So the vehicle detector boxes the whole car (tracked like any other class), and
    ``TrackManager`` emits the TOP ``track_cabin_frac`` of that box — where occupants sit at
    EVERY angle (windshield head-on, side windows on the pass). The vehicle detector is
    built only when ``track_cabin_frac`` > 0 and the model resolves; cars are large so it
    runs whole-frame (no tiling) on the base cadence. See ``TrackManager``.

    ``face_sample_fps`` (default = ``sample_fps``) gives faces their OWN, denser cadence:
    a small low-texture pedestrian face can't be optical-flow-tracked between sparse
    samples (its box freezes while it walks out of frame), so detecting it more often
    snaps the box to its real position; faces are cheap relative to plates, so the denser
    cadence is affordable. ``face_confirm_min_detections`` is the temporal-confirmation
    threshold: a face track is emitted only once it has accumulated this many associated
    detections (default 1 = no confirmation). A real face re-detects across consecutive
    samples as it persists → many detections → kept (RETROACTIVELY, including its low-conf
    discovery frames); a foliage/shadow/wall FP fires once then coasts → one detection →
    dropped. This is what lets ``face_conf`` chase a 0.19 real face without blurring the
    0.5-0.7 static-texture FPs that share its score range — confidence alone can't separate
    them, but persistence can. Plates are always emitted (recall-critical, aspect/area-gated).

    ``pre_roll_frames`` covers the HEAD a confirmed object is present-but-undetected: it extrapolates
    a track's box backward along its early velocity for up to this many frames before the first
    detection (a small/distant oncoming plate is readable, and a pedestrian present, a few frames
    before the detector first scores them). Emission also runs anchor interpolation — a track's
    between-detection gaps are filled with the straight line between bracketing detections, since flow
    freezes a small low-texture object between samples but two detections pin the true endpoints.
    """
    from .track import TrackManager

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    face_detector, plate_detector = build_detectors(
        plate_model_path,
        face_model_path,
        face_infer_shape,
        plate_infer_size,
        plate_iou,
        plate_min_aspect,
        plate_max_aspect,
        plate_threads,
    )
    # Vehicle detector is built only when the cabin blur is on (cabin_frac > 0) AND a
    # model resolves; cars are large so it runs whole-frame (no tiling).
    vehicle_detector = (
        build_vehicle_detector(vehicle_model_path, plate_infer_size, plate_iou, plate_threads)
        if track_cabin_frac > 0
        else None
    )
    # Whole-person detector is built only when person blur is on AND a model resolves;
    # people are large/high-confidence, so it runs whole-frame (no tiling) like vehicles.
    person_detector = (
        build_person_detector(person_model_path, plate_infer_size, plate_iou, plate_threads)
        if track_person
        else None
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, round(fps / sample_fps)) if fps > 0 and sample_fps > 0 else 1
    eff_face_conf = conf if face_conf is None else face_conf
    eff_face_fps = sample_fps if face_sample_fps is None else face_sample_fps
    face_step = max(1, round(fps / eff_face_fps)) if fps > 0 and eff_face_fps > 0 else 1

    tracker: TrackManager | None = None
    tiles: list[tuple[int, int, int, int]] | None = None
    n_face = n_plate = 0
    frame_idx = 0
    # Adaptive plate cadence: between sparse samples a plate accelerating through a
    # close pass moves farther per frame than flow can follow, so the box lags and
    # the plate escapes (readable). While a plate track is fast, force plate-only
    # re-detection every frame so the box snaps to ground truth; ``boost_run`` keeps
    # it on a few frames past the last fast reading to bridge a momentary dip and
    # self-clears when the plate retires (``max_speed`` → 0).
    boost_run = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if tracker is None:  # geometry is constant — build the grid + tracker once
                h, w = frame.shape[:2]
                tiles = frame_tiles(w, h, tile_px, tile_overlap)
                tracker = TrackManager(
                    w,
                    h,
                    downscale=track_downscale,
                    lk_win=track_lk_win,
                    lk_levels=track_lk_levels,
                    max_horizon_frames=track_max_horizon_frames,
                    face_coast_frames=track_face_coast_frames,
                    face_coast_moving_frames=track_face_coast_moving_frames,
                    coast_move_speed=track_coast_move_speed,
                    retire_edge_coast_margin=track_retire_edge_coast_margin,
                    face_sample_interval_frames=face_step,
                    min_points=track_min_points,
                    scale_min=track_scale_min,
                    scale_max=track_scale_max,
                    min_inliers=track_min_inliers,
                    motion_gain=track_motion_gain,
                    motion_max=track_motion_max_px,
                    face_motion_gain=track_face_motion_gain,
                    plate_margin_floor=track_plate_margin_floor_px,
                    face_margin_frac=track_face_margin_frac,
                    assoc_motion_gain=track_assoc_motion_gain,
                    assoc_min_gate=track_assoc_min_gate_px,
                    dedup_iou=track_dedup_iou,
                    cabin_frac=track_cabin_frac,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Faces and plates run on INDEPENDENT cadences: faces at ``face_step``
            # (denser, so a small face is snapped often enough to track its walk;
            # they're cheap), plates at the base ``step`` plus the speed-gated boost
            # (re-detect a fast oncoming plate every frame so its box snaps to the
            # readable plate through a close pass). Vehicle OCCUPANTS are NOT chased
            # by face detection — a face through a moving windshield is unreliable
            # (the model latches onto the wall/roof/reflections, never the actual
            # face); occupants are covered by the plate-derived cabin blur instead
            # (see ``TrackManager`` cabin_* / ``confirmed_frame_boxes``).
            do_face = frame_idx % face_step == 0
            do_plate = frame_idx % step == 0 or boost_run > 0
            # Vehicles drive the occupant cabin blur. They're large/high-confidence,
            # so detect whole-frame (no tiling). Honor the BOOST: a fast close-passing
            # car moves farther per frame than optical flow can follow, so flow drifts
            # the box onto the road and the cabin lands off the windshield — so when the
            # boost is armed (the plate is fast = the car is close) re-detect the vehicle
            # EVERY frame, snapping the box to ground truth instead of flow-tracking it.
            do_vehicle = vehicle_detector is not None and (frame_idx % step == 0 or boost_run > 0)
            # People are detected on the base cadence too (a standing operator is slow, so
            # optical flow bridges the gaps); the whole box is blurred, not just the head.
            do_person = person_detector is not None and (frame_idx % step == 0 or boost_run > 0)
            plates: list[Box] = []
            vehicles: list[Box] = []
            persons: list[Box] = []
            if do_face or do_plate or do_vehicle or do_person:
                faces = detect_tiled(frame, face_detector, eff_face_conf, tiles) if do_face else []
                if do_plate:
                    plates = detect_plates(frame, plate_detector, conf, tiles, plate_max_area_frac)
                if do_vehicle:
                    vehicles = filter_vehicles(
                        vehicle_detector.detect(frame, vehicle_conf),
                        frame.shape[0],
                        vehicle_min_height_frac,
                        vehicle_ignore_bottom_frac,
                    )
                if do_person:
                    # Height gate only (drop tiny distant dots); a person can legitimately
                    # sit in the bottom strip, so no ignore-bottom exclusion.
                    persons = filter_vehicles(
                        person_detector.detect(frame, person_conf),
                        frame.shape[0],
                        person_min_height_frac,
                        0.0,
                    )
                n_face += len(faces)
                n_plate += len(plates)
                dets = (
                    [("face", b) for b in faces]
                    + [("plate", b) for b in plates]
                    + [("vehicle", b) for b in vehicles]
                    + [("person", b) for b in persons]
                )
                tracker.step(frame_idx, gray, dets)
            else:
                tracker.step(frame_idx, gray, None)
            # Boost (per-frame re-detection) arms on a fast OR just-seen plate/vehicle.
            # A worth-blurring vehicle present arms it so the cabin is snapped every
            # frame of the pass — not just the close frames a fast plate covers — since
            # flow drifts a fast-scaling car box onto the road between sparse samples.
            boost_run = next_boost_run(
                boost_run,
                max(tracker.max_speed("plate"), tracker.max_speed("vehicle")),
                plate_boost_speed_px,
                plate_boost_max_run,
                plate_seen=len(plates) > 0 or len(vehicles) > 0,
            )
            frame_idx += 1
    finally:
        cap.release()

    # Temporal confirmation drops unconfirmed FACE tracks (one-detection foliage/
    # shadow flickers) while keeping every plate; interpolation fills each track's
    # between-detection gaps with a straight line (covers a small object flow can't
    # track between samples). ``face_confirm_min_detections=1`` + no interpolate is a
    # no-op (== frame_boxes).
    frame_boxes = (
        tracker.confirmed_frame_boxes(
            face_confirm_min_detections, interpolate=True, pre_roll=pre_roll_frames
        )
        if tracker is not None
        else {}
    )
    return frame_boxes, frame_idx, n_face, n_plate
