"""redactcam — tracked face, plate, occupant and person redaction for video.

Five stages, each usable on its own:

1. ``detect``   — tiled YOLO/ONNX detection of faces, plates, vehicles, people.
2. ``track``    — optical-flow propagation, so a blur follows a moving object.
3. ``timeline`` — dense per-frame boxes, dilated and serialised to JSON.
4. ``mask``     — the timeline rasterised to a lossless mask video.
5. ``coverage`` — an independent re-detection that proves the mask covers what
                  it claims to, run *before* the irreversible encode.

``pipeline.redact_video`` wires all five together for the common case.
"""

from .apply import apply_blur, blur_filter_complex, blur_radius
from .coverage import CoverageReport, Leak, cabin_region, plate_inside, verify_cabin_coverage
from .detect import (
    Box,
    CenterFaceDetector,
    FrameRegions,
    ModelUnavailableError,
    YoloDetector,
    build_detectors,
    build_face_detector,
    build_image_detectors,
    build_person_detector,
    build_plate_detector,
    build_vehicle_detector,
    detect_and_track,
    detect_image,
    detect_regions,
    iou,
)
from .mask import active_boxes_at, materialize_mask_video, render_mask_frame
from .models import (
    DEFAULT_MODELS,
    FACE_MODEL,
    PERSON_MODEL,
    PLATE_MODEL,
    VEHICLE_MODEL,
    ModelSpec,
    resolve_model,
)
from .pipeline import CoverageError, RedactionResult, file_hash, redact_video
from .presets import (
    ROAD_FOOTAGE,
    ROAD_FOOTAGE_BOX_DILATION_PX,
    ROAD_FOOTAGE_FEATHER_PX,
)
from .timeline import (
    BlurTimeline,
    build_timeline,
    dilate_box,
    load_sidecar,
    probe_video,
    write_sidecar,
)
from .track import Track, TrackManager

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # high level
    "redact_video",
    "RedactionResult",
    "CoverageError",
    "file_hash",
    "ROAD_FOOTAGE",
    "ROAD_FOOTAGE_BOX_DILATION_PX",
    "ROAD_FOOTAGE_FEATHER_PX",
    # detection
    "Box",
    "iou",
    "FrameRegions",
    "YoloDetector",
    "CenterFaceDetector",
    "ModelUnavailableError",
    "detect_regions",
    "detect_and_track",
    "detect_image",
    "build_detectors",
    "build_image_detectors",
    "build_face_detector",
    "build_plate_detector",
    "build_vehicle_detector",
    "build_person_detector",
    # tracking
    "Track",
    "TrackManager",
    # timeline
    "BlurTimeline",
    "build_timeline",
    "dilate_box",
    "probe_video",
    "write_sidecar",
    "load_sidecar",
    # mask
    "render_mask_frame",
    "active_boxes_at",
    "materialize_mask_video",
    # apply
    "blur_radius",
    "blur_filter_complex",
    "apply_blur",
    # coverage
    "CoverageReport",
    "Leak",
    "cabin_region",
    "plate_inside",
    "verify_cabin_coverage",
    # models
    "ModelSpec",
    "resolve_model",
    "DEFAULT_MODELS",
    "FACE_MODEL",
    "PLATE_MODEL",
    "VEHICLE_MODEL",
    "PERSON_MODEL",
]
