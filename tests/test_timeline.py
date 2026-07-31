"""Timeline and JSON sidecar tests.

Pure box arithmetic and serialisation — no models, no video, no network. The
dense ``frame_boxes`` input is what ``detect_and_track`` produces (one entry per
source frame); this module only dilates and persists it.
"""

from redactcam import timeline as br


def test_dilate_box_clamps_to_frame():
    assert br.dilate_box((5, 5, 10, 10), 8, 100, 100) == (0, 0, 23, 23)
    assert br.dilate_box((90, 90, 10, 10), 8, 100, 100) == (82, 82, 18, 18)


def test_build_timeline_dilates_each_frame_independently():
    # Dense per-frame boxes (the object at a different spot each frame); no
    # interval/union — each frame keeps exactly its own (dilated) box.
    frame_boxes = {0: [(10, 10, 4, 4)], 1: [(14, 10, 4, 4)], 2: [(18, 10, 4, 4)]}
    tl = br.build_timeline(
        frame_boxes,
        source_hash="h",
        source_fps=10.0,
        source_frame_count=20,
        frame_width=100,
        frame_height=100,
        box_dilation_px=2,
        sample_fps=8.0,
        conf=0.4,
    )
    assert tl.frame_boxes[0] == [(8, 8, 8, 8)]  # dilated by 2 on every side
    assert tl.frame_boxes[1] == [(12, 8, 8, 8)]  # moved with the object, not frozen
    assert tl.frame_boxes[2] == [(16, 8, 8, 8)]
    assert 3 not in tl.frame_boxes  # a frame with no tracked box carries no blur


def test_timeline_empty_when_no_detections():
    tl = br.build_timeline(
        {},
        source_hash="h",
        source_fps=10.0,
        source_frame_count=20,
        frame_width=100,
        frame_height=100,
        box_dilation_px=4,
        sample_fps=8.0,
        conf=0.4,
    )
    assert tl.frame_boxes == {}
    assert tl.source_frame_count == 20


def test_sidecar_round_trip(tmp_path):
    tl = br.build_timeline(
        {0: [(10, 10, 4, 4)], 5: [(50, 50, 4, 4)]},
        source_hash="deadbeef",
        source_fps=24.0,
        source_frame_count=48,
        frame_width=640,
        frame_height=480,
        box_dilation_px=2,
        sample_fps=8.0,
        conf=0.4,
    )
    path = tmp_path / "blur.json"
    br.write_sidecar(path, tl)
    loaded = br.load_sidecar(path)
    assert loaded.source_hash == "deadbeef"
    assert loaded.source_frame_count == 48
    assert loaded.frame_boxes[0] == [(8, 8, 8, 8)]  # int key + tuple box restored
    assert loaded.frame_boxes[5] == [(48, 48, 8, 8)]


def test_load_sidecar_rejects_wrong_schema_or_missing(tmp_path):
    """A sidecar written by a different (older, or foreign) schema is refused
    rather than half-read: reusing boxes whose meaning has changed silently
    blurs the wrong pixels. A schema mismatch simply means re-detect."""
    path = tmp_path / "blur.json"
    path.write_text('{"schema": "somethingelse/v1"}')
    assert br.load_sidecar(path) is None
    assert br.load_sidecar(tmp_path / "missing.json") is None
    path.write_text("{ not json")
    assert br.load_sidecar(path) is None
