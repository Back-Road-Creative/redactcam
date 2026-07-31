"""The tuned preset must stay wired to the function it configures."""

import inspect

from redactcam.detect import detect_and_track
from redactcam.presets import ROAD_FOOTAGE


def test_every_preset_key_is_a_real_parameter():
    """A preset key that no longer matches a parameter would be silently dropped
    by ``**kwargs``, quietly reverting a tuned value to its default. Checking the
    signature makes that impossible rather than merely unlikely."""
    params = set(inspect.signature(detect_and_track).parameters)
    unknown = sorted(set(ROAD_FOOTAGE) - params)
    assert unknown == [], f"preset keys not accepted by detect_and_track: {unknown}"


def test_preset_is_accepted_by_the_signature():
    """Binding the preset proves it can actually be splatted in, not just that
    the names look right."""
    sig = inspect.signature(detect_and_track)
    sig.bind("clip.mp4", **ROAD_FOOTAGE)


def test_face_confidence_is_paired_with_confirmation():
    """The low face confidence only works because persistence filters the false
    positives that share its score range. Shipping one without the other blurs
    every hedge in the frame."""
    assert ROAD_FOOTAGE["face_conf"] < 0.4
    assert ROAD_FOOTAGE["face_confirm_min_detections"] >= 2


def test_plate_gates_bracket_real_plate_geometry():
    """Real plates measured ≈2.4-3:1 at an angle and ≤4.7:1 head-on, so the band
    must contain that range while excluding squarish signs and long railings."""
    assert ROAD_FOOTAGE["plate_min_aspect"] <= 2.4
    assert ROAD_FOOTAGE["plate_max_aspect"] >= 4.7
    assert 0 < ROAD_FOOTAGE["plate_max_area_frac"] < 0.01
