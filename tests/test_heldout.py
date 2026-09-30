"""Held-out evaluation: predeclared cases, per-class misses, disclosed limits.

Everything here is synthetic boxes — the module scores labelled boxes against
predicted boxes and makes no claim about real footage.
"""

import pytest

from redactcam import heldout as ho
from redactcam.coverage import Status


def _manifest():
    return ho.HeldOutManifest(
        criteria="A reviewer marks a face/plate/person 'missed' if any identifying part is unblurred.",
        cases=(
            ho.HeldOutCase("dusk-1", {"face": [(0, 0, 10, 10), (50, 50, 10, 10)], "plate": [(20, 20, 30, 8)]}),
            ho.HeldOutCase("rain-2", {"face": [(5, 5, 10, 10)]}),
        ),
    )


def test_digest_is_stable_and_changes_with_the_labels():
    a = _manifest()
    assert a.digest() == _manifest().digest()
    b = ho.HeldOutManifest(a.criteria, a.cases[:1])
    assert a.digest() != b.digest()


def test_evaluation_requires_the_predeclared_digest():
    m = _manifest()
    with pytest.raises(ho.ManifestMismatch):
        ho.evaluate(m, {}, declared_digest="0" * 64)
    ho.evaluate(m, {}, declared_digest=m.digest())


def test_per_class_misses_are_disclosed():
    m = _manifest()
    predicted = {
        "dusk-1": {"face": [(0, 0, 10, 10)], "plate": [(20, 20, 30, 8)]},  # one face missed
        "rain-2": {"face": [(5, 5, 10, 10)]},
    }
    res = ho.evaluate(m, predicted, declared_digest=m.digest())
    face = res.classes["face"]
    assert face.checked == 3 and face.misses == 1
    assert face.status is Status.FAILED
    assert res.classes["plate"].status is Status.VERIFIED
    assert res.misses["face"] == [("dusk-1", (50, 50, 10, 10))]
    assert res.manifest_digest == m.digest()
    assert res.limitations  # always disclosed


def test_class_with_no_labelled_cases_is_unchecked_never_safe():
    m = _manifest()
    res = ho.evaluate(m, {}, declared_digest=m.digest())
    assert res.classes["person"].status is Status.UNCHECKED
    assert res.classes["cabin"].status is Status.UNCHECKED
    assert not res.all_verified


def test_zero_misses_is_still_bounded_by_the_sample_not_a_guarantee():
    m = _manifest()
    predicted = {
        "dusk-1": {"face": [(0, 0, 10, 10), (50, 50, 10, 10)], "plate": [(20, 20, 30, 8)]},
        "rain-2": {"face": [(5, 5, 10, 10)]},
    }
    res = ho.evaluate(m, predicted, declared_digest=m.digest())
    assert res.classes["face"].status is Status.VERIFIED
    assert "sample" in res.classes["face"].limitation.lower()
    assert "guarantee" in " ".join(res.limitations).lower()
    assert not res.all_verified  # person and cabin were never labelled


def test_to_dict_round_trips_as_json():
    import json

    m = _manifest()
    d = json.loads(json.dumps(ho.evaluate(m, {}, declared_digest=m.digest()).to_dict()))
    assert d["manifest_digest"] == m.digest()
    assert d["classes"]["face"]["status"] == "failed"  # labelled, all missed
    assert d["classes"]["person"]["status"] == "unchecked"
