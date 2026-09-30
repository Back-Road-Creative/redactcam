"""Held-out evaluation of recall, with the labels and the review criteria fixed first.

The independent verifier in :mod:`redactcam.coverage` checks vehicle cabins only.
Whether faces, plates and pedestrians are found is a *recall* question, and the
only honest answer is a labelled sample measured per class. This module is the
scoring half of that: it does not run a detector and it does not touch footage.

Two rules are built in rather than left to discipline:

* **Predeclare.** A :class:`HeldOutManifest` (the human review criteria plus every
  labelled case) has a content digest. :func:`evaluate` refuses to score unless the
  caller passes the digest recorded *before* any model or threshold was tuned, so
  the cases cannot be quietly swapped after seeing results.
* **Unknown is not safe.** A class with no labelled items is ``UNCHECKED``. A class
  with labelled items and zero misses is ``VERIFIED`` for that sample only, and
  says so; no result here is a guarantee that every person or plate is found.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .coverage import CLASSES, ClassCoverage, Status
from .detect import Box, iou

LIMITATIONS = (
    "Scores describe only the labelled held-out sample; they do not guarantee that every "
    "face, plate or person in other footage is detected.",
    "Synthetic tests and threshold changes alone cannot establish detection recall.",
    "A miss is a labelled item with no predicted box at IoU >= the threshold; whether the "
    "blur made the subject unidentifiable is the human review criteria's call, not this score.",
)


class ManifestMismatch(ValueError):
    """The manifest scored is not the one declared before evaluation."""


@dataclass(frozen=True)
class HeldOutCase:
    """One held-out clip: its id and the human-labelled boxes per class."""

    case_id: str
    labels: dict[str, list[Box]]


@dataclass(frozen=True)
class HeldOutManifest:
    """The predeclared evaluation set: review ``criteria`` plus every case."""

    criteria: str
    cases: tuple[HeldOutCase, ...]

    def digest(self) -> str:
        """SHA256 over a canonical form; record it before any tuning."""
        canon = {
            "criteria": self.criteria,
            "cases": sorted(
                (
                    c.case_id,
                    {k: sorted(list(b) for b in v) for k, v in sorted(c.labels.items())},
                )
                for c in self.cases
            ),
        }
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


@dataclass
class HeldOutResult:
    """Per-class verdicts, the exact misses, and the standing limitations."""

    manifest_digest: str
    classes: dict[str, ClassCoverage]
    misses: dict[str, list[tuple[str, Box]]] = field(default_factory=dict)
    limitations: tuple[str, ...] = LIMITATIONS

    @property
    def all_verified(self) -> bool:
        return all(c.status is Status.VERIFIED for c in self.classes.values())

    def to_dict(self) -> dict:
        return {
            "manifest_digest": self.manifest_digest,
            "all_verified": self.all_verified,
            "classes": {n: c.to_dict() for n, c in self.classes.items()},
            "misses": {
                n: [{"case": cid, "box": list(b)} for cid, b in m] for n, m in self.misses.items()
            },
            "limitations": list(self.limitations),
        }


def _matches(label: Box, predicted: list[Box], used: set[int], thresh: float) -> bool:
    best, best_i = 0.0, -1
    for i, p in enumerate(predicted):
        if i in used:
            continue
        v = iou(label, p)
        if v >= thresh and v > best:
            best, best_i = v, i
    if best_i < 0:
        return False
    used.add(best_i)
    return True


def evaluate(
    manifest: HeldOutManifest,
    predicted: dict[str, dict[str, list[Box]]],
    *,
    declared_digest: str,
    iou_threshold: float = 0.3,
) -> HeldOutResult:
    """Score ``predicted[case_id][class]`` boxes against the manifest's labels.

    Raises :class:`ManifestMismatch` unless ``declared_digest`` equals the
    manifest's digest. Each predicted box can account for one label only."""
    digest = manifest.digest()
    if declared_digest != digest:
        raise ManifestMismatch(
            "held-out manifest does not match the predeclared digest "
            f"({digest[:12]} != {declared_digest[:12]}); cases must be fixed before tuning"
        )
    checked = dict.fromkeys(CLASSES, 0)
    missed: dict[str, list[tuple[str, Box]]] = {n: [] for n in CLASSES}
    for case in manifest.cases:
        for name, labels in case.labels.items():
            if name not in checked:
                raise ValueError(f"unknown privacy class {name!r} in case {case.case_id!r}")
            preds = list(predicted.get(case.case_id, {}).get(name, []))
            used: set[int] = set()
            for lab in labels:
                checked[name] += 1
                if not _matches(lab, preds, used, iou_threshold):
                    missed[name].append((case.case_id, lab))
    classes: dict[str, ClassCoverage] = {}
    n_cases = {n: sum(1 for c in manifest.cases if c.labels.get(n)) for n in CLASSES}
    for name in CLASSES:
        n, bad = checked[name], len(missed[name])
        if n == 0:
            classes[name] = ClassCoverage(
                name,
                Status.UNCHECKED,
                "no labelled held-out items for this class",
                "recall is unmeasured; absence of labels is not evidence of safety",
            )
        elif bad:
            classes[name] = ClassCoverage(
                name,
                Status.FAILED,
                f"{bad} of {n} labelled items in {n_cases[name]} cases had no matching detection",
                "the labelled sample only; other footage may miss more or fewer",
                checked=n,
                misses=bad,
            )
        else:
            classes[name] = ClassCoverage(
                name,
                Status.VERIFIED,
                f"all {n} labelled items in {n_cases[name]} cases were matched",
                f"this is a sample of {n} items, not a guarantee for other footage",
                checked=n,
            )
    return HeldOutResult(digest, classes, {n: m for n, m in missed.items() if m})
