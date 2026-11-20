"""The box-region blur never runs on a crop small enough to hit OpenCV's
multithreaded GaussianBlur race (4.13: a 5x5 kernel on 6-17 row images returns
pixels below the true value), and odd boxes never drop out of the mask.

These tests deliberately do NOT pin cv2 to one thread: production does not."""

import cv2
import numpy as np
import pytest

from redactcam import mask as bm


def _reference(width, height, boxes, feather):
    """The old full-frame render, single-threaded (deterministic)."""
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        frame = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, thickness=-1)
        k = feather * 2 + 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    finally:
        cv2.setNumThreads(old)


def test_crop_is_never_smaller_than_the_minimum_when_the_frame_allows(monkeypatch):
    shapes = []
    real = cv2.GaussianBlur

    def spy(src, ksize, sigma):
        shapes.append(src.shape)
        return real(src, ksize, sigma)

    monkeypatch.setattr(bm.cv2, "GaussianBlur", spy)
    bm.render_mask_frame(500, 1080, [(0, 0, 499, 8)], 2)
    bm.render_mask_frame(1920, 1080, [(900, 500, 3, 3)], 2)
    bm.render_mask_frame(20, 10, [(2, 2, 3, 3)], 2)  # frame smaller than the floor
    assert shapes[0][0] >= bm.MIN_BLUR_CROP and shapes[0][1] >= bm.MIN_BLUR_CROP
    assert shapes[1][0] >= bm.MIN_BLUR_CROP and shapes[1][1] >= bm.MIN_BLUR_CROP
    assert shapes[2] == (10, 20)  # whole frame, same as the old full-frame path


@pytest.mark.parametrize("boxes", [[(0, 0, 499, 8)], [(100, 0, 300, 6)], [(0, 1070, 499, 9)]])
def test_thin_edge_boxes_never_lose_mask_pixels_at_default_threads(boxes):
    cv2.setNumThreads(max(4, cv2.getNumThreads()))
    ref = _reference(500, 1080, boxes, 2)
    for _ in range(150):
        got = bm.render_mask_frame(500, 1080, boxes, 2)
        assert (got >= ref).all()
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("box", [(60, 20, -30, 20), (20, 60, 20, -40), (60, 60, -30, -30)])
@pytest.mark.parametrize("feather", [0, 2, 8])
def test_negative_extent_boxes_match_the_full_frame_render(box, feather):
    boxes = [box, (150, 20, 10, 10)]
    ref = _reference(200, 120, boxes, feather) if feather else None
    got = bm.render_mask_frame(200, 120, boxes, feather)
    if ref is None:
        ref = np.zeros((120, 200), np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(ref, (x, y), (x + w, y + h), 255, thickness=-1)
    assert np.array_equal(got, ref)


def test_unsigned_numpy_coordinates_do_not_wrap():
    box = tuple(np.uint16(v) for v in (2, 3, 20, 20))
    got = bm.render_mask_frame(100, 80, [box], 8)
    assert np.array_equal(got, _reference(100, 80, [(2, 3, 20, 20)], 8))


@pytest.mark.parametrize("size", [(1, 1), (7, 1), (3, 1), (1, 7), (2, 2), (8, 8)])
def test_ffv1_args_never_ask_for_more_slices_than_a_tiny_frame_allows(size):
    args = bm.ffv1_threading_args(*size)
    if min(size) < 8:
        assert "-slices" not in args
    else:
        assert args[args.index("-slices") + 1] == str(bm.FFV1_SLICES)
