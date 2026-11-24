# -*- coding: utf-8 -*-
"""Tests fonctionnels du pipeline Panorama SIFT."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from panorama_sift import PanoramaParams, run_pipeline
from panorama_sift.image_processing import prepare_images
from panorama_sift.stitching import blend_images
from scripts.generate_test_images import generate_all


def read_files(paths):
    return [(Path(path).name, Path(path).read_bytes()) for path in paths]


@pytest.fixture(scope="module")
def image_sets(tmp_path_factory):
    return generate_all(str(tmp_path_factory.mktemp("panorama_images")))


def test_two_overlapping_images(image_sets):
    result = run_pipeline(read_files(image_sets["deux_images"]), PanoramaParams())
    assert result.error is None
    assert result.panorama is not None
    assert result.pairs[0].nb_inliers >= 50


def test_images_without_common_area_are_rejected(image_sets):
    result = run_pipeline(read_files(image_sets["sans_zone_commune"]), PanoramaParams())
    assert result.error is not None
    assert result.panorama is None


def test_three_successive_images(image_sets):
    result = run_pipeline(read_files(image_sets["trois_images"]), PanoramaParams())
    assert result.error is None
    assert result.panorama is not None
    assert len(result.pairs) == 2
    assert all(pair.ok for pair in result.pairs)
    assert result.panorama.shape[1] > 1200


def test_first_reference_and_inner_crop(image_sets):
    params = PanoramaParams(reference="premiere", crop_mode="interieur")
    result = run_pipeline(read_files(image_sets["trois_images"]), params)
    assert result.error is None
    assert result.panorama is not None


def test_invalid_file_is_rejected(image_sets):
    result = run_pipeline(read_files(image_sets["invalide"]), PanoramaParams())
    assert result.error is not None
    assert "pas une image valide" in result.error


def test_single_image_is_rejected(image_sets):
    result = run_pipeline(read_files(image_sets["deux_images"][:1]), PanoramaParams())
    assert result.error is not None
    assert "minimum deux images" in result.error


def test_insufficient_matches_are_rejected(image_sets):
    result = run_pipeline(read_files(image_sets["peu_de_correspondances"]), PanoramaParams())
    assert result.error is not None
    assert result.panorama is None


def _png_bytes(array, mode):
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_manual_diagonal_ignores_white_padding_even_when_connected_to_scene():
    rgb = np.full((200, 200, 3), 255, np.uint8)
    rgb[30:45, 100:120] = 80
    data = _png_bytes(rgb, "RGB")
    prepared = prepare_images(
        [("above.png", data), ("below.png", data)], 200,
        diagonal_crops={0: (0, 100, True), 1: (0, 100, False)},
    )
    panorama, valid = blend_images(
        [image.bgr for image in prepared], [image.valid_mask for image in prepared],
        [np.eye(3), np.eye(3)], np.eye(3), (200, 200),
    )
    assert prepared[0].valid_mask[150, 25] == 0
    assert prepared[1].valid_mask[35, 110] == 0
    assert valid[35, 110]
    assert np.all(panorama[35, 110] == 80)


def test_shallow_jpeg_padding_does_not_darken_the_other_image():
    height, width = 240, 500
    scene = np.full((height, width, 3), 160, np.uint8)
    # A real dark detail inside the scene must remain valid.
    scene[105:135, 220:280] = 0
    files = []
    for index, points in enumerate((
        [[0, 0], [499, 0], [499, 48]],
        [[0, 191], [0, 239], [499, 239]],
    )):
        image = scene.copy()
        cv2.fillPoly(image, [np.array(points)], (0, 0, 0))
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=70)
        files.append((f"{index}.jpeg", buffer.getvalue()))

    prepared = prepare_images(files, 1000)
    assert prepared[0].valid_mask[10, 400] == 0
    assert prepared[1].valid_mask[225, 100] == 0
    assert all(image.valid_mask[120, 250] == 255 for image in prepared)
    panorama, valid = blend_images(
        [image.bgr for image in prepared], [image.valid_mask for image in prepared],
        [np.eye(3), np.eye(3)], np.eye(3), (width, height),
    )
    assert valid.all()
    assert np.all(panorama[10, 400] >= 155)
    assert np.all(panorama[225, 100] >= 155)


def test_transparent_padding_is_excluded_from_blending():
    rng = np.random.default_rng(12)
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[:, :, :3] = rng.integers(0, 245, (200, 200, 3), dtype=np.uint8)
    alpha = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(alpha, [np.array([[0, 0], [199, 0], [199, 199]])], 255)
    rgba[:, :, 3] = alpha
    content = _png_bytes(rgba, "RGBA")

    prepared = prepare_images([("a.png", content), ("b.png", content)], 1000)[0]
    assert prepared.valid_mask[190, 10] == 0
    assert prepared.valid_mask[10, 190] == 255

    panorama, valid = blend_images(
        [prepared.bgr], [prepared.valid_mask], [np.eye(3)], np.eye(3), (200, 200)
    )
    assert not valid[190, 10]
    assert np.all(panorama[190, 10] == 0)


@pytest.mark.parametrize("padding_color", [(255, 255, 255), (0, 0, 0)])
def test_large_uniform_corner_triangle_is_detected_as_padding(padding_color):
    rng = np.random.default_rng(21)
    cases = (
        (np.array([[0, 100], [0, 199], [100, 199]]), (190, 10)),
        (np.array([[100, 0], [199, 0], [199, 100]]), (10, 190)),
    )
    for index, (triangle, invalid_point) in enumerate(cases):
        rgb = rng.integers(20, 230, (200, 200, 3), dtype=np.uint8)
        cv2.fillPoly(rgb, [triangle], padding_color)
        content = _png_bytes(rgb, "RGB")
        prepared = prepare_images(
            [(f"a{index}.png", content), (f"b{index}.png", content)], 1000
        )[0]
        assert prepared.valid_mask[invalid_point] == 0
        assert prepared.valid_mask[100, 100] == 255

