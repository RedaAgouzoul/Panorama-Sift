# -*- coding: utf-8 -*-
"""Tests fonctionnels du pipeline Panorama SIFT."""

from pathlib import Path

import pytest

from panorama_sift import PanoramaParams, run_pipeline
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

