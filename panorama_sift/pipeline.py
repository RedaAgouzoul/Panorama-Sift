# -*- coding: utf-8 -*-
"""Orchestration du pipeline complet de création du panorama."""

import time
from typing import Callable, List, Optional, Tuple

import cv2

from .image_processing import detect_sift_features, draw_keypoints_image, prepare_images
from .matching import match_pair
from .models import PanoramaError, PanoramaParams, PipelineResult
from .stitching import assemble_panorama

ProgressCallback = Callable[[float, str], None]

def run_pipeline(files: List[Tuple[str, bytes]], params: Optional[PanoramaParams] = None,
                 progress: Optional[ProgressCallback] = None, *,
                 diagonal_crops=None) -> PipelineResult:
    """Exécute toutes les étapes. Ne lève pas d'exception « métier » :
    en cas de problème, result.error et result.error_step sont renseignés et
    tout ce qui a pu être calculé avant l'erreur reste disponible."""
    params = params or PanoramaParams()
    result = PipelineResult()

    def report(fraction: float, text: str) -> None:
        if progress is not None:
            progress(fraction, text)

    current_step = "chargement"
    try:
        # --- Étape 1 : chargement -------------------------------------------------
        report(0.02, "Étape 1/5 : chargement des images…")
        start = time.perf_counter()
        result.images = prepare_images(files, params.max_side, diagonal_crops)
        result.durations["1. Chargement"] = time.perf_counter() - start

        # --- Étape 2 : SIFT ---------------------------------------------------------
        current_step = "détection SIFT"
        start = time.perf_counter()
        for position, prepared in enumerate(result.images):
            report(0.10 + 0.25 * position / len(result.images),
                   f"Étape 2/5 : SIFT sur l'image {position + 1}/{len(result.images)}…")
            features = detect_sift_features(prepared.bgr, prepared.valid_mask)
            result.features.append(features)
            result.keypoint_images.append(draw_keypoints_image(prepared.bgr, features))
        result.durations["2. Détection SIFT"] = time.perf_counter() - start

        for position, features in enumerate(result.features):
            if features.descriptors is None or len(features.keypoints) < 2:
                raise PanoramaError(
                    f"Aucun point clé exploitable dans « {result.images[position].name} » "
                    "(image trop uniforme, floue ou sous/sur-exposée)."
                )

        # --- Étapes 3 et 4 : appariement + homographie, paire par paire -------------
        current_step = "appariement / homographie"
        start = time.perf_counter()
        for i in range(len(result.images) - 1):
            report(0.40 + 0.35 * i / max(1, len(result.images) - 1),
                   f"Étapes 3-4/5 : appariement et RANSAC, images {i + 1} et {i + 2}…")
            result.pairs.append(match_pair(
                i, result.features[i], result.images[i].bgr,
                result.features[i + 1], result.images[i + 1].bgr, params
            ))
        result.durations["3-4. Appariement + RANSAC"] = time.perf_counter() - start

        failed = [pair for pair in result.pairs if not pair.ok]
        if failed:
            first = failed[0]
            raise PanoramaError(
                f"Impossible d'aligner l'image {first.index_left + 1} "
                f"(« {result.images[first.index_left].name} ») avec l'image {first.index_right + 1} "
                f"(« {result.images[first.index_right].name} »). {first.message}"
            )

        # --- Étape 5 : assemblage ------------------------------------------------------
        current_step = "assemblage"
        report(0.80, "Étape 5/5 : assemblage et fusion des images…")
        start = time.perf_counter()
        result.panorama = assemble_panorama(
            [prepared.bgr for prepared in result.images],
            [prepared.valid_mask for prepared in result.images],
            [pair.homography for pair in result.pairs],
            params, result
        )
        result.durations["5. Assemblage"] = time.perf_counter() - start
        report(1.0, "Terminé.")

    except PanoramaError as error:
        result.error = str(error)
        result.error_step = current_step
    except cv2.error as error:  # erreur interne OpenCV inattendue
        result.error = f"Erreur interne OpenCV pendant l'étape « {current_step} » : {error}"
        result.error_step = current_step
    except MemoryError:
        result.error = ("Mémoire insuffisante pour construire le panorama. "
                        "Réduisez la « taille maximale des images » dans les paramètres.")
        result.error_step = current_step
    return result

