# -*- coding: utf-8 -*-
"""Paramètres, modèles de résultats et exceptions du projet."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

class PanoramaError(Exception):
    """Erreur « métier » avec un message compréhensible par l'utilisateur."""


class HomographyError(PanoramaError):
    """Échec du calcul d'homographie ; garde le nombre d'inliers pour le diagnostic."""

    def __init__(self, message: str, nb_inliers: int = 0):
        super().__init__(message)
        self.nb_inliers = nb_inliers


@dataclass
class PanoramaParams:
    """Tous les paramètres réglables du traitement (valeurs par défaut sûres)."""

    max_side: int = 1000          # côté maximal (px) après redimensionnement
    lowe_ratio: float = 0.75      # seuil du ratio test de Lowe (0.7 à 0.8 typique)
    ransac_threshold: float = 4.0  # erreur de reprojection max (px) d'un inlier
    min_good_matches: int = 12    # minimum de correspondances après le ratio test
    min_inliers: int = 10         # minimum d'inliers RANSAC pour accepter la paire
    min_inlier_ratio: float = 0.25  # minimum de (inliers / correspondances conservées)
    reference: str = "centre"     # image de référence : "centre" ou "premiere"
    crop_mode: str = "bbox"       # "aucun", "bbox" (sans perte) ou "interieur"
    max_canvas_pixels: int = 25_000_000  # garde-fou mémoire sur la taille du panorama


@dataclass
class PreparedImage:
    """Une image chargée et prête pour OpenCV."""

    name: str
    bgr: np.ndarray                 # image redimensionnée (BGR, uint8)
    original_size: Tuple[int, int]  # (largeur, hauteur) avant redimensionnement


@dataclass
class FeatureSet:
    """Résultat de SIFT pour une image."""

    keypoints: tuple                # tuple de cv2.KeyPoint
    descriptors: Optional[np.ndarray]  # tableau (N, 128) float32, ou None


@dataclass
class PairResult:
    """Résultat de l'appariement entre l'image i (gauche) et l'image i+1 (droite)."""

    index_left: int
    index_right: int
    nb_initial: int = 0             # correspondances candidates (avant ratio test)
    nb_good: int = 0                # correspondances conservées par le ratio test
    nb_inliers: int = 0             # inliers trouvés par RANSAC
    homography: Optional[np.ndarray] = None  # H : coordonnées image i -> image i+1
    matches_image: Optional[np.ndarray] = None  # visualisation (BGR)
    ok: bool = False
    message: str = ""               # explication si la paire est refusée


@dataclass
class PipelineResult:
    """Tout ce qui est produit par run_pipeline, utilisé par l'interface."""

    images: List[PreparedImage] = field(default_factory=list)
    features: List[FeatureSet] = field(default_factory=list)
    keypoint_images: List[np.ndarray] = field(default_factory=list)
    pairs: List[PairResult] = field(default_factory=list)
    reference_index: int = 0
    homographies_to_reference: List[np.ndarray] = field(default_factory=list)
    panorama: Optional[np.ndarray] = None   # panorama final (BGR)
    canvas_size: Optional[Tuple[int, int]] = None   # taille avant recadrage
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, x1, y1)
    durations: dict = field(default_factory=dict)  # temps par étape (secondes)
    error: Optional[str] = None             # message d'erreur global éventuel
    error_step: Optional[str] = None        # étape où l'erreur est survenue

