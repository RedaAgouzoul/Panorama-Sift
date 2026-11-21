# -*- coding: utf-8 -*-
"""Chargement, préparation, détection SIFT et encodage des images."""

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import FeatureSet, PanoramaError, PreparedImage

def load_image_from_bytes(file_bytes: bytes, name: str) -> np.ndarray:
    """Décode un fichier image et renvoie un tableau BGR (format OpenCV).

    On passe par Pillow car il gère l'orientation EXIF (les photos de
    téléphone sont souvent « tournées » dans les métadonnées) puis on
    convertit RGB -> BGR pour OpenCV.
    """
    if not file_bytes:
        raise PanoramaError(f"« {name} » : le fichier est vide.")
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)  # corrige l'orientation
            pil_image = pil_image.convert("RGB")             # retire alpha / palette
            rgb = np.array(pil_image)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise PanoramaError(
            f"« {name} » n'est pas une image valide (fichier corrompu ou format "
            "non pris en charge). Formats acceptés : PNG, JPG, JPEG, BMP, WEBP, TIF."
        )
    if rgb.ndim != 3 or rgb.shape[0] < 100 or rgb.shape[1] < 100:
        raise PanoramaError(
            f"« {name} » est trop petite (minimum 100 x 100 pixels) pour être traitée."
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_keep_ratio(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Réduit l'image si son plus grand côté dépasse max_side (ratio conservé)."""
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_bgr
    scale = max_side / longest_side
    new_size = (int(round(width * scale)), int(round(height * scale)))
    # INTER_AREA est l'interpolation recommandée pour réduire une image
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def prepare_images(files: List[Tuple[str, bytes]], max_side: int) -> List[PreparedImage]:
    """Charge, valide et redimensionne une liste de (nom, contenu_binaire)."""
    if len(files) < 2:
        raise PanoramaError(
            "Il faut au minimum deux images pour créer un panorama "
            f"({len(files)} image(s) fournie(s))."
        )
    prepared: List[PreparedImage] = []
    for name, content in files:
        image_bgr = load_image_from_bytes(content, name)
        height, width = image_bgr.shape[:2]
        prepared.append(
            PreparedImage(
                name=name,
                bgr=resize_keep_ratio(image_bgr, max_side),
                original_size=(width, height),
            )
        )
    return prepared


# ---------------------------------------------------------------------------
# 2. Détection des points clés SIFT
# ---------------------------------------------------------------------------

def detect_sift_features(image_bgr: np.ndarray) -> FeatureSet:
    """Détecte les points clés SIFT et calcule leurs descripteurs.

    detectAndCompute(image, masque) renvoie DEUX résultats :
      - keypoints   : la liste des points clés (position x/y, taille,
                      orientation, force de réponse) ;
      - descriptors : un tableau de N lignes x 128 colonnes ; la ligne i est la
                      « carte d'identité » numérique du point clé i.
    SIFT travaille sur une image en niveaux de gris.
    """
    sift = cv2.SIFT_create()  # disponible dans le paquet principal depuis OpenCV 4.4
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return FeatureSet(keypoints=keypoints, descriptors=descriptors)


def draw_keypoints_image(image_bgr: np.ndarray, features: FeatureSet,
                         max_drawn: int = 500) -> np.ndarray:
    """Dessine les points clés (cercle = échelle, trait = orientation).

    Pour garder une image lisible, on ne dessine que les `max_drawn` points
    les plus « forts » (réponse la plus élevée).
    """
    strongest = sorted(features.keypoints, key=lambda kp: kp.response, reverse=True)
    return cv2.drawKeypoints(
        image_bgr,
        strongest[:max_drawn],
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

def encode_image(image_bgr: np.ndarray, extension: str = ".png") -> bytes:
    """Encode une image BGR en PNG ou JPEG (octets) pour le téléchargement."""
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in (".jpg", ".jpeg") else []
    success, buffer = cv2.imencode(extension, image_bgr, parameters)
    if not success:
        raise PanoramaError("Échec de l'encodage de l'image pour le téléchargement.")
    return buffer.tobytes()

