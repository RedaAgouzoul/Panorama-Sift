# -*- coding: utf-8 -*-
"""Chargement, préparation, détection SIFT et encodage des images."""

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import FeatureSet, PanoramaError, PreparedImage

def _triangular_padding_mask(rgb: np.ndarray) -> np.ndarray:
    """Repère les aplats noirs ou blancs délimités par une diagonale."""
    height, width = rgb.shape[:2]
    valid = np.full((height, width), 255, np.uint8)
    # Une petite tolérance absorbe les variations dues à la compression JPEG.
    for candidate in (np.all(rgb >= 245, axis=2), np.all(rgb <= 12, axis=2)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 0.02 * height * width:
                continue
            component = labels == label
            # Tester les lignes et les colonnes pour les diagonales peu inclinées.
            orientations = (component, component[:, ::-1],
                            component.T, component.T[:, ::-1])
            if any(_has_diagonal_edge(region) for region in orientations):
                valid[component] = 0
    return valid


def _has_diagonal_edge(region: np.ndarray) -> bool:
    """Vérifie qu'un aplat part d'un coin et suit le bord gauche."""
    height, width = region.shape
    if not (region[0, 0] or region[-1, 0]) or region[:, -1].sum() > 0.05 * height:
        return False
    counts = region.sum(axis=1)
    rows = np.flatnonzero(counts)
    if rows.size < max(12, 0.2 * height):
        return False
    # Les lignes de l'aplat doivent former un segment continu depuis le bord.
    ends = width - 1 - region[rows, ::-1].argmax(axis=1)
    good = region[rows, 0] & (counts[rows] == ends + 1)
    if good.sum() < 0.85 * rows.size:
        return False
    x, y = rows[good], ends[good]
    if np.ptp(y) < max(8, 0.08 * width):
        return False
    predicted = np.polyval(np.polyfit(x, y, 1), x)
    return np.mean(np.abs(y - predicted)) <= max(2.0, 0.02 * np.ptp(y))


def _load_image_and_mask(file_bytes: bytes, name: str,
                         detect_padding: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Décode une image BGR et son masque de pixels valides.

    Pillow corrige l'orientation EXIF. Le canal alpha est conservé sous forme
    de masque avant la conversion RGB vers BGR pour OpenCV.
    """
    if not file_bytes:
        raise PanoramaError(f"« {name} » : le fichier est vide.")
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            has_alpha = "A" in pil_image.getbands() or "transparency" in pil_image.info
            if has_alpha:
                rgba = np.array(pil_image.convert("RGBA"))
                rgb = rgba[:, :, :3]
                valid_mask = np.where(rgba[:, :, 3] > 8, 255, 0).astype(np.uint8)
            else:
                rgb = np.array(pil_image.convert("RGB"))
                valid_mask = (_triangular_padding_mask(rgb) if detect_padding
                              else np.full(rgb.shape[:2], 255, np.uint8))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise PanoramaError(
            f"« {name} » n'est pas une image valide (fichier corrompu ou format "
            "non pris en charge). Formats acceptés : PNG, JPG, JPEG, BMP, WEBP, TIF."
        )
    if rgb.ndim != 3 or rgb.shape[0] < 100 or rgb.shape[1] < 100:
        raise PanoramaError(
            f"« {name} » est trop petite (minimum 100 x 100 pixels) pour être traitée."
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[valid_mask == 0] = 0
    return bgr, valid_mask


def load_image_from_bytes(file_bytes: bytes, name: str) -> np.ndarray:
    """Décode un fichier image et renvoie son contenu BGR pour OpenCV."""
    image_bgr, _ = _load_image_and_mask(file_bytes, name)
    return image_bgr


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


def prepare_images(files: List[Tuple[str, bytes]], max_side: int,
                   diagonal_crops=None) -> List[PreparedImage]:
    """Charge, valide et redimensionne une liste de (nom, contenu_binaire)."""
    if len(files) < 2:
        raise PanoramaError(
            "Il faut au minimum deux images pour créer un panorama "
            f"({len(files)} image(s) fournie(s))."
        )
    prepared: List[PreparedImage] = []
    for index, (name, content) in enumerate(files):
        crop = diagonal_crops.get(index) if diagonal_crops else None
        image_bgr, valid_mask = _load_image_and_mask(content, name, detect_padding=crop is None)
        height, width = image_bgr.shape[:2]
        if crop is not None:
            left, right, keep_above = crop
            boundary = np.linspace(left, right, width) * (height - 1) / 100
            rows = np.arange(height)[:, None]
            keep = rows <= boundary if keep_above else rows >= boundary
            valid_mask[~keep] = 0
            image_bgr[valid_mask == 0] = 0
        resized_bgr = resize_keep_ratio(image_bgr, max_side)
        if resized_bgr.shape[:2] != valid_mask.shape:
            valid_mask = cv2.resize(
                valid_mask,
                (resized_bgr.shape[1], resized_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        prepared.append(
            PreparedImage(
                name=name,
                bgr=resized_bgr,
                valid_mask=valid_mask,
                original_size=(width, height),
            )
        )
    return prepared


# ---------------------------------------------------------------------------
# 2. Détection des points clés SIFT
# ---------------------------------------------------------------------------

def detect_sift_features(image_bgr: np.ndarray,
                         valid_mask: np.ndarray | None = None) -> FeatureSet:
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
    keypoints, descriptors = sift.detectAndCompute(gray, valid_mask)
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

