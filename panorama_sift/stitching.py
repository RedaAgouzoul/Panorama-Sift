# -*- coding: utf-8 -*-
"""Composition des homographies, projection, fusion et recadrage."""

from typing import List, Tuple

import cv2
import numpy as np

from .models import PanoramaError, PanoramaParams, PipelineResult

def compute_homographies_to_reference(pair_homographies: List[np.ndarray],
                                      reference_index: int) -> List[np.ndarray]:
    """Compose les homographies de paires pour tout ramener dans le repère de
    l'image de référence.

    pair_homographies[k] transforme l'image k vers l'image k+1.
      - Pour une image i à gauche de la référence : on enchaîne
        H_(ref-1) @ ... @ H_(i)   (on « avance » vers la référence).
      - Pour une image i à droite de la référence : on inverse l'enchaînement
        H_(i-1) @ ... @ H_(ref)   (on « recule » vers la référence).
    """
    image_count = len(pair_homographies) + 1
    homographies = [np.eye(3) for _ in range(image_count)]
    for i in range(reference_index - 1, -1, -1):          # images à gauche
        homographies[i] = homographies[i + 1] @ pair_homographies[i]
    for i in range(reference_index + 1, image_count):     # images à droite
        homographies[i] = homographies[i - 1] @ np.linalg.inv(pair_homographies[i - 1])
    return [h / h[2, 2] for h in homographies]


def compute_canvas(images: List[np.ndarray], homographies: List[np.ndarray],
                   max_canvas_pixels: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Calcule la taille du panorama et la translation qui évite les coordonnées négatives.

    On transforme les 4 coins de chaque image ; le rectangle englobant de tous
    ces coins donne les limites du canevas. Si le minimum est négatif, la
    matrice de translation décale tout pour que tout soit visible.
    """
    all_corners = []
    for image, homography in zip(images, homographies):
        height, width = image.shape[:2]
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float64)
        projected = corners @ homography.T
        if np.any(projected[:, 2] <= 1e-9):
            raise PanoramaError(
                "Les transformations calculées sont incohérentes (une image passerait « derrière » la caméra). "
                "Vérifiez l'ordre des images et réduisez les changements de perspective."
            )
        all_corners.append(projected[:, :2] / projected[:, 2:3])
    all_corners = np.vstack(all_corners)

    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_width, canvas_height = int(x_max - x_min), int(y_max - y_min)

    if canvas_width <= 0 or canvas_height <= 0:
        raise PanoramaError("Le panorama calculé est vide.")
    if canvas_width * canvas_height > max_canvas_pixels:
        raise PanoramaError(
            f"Le panorama serait démesuré ({canvas_width} x {canvas_height} px). Les homographies se sont "
            "probablement accumulées avec des erreurs : vérifiez l'ordre des images et le chevauchement."
        )

    # Matrice de translation : décale les coordonnées pour que x_min,y_min -> (0,0)
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    return translation, (canvas_width, canvas_height)


def blend_images(images: List[np.ndarray], masks: List[np.ndarray],
                 homographies: List[np.ndarray], translation: np.ndarray,
                 canvas_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Projette chaque image dans le canevas et fusionne les zones de chevauchement.

    Fusion « pondérée par la distance au bord » (feathering) : dans une zone
    commune, chaque image contribue d'autant plus que le pixel est loin de
    son bord. Cela adoucit la couture entre les images.

    Renvoie (panorama BGR uint8, masque booléen des pixels valides).
    """
    canvas_width, canvas_height = canvas_size
    weighted_sum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_total = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for image, source_mask, homography in zip(images, masks, homographies):
        full_transform = translation @ homography

        # warpPerspective : pour chaque pixel du canevas, va chercher le pixel
        # correspondant dans l'image source grâce à la matrice (interpolation bilinéaire)
        warped = cv2.warpPerspective(image, full_transform, canvas_size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Masque des pixels réellement couverts par cette image
        coverage = cv2.warpPerspective(source_mask, full_transform,
                                       canvas_size, flags=cv2.INTER_NEAREST)
        # On érode d'un pixel pour éviter le liseré sombre dû à l'interpolation
        coverage = cv2.erode(coverage, np.ones((3, 3), np.uint8))

        # Poids = distance de chaque pixel valide au bord de la zone couverte
        weights = cv2.distanceTransform(coverage, cv2.DIST_L2, 3)
        weighted_sum += warped.astype(np.float32) * weights[:, :, None]
        weight_total += weights

    valid_mask = weight_total > 0
    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    safe_total = np.where(valid_mask, weight_total, 1.0)[:, :, None]
    panorama[valid_mask] = np.clip(weighted_sum / safe_total, 0, 255).astype(np.uint8)[valid_mask]
    return panorama, valid_mask


def largest_inner_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Plus grand rectangle (x0, y0, x1, y1) entièrement contenu dans le masque.

    Algorithme classique du « plus grand rectangle dans un histogramme » appliqué
    ligne par ligne, avec une pile. x1 et y1 sont exclus (comme en NumPy).
    """
    height, width = mask.shape
    column_heights = [0] * width
    best_area, best_box = 0, (0, 0, 0, 0)
    for y in range(height):
        row = mask[y]
        column_heights = [h + 1 if row[x] else 0 for x, h in enumerate(column_heights)]
        stack: List[Tuple[int, int]] = []  # couples (x de départ, hauteur)
        for x in range(width + 1):
            current = column_heights[x] if x < width else 0
            start = x
            while stack and stack[-1][1] >= current:
                start, stack_height = stack.pop()
                area = stack_height * (x - start)
                if area > best_area:
                    best_area = area
                    best_box = (start, y - stack_height + 1, x, y + 1)
            stack.append((start, current))
    return best_box


def crop_panorama(panorama: np.ndarray, valid_mask: np.ndarray,
                  crop_mode: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Recadre le panorama.

    - "aucun"    : rien n'est retiré ;
    - "bbox"     : on retire seulement les marges noires inutiles autour du
                   contenu (rectangle englobant) : AUCUNE partie valide n'est coupée ;
    - "interieur": plus grand rectangle sans aucun pixel noir : plus propre,
                   mais coupe une partie du contenu valide.
    """
    height, width = valid_mask.shape
    if crop_mode == "aucun":
        return panorama, (0, 0, width, height)

    rows = np.where(valid_mask.any(axis=1))[0]
    cols = np.where(valid_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise PanoramaError("Le panorama obtenu est vide (aucun pixel valide).")
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1

    if crop_mode == "interieur":
        # Calcul sur une version réduite du masque pour rester rapide
        sub_mask = valid_mask[y0:y1, x0:x1]
        scale = max(1, int(np.ceil(max(sub_mask.shape) / 400)))
        small = cv2.resize(sub_mask.astype(np.uint8), None, fx=1 / scale, fy=1 / scale,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        small = cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        sx0, sy0, sx1, sy1 = largest_inner_rectangle(small)
        if (sx1 - sx0) * (sy1 - sy0) > 0:
            margin = scale + 1  # petite marge de sécurité après remise à l'échelle
            nx0, ny0 = x0 + sx0 * scale + margin, y0 + sy0 * scale + margin
            nx1, ny1 = x0 + sx1 * scale - margin, y0 + sy1 * scale - margin
            if nx1 > nx0 and ny1 > ny0:
                x0, y0, x1, y1 = nx0, ny0, nx1, ny1

    return panorama[y0:y1, x0:x1], (x0, y0, x1, y1)


def assemble_panorama(images: List[np.ndarray], masks: List[np.ndarray],
                      pair_homographies: List[np.ndarray],
                      params: PanoramaParams, result: PipelineResult) -> np.ndarray:
    """Étape d'assemblage complète ; remplit aussi les champs de `result`."""
    image_count = len(images)
    reference_index = image_count // 2 if params.reference == "centre" else 0
    result.reference_index = reference_index

    homographies = compute_homographies_to_reference(pair_homographies, reference_index)
    result.homographies_to_reference = homographies

    translation, canvas_size = compute_canvas(images, homographies, params.max_canvas_pixels)
    result.canvas_size = canvas_size

    panorama, valid_mask = blend_images(images, masks, homographies, translation, canvas_size)
    cropped, crop_box = crop_panorama(panorama, valid_mask, params.crop_mode)
    result.crop_box = crop_box
    if cropped.size == 0:
        raise PanoramaError("Le panorama final est vide après recadrage.")
    return cropped

