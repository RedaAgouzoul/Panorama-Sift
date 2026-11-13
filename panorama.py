# -*- coding: utf-8 -*-
"""
panorama.py
===========
Module de traitement du projet « Création d'un panorama d'images par
appariement de caractéristiques SIFT ».

Ce fichier ne contient AUCUN code d'interface : il peut être utilisé seul
(par exemple depuis tests_automatiques.py) et il est appelé par app.py.

Pipeline complet (fonction run_pipeline) :
    1. Chargement et préparation des images (validation, EXIF, redimensionnement)
    2. Détection des points clés + descripteurs SIFT
    3. Appariement BFMatcher (KNN) + ratio test de Lowe
    4. Homographie avec RANSAC pour chaque paire d'images successives
    5. Assemblage : canevas, warpPerspective, fusion, recadrage

Convention de couleurs : toutes les images internes sont en BGR (convention
d'OpenCV). La conversion en RGB (pour Streamlit) est faite dans app.py.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------------------------------------------------------------------
# 0. Paramètres, structures de données et exceptions
# ---------------------------------------------------------------------------

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


# Signature d'une fonction de progression : (fraction entre 0 et 1, texte)
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# 1. Chargement et préparation des images
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Appariement : BFMatcher (KNN) + ratio test de Lowe
# ---------------------------------------------------------------------------

def match_descriptors(descriptors_left: Optional[np.ndarray],
                      descriptors_right: Optional[np.ndarray],
                      lowe_ratio: float) -> Tuple[int, List[cv2.DMatch]]:
    """Compare les descripteurs de deux images.

    Renvoie (nombre de correspondances candidates, correspondances conservées).

    Principe :
      1. BFMatcher (« force brute ») compare chaque descripteur de l'image de
         gauche à TOUS ceux de l'image de droite avec la distance euclidienne
         (NORM_L2, adaptée à SIFT).
      2. knnMatch(k=2) garde les 2 meilleurs candidats pour chaque point.
      3. Ratio test de Lowe : on ne garde le meilleur candidat que s'il est
         nettement meilleur que le deuxième (distance1 < ratio x distance2).
         Si les deux sont aussi proches, le point est « ambigu » : on l'écarte.
    """
    if descriptors_left is None or descriptors_right is None:
        raise PanoramaError("Aucun descripteur SIFT : l'image est peut-être trop uniforme ou floue.")
    if len(descriptors_left) < 2 or len(descriptors_right) < 2:
        raise PanoramaError("Trop peu de points clés SIFT détectés pour comparer les images.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_pairs = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

    # Certaines paires peuvent contenir moins de 2 éléments : on les ignore
    knn_pairs = [pair for pair in knn_pairs if len(pair) == 2]
    good_matches = [best for best, second in knn_pairs if best.distance < lowe_ratio * second.distance]
    return len(knn_pairs), good_matches


# ---------------------------------------------------------------------------
# 4. Homographie avec RANSAC
# ---------------------------------------------------------------------------

def is_homography_plausible(homography: np.ndarray, width: int, height: int) -> bool:
    """Vérifie qu'une homographie est « raisonnable » (anti-résultats absurdes).

    On transforme les 4 coins de l'image et on vérifie que le quadrilatère obtenu :
      - est devant la caméra (coordonnée homogène positive),
      - est convexe (pas de papillon / repli),
      - n'est ni minuscule ni gigantesque par rapport à l'image d'origine.
    """
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    homogeneous = np.hstack([corners, np.ones((4, 1))]) @ homography.T
    if np.any(homogeneous[:, 2] <= 1e-9):
        return False
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]

    # Convexité : les produits vectoriels des côtés consécutifs ont le même signe
    cross_signs = []
    for i in range(4):
        edge_a = projected[(i + 1) % 4] - projected[i]
        edge_b = projected[(i + 2) % 4] - projected[(i + 1) % 4]
        cross_signs.append(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    if not (all(c > 0 for c in cross_signs) or all(c < 0 for c in cross_signs)):
        return False

    # Rapport des aires (formule du lacet)
    x, y = projected[:, 0], projected[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    ratio = area / float(width * height)
    return 0.1 < ratio < 10.0


def estimate_homography(keypoints_left: tuple, keypoints_right: tuple,
                        good_matches: List[cv2.DMatch], params: PanoramaParams,
                        image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule H (image de gauche -> image de droite) avec RANSAC.

    Renvoie (H, masque_inliers) où masque_inliers[i] = 1 si la correspondance i
    est un inlier. Lève PanoramaError si le calcul n'est pas fiable.
    """
    if len(good_matches) < params.min_good_matches:
        raise PanoramaError(
            f"Seulement {len(good_matches)} correspondances fiables après le ratio test "
            f"(minimum requis : {params.min_good_matches}). Les images ont probablement "
            "trop peu de zone commune, ou trop peu de détails."
        )

    # Coordonnées (x, y) des points appariés dans chaque image, forme (N, 1, 2)
    points_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        points_left, points_right, cv2.RANSAC, params.ransac_threshold
    )
    if homography is None or inlier_mask is None:
        raise PanoramaError("RANSAC n'a pas réussi à calculer une homographie (correspondances incohérentes).")

    nb_inliers = int(inlier_mask.sum())
    inlier_ratio = nb_inliers / len(good_matches)
    if nb_inliers < params.min_inliers:
        raise HomographyError(
            f"RANSAC ne retient que {nb_inliers} inliers sur {len(good_matches)} correspondances "
            f"(minimum requis : {params.min_inliers}). Les images ne semblent pas montrer la même scène.",
            nb_inliers,
        )
    if inlier_ratio < params.min_inlier_ratio:
        raise HomographyError(
            f"Seulement {inlier_ratio:.0%} des correspondances sont cohérentes entre elles "
            f"({nb_inliers} inliers sur {len(good_matches)}, minimum requis : {params.min_inlier_ratio:.0%}). "
            "Il s'agit probablement de fausses correspondances : les images n'ont sans doute pas de zone commune.",
            nb_inliers,
        )
    if not is_homography_plausible(homography, image_size[0], image_size[1]):
        raise HomographyError(
            "L'homographie calculée est géométriquement incohérente (image déformée à l'extrême). "
            "Vérifiez l'ordre des images et le chevauchement.",
            nb_inliers,
        )
    return homography, inlier_mask


def draw_matches_image(image_left: np.ndarray, keypoints_left: tuple,
                       image_right: np.ndarray, keypoints_right: tuple,
                       good_matches: List[cv2.DMatch],
                       inlier_mask: Optional[np.ndarray],
                       max_lines: int = 120) -> np.ndarray:
    """Visualise les correspondances avec cv2.drawMatches.

    Vert = inliers (cohérents avec l'homographie), rouge = écartés par RANSAC.
    Si aucun masque n'est fourni, tout est dessiné en vert.
    """
    if inlier_mask is None:
        inliers = sorted(good_matches, key=lambda m: m.distance)[:max_lines]
        outliers: List[cv2.DMatch] = []
    else:
        flags = inlier_mask.ravel().astype(bool)
        inliers = sorted((m for m, keep in zip(good_matches, flags) if keep), key=lambda m: m.distance)[:max_lines]
        outliers = sorted((m for m, keep in zip(good_matches, flags) if not keep), key=lambda m: m.distance)[:40]

    no_single_points = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # 1er appel : crée l'image « côte à côte » et dessine les rejetés en rouge
    canvas = cv2.drawMatches(image_left, keypoints_left, image_right, keypoints_right,
                             outliers, None, matchColor=(0, 0, 255), flags=no_single_points)
    # 2e appel : dessine les inliers en vert par-dessus la même image
    canvas = cv2.drawMatches(image_left, keypoints_left, image_right, keypoints_right,
                             inliers, canvas, matchColor=(0, 220, 0),
                             flags=no_single_points | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    return canvas


def match_pair(index_left: int, features_left: FeatureSet, image_left: np.ndarray,
               features_right: FeatureSet, image_right: np.ndarray,
               params: PanoramaParams) -> PairResult:
    """Appariement + homographie pour une paire (ne lève jamais d'exception :
    les problèmes sont rapportés dans PairResult.message)."""
    result = PairResult(index_left=index_left, index_right=index_left + 1)
    good_matches: List[cv2.DMatch] = []
    try:
        result.nb_initial, good_matches = match_descriptors(
            features_left.descriptors, features_right.descriptors, params.lowe_ratio
        )
        result.nb_good = len(good_matches)
        height, width = image_left.shape[:2]
        homography, inlier_mask = estimate_homography(
            features_left.keypoints, features_right.keypoints, good_matches, params, (width, height)
        )
        result.homography = homography
        result.nb_inliers = int(inlier_mask.sum())
        result.ok = True
        result.message = "Paire alignée avec succès."
        result.matches_image = draw_matches_image(
            image_left, features_left.keypoints, image_right, features_right.keypoints,
            good_matches, inlier_mask
        )
    except PanoramaError as error:
        result.message = str(error)
        result.nb_inliers = getattr(error, "nb_inliers", 0)
        # Même en cas d'échec, on montre les correspondances pour diagnostiquer
        if good_matches:
            result.matches_image = draw_matches_image(
                image_left, features_left.keypoints, image_right, features_right.keypoints,
                good_matches, None
            )
    return result


# ---------------------------------------------------------------------------
# 5. Assemblage : transformations, canevas, fusion, recadrage
# ---------------------------------------------------------------------------

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


def blend_images(images: List[np.ndarray], homographies: List[np.ndarray],
                 translation: np.ndarray, canvas_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Projette chaque image dans le canevas et fusionne les zones de chevauchement.

    Fusion « pondérée par la distance au bord » (feathering) : dans une zone
    commune, chaque image contribue d'autant plus que le pixel est loin de
    son bord. Cela adoucit la couture entre les images.

    Renvoie (panorama BGR uint8, masque booléen des pixels valides).
    """
    canvas_width, canvas_height = canvas_size
    weighted_sum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_total = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for image, homography in zip(images, homographies):
        height, width = image.shape[:2]
        full_transform = translation @ homography

        # warpPerspective : pour chaque pixel du canevas, va chercher le pixel
        # correspondant dans l'image source grâce à la matrice (interpolation bilinéaire)
        warped = cv2.warpPerspective(image, full_transform, canvas_size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Masque des pixels réellement couverts par cette image
        coverage = cv2.warpPerspective(np.full((height, width), 255, np.uint8), full_transform,
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


def assemble_panorama(images: List[np.ndarray], pair_homographies: List[np.ndarray],
                      params: PanoramaParams, result: PipelineResult) -> np.ndarray:
    """Étape d'assemblage complète ; remplit aussi les champs de `result`."""
    image_count = len(images)
    reference_index = image_count // 2 if params.reference == "centre" else 0
    result.reference_index = reference_index

    homographies = compute_homographies_to_reference(pair_homographies, reference_index)
    result.homographies_to_reference = homographies

    translation, canvas_size = compute_canvas(images, homographies, params.max_canvas_pixels)
    result.canvas_size = canvas_size

    panorama, valid_mask = blend_images(images, homographies, translation, canvas_size)
    cropped, crop_box = crop_panorama(panorama, valid_mask, params.crop_mode)
    result.crop_box = crop_box
    if cropped.size == 0:
        raise PanoramaError("Le panorama final est vide après recadrage.")
    return cropped


# ---------------------------------------------------------------------------
# 6. Pipeline complet
# ---------------------------------------------------------------------------

def run_pipeline(files: List[Tuple[str, bytes]], params: Optional[PanoramaParams] = None,
                 progress: Optional[ProgressCallback] = None) -> PipelineResult:
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
        result.images = prepare_images(files, params.max_side)
        result.durations["1. Chargement"] = time.perf_counter() - start

        # --- Étape 2 : SIFT ---------------------------------------------------------
        current_step = "détection SIFT"
        start = time.perf_counter()
        for position, prepared in enumerate(result.images):
            report(0.10 + 0.25 * position / len(result.images),
                   f"Étape 2/5 : SIFT sur l'image {position + 1}/{len(result.images)}…")
            features = detect_sift_features(prepared.bgr)
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


def encode_image(image_bgr: np.ndarray, extension: str = ".png") -> bytes:
    """Encode une image BGR en PNG ou JPEG (octets) pour le téléchargement."""
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in (".jpg", ".jpeg") else []
    success, buffer = cv2.imencode(extension, image_bgr, parameters)
    if not success:
        raise PanoramaError("Échec de l'encodage de l'image pour le téléchargement.")
    return buffer.tobytes()
