# -*- coding: utf-8 -*-
"""Appariement SIFT, ratio de Lowe, homographie et RANSAC."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import FeatureSet, HomographyError, PairResult, PanoramaError, PanoramaParams

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

