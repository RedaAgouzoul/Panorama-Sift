# -*- coding: utf-8 -*-
"""
tests_automatiques.py
=====================
Vérifie automatiquement les 5 scénarios de test du projet, avec des images
synthétiques générées par exemples/generer_jeux_de_test.py.

Lancement :   python tests_automatiques.py
Résultat  :   une ligne PASS / FAIL par scénario (code de sortie 0 si tout passe).
"""

import os
import sys
import tempfile

import cv2

from panorama import PanoramaParams, run_pipeline
from exemples.generer_jeux_de_test import generate_all


def read_files(paths):
    """Transforme des chemins en liste de (nom, octets), comme le fait l'interface."""
    files = []
    for path in paths:
        with open(path, "rb") as handle:
            files.append((os.path.basename(path), handle.read()))
    return files


def main() -> int:
    results = []

    def check(label, condition, detail=""):
        results.append(condition)
        print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f"  ->  {detail}" if detail else ""))

    with tempfile.TemporaryDirectory() as folder:
        sets = generate_all(folder)
        params = PanoramaParams()

        # Test 1 : deux images qui se chevauchent
        r = run_pipeline(read_files(sets["deux_images"]), params)
        ok = r.error is None and r.panorama is not None and r.pairs[0].nb_inliers >= 50
        check("1. Deux images qui se chevauchent", ok,
              f"inliers={r.pairs[0].nb_inliers if r.pairs else '?'}, "
              f"panorama={r.panorama.shape[1]}x{r.panorama.shape[0] if r.panorama is not None else ''}"
              if r.panorama is not None else (r.error or ""))
        if r.panorama is not None:
            cv2.imwrite(os.path.join(os.getcwd(), "_resultat_test_2_images.png"), r.panorama)

        # Test 2 : aucune zone commune -> erreur claire, pas de panorama
        r = run_pipeline(read_files(sets["sans_zone_commune"]), params)
        check("2. Deux images sans zone commune", r.error is not None and r.panorama is None, r.error or "aucune erreur !")

        # Test 3 : trois images successives
        r = run_pipeline(read_files(sets["trois_images"]), params)
        ok = r.error is None and r.panorama is not None and len(r.pairs) == 2 and all(p.ok for p in r.pairs)
        wide_enough = r.panorama is not None and r.panorama.shape[1] > 1200
        check("3. Trois images successives", ok and wide_enough,
              f"inliers={[p.nb_inliers for p in r.pairs]}, "
              f"taille={r.panorama.shape[1]}x{r.panorama.shape[0]}" if r.panorama is not None else (r.error or ""))
        if r.panorama is not None:
            cv2.imwrite(os.path.join(os.getcwd(), "_resultat_test_3_images.png"), r.panorama)

        # Test 3 bis : trois images avec la 1re image comme référence + recadrage intérieur
        r2 = run_pipeline(read_files(sets["trois_images"]),
                          PanoramaParams(reference="premiere", crop_mode="interieur"))
        check("3 bis. Trois images, référence = 1re image, recadrage intérieur",
              r2.error is None and r2.panorama is not None,
              f"taille={r2.panorama.shape[1]}x{r2.panorama.shape[0]}" if r2.panorama is not None else (r2.error or ""))

        # Test 4 : fichier invalide
        r = run_pipeline(read_files(sets["invalide"]), params)
        check("4. Fichier invalide (texte renommé .jpg)", r.error is not None and "pas une image valide" in r.error, r.error or "")

        # Test 4 bis : une seule image
        r = run_pipeline(read_files(sets["deux_images"][:1]), params)
        check("4 bis. Une seule image fournie", r.error is not None and "minimum deux images" in r.error, r.error or "")

        # Test 5 : trop peu de correspondances fiables
        r = run_pipeline(read_files(sets["peu_de_correspondances"]), params)
        check("5. Trop peu de correspondances fiables", r.error is not None and r.panorama is None, r.error or "aucune erreur !")

    print()
    print(f"{sum(results)}/{len(results)} tests réussis.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
