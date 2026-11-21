# -*- coding: utf-8 -*-
"""
generer_jeux_de_test.py
=======================
Génère des images SYNTHÉTIQUES (créées par ordinateur, pas des photos) pour
tester l'application sans avoir de photos sous la main.

Attention : ce ne sont pas de vraies photographies. Elles servent à vérifier
que le programme fonctionne et à reproduire les cas d'erreur. Pour la
une validation réaliste, utilisez de vraies photos (voir examples/conseils-prise-photos.md).

Utilisation (depuis le dossier panorama_sift, environnement virtuel activé) :
    python scripts/generate_test_images.py

Les images sont créées dans examples/jeux_de_test/ .
"""

import os

import cv2
import numpy as np

SCENE_WIDTH, SCENE_HEIGHT = 2000, 700


def create_scene(seed: int = 7, width: int = SCENE_WIDTH, height: int = SCENE_HEIGHT) -> np.ndarray:
    """Crée une grande « scène » riche en détails (bruit lissé, formes, textes)."""
    rng = np.random.default_rng(seed)

    # Fond : bruit coloré lissé à plusieurs échelles (donne de la texture naturelle)
    scene = np.zeros((height, width, 3), np.float32)
    for sigma, weight in ((40, 0.5), (15, 0.35), (5, 0.25)):
        noise = rng.random((height, width, 3)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        scene += weight * noise
    scene = np.clip(scene / scene.max() * 255, 0, 255).astype(np.uint8)

    # Formes géométriques colorées
    for _ in range(140):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(scene, center, int(rng.integers(8, 45)), color, int(rng.choice([-1, 2, 3])))
        elif kind == 1:
            size = int(rng.integers(15, 80))
            cv2.rectangle(scene, center, (center[0] + size, center[1] + int(size * 0.7)), color,
                          int(rng.choice([-1, 2])))
        else:
            end = (center[0] + int(rng.integers(-90, 90)), center[1] + int(rng.integers(-90, 90)))
            cv2.line(scene, center, end, color, int(rng.integers(2, 5)))

    # Textes (des coins très distinctifs pour SIFT)
    words = ["CAFE", "PARIS", "NORD", "SIFT", "VISION", "42", "OPEN", "BIG", "DATA", "CLOUD", "STOP", "EXIT"]
    for _ in range(45):
        position = (int(rng.integers(0, width - 120)), int(rng.integers(40, height - 10)))
        cv2.putText(scene, str(rng.choice(words)), position, cv2.FONT_HERSHEY_SIMPLEX,
                    float(rng.uniform(0.7, 1.8)), tuple(int(c) for c in rng.integers(0, 255, 3)),
                    int(rng.integers(2, 4)), cv2.LINE_AA)
    return scene


def extract_view(scene: np.ndarray, x_start: int, width: int = 800, rotation_deg: float = 0.0,
                 gain: float = 1.0, seed: int = 0) -> np.ndarray:
    """Simule une photo : fenêtre de la scène + petite rotation + variation de luminosité."""
    height = scene.shape[0]
    view = scene[:, x_start:x_start + width].copy()
    if rotation_deg != 0.0:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, 1.0)
        view = cv2.warpAffine(view, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    view = np.clip(view.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    # Léger bruit de capteur
    noise = np.random.default_rng(seed).normal(0, 2.0, view.shape)
    return np.clip(view + noise, 0, 255).astype(np.uint8)


def create_low_texture_scene(width: int = 2000, height: int = 700) -> np.ndarray:
    """Scène quasi uniforme (dégradé flou) avec seulement 2 petits disques dans la
    zone commune : il y a quelques points clés, mais beaucoup trop peu pour aligner."""
    x_ramp = np.tile(np.linspace(60, 190, width, dtype=np.float32), (height, 1))
    scene = np.dstack([x_ramp, x_ramp * 0.9, x_ramp * 0.8]).astype(np.uint8)
    scene = cv2.GaussianBlur(scene, (0, 0), 25)
    rng = np.random.default_rng(5)
    for _ in range(2):
        center = (int(rng.integers(520, 780)), int(rng.integers(100, 600)))
        cv2.circle(scene, center, int(rng.integers(12, 30)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return cv2.GaussianBlur(scene, (0, 0), 2)


def save_png(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)


def generate_all(output_dir: str) -> dict:
    """Crée tous les jeux de test et renvoie leurs chemins."""
    os.makedirs(output_dir, exist_ok=True)
    scene = create_scene(seed=7)
    other_scene = create_scene(seed=99)

    paths = {}

    # Jeu 1 : deux images avec ~40 % de chevauchement
    two_dir = os.path.join(output_dir, "1_deux_images_ok")
    os.makedirs(two_dir, exist_ok=True)
    views = [extract_view(scene, 0, rotation_deg=0.0, gain=1.0, seed=1),
             extract_view(scene, 480, rotation_deg=1.5, gain=0.92, seed=2)]
    paths["deux_images"] = []
    for i, view in enumerate(views, start=1):
        path = os.path.join(two_dir, f"{i:02d}_vue.png")
        save_png(path, view)
        paths["deux_images"].append(path)

    # Jeu 2 : deux images sans aucune zone commune (deux scènes différentes)
    none_dir = os.path.join(output_dir, "2_sans_zone_commune")
    os.makedirs(none_dir, exist_ok=True)
    paths["sans_zone_commune"] = []
    for i, source in enumerate((scene, other_scene), start=1):
        path = os.path.join(none_dir, f"{i:02d}_vue.png")
        save_png(path, extract_view(source, 100, seed=10 + i))
        paths["sans_zone_commune"].append(path)

    # Jeu 3 : trois images successives
    three_dir = os.path.join(output_dir, "3_trois_images_ok")
    os.makedirs(three_dir, exist_ok=True)
    paths["trois_images"] = []
    for i, (x_start, rotation, gain) in enumerate(((0, 0.0, 1.0), (450, -1.0, 0.95), (900, 1.2, 1.05)), start=1):
        path = os.path.join(three_dir, f"{i:02d}_vue.png")
        save_png(path, extract_view(scene, x_start, rotation_deg=rotation, gain=gain, seed=20 + i))
        paths["trois_images"].append(path)

    # Jeu 4 : image invalide (un fichier texte déguisé en .jpg)
    invalid_dir = os.path.join(output_dir, "4_fichier_invalide")
    os.makedirs(invalid_dir, exist_ok=True)
    invalid_path = os.path.join(invalid_dir, "faux_fichier.jpg")
    with open(invalid_path, "w", encoding="utf-8") as handle:
        handle.write("Ceci n'est pas une image, juste du texte renommé en .jpg")
    valid_path = os.path.join(invalid_dir, "01_vue.png")
    save_png(valid_path, views[0])
    paths["invalide"] = [valid_path, invalid_path]

    # Jeu 5 : trop peu de correspondances (scène quasi uniforme)
    low_dir = os.path.join(output_dir, "5_peu_de_correspondances")
    os.makedirs(low_dir, exist_ok=True)
    flat_scene = create_low_texture_scene()
    paths["peu_de_correspondances"] = []
    for i, x_start in enumerate((0, 500), start=1):
        path = os.path.join(low_dir, f"{i:02d}_vue.png")
        save_png(path, extract_view(flat_scene, x_start, seed=30 + i))
        paths["peu_de_correspondances"].append(path)

    return paths


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    destination = os.path.join(project_root, "examples", "jeux_de_test")
    created = generate_all(destination)
    print(f"Jeux de test créés dans : {destination}")
    for name, files in created.items():
        print(f"  - {name} : {len(files)} fichier(s)")

