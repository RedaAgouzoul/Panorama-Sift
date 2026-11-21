# Panorama SIFT

Application web locale qui assemble plusieurs photographies superposées en un panorama avec des méthodes classiques
de vision par ordinateur : SIFT, BFMatcher, ratio de Lowe, homographie, RANSAC et fusion pondérée.

## Fonctionnalités

- Import de deux images ou plus avec Streamlit.
- Correction de l'orientation EXIF et redimensionnement automatique.
- Visualisation des points clés, correspondances, inliers et homographies.
- Choix de l'image de référence et du mode de recadrage.
- Messages d'erreur explicites et téléchargement en PNG ou JPEG.
- Traitement entièrement local, sans API ni cloud.

## Installation

Prérequis : Python 3.10 à 3.13.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Importez les images dans l'ordre de gauche à droite, avec idéalement 30 à 50 % de chevauchement.

## Tests

```bash
pytest
```

Les sept tests couvrent les panoramas de deux et trois images ainsi que les principales erreurs utilisateur. Pour
générer manuellement les images synthétiques :

```bash
python scripts/generate_test_images.py
```

## Structure

```text
.
├── app.py
├── panorama_sift/
│   ├── models.py
│   ├── image_processing.py
│   ├── matching.py
│   ├── stitching.py
│   └── pipeline.py
├── tests/test_pipeline.py
├── scripts/generate_test_images.py
├── examples/
├── docs/documentation-projet.md
├── LICENSE
└── requirements.txt
```

La [documentation complète du projet](docs/documentation-projet.md) décrit l'architecture, les algorithmes, les
modules, le pipeline, les tests, les limites et la maintenance.

## Membres du groupe

- ABBA Badereddine
- AGOUZOUL Mohamed Reda
- BAROUDI Yassine
- BITY Hiba
- ELKHALFI Imane

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).
