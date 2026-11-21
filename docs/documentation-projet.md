# Documentation du projet Panorama SIFT

## Présentation

Panorama SIFT est une application web locale qui assemble plusieurs photographies d'une même scène en une image
panoramique. Elle repose uniquement sur des méthodes classiques de vision par ordinateur et ne transmet aucune image
à un service externe.

Le projet utilise :

- Streamlit pour l'interface web ;
- Pillow pour le chargement et la correction EXIF ;
- OpenCV pour SIFT, l'appariement, RANSAC et les transformations géométriques ;
- NumPy pour les calculs matriciels ;
- pytest pour les tests automatisés.

## Architecture

```text
.
├── app.py
├── panorama_sift/
│   ├── __init__.py
│   ├── models.py
│   ├── image_processing.py
│   ├── matching.py
│   ├── stitching.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── scripts/
│   └── generate_test_images.py
├── examples/
├── docs/
│   └── documentation-projet.md
├── README.md
├── LICENSE
└── requirements.txt
```

### Interface

Le fichier `app.py` contient uniquement la couche de présentation :

- importation des fichiers ;
- configuration des paramètres ;
- aperçu des images ;
- suivi de la progression ;
- affichage des étapes et des erreurs ;
- téléchargement du résultat.

Les algorithmes ne dépendent pas de Streamlit et peuvent être utilisés séparément.

### Modèles

`panorama_sift/models.py` définit les objets échangés entre les modules :

- `PanoramaParams` : paramètres du traitement ;
- `PreparedImage` : image chargée et redimensionnée ;
- `FeatureSet` : points clés et descripteurs SIFT ;
- `PairResult` : résultat de l'alignement d'une paire ;
- `PipelineResult` : résultat complet du pipeline ;
- `PanoramaError` et `HomographyError` : erreurs métier.

### Traitement des images

`panorama_sift/image_processing.py` assure :

1. la validation du fichier ;
2. la correction de l'orientation EXIF ;
3. la conversion vers le format BGR d'OpenCV ;
4. le redimensionnement avec conservation des proportions ;
5. la détection des points clés SIFT ;
6. l'encodage du résultat en PNG ou JPEG.

### Appariement

`panorama_sift/matching.py` compare chaque paire d'images successives :

1. BFMatcher recherche les deux descripteurs les plus proches ;
2. le ratio de Lowe élimine les associations ambiguës ;
3. RANSAC estime une homographie robuste ;
4. les inliers et les correspondances rejetées sont identifiés ;
5. la plausibilité géométrique de la transformation est contrôlée.

Une paire est refusée si elle ne possède pas assez de correspondances ou si son homographie est incohérente.

### Assemblage

`panorama_sift/stitching.py` réalise l'assemblage final :

1. composition des homographies vers l'image de référence ;
2. calcul des dimensions du canevas ;
3. translation pour éviter les coordonnées négatives ;
4. projection avec `warpPerspective` ;
5. fusion pondérée des zones communes ;
6. recadrage selon le mode sélectionné.

### Pipeline

`panorama_sift/pipeline.py` orchestre les modules précédents et renvoie toujours un `PipelineResult`.
Les erreurs prévisibles sont converties en messages compréhensibles afin que l'interface puisse présenter les
résultats intermédiaires sans interrompre brutalement l'application.

## Flux de traitement

```text
Fichiers importés
      ↓
Validation et préparation
      ↓
Détection SIFT
      ↓
Appariement BFMatcher + ratio de Lowe
      ↓
Homographie + RANSAC
      ↓
Composition des transformations
      ↓
Projection et fusion
      ↓
Recadrage et export
```

## Paramètres

| Paramètre | Valeur par défaut | Description |
|---|---:|---|
| `max_side` | 1000 px | Dimension maximale après redimensionnement |
| `lowe_ratio` | 0,75 | Seuil de sélection des correspondances |
| `ransac_threshold` | 4 px | Tolérance de reprojection RANSAC |
| `min_good_matches` | 12 | Correspondances minimales après Lowe |
| `min_inliers` | 10 | Inliers minimaux après RANSAC |
| `min_inlier_ratio` | 25 % | Proportion minimale d'inliers |
| `reference` | centre | Image utilisée comme repère |
| `crop_mode` | bbox | Mode de recadrage |
| `max_canvas_pixels` | 25 000 000 | Limite de sécurité mémoire |

## Tests

La suite de tests vérifie :

- l'assemblage de deux images ;
- l'assemblage de trois images ;
- les deux choix d'image de référence ;
- le recadrage intérieur ;
- le rejet de scènes différentes ;
- le rejet d'un fichier invalide ;
- le rejet d'une seule image ;
- le rejet d'images contenant trop peu de correspondances.

Exécution :

```bash
pytest
```

Les données synthétiques peuvent être générées séparément :

```bash
python scripts/generate_test_images.py
```

## Gestion des erreurs

Les erreurs suivantes sont traitées explicitement :

- fichier vide ou format non reconnu ;
- image trop petite ;
- absence de points SIFT exploitables ;
- nombre insuffisant de correspondances ;
- nombre ou ratio d'inliers insuffisant ;
- homographie géométriquement incohérente ;
- panorama vide ou démesuré ;
- erreur interne OpenCV ;
- mémoire insuffisante.

## Limites techniques

- Les images doivent être fournies dans leur ordre spatial.
- Chaque image doit chevaucher l'image suivante.
- Une homographie représente mal les scènes présentant une forte parallaxe.
- Les objets en mouvement peuvent produire des images fantômes.
- La fusion ne compense pas complètement les différences d'exposition.
- Les panoramas très larges peuvent subir une déformation aux extrémités.
- Les erreurs peuvent s'accumuler lorsque le nombre d'images augmente.

## Extensions possibles

- classement automatique des images ;
- projection cylindrique ou sphérique ;
- compensation automatique de l'exposition ;
- détection optimisée des coutures ;
- fusion multibande ;
- ajustement global des transformations ;
- prise en charge de très grandes images par traitement en tuiles.

## Maintenance

Lors d'une modification du moteur :

1. conserver l'indépendance entre l'interface et les algorithmes ;
2. ajouter ou adapter un test ;
3. exécuter `pytest` ;
4. vérifier manuellement au moins un panorama réussi et un cas d'erreur ;
5. mettre à jour cette documentation si l'architecture ou les paramètres changent.

