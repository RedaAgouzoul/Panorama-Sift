# 🏞️ Panorama par appariement de points SIFT

Application web **locale** qui assemble automatiquement plusieurs photographies se chevauchant en un seul
panorama, avec les méthodes classiques de vision par ordinateur (sans deep learning, sans API, sans cloud).

> Projet du module **Computer Vision** — Master Big Data & Cloud Computing.

---

## Sommaire

1. [Fonctionnalités](#fonctionnalités)
2. [Comment ça marche](#comment-ça-marche)
3. [Démarrage rapide](#démarrage-rapide)
4. [Utiliser l'interface avec vos propres images](#utiliser-linterface-avec-vos-propres-images)
5. [Tests](#tests)
6. [Réglages disponibles](#réglages-disponibles)
7. [Structure du projet](#structure-du-projet)
8. [Limites connues](#limites-connues)
9. [Pistes d'amélioration](#pistes-damélioration)
10. [Dépannage](#dépannage)

---

## Fonctionnalités

- Import de **2 images ou plus** directement dans le navigateur (glisser-déposer).
- Visualisation de **chaque étape** : images d'origine, points clés SIFT, correspondances (pour une paire
  acceptée : inliers en vert, correspondances rejetées par RANSAC en rouge), matrice d'homographie, panorama final.
- **Messages d'erreur explicites** quand l'assemblage est impossible (pas de zone commune, trop peu de
  correspondances, fichier invalide…) : l'application ne plante jamais. Les onglets restent consultables après un
  échec pour comprendre où le traitement a bloqué.
- Choix de l'**image de référence** (centrale ou première) et du **mode de recadrage**.
- Paramètres réglables en direct dans la barre latérale (taille des images, ratio de Lowe, seuil RANSAC…) pour
  observer leur effet.
- Téléchargement du résultat en **PNG** (sans perte) ou **JPEG** (léger).
- Correction automatique de l'**orientation EXIF** des photos de téléphone.

## Comment ça marche

```
Photos → Chargement → SIFT → Appariement → Homographie + RANSAC → Assemblage → Panorama
```

| Étape | Méthode | Idée |
|---|---|---|
| 1. Chargement | Pillow + redimensionnement (1000 px max par défaut) | Décoder, corriger l'orientation, accélérer les calculs |
| 2. Détection | **SIFT** (`cv2.SIFT_create`) | Repérer des détails invariants à l'échelle, à la rotation et en partie à la luminosité (128 valeurs par point) |
| 3. Appariement | **BFMatcher** KNN (k=2) + **ratio test de Lowe** (0,75) | Retrouver les mêmes détails d'une photo à l'autre en écartant les cas ambigus |
| 4. Géométrie | `findHomography` + **RANSAC** (4 px) | Estimer la déformation entre deux photos en ignorant les fausses correspondances |
| 5. Assemblage | `warpPerspective` + fusion pondérée (*feathering*) | Projeter toutes les images sur une même toile et adoucir les coutures |
| 6. Recadrage | Marges noires / rectangle maximal / aucun | Retirer les coins noirs dus à la perspective |

Une paire d'images n'est acceptée que si elle passe **4 contrôles de fiabilité** : au moins 12 correspondances
après Lowe, au moins 10 inliers, au moins 25 % d'inliers, et une homographie géométriquement plausible.

📘 Explications détaillées (et 14 questions probables de soutenance) :
[`GUIDE_COMPREHENSION_SOUTENANCE.md`](GUIDE_COMPREHENSION_SOUTENANCE.md)

---

## Démarrage rapide

Aucun serveur externe n'est nécessaire : tout tourne sur votre ordinateur.

**Prérequis** : Python **3.10 à 3.13** et `git` (uniquement pour cloner le dépôt).
Testé avec Python 3.12 (numpy 2.4, OpenCV 4.13, Pillow 12.1, Streamlit 1.66, sous Linux). Les toutes dernières
versions de Python peuvent ne pas encore avoir de paquets OpenCV précompilés.

### 1. Récupérer le projet

```bash
git clone https://github.com/RedaAgouzoul/Panorama-Sift.git
cd Panorama-Sift
```

### 2. Créer et activer un environnement virtuel

<details open>
<summary><b>Windows – PowerShell</b></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Si l'activation est refusée (« l'exécution de scripts est désactivée »), lancez une fois
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utilisez `venv\Scripts\activate.bat` dans `cmd`.
</details>

<details>
<summary><b>Windows – Git Bash</b></summary>

```bash
python -m venv venv
source venv/Scripts/activate
```
</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```
</details>

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancer l'interface

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur **http://localhost:8501** (sinon, copiez cette adresse). Au premier lancement,
Streamlit peut demander un e-mail : appuyez simplement sur Entrée.

`Ctrl + C` dans le terminal arrête l'application. Pour la relancer : `cd` dans le dossier du projet, réactivez
l'environnement virtuel (étape 2), puis `streamlit run app.py`.

---

## Utiliser l'interface avec vos propres images

Toute personne qui clone le dépôt peut tester l'application **avec ses propres photos** :

1. Lancez `streamlit run app.py`.
2. Glissez vos images dans la zone d'import, **dans l'ordre de gauche à droite**
   — ou cochez « Trier les images par nom de fichier » dans la barre latérale (ex. `01_gauche.jpg`,
   `02_milieu.jpg`).
   Formats acceptés : **PNG, JPG/JPEG, BMP, WEBP, TIF/TIFF**. Le format **HEIC** (iPhone) n'est pas pris en
   charge : réglez l'appareil sur le format « le plus compatible » ou exportez les photos en JPEG.
3. Ajustez si besoin les paramètres de la barre latérale.
4. Cliquez sur **Générer le panorama**.
5. Parcourez les onglets, puis téléchargez le résultat. **Réinitialiser** vide tout pour recommencer.

Vos images restent sur votre machine : rien n'est envoyé sur Internet.

### Prendre de bonnes photos

- **Pivotez sur place** (comme sur un trépied) sans vous déplacer entre les prises.
- Photographiez de **gauche à droite**, avec **30 à 50 % de chevauchement** entre deux photos.
- Choisissez une scène riche en détails (bâtiments, enseignes, arbres), pas un ciel ou un mur uniforme.
- Gardez la même orientation, le même zoom et, si possible, la même exposition.
- Évitez les objets qui bougent (personnes, véhicules) dans la zone commune.

Plus de conseils : [`exemples/instructions.txt`](exemples/instructions.txt).

---

## Tests

### Tests automatiques (sans photos)

```bash
python exemples/generer_jeux_de_test.py   # génère des images synthétiques dans exemples/jeux_de_test/
python tests_automatiques.py              # exécute les vérifications et affiche PASS / FAIL
```

Résultat attendu : `7/7 tests réussis`. Le script écrit aussi deux fichiers `_resultat_test_*.png` dans le
dossier courant (ils peuvent être supprimés).

| Scénario | Comportement attendu |
|---|---|
| Deux images qui se chevauchent | Panorama généré |
| Deux images sans zone commune | Message d'erreur clair, pas de panorama |
| Trois images successives | Panorama plus large que chaque image |
| Trois images, référence = 1re image, recadrage intérieur | Panorama généré |
| Fichier invalide (texte renommé en `.jpg`) | Erreur « pas une image valide » |
| Une seule image | Erreur « minimum deux images » |
| Trop peu de correspondances | Message indiquant le minimum requis (12) |

### Tests manuels dans l'interface

Après avoir généré les jeux de test, importez dans l'application les images d'un sous-dossier de
`exemples/jeux_de_test/` (par exemple `1_deux_images_ok/`) pour reproduire chaque cas à la main.

> ⚠️ Les images générées sont **synthétiques** : elles servent à vérifier le programme, pas à juger la qualité
> visuelle.

### Tests avec de vraies photos

Le dossier `exemples/photos_reelles/` contient de vraies photos (source et résultats attendus dans
`exemples/photos_reelles/SOURCES.txt`). Importez les images d'un sous-dossier à la fois :

| Dossier | Résultat attendu |
|---|---|
| `1_deux_photos_ok/` | Succès (environ 500 inliers) |
| `2_trois_photos_ok/` | Succès, 2 paires, panorama plus large |
| `3_six_photos_limite/` | Succès mais très déformé : illustre les limites de la projection plane |
| `4_chevauchement_reduit/` | Succès avec un chevauchement plus faible (1re + 3e photo) |
| `5_scenes_differentes_erreur/` | Erreur : trop peu de correspondances fiables |
| `6_autre_scene/` | Succès |

---

## Réglages disponibles

**Dans la barre latérale de l'interface :**

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| Taille maximale des images | 1000 px (500 à 2000) | Plus grand → plus de détails, mais plus lent |
| Ratio de Lowe | 0,75 (0,60 à 0,90) | Plus bas → moins de correspondances mais plus sûres |
| Seuil RANSAC | 4 px (1 à 10) | Tolérance pour considérer une correspondance comme cohérente |
| Image de référence | Centrale | Répartit les déformations (la première image est aussi proposée) |
| Recadrage | Retirer les marges noires | Ou rectangle maximal, ou aucun |
| Tri par nom de fichier | Désactivé | Ordonne les images par nom au lieu de l'ordre d'import |

**Dans le code uniquement** (classe `PanoramaParams`, `panorama.py`) :

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| `min_good_matches` | 12 | Minimum de correspondances après le ratio test |
| `min_inliers` | 10 | Minimum d'inliers après RANSAC |
| `min_inlier_ratio` | 25 % | Rejette les appariements dus au hasard |
| `max_canvas_pixels` | 25 000 000 | Garde-fou mémoire sur la taille du panorama |

---

## Structure du projet

```
.
├── app.py                              Interface Streamlit (aucun algorithme ici)
├── panorama.py                         Moteur : SIFT, appariement, RANSAC, assemblage
├── requirements.txt                    Dépendances
├── tests_automatiques.py               Scénarios de test automatiques
├── GUIDE_COMPREHENSION_SOUTENANCE.md   Explications pédagogiques + préparation de la soutenance
├── README.md
└── exemples/
    ├── __init__.py
    ├── generer_jeux_de_test.py         Génère des images synthétiques de test
    ├── instructions.txt                Conseils pour prendre de bonnes photos
    └── photos_reelles/                 Vraies photos de test + SOURCES.txt
```

`panorama.py` est indépendant de l'interface : il peut être testé et réutilisé sans Streamlit.

## Limites connues

- **Parallaxe** : l'homographie suppose une scène plane ou lointaine, ou un appareil qui tourne autour de son
  centre. Sinon : doublons et décalages.
- **Ordre des images** fourni par l'utilisateur ; il n'est pas deviné automatiquement.
- **Fusion simple** : pas de compensation d'exposition, des différences de luminosité peuvent rester visibles ;
  les objets en mouvement créent des « fantômes ».
- **Très grands champs de vision** (plus de ~120°) : la projection plane étire fortement les bords.
- **Erreurs cumulées** avec beaucoup d'images (pas d'optimisation globale).
- **Paire refusée** : les correspondances affichées dans l'onglet dédié ne distinguent pas inliers et
  correspondances rejetées (tous les traits sont verts).

## Pistes d'amélioration

- Ajustement de faisceaux (*bundle adjustment*) pour une estimation globale des positions.
- Projection cylindrique ou sphérique pour les grands champs de vision.
- Compensation d'exposition et fusion multi-bandes.
- Recherche d'une couture optimale entre images.
- Détection automatique de l'ordre des images (comparaison de toutes les paires).
- Accélération de l'appariement avec FLANN.
- Support du format HEIC (`pillow-heif`).

## Dépannage

| Problème | Solution |
|---|---|
| `python` n'est pas reconnu | Réinstallez Python depuis python.org en cochant « Add Python to PATH » |
| « l'exécution de scripts est désactivée » à l'activation | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou `venv\Scripts\activate.bat` dans `cmd` |
| `numpy.core.multiarray failed to import` | `pip install -U numpy opencv-python` |
| `cv2.SIFT_create` introuvable ou conflit OpenCV | `pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless`, puis `pip install opencv-python` |
| `DLL load failed` (Windows) | Installez le « Microsoft Visual C++ Redistributable » |
| Le port 8501 est déjà utilisé | `streamlit run app.py --server.port 8502` |
| Une photo d'iPhone est refusée | Format HEIC non pris en charge : exportez en JPEG |
| Panorama déformé ou refusé | Vérifiez l'ordre (gauche → droite), le chevauchement (30 à 50 %) et que vous avez pivoté sans vous déplacer |

## Technologies

Python · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Pillow](https://python-pillow.org/) ·
[Streamlit](https://streamlit.io/)

Référence : D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV, 2004.

## Auteurs

- ABBA Badereddine
- AGOUZOUL Mohamed Reda
- BAROUDI Yassine
- BITY Hiba
- ELKHALFI Imane

Master Big Data & Cloud Computing
