# 🏞️ Panorama par appariement de points SIFT


Application web **locale** qui assemble automatiquement plusieurs photographies se chevauchant en un seul
panorama, avec les méthodes classiques de vision par ordinateur (sans deep learning, sans API, sans cloud).

> Projet du module **Computer Vision** — Master Big Data & Cloud Computing.



---

## 📑 Sommaire

1. [Fonctionnalités](#-fonctionnalités)
2. [Comment ça marche](#-comment-ça-marche)
3. [Démarrage rapide](#-démarrage-rapide)
4. [Utiliser l'interface avec vos propres images](#-utiliser-linterface-avec-vos-propres-images)
5. [Tests](#-tests)
6. [Réglages disponibles](#-réglages-disponibles)
7. [Structure du projet](#-structure-du-projet)
8. [Conditions pour un bon résultat](#-conditions-pour-un-bon-résultat)
9. [Limites connues](#-limites-connues)
10. [Pistes d'amélioration](#-pistes-damélioration)

---

## ✨ Fonctionnalités

- Import de **2 images ou plus** directement dans le navigateur (glisser-déposer).
- Visualisation de **chaque étape** : images d'origine, points clés SIFT, correspondances (inliers en vert,
  rejetées en rouge), matrice d'homographie, panorama final.
- **Messages d'erreur explicites** quand l'assemblage est impossible (pas de zone commune, trop peu de
  correspondances, fichier invalide…) : l'application ne plante jamais.
- Choix de l'**image de référence** (centrale ou première) et du **mode de recadrage**.
- Paramètres réglables en direct (ratio de Lowe, seuil RANSAC…) pour observer leur effet.
- Téléchargement du résultat en **PNG** (sans perte) ou **JPEG** (léger).
- Correction automatique de l'**orientation EXIF** des photos de téléphone.

## 🧠 Comment ça marche

```
Photos → Chargement → SIFT → Appariement → Homographie + RANSAC → Assemblage → Panorama
```

| Étape | Méthode | Idée |
|---|---|---|
| 1. Chargement | Pillow + redimensionnement (1000 px max) | Décoder, corriger l'orientation, accélérer les calculs |
| 2. Détection | **SIFT** (`cv2.SIFT_create`) | Repérer des détails invariants à l'échelle, à la rotation et à la luminosité (128 valeurs par point) |
| 3. Appariement | **BFMatcher** KNN (k=2) + **ratio test de Lowe** (0,75) | Retrouver les mêmes détails d'une photo à l'autre en écartant les cas ambigus |
| 4. Géométrie | `findHomography` + **RANSAC** (4 px) | Estimer la déformation entre deux photos en ignorant les fausses correspondances |
| 5. Assemblage | `warpPerspective` + fusion pondérée (*feathering*) | Projeter toutes les images sur une même toile et adoucir les coutures |
| 6. Recadrage | Marges noires / rectangle maximal / aucun | Retirer les coins noirs dus à la perspective |

Une paire d'images n'est acceptée que si elle passe **4 contrôles de fiabilité** : au moins 12 correspondances
après Lowe, au moins 10 inliers, au moins 25 % d'inliers, et une homographie géométriquement plausible.

📘 Explications détaillées (et questions probables de soutenance) :
[`GUIDE_COMPREHENSION_SOUTENANCE.md`](GUIDE_COMPREHENSION_SOUTENANCE.md)

---

## 🚀 Démarrage rapide

Aucun serveur externe n'est nécessaire : tout tourne sur votre ordinateur.

**Prérequis** : Python **3.10 à 3.13** et `git`.
(Testé avec Python 3.12. Les toutes dernières versions de Python peuvent ne pas encore avoir de paquets
OpenCV précompilés.)

### 1. Récupérer le projet

```bash
git clone https://github.com/VOTRE_COMPTE/VOTRE_DEPOT.git
cd VOTRE_DEPOT
```

### 2. Créer et activer un environnement virtuel

<details open>
<summary><b>Windows – PowerShell</b></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
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

Le navigateur s'ouvre sur **http://localhost:8501** (sinon, copiez cette adresse).
`Ctrl + C` dans le terminal arrête l'application.

---

## 🖼️ Utiliser l'interface avec vos propres images

Toute personne qui clone le dépôt peut tester l'application **avec ses propres photos** :

1. Lancez `streamlit run app.py`.
2. Glissez vos images dans la zone d'import (formats courants : JPG, PNG…), **dans l'ordre de gauche à droite**
   — ou cochez l'option de tri par nom dans la barre latérale (ex. `01_gauche.jpg`, `02_milieu.jpg`).
3. Ajustez si besoin les paramètres de la barre latérale.
4. Cliquez sur **Générer le panorama**.
5. Parcourez les onglets, puis téléchargez le résultat. **Réinitialiser** vide tout pour recommencer.

Vos images restent sur votre machine : rien n'est envoyé sur Internet.

### Prendre de bonnes photos

- **Pivotez sur place** (comme sur un trépied) sans vous déplacer entre les prises.
- Photographiez de **gauche à droite**, avec **30 à 50 % de chevauchement** entre deux photos.
- Choisissez une scène riche en détails (bâtiments, enseignes, arbres), pas un ciel ou un mur uniforme.
- Gardez la même orientation, le même zoom et, si possible, la même exposition.
- Évitez les objets qui bougent dans la zone commune.

Plus de conseils : [`exemples/instructions.txt`](exemples/instructions.txt).

---

## 🧪 Tests

### Tests automatiques (sans photos)

```bash
python exemples/generer_jeux_de_test.py   # génère des images synthétiques dans exemples/jeux_de_test/
python tests_automatiques.py              # exécute les scénarios et affiche PASS / FAIL
```

Résultat attendu : tous les tests en `PASS`.

| Scénario | Comportement attendu |
|---|---|
| Deux images qui se chevauchent | Panorama généré |
| Deux images sans zone commune | Message d'erreur clair, pas de panorama |
| Trois images successives | Panorama plus large que chaque image |
| Fichier invalide (texte renommé en `.jpg`) | Erreur « pas une image valide » |
| Une seule image | Erreur « minimum deux images » |
| Trop peu de correspondances | Message indiquant le minimum requis |

### Tests manuels dans l'interface

Après avoir généré les jeux de test, importez dans l'application les images d'un sous-dossier de
`exemples/jeux_de_test/` (par exemple `1_deux_images_ok/`) pour reproduire chaque cas à la main.

> ⚠️ Les images générées sont **synthétiques** : elles servent à vérifier le programme, pas à juger la qualité
> visuelle. Pour cela, utilisez de vraies photos.

---

## ⚙️ Réglages disponibles

| Paramètre | Valeur par défaut | Effet |
|---|---|---|
| Ratio de Lowe | 0,75 | Plus bas → moins de correspondances mais plus sûres |
| Seuil RANSAC | 4 px | Tolérance pour considérer une correspondance comme cohérente |
| Correspondances minimales | 12 | Seuil de fiabilité après le ratio test |
| Inliers minimaux | 10 | Seuil de fiabilité après RANSAC |
| Part minimale d'inliers | 25 % | Rejette les appariements dus au hasard |
| Image de référence | Centrale | Répartit les déformations (la première image est aussi proposée) |
| Recadrage | Retirer les marges noires | Ou rectangle maximal, ou aucun |

Les valeurs par défaut sont définies dans `PanoramaParams` (`panorama.py`).

---

## 📁 Structure du projet

```
.
├── app.py                              Interface Streamlit (aucun algorithme ici)
├── panorama.py                         Moteur : SIFT, appariement, RANSAC, assemblage
├── requirements.txt                    Dépendances
├── tests_automatiques.py               Scénarios de test automatiques
├── GUIDE_COMPREHENSION_SOUTENANCE.md   Explications pédagogiques + préparation de la soutenance
├── README.md
└── exemples/
    ├── generer_jeux_de_test.py         Génère des images synthétiques de test
    ├── instructions.txt                Conseils pour prendre de bonnes photos
    └── __init__.py
```

`panorama.py` est indépendant de l'interface : il peut être testé et réutilisé sans Streamlit.

## ✅ Conditions pour un bon résultat

- Chevauchement de 30 à 50 % entre photos successives.
- Scène avec des détails distinctifs.
- Appareil qui pivote sur place, exposition constante.
- Peu ou pas de personnes et de véhicules en mouvement dans la zone commune.

## ⚠️ Limites connues

- **Parallaxe** : l'homographie suppose une scène plane ou lointaine, ou un appareil qui tourne autour de son
  centre. Sinon : doublons et décalages.
- **Ordre des images** fourni par l'utilisateur ; il n'est pas deviné automatiquement.
- **Fusion simple** : pas de compensation d'exposition, des différences de luminosité peuvent rester visibles ;
  les objets en mouvement créent des « fantômes ».
- **Très grands champs de vision** (plus de ~120°) : la projection plane étire fortement les bords.
- **Erreurs cumulées** avec beaucoup d'images (pas d'optimisation globale).

## 🔭 Pistes d'amélioration

- Ajustement de faisceaux (*bundle adjustment*) pour une estimation globale des positions.
- Projection cylindrique ou sphérique pour les grands champs de vision.
- Compensation d'exposition et fusion multi-bandes.
- Recherche d'une couture optimale entre images.
- Détection automatique de l'ordre des images (comparaison de toutes les paires).
- Accélération de l'appariement avec FLANN.

## 🛠️ Technologies

Python · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Pillow](https://python-pillow.org/) ·
[Streamlit](https://streamlit.io/)

Référence : D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV, 2004.

## 👤 Auteur

ABBA Badereddine
AGOUZOUL Mohamed Reda
BAROUDI Yassine
Bity Hiba
ELKHLIFI Hiba


Master Big Data & Cloud Computing