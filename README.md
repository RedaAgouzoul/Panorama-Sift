# Panorama d'images par appariement de caractéristiques SIFT

Application locale (Python + OpenCV + Streamlit) qui assemble plusieurs photographies
se chevauchant en un panorama, avec les méthodes classiques de vision par ordinateur :

**SIFT** → **BFMatcher (KNN) + ratio test de Lowe** → **homographie + RANSAC** → **fusion des images**

Aucun serveur externe, aucune API payante, aucun cloud : tout tourne sur votre ordinateur.

## Structure du projet

```
panorama_sift/
├── app.py                         Interface Streamlit (aucun algorithme ici)
├── panorama.py                    Tous les algorithmes (SIFT, appariement, RANSAC, assemblage)
├── requirements.txt               Dépendances
├── README.md                      Ce fichier
├── GUIDE_COMPREHENSION_SOUTENANCE.md   Explications pédagogiques + préparation de la soutenance
├── tests_automatiques.py          Vérifie les 5 scénarios de test d'un seul coup
└── exemples/
    ├── instructions.txt           Comment préparer de bonnes photos de test
    ├── generer_jeux_de_test.py    Génère des images SYNTHÉTIQUES de test
    └── __init__.py                (fichier vide, nécessaire pour l'import)
```

Pourquoi deux fichiers principaux ? `panorama.py` peut être testé sans interface, et `app.py`
reste court et lisible. Les deux ajouts (`tests_automatiques.py`, `generer_jeux_de_test.py`)
permettent de tester sans avoir de photos sous la main.

## Installation et lancement (Windows, PowerShell)

```powershell
# 1. Vérifier Python (3.10 ou plus recommandé)
python --version

# 2. Se placer dans le dossier du projet (adaptez le chemin)
cd C:\Users\VOTRE_NOM\Documents\panorama_sift

# 3. Créer l'environnement virtuel (une seule fois)
python -m venv venv

# 4. Activer l'environnement virtuel (à refaire à chaque nouveau terminal)
.\venv\Scripts\Activate.ps1

# 5. Installer les dépendances (une seule fois)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Lancer l'application
streamlit run app.py
```

Le navigateur s'ouvre automatiquement sur **http://localhost:8501**
(sinon, copiez cette adresse dans votre navigateur).

- **Arrêter** l'application : dans le terminal, `Ctrl + C`.
- **Relancer plus tard** : ouvrez PowerShell, `cd` vers le dossier, réactivez
  l'environnement (étape 4), puis `streamlit run app.py`.

## Utilisation

1. Importez 2 images ou plus, **dans l'ordre de gauche à droite**.
2. Cliquez sur **Générer le panorama**.
3. Parcourez les onglets : images originales, points clés, correspondances,
   homographie, panorama final.
4. Téléchargez le résultat en PNG ou JPEG. Le bouton **Réinitialiser** vide tout.

## Tester sans photos

```powershell
python exemples\generer_jeux_de_test.py   # crée exemples\jeux_de_test\ (images synthétiques)
python tests_automatiques.py              # lance les 5 scénarios de test
```

Pour des tests réalistes (et pour la soutenance), utilisez de vraies photos :
voir `exemples/instructions.txt`.

## Conditions pour que ça marche bien

- Chevauchement de **30 à 50 %** entre deux photos successives.
- Scène avec des détails distinctifs (bâtiments, panneaux, textures), pas un ciel ou un mur uniforme.
- Appareil qui **pivote sur place** (ne pas se déplacer entre les photos), exposition constante.
- Pas (ou peu) de personnes ou de véhicules en mouvement dans la zone commune.

## Limites connues

- La méthode suppose une scène plane ou lointaine, ou un appareil qui tourne autour de son centre :
  sinon l'effet de **parallaxe** crée des doublons et des décalages.
- Les images doivent être fournies **dans l'ordre** ; l'ordre n'est pas deviné automatiquement.
- Fusion simple (pondération par la distance au bord) : pas de correction d'exposition, des
  différences de luminosité peuvent rester visibles ; les objets en mouvement donnent des « fantômes ».
- Panoramas très larges (plus de ~120° de rotation) : la projection par homographie étire les bords.
- Les erreurs s'accumulent légèrement quand le nombre d'images augmente (pas d'optimisation globale).

## Versions testées

Python 3.12 · numpy 2.4 · opencv 4.13 · Pillow 12.1 · streamlit 1.66 (testé sous Linux).
