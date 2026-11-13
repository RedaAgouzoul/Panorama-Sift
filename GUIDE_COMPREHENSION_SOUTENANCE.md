# Guide de compréhension et préparation de la soutenance

Ce guide explique le projet en français simple, puis prépare la démonstration.
Les noms entre `code` sont de vraies fonctions de `panorama.py`.

---

# PARTIE A — Comprendre le fonctionnement

## Vue d'ensemble en une phrase

> On repère dans chaque photo des **détails reconnaissables** (SIFT), on **retrouve les mêmes détails** d'une
> photo à l'autre (appariement), on en déduit **comment déformer** une photo pour la superposer à l'autre
> (homographie + RANSAC), puis on **colle et fusionne** le tout sur une grande toile (panorama).

```
Photos → [1 Chargement] → [2 SIFT] → [3 Appariement] → [4 Homographie+RANSAC] → [5 Assemblage] → Panorama
          prepare_images   detect_sift_   match_descriptors   estimate_homography   assemble_panorama
                           features
```

## 1. Qu'est-ce qu'un point clé SIFT ?

SIFT (*Scale-Invariant Feature Transform*, D. Lowe, 2004) repère des **zones de l'image faciles à
reconnaître** : coins, taches, motifs. Sa force : il retrouve le même détail même si la photo est un peu
**agrandie, tournée ou plus claire/sombre**. Un point clé contient :

- sa **position** (x, y) ;
- son **échelle** (la taille du détail) ;
- son **orientation** (la direction dominante autour du point) ;
- une **réponse** (sa « force »).

`detectAndCompute(image_gris, None)` renvoie **deux résultats** :

1. `keypoints` : la liste des points clés ;
2. `descriptors` : un tableau de N lignes × 128 colonnes (une ligne par point clé).

## 2. Que contient un descripteur de 128 nombres ?

C'est la « carte d'identité » du voisinage d'un point clé. SIFT prend une fenêtre de 16×16 pixels autour du
point (orientée selon l'orientation du point), la découpe en **4 × 4 = 16 cases**, et pour chaque case calcule
un petit histogramme de **8 directions** de variation de luminosité (gradients).
**16 cases × 8 directions = 128 nombres.** Deux points qui représentent le même détail ont des descripteurs
**proches** (distance euclidienne faible) ; des détails différents ont des descripteurs **éloignés**.

## 3. Pourquoi BFMatcher et le ratio test de Lowe ?

- **BFMatcher** (*Brute-Force*) compare chaque descripteur de l'image de gauche à **tous** ceux de l'image de
  droite. C'est simple, exact et suffisant pour quelques milliers de points. On utilise la norme `NORM_L2`
  (distance euclidienne), adaptée aux descripteurs SIFT (qui sont des nombres décimaux).
- **KNN avec k=2** : pour chaque point, on garde ses **2 meilleurs candidats**.
- **Ratio test de Lowe** : on accepte le meilleur candidat seulement si
  `distance(1er) < 0,75 × distance(2e)`. Si les deux candidats sont presque aussi bons, le point est **ambigu**
  (ex. une fenêtre parmi 20 fenêtres identiques) : mieux vaut l'écarter. Lowe a montré que ce test élimine
  la grande majorité des fausses correspondances en perdant très peu de bonnes.

## 4. Qu'est-ce qu'une homographie ?

Une **matrice 3×3** (8 paramètres libres) qui transforme les coordonnées d'une image vers celles d'une autre :
elle combine translation, rotation, changement d'échelle **et effet de perspective**. Elle décrit exactement
le lien entre deux photos si la scène est **plane**, **lointaine**, ou si l'appareil **tourne sur place**
(c'est le cas d'un panorama bien pris).

On l'applique à un point (x, y) ainsi : on calcule `[x', y', w] = H · [x, y, 1]`, puis la position finale est
`(x'/w, y'/w)` — la division par `w` produit l'effet de perspective.

## 5. Le rôle de RANSAC

Parmi les correspondances conservées, quelques-unes restent **fausses** (des « outliers »). Si on calculait
l'homographie avec toutes, elles la fausseraient. **RANSAC** (*RANdom SAmple Consensus*) :

1. tire au hasard **4 correspondances** (le minimum pour définir une homographie) ;
2. calcule une homographie candidate ;
3. compte combien d'**autres** correspondances sont d'accord (leur point transformé tombe à moins de
   `4 pixels` de son vrai partenaire) : ce sont les **inliers** ;
4. recommence de nombreuses fois et **garde la candidate qui a le plus d'inliers**, puis la réaffine avec eux.

Résultat : les mauvaises correspondances sont **ignorées** sans avoir eu à les identifier une par une.
`cv2.findHomography(src, dst, cv2.RANSAC, 4.0)` renvoie la matrice **et** un masque indiquant les inliers.

## 6. Comment `warpPerspective()` transforme une image

`cv2.warpPerspective(image, M, (largeur, hauteur))` fabrique une nouvelle image de la taille demandée.
Pour **chaque pixel de la nouvelle image**, OpenCV utilise l'inverse de `M` pour savoir de quel endroit
de l'image d'origine il vient, puis lit la valeur par **interpolation bilinéaire** (moyenne pondérée des 4 pixels
voisins). Parcourir la destination (et non la source) évite les trous dans l'image obtenue.
Les zones que l'image ne couvre pas restent noires.

## 7. Comment le canevas du panorama est calculé (`compute_canvas`)

1. On prend les **4 coins** de chaque image, qu'on transforme avec son homographie vers le repère de référence.
2. On cherche le plus petit rectangle qui contient **tous** les coins transformés : `x_min, y_min, x_max, y_max`.
3. Les coordonnées peuvent être **négatives** (une image qui déborde à gauche ou en haut de l'image de
   référence). On construit une **matrice de translation** `[[1,0,-x_min],[0,1,-y_min],[0,0,1]]` qui décale
   tout pour que le coin supérieur gauche devienne (0, 0).
4. Largeur = `x_max − x_min`, hauteur = `y_max − y_min`. Chaque image est ensuite projetée avec
   `translation @ homographie`.

## 8. Comment les images sont fusionnées (`blend_images`)

Dans une zone de chevauchement, copier simplement une image sur l'autre ferait apparaître une **couture**
nette. On utilise une fusion **pondérée par la distance au bord** (*feathering*) : pour chaque image, un
poids = distance du pixel au bord de la zone couverte (`cv2.distanceTransform`). Au centre d'une image, son
poids est fort ; près de son bord, il tend vers 0. Le pixel final est la **moyenne pondérée** des images qui le
couvrent : la transition est progressive.

## 9. Stratégie pour plusieurs images (`compute_homographies_to_reference`)

1. On calcule une homographie pour chaque **paire successive** (1→2, 2→3, …).
2. On choisit une **image de référence** (par défaut l'image **centrale**, car cela répartit les déformations ;
   la première image est aussi proposée).
3. On **compose** les matrices pour ramener chaque image dans le repère de la référence. Exemple avec 3 images
   et la référence = image 2 : `H₁→réf = H₁→₂` ; `H₃→réf = (H₂→₃)⁻¹`.
4. On construit **un seul canevas** et on y projette toutes les images en une fois (une seule interpolation
   par image : pas de flou accumulé).

C'est une approche progressive et simple, sans optimisation globale : les petites erreurs de chaque paire
s'additionnent un peu quand le nombre d'images augmente.

## 10. Recadrage (`crop_panorama`)

Après perspective, le panorama a des **coins noirs**. Trois modes :
- **Retirer les marges noires** (défaut) : rectangle englobant du contenu → on ne coupe **aucune** partie valide ;
  quelques coins noirs peuvent rester.
- **Rectangle maximal** : plus grand rectangle sans aucun pixel noir → plus propre, mais coupe du contenu.
- **Aucun** : le canevas brut.

## 11. Les fonctions et paramètres importants

| Élément | Rôle |
|---|---|
| `load_image_from_bytes` | Décode via Pillow (corrige l'orientation EXIF), passe en BGR pour OpenCV |
| `resize_keep_ratio` | Réduit à 1000 px max (`INTER_AREA`) : plus rapide, ratio conservé |
| `detect_sift_features` | `cv2.SIFT_create()` + `detectAndCompute` |
| `match_descriptors` | `BFMatcher(NORM_L2).knnMatch(k=2)` + ratio de Lowe (**0,75**) |
| `estimate_homography` | `findHomography(RANSAC, 4,0 px)` + 4 contrôles de fiabilité (voir ci-dessous) |
| `is_homography_plausible` | Rejette les homographies absurdes (quadrilatère non convexe, aire ×10…) |
| `compute_canvas` | Taille du panorama + translation |
| `blend_images` | `warpPerspective` + fusion pondérée |
| `crop_panorama` | Recadrage |
| `run_pipeline` | Enchaîne tout et rapporte les erreurs sans planter |

**Contrôles de fiabilité d'une paire** : ≥ 12 correspondances après Lowe ; ≥ 10 inliers ; ≥ 25 % d'inliers
parmi les correspondances ; homographie plausible. Ils sont réglables dans `PanoramaParams`.
*(Le contrôle du ratio d'inliers a été ajouté pendant les tests : deux scènes différentes mais contenant des
lettres similaires donnaient 61 correspondances « fiables » dont 11 inliers par hasard.)*

## 12. Limites de la méthode

- **Parallaxe** : si l'appareil se déplace ou si la scène a du relief proche, une seule homographie ne suffit
  pas → doublons, décalages.
- **Objets en mouvement** → fantômes dans les zones fusionnées.
- **Exposition différente** entre photos → transitions visibles (pas de compensation d'exposition).
- **Très grands champs de vision** : la projection plane étire fortement les bords.
- **Ordre à fournir** par l'utilisateur ; scènes pauvres en détails ou très répétitives (façades régulières) →
  appariements ambigus.
- **Erreurs cumulées** avec beaucoup d'images (pas d'ajustement global).

---

# PARTIE B — Préparer la soutenance

## 1. Le problème

Un appareil photo a un champ de vision limité. Pour voir une scène plus large, on prend plusieurs photos qui se
chevauchent ; il faut les **assembler automatiquement** en corrigeant leurs différences de position et de
perspective.

## 2. Objectifs

Détecter des points caractéristiques invariants (SIFT) → les apparier de façon fiable → estimer la
transformation géométrique robuste aux erreurs (RANSAC) → produire un panorama cohérent, dans une application
locale simple à utiliser.

## 3. Architecture

- `panorama.py` = « moteur » (algorithmes, sans interface) ;
- `app.py` = interface Streamlit (import, paramètres, onglets de résultats, téléchargement) ;
- `tests_automatiques.py` + `exemples/` = tests reproductibles.
`app.py` appelle `run_pipeline()`, qui renvoie un objet `PipelineResult` contenant tous les résultats
intermédiaires (c'est ce qui permet d'afficher chaque étape, même en cas d'échec).

## 4. Pipeline complet

Chargement/validation → redimensionnement → SIFT → BFMatcher KNN + Lowe → homographie RANSAC (par paire) →
composition vers l'image de référence → canevas + translation → `warpPerspective` → fusion pondérée → recadrage.

## 5. Technologies et rôles

| Technologie | Rôle |
|---|---|
| Python 3 | Langage |
| OpenCV | SIFT, BFMatcher, findHomography, warpPerspective, dessin des correspondances |
| NumPy | Tableaux d'images, calcul matriciel (composition des homographies) |
| Pillow | Lecture des fichiers + orientation EXIF des photos de téléphone |
| Streamlit | Interface web locale |

## 6. Résultats à montrer (ordre conseillé, ~5 min)

1. L'**image d'origine** (2–3 photos prises par vous).
2. Les **points clés** : « SIFT trouve N détails par image ».
3. Les **correspondances** : lignes vertes (inliers) et rouges (rejetées par RANSAC) ; montrer les chiffres
   *initiales → conservées (Lowe) → inliers (RANSAC)*.
4. La **matrice d'homographie** (expander « Détails techniques »).
5. Le **panorama final**, comparé aux originaux ; téléchargement.
6. Un **cas d'échec** volontaire (deux images sans lien) : le programme explique pourquoi il refuse.
7. Option : changer le ratio de Lowe ou le seuil RANSAC en direct pour montrer leur effet.

## 7. Limites à annoncer honnêtement

Parallaxe, objets mobiles, exposition, ordre des images fourni par l'utilisateur, pas d'optimisation globale.
Précisez aussi que les tests automatiques utilisent des **images synthétiques** et que vos photos réelles
sont à tester avant la soutenance (les résultats dépendent de la qualité des photos).

## 8. Questions probables et réponses simples

**Pourquoi SIFT plutôt qu'une autre méthode ?** Il est invariant à l'échelle et à la rotation, robuste aux
changements de luminosité, et disponible dans OpenCV standard depuis la 4.4 (brevet expiré en 2020). Contre-
partie : plus lent que ORB.

**À quoi sert le ratio test ?** Écarter les correspondances ambiguës : un bon appariement doit être nettement
meilleur que le deuxième candidat.

**Pourquoi 0,75 ?** Lowe propose 0,8 ; 0,75 est un peu plus strict. Plus bas = moins de correspondances mais plus
sûres ; plus haut = plus de correspondances mais plus de fausses.

**Pourquoi RANSAC alors qu'on a déjà filtré avec Lowe ?** Lowe filtre selon l'*apparence*, pas selon la
*géométrie*. RANSAC vérifie que les correspondances sont **cohérentes entre elles** avec une même transformation.

**Combien de points faut-il au minimum ?** 4 correspondances définissent une homographie (8 inconnues, 2 équations
par point). On en exige beaucoup plus (12, dont 10 inliers) pour la fiabilité.

**Pourquoi une homographie et pas une simple translation ?** Les photos prises en tournant ont des différences de
perspective et de rotation ; une translation ne suffirait pas.

**Pourquoi l'image centrale comme référence ?** Elle reste non déformée et les autres sont déformées de façon
plus équilibrée ; avec la première image, les images lointaines subissent de grandes déformations.

**Pourquoi composer les homographies ?** On ne peut apparier de façon fiable que des images voisines ; en
multipliant les matrices on obtient le lien entre une image éloignée et la référence.

**Comment gérez-vous les coordonnées négatives ?** Par une matrice de translation calculée à partir des coins transformés.

**Pourquoi fusionner avec des poids plutôt que superposer ?** Pour adoucir les coutures dues aux petites différences
de luminosité.

**Que se passe-t-il si les images sont dans le mauvais ordre ?** Les paires successives n'ont pas de zone commune :
le programme détecte l'échec (peu d'inliers ou homographie incohérente) et affiche un message ; il ne réordonne pas
automatiquement.

**Comment l'améliorer ?** Estimation globale des positions (*bundle adjustment*), projection cylindrique/sphérique,
compensation d'exposition, fusion multi-bandes, recherche de couture optimale, détection automatique de l'ordre
(comparer toutes les paires), accélération par FLANN.

**Pourquoi avoir redimensionné les images ?** Le temps de calcul de SIFT et de l'appariement croît fortement avec le
nombre de pixels ; 1000 px suffisent pour un panorama d'écran.

**Comment avez-vous testé ?** Scénarios automatisés (`tests_automatiques.py`) : 2 images, 3 images, sans zone commune,
fichier invalide, trop peu de correspondances ; plus des essais avec des photos réelles *(à compléter avec vos résultats)*.
