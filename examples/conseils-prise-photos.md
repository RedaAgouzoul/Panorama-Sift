PRÉPARER DES IMAGES DE TEST POUR LE PANORAMA
=============================================

Aucune photo n'est fournie avec le projet : vous devez prendre les vôtres
(ou utiliser les images synthétiques décrites plus bas pour de simples tests techniques).


1. PRENDRE DE BONNES PHOTOS
-----------------------------------------------------------
- Placez-vous face à la scène (paysage, bâtiment, rue) et PIVOTEZ sur vous-même
  entre les photos, sans vous déplacer (comme un trépied). Évitez de marcher.
- Prenez 2 ou 3 photos de GAUCHE à DROITE.
- Chaque photo doit montrer 30 à 50 % de ce que montrait la précédente
  (repérez un élément de la photo 1 qui doit réapparaître sur le bord de la photo 2).
- Choisissez une scène avec des détails distinctifs : bâtiments, fenêtres, enseignes,
  arbres, textures. Évitez ciel uniforme, mur blanc, étendue d'eau, vitres reflétantes.
- Gardez la même orientation (toutes en paysage ou toutes en portrait), la même
  focale (pas de zoom) et, si possible, une exposition identique (verrouillage AE/AF).
- Évitez les objets qui bougent (voitures, piétons) dans la zone commune.
- Nommez les fichiers dans l'ordre : 01_gauche.jpg, 02_milieu.jpg, 03_droite.jpg
  (l'application peut les trier par nom si vous cochez l'option dans la barre latérale).


2. IMAGES SYNTHÉTIQUES (tests techniques uniquement)
----------------------------------------------------
Depuis le dossier du projet, environnement virtuel activé :

    python scripts\generate_test_images.py

Cela crée le dossier examples\jeux_de_test\ avec 5 sous-dossiers :
  1_deux_images_ok          -> 2 images avec ~40 % de chevauchement
  2_sans_zone_commune       -> 2 images de scènes différentes
  3_trois_images_ok         -> 3 images successives
  4_fichier_invalide        -> 1 image valide + 1 fichier texte renommé .jpg
  5_peu_de_correspondances  -> 2 images presque uniformes

ATTENTION : ce ne sont PAS des photographies, seulement des dessins générés par ordinateur.


3. LES 5 TESTS À FAIRE DANS L'APPLICATION
-----------------------------------------
Test 1 - Deux images qui se chevauchent (dossier 1_...)
   Attendu : message vert « Panorama généré avec succès », panorama visible, boutons de téléchargement.

Test 2 - Deux images sans zone commune (dossier 2_...)
   Attendu : message rouge. Exemple de message :
   « Impossible d'aligner l'image 1 avec l'image 2. Seulement 18 % des correspondances sont
     cohérentes entre elles ... les images n'ont sans doute pas de zone commune. »
   Pas de panorama ; les onglets montrent les correspondances (désordonnées, rouges).

Test 3 - Trois images successives (dossier 3_...)
   Attendu : succès, 2 paires affichées, panorama plus large que chacune des images.

Test 4 - Image invalide (dossier 4_...)
   Attendu : l'aperçu affiche « faux_fichier.jpg n'est pas une image valide ... » en rouge ;
   en cliquant sur Générer : « Échec à l'étape chargement ». Un fichier d'une extension non
   prévue (.pdf, .txt...) est refusé directement par la zone d'import.

Test 5 - Trop peu de correspondances (dossier 5_...)
   Attendu : « Seulement 5 correspondances fiables après le ratio test (minimum requis : 12) ... »

Les chiffres exacts peuvent varier légèrement selon les versions d'OpenCV.

