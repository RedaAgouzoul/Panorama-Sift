# -*- coding: utf-8 -*-
"""
app.py
======
Interface Streamlit du projet « Création d'un panorama d'images par
appariement de caractéristiques SIFT ».

Lancement :   streamlit run app.py

Ce fichier ne contient que l'interface. Tous les algorithmes (SIFT, appariement,
RANSAC, assemblage) sont dans le package panorama_sift.
"""

import cv2
import numpy as np
import streamlit as st

from panorama_sift import (PanoramaError, PanoramaParams, PipelineResult, encode_image,
                           load_image_from_bytes, resize_keep_ratio, run_pipeline)

EXTENSIONS_ACCEPTEES = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]

st.set_page_config(page_title="Panorama SIFT", page_icon="🏞️", layout="wide")


# ---------------------------------------------------------------------------
# Petites fonctions utilitaires d'affichage
# ---------------------------------------------------------------------------

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV travaille en BGR, Streamlit affiche en RGB : on inverse les canaux."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_image(image_bgr: np.ndarray, caption: str = "") -> None:
    """Affiche une image BGR en pleine largeur de sa colonne.

    Streamlit a renommé le paramètre de largeur selon les versions : on essaie
    d'abord la syntaxe récente, puis l'ancienne."""
    rgb = bgr_to_rgb(image_bgr)
    try:
        st.image(rgb, caption=caption, width="stretch")
    except Exception:
        st.image(rgb, caption=caption, use_container_width=True)


@st.cache_data(show_spinner=False)
def make_preview(content: bytes, name: str):
    """Miniature RGB d'un fichier importé (mise en cache pour ne pas tout recalculer
    à chaque interaction). Renvoie (image_rgb, None) ou (None, message_d_erreur)."""
    try:
        image_bgr = load_image_from_bytes(content, name)
    except PanoramaError as error:
        return None, str(error)
    return bgr_to_rgb(resize_keep_ratio(image_bgr, 600)), None


def reset_everything() -> None:
    """Efface les résultats et vide la zone d'import (en changeant sa clé)."""
    st.session_state["result"] = None
    st.session_state["signature"] = None
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=5, suppress_small=True, floatmode="fixed")


# ---------------------------------------------------------------------------
# Initialisation de l'état de session
# ---------------------------------------------------------------------------

st.session_state.setdefault("result", None)       # dernier PipelineResult
st.session_state.setdefault("signature", None)    # empreinte des entrées du dernier calcul
st.session_state.setdefault("uploader_key", 0)    # change pour vider l'uploader


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------

st.title("🏞️ Panorama d'images par appariement SIFT")
st.markdown(
    "Cette application assemble plusieurs photographies qui se chevauchent en un **panorama**. "
    "Elle utilise uniquement des méthodes classiques de vision par ordinateur : "
    "**SIFT** (points clés) → **BFMatcher + ratio test de Lowe** (appariement) → "
    "**homographie + RANSAC** (alignement) → **fusion** des images."
)

# ---------------------------------------------------------------------------
# Barre latérale : paramètres
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Paramètres")
    st.caption("Les valeurs par défaut conviennent à la plupart des photos.")
    max_side = st.slider("Taille maximale des images (px)", 500, 2000, 1000, 100,
                         help="Les images plus grandes sont réduites (ratio conservé). "
                              "Plus petit = plus rapide ; plus grand = plus précis.")
    lowe_ratio = st.slider("Ratio test de Lowe", 0.60, 0.90, 0.75, 0.01,
                           help="Plus bas = appariements plus stricts (moins nombreux mais plus sûrs).")
    ransac_threshold = st.slider("Seuil RANSAC (px)", 1.0, 10.0, 4.0, 0.5,
                                 help="Erreur maximale (en pixels) pour qu'une correspondance soit un inlier.")
    reference_label = st.radio("Image de référence", ["Image centrale (recommandé)", "Première image"],
                               help="L'image de référence reste non déformée ; les autres sont alignées sur elle. "
                                    "Choisir l'image centrale réduit les déformations.")
    crop_label = st.radio(
        "Recadrage du panorama",
        ["Retirer les marges noires (sans rien couper)", "Rectangle maximal (sans aucun bord noir)", "Aucun recadrage"],
        help="« Rectangle maximal » donne un résultat plus propre mais supprime une partie du contenu.")
    sort_by_name = st.checkbox("Trier les images par nom de fichier", value=False,
                               help="Utile si l'ordre d'import n'est pas celui de la prise de vue "
                                    "(ex. 01_gauche.jpg, 02_milieu.jpg, 03_droite.jpg).")
    st.divider()
    st.button("🔄 Réinitialiser / recommencer", on_click=reset_everything)

params = PanoramaParams(
    max_side=max_side,
    lowe_ratio=lowe_ratio,
    ransac_threshold=ransac_threshold,
    reference="centre" if reference_label.startswith("Image centrale") else "premiere",
    crop_mode={"Retirer": "bbox", "Rectangle": "interieur", "Aucun": "aucun"}[crop_label.split()[0]],
)

# ---------------------------------------------------------------------------
# Import des images
# ---------------------------------------------------------------------------

st.header("1️⃣ Import des images")
st.info(
    "Importez **au minimum 2 images**, **dans l'ordre de gauche à droite** de la scène. "
    "Chaque image doit montrer environ **30 à 50 % de la scène de la précédente**."
)
uploaded_files = st.file_uploader(
    "Choisissez vos photos (PNG, JPG, JPEG, BMP, WEBP, TIF)",
    type=EXTENSIONS_ACCEPTEES,
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

files_payload = []  # liste de (nom, octets)
if uploaded_files:
    files_payload = [(f.name, f.getvalue()) for f in uploaded_files]
    if sort_by_name:
        files_payload.sort(key=lambda item: item[0].lower())

if files_payload:
    st.subheader("Aperçu dans l'ordre de traitement")
    preview_columns = st.columns(min(len(files_payload), 4))
    for position, (name, content) in enumerate(files_payload):
        with preview_columns[position % len(preview_columns)]:
            preview_rgb, preview_error = make_preview(content, name)
            if preview_error:
                st.error(f"{position + 1}. {preview_error}")
            else:
                st.image(preview_rgb, caption=f"{position + 1}. {name}")

# Empreinte des entrées : permet de détecter que les images/paramètres ont changé
current_signature = (tuple((name, len(content)) for name, content in files_payload), params)

generate_clicked = st.button(
    "🚀 Générer le panorama",
    type="primary",
    disabled=len(files_payload) == 0,
)
if len(files_payload) == 1:
    st.warning("Une seule image importée : ajoutez-en au moins une autre pour créer un panorama.")

# ---------------------------------------------------------------------------
# Lancement du traitement
# ---------------------------------------------------------------------------

if generate_clicked:
    progress_bar = st.progress(0.0, text="Démarrage…")

    def update_progress(fraction: float, text: str) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=text)

    st.session_state["result"] = run_pipeline(files_payload, params, progress=update_progress)
    st.session_state["signature"] = current_signature
    progress_bar.empty()

# ---------------------------------------------------------------------------
# Affichage des résultats
# ---------------------------------------------------------------------------

result: PipelineResult = st.session_state["result"]

if result is not None:
    st.divider()
    st.header("2️⃣ Résultats")

    if st.session_state["signature"] != current_signature:
        st.warning("Les images ou les paramètres ont changé depuis le dernier calcul : "
                   "cliquez sur « Générer le panorama » pour mettre à jour.")

    # --- Message global de succès ou d'erreur ---
    if result.error:
        st.error(f"❌ Échec à l'étape « {result.error_step} » : {result.error}")
        st.caption("Les étapes réussies avant l'erreur restent consultables dans les onglets ci-dessous "
                   "pour vous aider à comprendre le problème.")
    else:
        panorama_height, panorama_width = result.panorama.shape[:2]
        st.success(f"✅ Panorama généré avec succès : {panorama_width} × {panorama_height} pixels, "
                   f"à partir de {len(result.images)} images.")

    # --- Informations principales sur le traitement ---
    if result.images:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Images traitées", len(result.images))
        metric_columns[1].metric("Points clés SIFT (total)", sum(len(f.keypoints) for f in result.features))
        if result.pairs:
            metric_columns[2].metric("Inliers RANSAC (total)", sum(p.nb_inliers for p in result.pairs))
        if result.durations:
            metric_columns[3].metric("Durée totale", f"{sum(result.durations.values()):.2f} s")

    tab_originals, tab_keypoints, tab_matches, tab_homography, tab_final = st.tabs([
        "🖼️ Images originales", "🔑 Points clés SIFT", "🔗 Correspondances",
        "📐 Homographie (RANSAC)", "🏞️ Panorama final"])

    # Onglet : images originales ------------------------------------------------
    with tab_originals:
        st.caption("Images après chargement (et redimensionnement éventuel), dans l'ordre de traitement.")
        columns = st.columns(min(max(len(result.images), 1), 3))
        for position, prepared in enumerate(result.images):
            with columns[position % len(columns)]:
                height, width = prepared.bgr.shape[:2]
                show_image(prepared.bgr, f"{position + 1}. {prepared.name} — {width}×{height} px "
                                         f"(original : {prepared.original_size[0]}×{prepared.original_size[1]})")

    # Onglet : points clés ------------------------------------------------------------
    with tab_keypoints:
        with st.expander("💡 Qu'est-ce qu'un point clé SIFT ?"):
            st.markdown(
                "Un **point clé** est un petit détail de l'image (coin, tache, motif) que SIFT sait "
                "**retrouver même si l'image change d'échelle, tourne ou change de luminosité**. "
                "Sur le dessin : le **centre** du cercle est la position, le **rayon** l'échelle du détail, "
                "et le **trait** son orientation. Chaque point clé a un **descripteur** de 128 nombres qui "
                "décrit l'aspect de son voisinage."
            )
        st.caption("Pour rester lisible, seuls les 500 points clés les plus forts de chaque image sont dessinés.")
        columns = st.columns(min(max(len(result.keypoint_images), 1), 2))
        for position, keypoint_image in enumerate(result.keypoint_images):
            with columns[position % len(columns)]:
                show_image(keypoint_image,
                           f"Image {position + 1} : {len(result.features[position].keypoints)} points clés détectés")

    # Onglet : correspondances -----------------------------------------------------------
    with tab_matches:
        with st.expander("💡 Comment fonctionne l'appariement ?"):
            st.markdown(
                "Pour chaque point clé de l'image de gauche, **BFMatcher** cherche (par force brute) les "
                "**2 descripteurs les plus proches** dans l'image de droite (KNN, k=2). Le **ratio test de Lowe** "
                "garde la correspondance seulement si le meilleur candidat est **nettement** plus proche que le "
                "deuxième : sinon, le point est ambigu et on l'écarte."
            )
        if not result.pairs:
            st.info("Aucune paire n'a pu être traitée.")
        for pair in result.pairs:
            st.subheader(f"Image {pair.index_left + 1} ↔ Image {pair.index_right + 1}")
            columns = st.columns(3)
            columns[0].metric("Correspondances initiales", pair.nb_initial)
            columns[1].metric("Conservées (ratio test)", pair.nb_good)
            columns[2].metric("Inliers RANSAC", pair.nb_inliers)
            if pair.ok:
                st.success(pair.message)
            else:
                st.error(pair.message)
            if pair.matches_image is not None:
                show_image(pair.matches_image,
                           "Vert : correspondances cohérentes (inliers). Rouge : écartées par RANSAC. "
                           "Seules les meilleures lignes sont dessinées.")

    # Onglet : homographie ------------------------------------------------------------------
    with tab_homography:
        with st.expander("💡 Homographie et RANSAC : de quoi parle-t-on ?", expanded=True):
            st.markdown(
                "Une **homographie** est une matrice 3×3 qui indique comment déformer une image "
                "(translation, rotation, échelle, perspective) pour qu'elle se superpose à une autre. "
                "Elle est calculée à partir des points appariés. Comme certains appariements sont faux, "
                "**RANSAC** tire au hasard de petits groupes de 4 correspondances, calcule une homographie, "
                "compte combien d'autres correspondances sont d'accord (**inliers**) et garde la meilleure. "
                "Les correspondances en désaccord (**outliers**) sont ignorées."
            )
        valid_pairs = [pair for pair in result.pairs if pair.homography is not None]
        if not valid_pairs:
            st.info("Aucune homographie n'a pu être calculée.")
        for pair in valid_pairs:
            st.markdown(f"**Image {pair.index_left + 1} → Image {pair.index_right + 1}** : "
                        f"{pair.nb_inliers} inliers sur {pair.nb_good} correspondances conservées "
                        f"({pair.nb_inliers / max(pair.nb_good, 1):.0%})")
            with st.expander("Détails techniques : matrice d'homographie 3×3"):
                st.code(format_matrix(pair.homography), language="text")
        if result.homographies_to_reference:
            st.markdown(f"**Image de référence : image {result.reference_index + 1}** "
                        "(les autres images sont alignées sur elle).")
            with st.expander("Détails techniques : homographie de chaque image vers l'image de référence"):
                for position, homography in enumerate(result.homographies_to_reference):
                    st.text(f"Image {position + 1} → référence")
                    st.code(format_matrix(homography), language="text")

    # Onglet : panorama final -------------------------------------------------------------------
    with tab_final:
        if result.panorama is None:
            st.info("Le panorama n'a pas pu être généré (voir le message d'erreur ci-dessus).")
        else:
            show_image(result.panorama, "Panorama final")

            download_columns = st.columns(2)
            download_columns[0].download_button(
                "⬇️ Télécharger en PNG (sans perte)", data=encode_image(result.panorama, ".png"),
                file_name="panorama.png", mime="image/png")
            download_columns[1].download_button(
                "⬇️ Télécharger en JPEG (plus léger)", data=encode_image(result.panorama, ".jpg"),
                file_name="panorama.jpg", mime="image/jpeg")

            st.subheader("Comparaison avec les images originales")
            comparison_columns = st.columns(len(result.images))
            for position, prepared in enumerate(result.images):
                with comparison_columns[position]:
                    show_image(prepared.bgr, f"Image {position + 1}")

            with st.expander("📋 Informations sur le traitement"):
                canvas_width, canvas_height = result.canvas_size
                crop_x0, crop_y0, crop_x1, crop_y1 = result.crop_box
                st.markdown(
                    f"- Images assemblées : **{len(result.images)}** (image de référence : n°{result.reference_index + 1})\n"
                    f"- Canevas avant recadrage : **{canvas_width} × {canvas_height}** px\n"
                    f"- Zone conservée après recadrage : x de {crop_x0} à {crop_x1}, y de {crop_y0} à {crop_y1}\n"
                    f"- Taille finale : **{crop_x1 - crop_x0} × {crop_y1 - crop_y0}** px\n"
                    f"- Paramètres : ratio de Lowe = {params.lowe_ratio}, seuil RANSAC = {params.ransac_threshold} px, "
                    f"taille max = {params.max_side} px"
                )
                st.markdown("**Durée par étape :**")
                for step_name, seconds in result.durations.items():
                    st.markdown(f"- {step_name} : {seconds:.2f} s")
else:
    st.caption("Importez vos images puis cliquez sur « Générer le panorama ».")
