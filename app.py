"""
app.py — Interface de démonstration pour la soutenance du PFA
"OCR Layout Semantique : restauration et extraction fiable d'informations
dans des documents scannes bruites"

Lancement (depuis la racine du projet, venv active) :
    streamlit run app.py

Cette application fait tourner EN DIRECT le vrai pipeline développé pendant
le projet (Phases 2 a 6) : dégradation -> prétraitement -> OCR -> extraction
-> score de confiance. Rien n'est simulé ou pré-calculé pour la démo.

Auteur : Oumaima - PFA Document AI (encadrant : Pr. Hafidi Imad)
"""

import sys
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent))

from src.ocr_engine import run_ocr
from src.extraction import extract_key_value_pairs_spatial, ground_truth_pairs_from_annotation, evaluate_extraction
from src.validation import compute_confidence_score, pair_is_correct
from src.preprocessing import apply_pipeline, PREPROCESSING_PIPELINES
from src.degradation import apply_degradation, DEGRADATIONS, LEVELS

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"


def apply_degradation_safe(img, degradation_type, level, seed=None):
    """Compatible avec les deux versions de apply_degradation rencontrees dans
    le projet : celle de la Phase 2 qui retourne (image, params_dict), et une
    version plus simple qui retourne directement l'image."""
    result = apply_degradation(img, degradation_type, level, seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Palette 100% bleue (degrades)
# ---------------------------------------------------------------------------

BLEU_TRES_FONCE = "#081B33"
BLEU_FONCE = "#0A2540"
BLEU_MOYEN = "#14468C"
BLEU = "#1E6FD9"
BLEU_CLAIR = "#4FC3F7"
BLEU_TRES_CLAIR = "#EAF3FB"
BLEU_PALE = "#BBDEFB"
BLANC = "#FFFFFF"


# ---------------------------------------------------------------------------
# Icones SVG (remplacent les emojis)
# ---------------------------------------------------------------------------

ICON_DOCUMENT = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg>"""

ICON_PLAY = """<svg width="18" height="18" viewBox="0 0 24 24" fill="white"><polygon points="6 3 20 12 6 21 6 3"/></svg>"""

ICON_ALERT = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0A2540" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""


def icon_html(svg: str, size: int = 18, valign: str = "middle") -> str:
    return f'<span style="display:inline-block;vertical-align:{valign};width:{size}px;height:{size}px;">{svg}</span>'


# ---------------------------------------------------------------------------
# Configuration de la page + style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OCR Layout Semantique - Demo PFA",
    page_icon="📄",
    layout="wide",
)

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {BLEU_TRES_FONCE} 0%, {BLEU_MOYEN} 55%, {BLEU} 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 1.9rem; }}
    .main-header p {{ color: {BLEU_TRES_CLAIR}; margin: 0.3rem 0 0 0; font-size: 1rem; }}

    .badge-haute {{
        background: linear-gradient(135deg, {BLEU_TRES_FONCE}, {BLEU_MOYEN});
        color: white; padding: 3px 12px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;
    }}
    .badge-moyenne {{
        background: linear-gradient(135deg, {BLEU_MOYEN}, {BLEU});
        color: white; padding: 3px 12px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;
    }}
    .badge-faible {{
        background: linear-gradient(135deg, {BLEU_CLAIR}, {BLEU_PALE});
        color: {BLEU_FONCE}; padding: 3px 12px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;
    }}

    .legend-swatch {{
        display: inline-block; width: 13px; height: 13px; border-radius: 3px;
        margin-right: 6px; vertical-align: middle;
    }}

    /* Boutons : degrade de bleu */
    div.stButton > button {{
        background: linear-gradient(135deg, {BLEU_MOYEN} 0%, {BLEU} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem !important;
        transition: filter 0.15s ease;
    }}
    div.stButton > button:hover {{
        filter: brightness(1.12);
        color: white !important;
    }}
    div.stButton > button:active {{
        filter: brightness(0.95);
    }}

    /* Metriques : legere teinte bleue */
    div[data-testid="stMetric"] {{
        background: {BLEU_TRES_CLAIR};
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid {BLEU_PALE};
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {BLEU_PALE} !important;
        border-radius: 10px !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    {icon_html(ICON_DOCUMENT, size=38)}
    <div>
        <h1>OCR Layout Semantique</h1>
        <p>Restauration et extraction fiable d'informations dans des documents scannes bruites — Demonstration live du pipeline</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def get_sample_documents():
    if not (RAW_DIR / "images").exists():
        return []
    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    images = sorted((RAW_DIR / "images").glob("*.png"))
    return [{"image": p.name, "annotation": p.stem + ".json"} for p in images]


def badge_html(niveau: str) -> str:
    return f'<span class="badge-{niveau}">{niveau.upper()}</span>'


LABEL_BOX_COLOR = "#1E8E3E"   # vert - label detecte
VALUE_BOX_COLOR = "#D93025"   # rouge - valeur associee


def draw_boxes(img: Image.Image, pairs: list) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for p in pairs:
        lb = p.get("label_box")
        if lb:
            x, y, w, h = lb
            draw.rectangle([x, y, x + w, y + h], outline=LABEL_BOX_COLOR, width=3)
        for vb in p.get("value_boxes", []):
            x, y, w, h = vb
            draw.rectangle([x, y, x + w, y + h], outline=VALUE_BOX_COLOR, width=3)
    return out


# ---------------------------------------------------------------------------
# Interface (page unique)
# ---------------------------------------------------------------------------

st.subheader("Teste le pipeline en direct")

col_source, col_options = st.columns([1, 1])

with col_source:
    source = st.radio(
        "Source du document",
        ["Utiliser un document d'exemple (FUNSD)", "Uploader mon propre document"],
        horizontal=False,
    )

    base_image = None
    gt_pairs = None
    doc_label = None

    if source == "Utiliser un document d'exemple (FUNSD)":
        samples = get_sample_documents()
        if not samples:
            st.warning("Aucun document d'exemple trouve dans data/raw/images/.")
        else:
            choix = st.selectbox("Choisir un document", [s["image"] for s in samples])
            img_path = RAW_DIR / "images" / choix
            ann_path = RAW_DIR / "annotations" / (Path(choix).stem + ".json")
            base_image = Image.open(img_path).convert("RGB")
            doc_label = choix
            if ann_path.exists():
                gt_pairs = ground_truth_pairs_from_annotation(ann_path)
    else:
        uploaded = st.file_uploader("Choisir une image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            base_image = Image.open(uploaded).convert("RGB")
            doc_label = uploaded.name

with col_options:
    st.markdown("**Options de test (facultatif)**")

    simuler_degradation = st.checkbox("Simuler une degradation (Phase 2) — tester la robustesse")
    degradation_choisie, niveau_choisi = None, None
    if simuler_degradation:
        degradation_choisie = st.selectbox("Type de degradation", list(DEGRADATIONS.keys()))
        niveau_choisi = st.select_slider("Niveau", options=LEVELS, value="moyen")

    appliquer_pretraitement = st.checkbox("Appliquer un pretraitement (Phase 4)")
    pipeline_choisi = "aucun"
    if appliquer_pretraitement:
        pipeline_choisi = st.selectbox("Pipeline de pretraitement", list(PREPROCESSING_PIPELINES.keys()))

st.divider()

if base_image is not None:
    image_travail = base_image
    if simuler_degradation:
        image_travail = apply_degradation_safe(image_travail, degradation_choisie, niveau_choisi, seed=42)
    if appliquer_pretraitement:
        image_travail = apply_pipeline(image_travail, pipeline_choisi)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Document original**")
        st.image(base_image, use_container_width=True)
    with col_img2:
        label_img2 = "Document transforme (degrade/pretraite)" if (simuler_degradation or appliquer_pretraitement) else "Document (aucune transformation)"
        st.markdown(f"**{label_img2}**")
        st.image(image_travail, use_container_width=True)

    lancer = st.button("Lancer l'analyse (OCR + extraction + confiance)", type="primary")

    if lancer:
        with st.spinner("OCR en cours (Tesseract)..."):
            tmp_path = ROOT / "data" / "_demo_tmp.png"
            image_travail.save(tmp_path)
            ocr_result = run_ocr(tmp_path)
            tmp_path.unlink(missing_ok=True)

        with st.spinner("Extraction des paires cle-valeur..."):
            pred_pairs = extract_key_value_pairs_spatial(ocr_result["words"])

        with st.spinner("Calcul des scores de confiance..."):
            scored_pairs = []
            for p in pred_pairs:
                s = compute_confidence_score(p)
                s["label_box"] = p["label_box"]
                s["value_boxes"] = p["value_boxes"]
                if gt_pairs is not None:
                    s["est_correcte"] = pair_is_correct(p, gt_pairs)
                scored_pairs.append(s)

        st.success(f"Analyse terminee : {len(ocr_result['words'])} mots reconnus, {len(pred_pairs)} paires extraites.")

        if len(pred_pairs) == 0:
            st.markdown(
                f"""
                <div style="background:{BLEU_TRES_CLAIR}; border-left:4px solid {BLEU}; border-radius:8px; padding:0.9rem 1.1rem; display:flex; gap:0.7rem; align-items:flex-start;">
                    {icon_html(ICON_ALERT, size=22)}
                    <div style="color:{BLEU_FONCE};">
                        Aucune paire cle-valeur extraite avec ces reglages. Ce n'est pas un bug :
                        c'est precisement le resultat attendu quand la degradation est trop forte pour
                        que l'OCR reconnaisse du texte exploitable. Essaie un niveau de degradation plus
                        faible, ou active un pretraitement (Phase 4) pour observer une amelioration.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mots reconnus (OCR)", len(ocr_result["words"]))
        m2.metric("Paires extraites", len(pred_pairs))
        nb_haute = sum(1 for s in scored_pairs if s["niveau_confiance"] == "haute")
        m3.metric("Paires 'haute confiance'", nb_haute)
        if gt_pairs is not None:
            eval_metrics = evaluate_extraction(pred_pairs, gt_pairs)
            m4.metric("F1-score (partial match)", f"{eval_metrics['f1_partial']:.1%}")
        else:
            m4.metric("Verite terrain", "non disponible")

        st.markdown("### Zones detectees sur le document")
        st.markdown(
            f'<span class="legend-swatch" style="background:{LABEL_BOX_COLOR};"></span> Label detecte'
            f'&nbsp;&nbsp;&nbsp;'
            f'<span class="legend-swatch" style="background:{VALUE_BOX_COLOR};"></span> Valeur associee',
            unsafe_allow_html=True,
        )
        img_boxes = draw_boxes(image_travail, pred_pairs)
        st.image(img_boxes, use_container_width=True)

        st.markdown("### Paires cle-valeur extraites, avec score de confiance")
        rows_html = "<table style='width:100%; border-collapse: collapse;'>"
        rows_html += (
            f"<tr style='background:linear-gradient(135deg,{BLEU_TRES_FONCE},{BLEU_MOYEN}); background-color:{BLEU_FONCE}; color:white;'>"
            "<th style='padding:8px;text-align:left;'>Label</th><th style='padding:8px;text-align:left;'>Valeur</th>"
            "<th style='padding:8px;'>Confiance</th><th style='padding:8px;'>Score</th></tr>"
        )
        for i, s in enumerate(scored_pairs):
            bg = BLEU_TRES_CLAIR if i % 2 == 0 else "white"
            rows_html += (
                f"<tr style='background:{bg};'>"
                f"<td style='padding:8px;'>{s['label']}</td>"
                f"<td style='padding:8px;'>{s['valeur']}</td>"
                f"<td style='padding:8px;'>{badge_html(s['niveau_confiance'])}</td>"
                f"<td style='padding:8px;'>{s['score_global']:.2f}</td>"
                "</tr>"
            )
        rows_html += "</table>"
        st.markdown(rows_html, unsafe_allow_html=True)

        with st.expander("Voir le texte OCR brut complet"):
            st.text(ocr_result["text"])

        if gt_pairs is not None:
            with st.expander("Comparer avec la verite terrain FUNSD"):
                st.json(gt_pairs)
else:
    st.info("Choisis un document d'exemple ou uploade une image pour lancer la demo.")