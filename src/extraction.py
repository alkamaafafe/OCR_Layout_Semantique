"""
src/extraction.py

Phase 5 : extraction d'entités / paires clé-valeur à partir des sorties OCR.

Deux approches implémentées (cf. taxonomie du cahier des charges) :

1. Approche "règles + OCR" (extract_entities_by_rules)
   Recherche, dans le texte OCR brut, des motifs typiques (dates, montants,
   références, emails) via expressions régulières et mots-clés proches.

2. Approche "OCR + layout" (extract_key_value_pairs_spatial)
   Regroupe les mots OCR en lignes/blocs à partir de leurs coordonnées
   (bounding boxes), repère les mots qui ressemblent à un "label" (se
   terminant par ':', ou mot-clé connu), puis associe chaque label à la
   valeur la plus proche spatialement (à droite sur la même ligne, ou juste
   en dessous).

Une troisième approche, "layout-aware" (LayoutLM/LayoutLMv3), est documentée
en bas de fichier mais volontairement NON implémentée ici : elle nécessite un
GPU et le téléchargement de plusieurs centaines de Mo de poids de modèle, ce
qui dépasse le cadre matériel de ce PFA (cf. section "Risques et stratégies
de mitigation" du cahier des charges : "Manque de GPU -> utiliser des
modèles pré-entraînés, réduire le volume ou privilégier les règles/layout").

Une fonction d'évaluation compare les paires extraites à la vérité terrain
FUNSD (via le champ "linking" des annotations).

Auteur : Oumaima - PFA Document AI (encadrant : Pr. Hafidi Imad)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Approche 1 : règles + OCR (motifs génériques dans le texte brut)
# ---------------------------------------------------------------------------

REGEX_PATTERNS = {
    # NB : on évite \b devant/après des chiffres. En regex, "_" est considéré
    # comme un caractère de mot (\w) au même titre qu'une lettre ou un chiffre :
    # \b ne détecte donc PAS de frontière entre "_" et "9" (ex: "__9/3/92",
    # très fréquent en sortie OCR de formulaires avec lignes à remplir
    # soulignées). On utilise donc des lookarounds explicites sur les chiffres.
    "date": re.compile(r"(?<!\d)\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?!\d)"),
    "montant": re.compile(r"(?<!\d)\d+[.,]\d{2}\s?(?:\$|€|MAD|DH)?(?!\d)"),
    "reference": re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,4}[-#]?\d{3,}(?!\d)"),
    "email": re.compile(r"(?<![\w.\-])[\w.\-]+@[\w.\-]+\.\w+(?![\w])"),
    "telephone": re.compile(r"(?<!\d)\d{2,3}[\s.\-]\d{2,4}[\s.\-]\d{2,4}(?!\d)"),
}

# Mots-clés dont la présence juste avant un motif renforce la confiance
KEYWORDS_BY_TYPE = {
    "date": ["date", "le", "du"],
    "montant": ["montant", "total", "prix", "amount"],
    "reference": ["ref", "reference", "référence", "no", "n°"],
    "email": ["email", "mail", "e-mail"],
    "telephone": ["tel", "tél", "phone", "ext"],
}


def extract_entities_by_rules(text: str) -> List[Dict]:
    """
    Approche 1 : cherche, dans un texte OCR brut (une seule chaîne), tous les
    motifs correspondant aux types définis dans REGEX_PATTERNS.

    Retourne une liste de dicts :
        {"type": "date", "valeur": "9/3/92", "position_char": 123}
    """
    entities = []
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "type": ent_type,
                "valeur": match.group(),
                "position_char": match.start(),
            })
    return sorted(entities, key=lambda e: e["position_char"])


# ---------------------------------------------------------------------------
# Approche 2 : OCR + layout (association spatiale label -> valeur)
# ---------------------------------------------------------------------------

LABEL_KEYWORDS = [
    "name", "date", "supervisor", "manager", "group", "suggestion",
    "reference", "amount", "total", "phone", "ext", "signature", "manageг",
]


def _word_center(word: Dict) -> Tuple[float, float]:
    x, y, w, h = word["box"]
    return x + w / 2, y + h / 2


def group_words_into_lines(words: List[Dict], y_tolerance: int = 12) -> List[List[Dict]]:
    """Regroupe une liste de mots OCR (avec bounding boxes) en lignes,
    en fonction de leur coordonnée verticale (y). Les mots d'une même ligne
    sont ensuite triés de gauche à droite."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: w["box"][1])
    lines: List[List[Dict]] = []

    for word in sorted_words:
        _, cy = _word_center(word)
        placed = False
        for line in lines:
            _, line_cy = _word_center(line[0])
            if abs(cy - line_cy) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda w: w["box"][0])

    lines.sort(key=lambda line: line[0]["box"][1])
    return lines


def _looks_like_label(word_text: str) -> bool:
    clean = word_text.lower().strip(":.,")
    return word_text.strip().endswith(":") or clean in LABEL_KEYWORDS


def extract_key_value_pairs_spatial(words: List[Dict]) -> List[Dict]:
    """
    Approche 2 : identifie les mots "label" (finissant par ':' ou mot-clé
    connu) et leur associe la valeur la plus proche spatialement :
        1. le(s) mot(s) suivant(s) sur la MÊME ligne, à droite du label
        2. sinon, le premier mot de la ligne juste EN DESSOUS

    Retourne une liste de dicts :
        {"label": "Date:", "valeur": "9/3/92", "methode": "meme_ligne"}
    """
    lines = group_words_into_lines(words)
    pairs = []

    for i, line in enumerate(lines):
        for j, word in enumerate(line):
            if not _looks_like_label(word["text"]):
                continue

            # 1. valeur sur la meme ligne, a droite
            reste_ligne = line[j + 1:]
            if reste_ligne:
                valeur = " ".join(w["text"] for w in reste_ligne[:4])
                pairs.append({"label": word["text"], "valeur": valeur, "methode": "meme_ligne"})
                continue

            # 2. sinon, ligne suivante
            if i + 1 < len(lines):
                ligne_suivante = lines[i + 1]
                valeur = " ".join(w["text"] for w in ligne_suivante[:4])
                pairs.append({"label": word["text"], "valeur": valeur, "methode": "ligne_suivante"})

    return pairs


# ---------------------------------------------------------------------------
# Vérité terrain FUNSD (pour l'évaluation)
# ---------------------------------------------------------------------------

def ground_truth_pairs_from_annotation(annotation_path: Path) -> List[Dict]:
    """
    Reconstruit les paires question -> réponse "vérité terrain" à partir
    d'une annotation FUNSD, en suivant le champ "linking" (liste de [id1,id2]
    reliant un bloc "question" à un bloc "answer").
    """
    ann = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    blocks_by_id = {b["id"]: b for b in ann["form"]}

    pairs = []
    seen = set()
    for block in ann["form"]:
        for link in block.get("linking", []):
            id_a, id_b = link
            key = tuple(sorted(link))
            if key in seen:
                continue
            seen.add(key)

            block_a = blocks_by_id.get(id_a)
            block_b = blocks_by_id.get(id_b)
            if block_a is None or block_b is None:
                continue

            if block_a["label"] == "question" and block_b["label"] == "answer":
                q, a = block_a, block_b
            elif block_b["label"] == "question" and block_a["label"] == "answer":
                q, a = block_b, block_a
            else:
                continue

            pairs.append({"label": q["text"], "valeur": a["text"]})

    return pairs


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().strip(":."))


def _partial_overlap(a: str, b: str) -> float:
    """Similarité simple entre 2 chaînes normalisées : proportion de mots de
    la plus courte chaîne retrouvés dans la plus longue. 1.0 = incluse en
    totalité, 0.0 = aucun mot commun."""
    words_a, words_b = set(a.split()), set(b.split())
    if not words_a or not words_b:
        return 0.0
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    return len(shorter & longer) / len(shorter)


def evaluate_extraction(predicted_pairs: List[Dict], gt_pairs: List[Dict],
                         partial_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compare les paires prédites aux paires de vérité terrain, avec 2 niveaux
    d'exigence (cf. cahier des charges, critère "Évaluation extraction
    d'information") :

    - "exact_match" : label ET valeur identiques après normalisation.
    - "partial_match" : le label correspond, ET au moins `partial_threshold`
      (50% par défaut) des mots de la valeur (la plus courte des deux) sont
      retrouvés dans l'autre — tolère les variations d'OCR (ponctuation,
      mots manquants en bord de valeur, etc.).

    Retourne précision / rappel / F1 pour les deux niveaux.
    """
    gt_norm = [(_normalize(p["label"]), _normalize(p["valeur"])) for p in gt_pairs]
    pred_norm = [(_normalize(p["label"]), _normalize(p["valeur"])) for p in predicted_pairs]

    gt_set = set(gt_norm)
    pred_set = set(pred_norm)

    # --- exact match ---
    true_positives_exact = len(gt_set & pred_set)
    precision_exact = true_positives_exact / len(pred_set) if pred_set else 0.0
    recall_exact = true_positives_exact / len(gt_set) if gt_set else 0.0
    f1_exact = (2 * precision_exact * recall_exact / (precision_exact + recall_exact)
                if (precision_exact + recall_exact) > 0 else 0.0)

    # --- partial match : label identique + valeur partiellement correcte ---
    gt_by_label = {}
    for label, valeur in gt_norm:
        gt_by_label.setdefault(label, []).append(valeur)

    true_positives_partial = 0
    for label, valeur in pred_norm:
        candidates = gt_by_label.get(label, [])
        if any(_partial_overlap(valeur, gt_valeur) >= partial_threshold for gt_valeur in candidates):
            true_positives_partial += 1

    precision_partial = true_positives_partial / len(pred_norm) if pred_norm else 0.0
    recall_partial = true_positives_partial / len(gt_norm) if gt_norm else 0.0
    f1_partial = (2 * precision_partial * recall_partial / (precision_partial + recall_partial)
                  if (precision_partial + recall_partial) > 0 else 0.0)

    # match du label seul (la valeur peut etre completement fausse)
    gt_labels = {l for l, v in gt_norm}
    pred_labels = {l for l, v in pred_norm}
    label_recall = len(gt_labels & pred_labels) / len(gt_labels) if gt_labels else 0.0

    return {
        "precision_exact": round(precision_exact, 3),
        "recall_exact": round(recall_exact, 3),
        "f1_exact": round(f1_exact, 3),
        "precision_partial": round(precision_partial, 3),
        "recall_partial": round(recall_partial, 3),
        "f1_partial": round(f1_partial, 3),
        "taux_labels_retrouves": round(label_recall, 3),
        "nb_paires_verite_terrain": len(gt_set),
        "nb_paires_predites": len(pred_set),
    }


# ---------------------------------------------------------------------------
# Note sur l'approche 3 (layout-aware, non implémentée)
# ---------------------------------------------------------------------------
#
# LayoutLM/LayoutLMv3 combinent texte + position + image dans un même modèle
# Transformer pré-entraîné, avec un fine-tuning supervisé sur FUNSD pour la
# tâche de classification de tokens (question/answer/header/other) et de
# liaison. Une inférence "telle quelle" (zero-shot) donne des résultats
# faibles sans fine-tuning ; un fine-tuning correct nécessite un GPU et
# plusieurs heures d'entraînement.
#
# Piste pour aller plus loin (si accès GPU, ex: Google Colab) :
#   from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
#   processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
#   model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
#   # fine-tuning sur FUNSD (dataset "nielsr/funsd-layoutlmv3" sur HuggingFace)


if __name__ == "__main__":
    print("Types d'entites (regles) :", list(REGEX_PATTERNS.keys()))
    print("Mots-cles labels (spatial) :", LABEL_KEYWORDS)