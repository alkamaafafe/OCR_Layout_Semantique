"""
src/evaluation.py

Métriques d'évaluation OCR pour la Phase 3 :
    - CER : Character Error Rate
    - WER : Word Error Rate
    - taux de mots correctement reconnus
    - taux de mots non détectés (mots de la vérité terrain absents de l'OCR)

Implémentation "from scratch" (distance de Levenshtein) pour éviter une
dépendance externe supplémentaire.

Auteur : Oumaima - PFA Document AI (encadrant : Pr. Hafidi Imad)
"""

from __future__ import annotations

from typing import Dict, List


def _levenshtein(seq_ref: List, seq_hyp: List) -> int:
    """Distance de Levenshtein classique (nombre min. d'insertions/suppressions/
    substitutions pour passer de seq_ref à seq_hyp), sur une séquence générique
    (caractères ou mots)."""
    n, m = len(seq_ref), len(seq_hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq_ref[i - 1] == seq_hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # suppression
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = distance de Levenshtein (caractères) / nb caractères référence."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = distance de Levenshtein (mots) / nb mots référence."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_word_detection_rate(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Taux de mots de la référence correctement retrouvés dans l'hypothèse
    (comparaison par ensemble de mots, insensible à l'ordre et aux doublons).

    Retourne :
        {
            "taux_mots_detectes": proportion de mots-référence présents dans l'OCR,
            "taux_mots_manquants": 1 - taux_mots_detectes,
        }
    """
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    if len(ref_words) == 0:
        return {"taux_mots_detectes": 1.0, "taux_mots_manquants": 0.0}

    detected = ref_words & hyp_words
    taux_detectes = len(detected) / len(ref_words)
    return {
        "taux_mots_detectes": taux_detectes,
        "taux_mots_manquants": 1.0 - taux_detectes,
    }


def evaluate_ocr_result(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calcule toutes les métriques d'un coup pour une paire (référence, OCR)."""
    metrics = {
        "cer": compute_cer(reference, hypothesis),
        "wer": compute_wer(reference, hypothesis),
    }
    metrics.update(compute_word_detection_rate(reference, hypothesis))
    return metrics


if __name__ == "__main__":
    # petit test manuel
    ref = "Discontinue coal retention analyses on licensee submitted"
    hyp = "Discontlnue coal retentlon analyses on licensee submited"
    print(evaluate_ocr_result(ref, hyp))
