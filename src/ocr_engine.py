"""
src/ocr_engine.py

Moteur OCR "baseline brut" pour la Phase 3 : applique Tesseract OCR
directement sur une image (sans prétraitement) et retourne :
    - le texte reconnu complet
    - les mots reconnus individuellement, avec leurs bounding boxes et un
      score de confiance par mot (donné par Tesseract)

Auteur : Oumaima - PFA Document AI (encadrant : Pr. Hafidi Imad)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
from PIL import Image


def run_ocr(image_path: Path, lang: str = "eng") -> Dict:
    """
    Applique Tesseract OCR brut sur une image.

    Retourne un dict :
        {
            "text": "texte complet reconnu, mots séparés par des espaces",
            "words": [
                {"text": "R&D", "conf": 91.2, "box": [x, y, w, h]},
                ...
            ]
        }

    Les mots avec une confiance négative (Tesseract renvoie -1 pour les
    zones vides / non reconnues) sont ignorés.
    """
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        raw_text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if raw_text == "" or conf < 0:
            continue
        words.append({
            "text": raw_text,
            "conf": conf,
            "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
        })

    full_text = " ".join(w["text"] for w in words)
    return {"text": full_text, "words": words}


def ground_truth_text_from_annotation(annotation_path: Path) -> str:
    """
    Extrait le texte "vérité terrain" d'une annotation FUNSD, en concaténant
    tous les mots individuels du champ "words" de chaque bloc "form".

    On utilise le niveau "word" (pas "text" par bloc) pour être comparable
    mot à mot avec la sortie de l'OCR.
    """
    import json
    ann = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    words = []
    for block in ann.get("form", []):
        for w in block.get("words", []):
            t = w.get("text", "").strip()
            if t:
                words.append(t)
    return " ".join(words)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test rapide de l'OCR baseline sur une image.")
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    result = run_ocr(Path(args.image_path))
    print(f"Mots détectés : {len(result['words'])}")
    print(f"Texte (200 premiers caractères) : {result['text'][:200]}")
