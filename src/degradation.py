"""
src/degradation.py

Module de génération de versions artificiellement dégradées de documents
scannés, pour le benchmark de robustesse (Phase 2 du PFA).

8 types de dégradation, chacun paramétrable sur 3 niveaux de difficulté :
    - faible
    - moyen
    - fort

Chaque fonction :
    - prend une image PIL.Image (RGB) en entrée
    - retourne (image_degradee: PIL.Image, params_utilises: dict)

Une fonction générique `apply_degradation(image, degradation_type, level, seed)`
permet d'appeler n'importe quelle dégradation par son nom.

Une fonction `generate_degraded_dataset(...)` applique toutes les dégradations
à tous les niveaux sur un dataset entier (ex: FUNSD) et sauvegarde :
    - les images dégradées (.png)
    - un manifest JSON listant, pour chaque image générée, le document
      d'origine, le type de dégradation, le niveau et les paramètres exacts.

Auteur : Oumaima - PFA Document AI (encadrant : Pr. Hafidi Imad)
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Utilitaires de conversion PIL <-> OpenCV
# ---------------------------------------------------------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL (RGB) en tableau OpenCV (BGR)."""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau OpenCV (BGR) en image PIL (RGB)."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _set_seed(seed: int | None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


# ---------------------------------------------------------------------------
# Grille de paramètres par niveau de difficulté
# ---------------------------------------------------------------------------
# Chaque dégradation a 3 jeux de paramètres (faible / moyen / fort).
# Ces valeurs sont volontairement documentées ici pour que le rapport
# puisse justifier chaque choix.

LEVELS = ["faible", "moyen", "fort"]

PARAM_GRID: Dict[str, Dict[str, dict]] = {
    "flou_gaussien": {
        "faible": {"kernel": 3, "sigma": 1.0},
        "moyen": {"kernel": 7, "sigma": 2.5},
        "fort": {"kernel": 13, "sigma": 5.0},
    },
    "bruit_aleatoire": {
        "faible": {"type": "gaussian", "std": 8},
        "moyen": {"type": "gaussian", "std": 20},
        "fort": {"type": "gaussian", "std": 40},
    },
    "rotation_legere": {
        "faible": {"angle_max": 2},
        "moyen": {"angle_max": 5},
        "fort": {"angle_max": 10},
    },
    "faible_contraste": {
        "faible": {"alpha": 0.85, "beta": 10},
        "moyen": {"alpha": 0.65, "beta": 20},
        "fort": {"alpha": 0.45, "beta": 30},
    },
    "compression_jpeg": {
        "faible": {"quality": 50},
        "moyen": {"quality": 25},
        "fort": {"quality": 10},
    },
    "effet_scan_degrade": {
        "faible": {"threshold_noise": 10, "blur": 1},
        "moyen": {"threshold_noise": 25, "blur": 2},
        "fort": {"threshold_noise": 45, "blur": 3},
    },
    "distorsion_decalage": {
        "faible": {"shift_px": 3, "shear": 0.01},
        "moyen": {"shift_px": 8, "shear": 0.03},
        "fort": {"shift_px": 15, "shear": 0.06},
    },
    "ombres": {
        "faible": {"n_shadows": 1, "opacity": 0.15},
        "moyen": {"n_shadows": 2, "opacity": 0.30},
        "fort": {"n_shadows": 3, "opacity": 0.50},
    },
}


# ---------------------------------------------------------------------------
# 1. Flou gaussien
# ---------------------------------------------------------------------------

def flou_gaussien(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["flou_gaussien"][level])
    out = img.filter(ImageFilter.GaussianBlur(radius=params["sigma"]))
    return out, {"type": "flou_gaussien", "level": level, **params}


# ---------------------------------------------------------------------------
# 2. Bruit aléatoire
# ---------------------------------------------------------------------------

def bruit_aleatoire(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["bruit_aleatoire"][level])
    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, params["std"], arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), {"type": "bruit_aleatoire", "level": level, **params}


# ---------------------------------------------------------------------------
# 3. Rotation légère
# ---------------------------------------------------------------------------

def rotation_legere(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["rotation_legere"][level])
    angle = random.uniform(-params["angle_max"], params["angle_max"])
    out = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return out, {"type": "rotation_legere", "level": level, "angle_applique": round(angle, 3), **params}


# ---------------------------------------------------------------------------
# 4. Faible contraste
# ---------------------------------------------------------------------------

def faible_contraste(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["faible_contraste"][level])
    arr = pil_to_cv(img).astype(np.float32)
    out = arr * params["alpha"] + params["beta"]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv_to_pil(out), {"type": "faible_contraste", "level": level, **params}


# ---------------------------------------------------------------------------
# 5. Compression JPEG (artefacts de compression forte)
# ---------------------------------------------------------------------------

def compression_jpeg(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["compression_jpeg"][level])
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=params["quality"])
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    return out, {"type": "compression_jpeg", "level": level, **params}


# ---------------------------------------------------------------------------
# 6. Effet "scan dégradé" (grain + légers artefacts de seuillage)
# ---------------------------------------------------------------------------

def effet_scan_degrade(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["effet_scan_degrade"][level])
    arr = pil_to_cv(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    speckle = np.random.randint(0, params["threshold_noise"], gray.shape, dtype=np.uint8)
    gray_noisy = cv2.subtract(gray, speckle // 2)
    gray_noisy = cv2.add(gray_noisy, speckle // 2)

    if params["blur"] > 0:
        k = params["blur"] * 2 + 1
        gray_noisy = cv2.GaussianBlur(gray_noisy, (k, k), 0)

    out = cv2.cvtColor(gray_noisy, cv2.COLOR_GRAY2BGR)
    return cv_to_pil(out), {"type": "effet_scan_degrade", "level": level, **params}


# ---------------------------------------------------------------------------
# 7. Distorsion / décalage géométrique léger
# ---------------------------------------------------------------------------

def distorsion_decalage(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["distorsion_decalage"][level])
    arr = pil_to_cv(img)
    h, w = arr.shape[:2]

    shift_x = random.uniform(-params["shift_px"], params["shift_px"])
    shift_y = random.uniform(-params["shift_px"], params["shift_px"])
    shear = random.uniform(-params["shear"], params["shear"])

    M = np.array([[1, shear, shift_x],
                  [shear, 1, shift_y]], dtype=np.float32)
    out = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return cv_to_pil(out), {
        "type": "distorsion_decalage", "level": level,
        "shift_x": round(shift_x, 2), "shift_y": round(shift_y, 2), "shear": round(shear, 4),
        **params,
    }


# ---------------------------------------------------------------------------
# 8. Ombres / zones assombries
# ---------------------------------------------------------------------------

def ombres(img: Image.Image, level: str = "moyen", seed: int | None = None) -> Tuple[Image.Image, dict]:
    _set_seed(seed)
    params = dict(PARAM_GRID["ombres"][level])
    out = img.convert("RGB").copy()
    w, h = out.size
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    shadow_boxes = []
    for _ in range(params["n_shadows"]):
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        x1 = x0 + random.randint(w // 4, w // 2)
        y1 = y0 + random.randint(h // 4, h // 2)
        alpha = int(255 * params["opacity"])
        draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, alpha))
        shadow_boxes.append([x0, y0, x1, y1])

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=25))
    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    return out, {"type": "ombres", "level": level, "zones": shadow_boxes, **params}


# ---------------------------------------------------------------------------
# Registre des dégradations
# ---------------------------------------------------------------------------

DEGRADATIONS: Dict[str, Callable] = {
    "flou_gaussien": flou_gaussien,
    "bruit_aleatoire": bruit_aleatoire,
    "rotation_legere": rotation_legere,
    "faible_contraste": faible_contraste,
    "compression_jpeg": compression_jpeg,
    "effet_scan_degrade": effet_scan_degrade,
    "distorsion_decalage": distorsion_decalage,
    "ombres": ombres,
}


def apply_degradation(img: Image.Image, degradation_type: str, level: str = "moyen",
                       seed: int | None = None) -> Tuple[Image.Image, dict]:
    """Applique UNE dégradation nommée à une image, à un niveau donné."""
    if degradation_type not in DEGRADATIONS:
        raise ValueError(f"Dégradation inconnue : {degradation_type}. "
                          f"Choix possibles : {list(DEGRADATIONS.keys())}")
    if level not in LEVELS:
        raise ValueError(f"Niveau inconnu : {level}. Choix possibles : {LEVELS}")
    return DEGRADATIONS[degradation_type](img, level=level, seed=seed)


# ---------------------------------------------------------------------------
# Génération du benchmark complet sur un dataset (ex : FUNSD)
# ---------------------------------------------------------------------------

def generate_degraded_dataset(
    manifest_path: Path,
    raw_images_dir: Path,
    output_dir: Path,
    degradations: List[str] | None = None,
    levels: List[str] | None = None,
    seed: int = 42,
) -> Path:
    """
    Applique toutes les dégradations x tous les niveaux à tous les documents
    listés dans `manifest_path`, sauvegarde les images dégradées dans
    `output_dir/images/` et écrit un manifest JSON récapitulatif.

    Nom de fichier de sortie :
        <nom_original_sans_ext>__<degradation>__<level>.png

    Retourne le chemin du manifest JSON généré.
    """
    degradations = degradations or list(DEGRADATIONS.keys())
    levels = levels or LEVELS

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(manifest) * len(degradations) * len(levels)
    done = 0

    for entry in manifest:
        img_name = entry["image"]
        img_path = Path(raw_images_dir) / img_name
        base_img = Image.open(img_path).convert("RGB")
        stem = Path(img_name).stem

        for deg_name in degradations:
            for level in levels:
                # seed dérivée pour reproductibilité par (doc, deg, level)
                local_seed = seed + hash((stem, deg_name, level)) % 10_000
                degraded_img, params = apply_degradation(base_img, deg_name, level, seed=local_seed)

                out_filename = f"{stem}__{deg_name}__{level}.png"
                degraded_img.save(images_out / out_filename)

                records.append({
                    "document_original": img_name,
                    "annotation_originale": entry.get("annotation"),
                    "image_degradee": out_filename,
                    "degradation": deg_name,
                    "level": level,
                    "seed": local_seed,
                    "params": params,
                })
                done += 1

    manifest_out_path = output_dir / "manifest_degraded.json"
    manifest_out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {done}/{total} images dégradées générées dans {images_out}")
    print(f"[OK] Manifest écrit : {manifest_out_path}")
    return manifest_out_path


if __name__ == "__main__":
    # Petit test manuel : python src/degradation.py
    # (nécessite data/raw/manifest.json + data/raw/images/ générés en Phase 1)
    import argparse

    parser = argparse.ArgumentParser(description="Génère le benchmark de documents dégradés (Phase 2).")
    parser.add_argument("--manifest", type=str, default="data/raw/manifest.json")
    parser.add_argument("--images_dir", type=str, default="data/raw/images")
    parser.add_argument("--output_dir", type=str, default="data/degraded")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_degraded_dataset(
        manifest_path=Path(args.manifest),
        raw_images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
