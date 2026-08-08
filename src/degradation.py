"""
degradation.py
--------------------------------
Phase 2 : Construction d'un benchmark de dégradation documentaire.

Fournit une fonction de dégradation paramétrable, capable de simuler :
- flou gaussien
- bruit aléatoire (gaussien / sel & poivre)
- rotation légère
- faible contraste
- compression JPEG
- effet "scan dégradé" (bruit + contraste + légère binarisation)
- décalage / distorsion légère
- ombres / zones assombries

Chaque dégradation est paramétrable et les transformations appliquées
sont journalisées pour la reproductibilité.
"""

import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)

LEVELS = {
    "faible": 0.33,
    "moyen": 0.66,
    "fort": 1.0,
}


def gaussian_blur(img, level):
    k = max(1, int(round(2 + 6 * level)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def random_noise(img, level):
    sigma = 8 + 40 * level
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate_slight(img, level):
    angle = random.uniform(-1, 1) * (1 + 6 * level)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255)), angle


def low_contrast(img, level):
    alpha = 1.0 - 0.6 * level  # réduit le contraste
    beta = 40 * level  # éclaircit / grise l'image
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def jpeg_compression(img, level):
    quality = int(round(80 - 65 * level))
    quality = max(quality, 5)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec, quality


def scan_effect(img, level):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy = random_noise(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), level * 0.5)
    contrast = low_contrast(noisy, level * 0.6)
    return contrast


def shift_distortion(img, level):
    h, w = img.shape[:2]
    dx = int(round(random.uniform(-3, 3) * (1 + 4 * level)))
    dy = int(round(random.uniform(-3, 3) * (1 + 4 * level)))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255)), (dx, dy)


def shadow(img, level):
    h, w = img.shape[:2]
    overlay = img.copy()
    x0 = random.randint(0, w // 2)
    y0 = random.randint(0, h // 2)
    x1 = x0 + random.randint(w // 4, w // 2)
    y1 = y0 + random.randint(h // 4, h // 2)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    alpha = 0.15 + 0.35 * level
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


DEGRADATIONS = {
    "blur": gaussian_blur,
    "noise": random_noise,
    "rotation": rotate_slight,
    "low_contrast": low_contrast,
    "jpeg": jpeg_compression,
    "scan_effect": scan_effect,
    "shift": shift_distortion,
    "shadow": shadow,
}


def apply_degradation(image_path, degradation_name, level_name="moyen"):
    """Applique une dégradation nommée à un niveau donné (faible/moyen/fort)."""
    assert degradation_name in DEGRADATIONS, f"dégradation inconnue: {degradation_name}"
    assert level_name in LEVELS, f"niveau inconnu: {level_name}"

    img = cv2.imread(str(image_path))
    level = LEVELS[level_name]
    fn = DEGRADATIONS[degradation_name]
    result = fn(img, level)

    params = {}
    if isinstance(result, tuple):
        out_img, extra = result
        params["extra"] = extra
    else:
        out_img = result

    return out_img, {
        "degradation": degradation_name,
        "level_name": level_name,
        "level_value": level,
        **params,
    }


def build_degraded_dataset(raw_dir="data/raw", out_dir="data/degraded",
                            degradations=None, levels=("faible", "moyen", "fort")):
    """
    Génère, pour chaque image brute et chaque (dégradation, niveau),
    une version dégradée + sauvegarde les paramètres appliqués.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degradations = degradations or list(DEGRADATIONS.keys())
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    log = []
    for entry in manifest:
        img_path = raw_dir / "images" / entry["image"]
        stem = Path(entry["image"]).stem

        for deg_name in degradations:
            for level_name in levels:
                out_img, meta = apply_degradation(img_path, deg_name, level_name)
                out_name = f"{stem}__{deg_name}__{level_name}.png"
                cv2.imwrite(str(out_dir / out_name), out_img)
                meta.update({
                    "source_image": entry["image"],
                    "annotation": entry["annotation"],
                    "output_image": out_name,
                })
                log.append(meta)

    with open(out_dir / "degradation_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(log)} images dégradées générées dans {out_dir}")
    return log


if __name__ == "__main__":
    build_degraded_dataset()
