"""
build_manifest.py

Construit data/raw/manifest.json et data/raw_test/manifest.json à partir du
dataset FUNSD officiel (dossiers training_data/ et testing_data/, chacun avec
un sous-dossier images/ et annotations/).

Usage (depuis la racine du repo OCR_Layout_Semantique) :

    python build_manifest.py --dataset_dir chemin/vers/dataset

Ça va créer :
    data/raw/manifest.json
    data/raw/images/...        (copie des images d'entraînement)
    data/raw/annotations/...   (copie des annotations d'entraînement)
    data/raw_test/manifest.json
    data/raw_test/images/...
    data/raw_test/annotations/...
"""

import argparse
import json
import shutil
from pathlib import Path


def build_split(src_dir: Path, dst_dir: Path, split_name: str) -> Path:
    src_images = src_dir / "images"
    src_annotations = src_dir / "annotations"

    dst_images = dst_dir / "images"
    dst_annotations = dst_dir / "annotations"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_annotations.mkdir(parents=True, exist_ok=True)

    manifest = []
    image_files = sorted(src_images.glob("*.png"))

    for img_path in image_files:
        stem = img_path.stem
        ann_path = src_annotations / f"{stem}.json"
        if not ann_path.exists():
            print(f"[ATTENTION] annotation manquante pour {img_path.name}, ignoré.")
            continue

        shutil.copy2(img_path, dst_images / img_path.name)
        shutil.copy2(ann_path, dst_annotations / ann_path.name)

        manifest.append({
            "image": img_path.name,
            "annotation": ann_path.name,
        })

    manifest_path = dst_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {split_name} : {len(manifest)} documents -> {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Construit les manifest.json à partir du dataset FUNSD brut.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="Dossier contenant training_data/ et testing_data/ (le dataset FUNSD dézippé).")
    parser.add_argument("--output_root", type=str, default="data",
                         help="Dossier de sortie racine (par défaut : data/).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_root)

    train_src = dataset_dir / "training_data"
    test_src = dataset_dir / "testing_data"

    if not train_src.exists():
        raise FileNotFoundError(f"Dossier introuvable : {train_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"Dossier introuvable : {test_src}")

    build_split(train_src, output_root / "raw", "train")
    build_split(test_src, output_root / "raw_test", "test")

    print("\nTerminé. Structure générée :")
    print(f"  {output_root}/raw/manifest.json")
    print(f"  {output_root}/raw/images/")
    print(f"  {output_root}/raw/annotations/")
    print(f"  {output_root}/raw_test/manifest.json")
    print(f"  {output_root}/raw_test/images/")
    print(f"  {output_root}/raw_test/annotations/")


if __name__ == "__main__":
    main()
