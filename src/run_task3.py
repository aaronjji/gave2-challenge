"""Compute Task 3 vascular biomarkers from Task 1/2 AV predictions.

Usage:
    python src/run_task3.py --av-dir predictions/task1/validation \
        --images-dir data/raw/GAVE2_preliminary/validation/images \
        --masks-dir data/raw/GAVE2_preliminary/validation/masks \
        --out-dir predictions/task3/validation
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biomarkers.compute import compute_biomarkers  # noqa: E402
from od_localization.heuristic import find_od_heuristic  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-dir", type=str, required=True, help="Directory of prediction-format AV PNGs (R=artery,G=vessel,B=vein)")
    p.add_argument("--images-dir", type=str, required=True, help="Original CFP images (for heuristic OD detection)")
    p.add_argument("--masks-dir", type=str, required=True, help="ROI masks (for heuristic OD detection)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--threshold", type=int, default=127)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    av_dir = Path(args.av_dir)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    av_paths = sorted(av_dir.glob("*.png"))
    print(f"Computing Task3 biomarkers for {len(av_paths)} cases -> {out_dir}")
    for av_path in av_paths:
        name = av_path.stem
        try:
            av_prob = np.array(Image.open(av_path).convert("RGB"))
            av_bin = (av_prob > args.threshold).astype(np.uint8) * 255

            image = np.array(Image.open(images_dir / f"{name}.png").convert("RGB"))
            roi = np.array(Image.open(masks_dir / f"{name}.png").convert("L"))
            od_mask = find_od_heuristic(image, roi)

            result = compute_biomarkers(av_bin, od_mask)

            txt_path = out_dir / f"{name}.txt"
            with open(txt_path, "w") as f:
                for key in ["AVR", "artery_density", "vein_density", "artery_fractal_dimension", "vein_fractal_dimension"]:
                    value = result[key]
                    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
                        f.write(f"{key} 0.0\n")
                    else:
                        f.write(f"{key} {value:.6f}\n")
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            traceback.print_exc()
            # Write a zero-valued fallback so the submission stays complete.
            with open(out_dir / f"{name}.txt", "w") as f:
                for key in ["AVR", "artery_density", "vein_density", "artery_fractal_dimension", "vein_fractal_dimension"]:
                    f.write(f"{key} 0.0\n")


if __name__ == "__main__":
    main()
