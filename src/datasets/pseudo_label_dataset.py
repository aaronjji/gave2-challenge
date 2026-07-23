"""Self-training dataset: pseudo-labels derived from our own validated 5-fold
ensemble's predictions on the unlabeled competition validation set.

Confidence-masked: only pixels the ensemble is confident about (probability
far from 0.5) contribute to the loss, via the same `roi` masking mechanism
already used for field-of-view -- low-confidence pixels are treated as
"outside ROI" for this sample. This is the key safeguard against confirmation
bias (a model reinforcing its own systematic mistakes, e.g. on vein topology,
by training on its own uncertain guesses there).
"""
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("L"))


class PseudoLabelDataset(Dataset):
    """use_ffa=False -> Task1 (3ch CFP), True -> Task2 (5ch CFP+FFA_A+FFA_AV).

    Expects:
      images_dir/{name}.png                 -- original CFP image
      masks_dir/{name}.png                  -- real field-of-view ROI mask
      pred_dir/{name}.png                   -- ensemble's quantized probability map (R=artery,G=vessel,B=vein)
      ffa_root/FFA_A, FFA_AV/{name}.png     -- Task2 only
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        pred_dir: str,
        case_names: list[str],
        patch_size: int = 384,
        use_ffa: bool = False,
        ffa_root: str | None = None,
        confidence_low: int = 40,   # ~0.157 * 255 -- prob below this = confidently "not this class"
        confidence_high: int = 215,  # ~0.843 * 255 -- prob above this = confidently "this class"
        min_roi_frac: float = 0.05,  # retry the crop if fewer than this fraction of pixels are usable (confident+in-FOV) -- an empty ROI makes BCE NaN (0/0)
        max_crop_attempts: int = 10,
        seed: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.pred_dir = Path(pred_dir)
        self.case_names = case_names
        self.patch_size = patch_size
        self.use_ffa = use_ffa
        self.ffa_root = Path(ffa_root) if ffa_root else None
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.min_roi_frac = min_roi_frac
        self.max_crop_attempts = max_crop_attempts
        self.rng = np.random.default_rng(seed)

        geo = [
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), shear=(-5, 5), p=0.5, border_mode=0),
        ]
        photo = [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3),
            A.CLAHE(p=0.3),
        ]
        additional_targets = {"label": "image", "roi": "mask"}
        if use_ffa:
            additional_targets["ffa_a"] = "image"
            additional_targets["ffa_av"] = "image"
        self.transform = A.Compose(geo + photo, additional_targets=additional_targets)

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        name = self.case_names[idx]
        image = _read_rgb(self.images_dir / f"{name}.png")
        roi = _read_gray(self.masks_dir / f"{name}.png")
        probs = _read_rgb(self.pred_dir / f"{name}.png")  # quantized 0-255 probability, R/G/B

        # binary pseudo-label + confidence mask (per-channel: confident where far from 0.5)
        pseudo_label = (probs > 127).astype(np.uint8) * 255
        confident = (probs < self.confidence_low) | (probs > self.confidence_high)  # HxWx3 bool
        confident_all_ch = confident.all(axis=-1)  # require all 3 channels confident to keep it simple
        effective_roi = ((roi > 0) & confident_all_ch).astype(np.uint8) * 255

        kwargs = {"image": image, "label": pseudo_label, "roi": effective_roi}
        if self.use_ffa:
            kwargs["ffa_a"] = _read_gray(self.ffa_root / "FFA_A" / f"{name}.png")[..., None].repeat(3, axis=2)
            kwargs["ffa_av"] = _read_gray(self.ffa_root / "FFA_AV" / f"{name}.png")[..., None].repeat(3, axis=2)

        for attempt in range(self.max_crop_attempts):
            out = self.transform(**kwargs)
            if (out["roi"] > 0).mean() >= self.min_roi_frac:
                break

        img_t = torch.from_numpy(out["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.use_ffa:
            a_t = torch.from_numpy(out["ffa_a"][..., :1].astype(np.float32) / 255.0).permute(2, 0, 1)
            av_t = torch.from_numpy(out["ffa_av"][..., :1].astype(np.float32) / 255.0).permute(2, 0, 1)
            img_t = torch.cat([img_t, a_t, av_t], dim=0)

        label_t = torch.from_numpy(out["label"].astype(np.float32) / 255.0).permute(2, 0, 1)
        roi_t = torch.from_numpy((out["roi"] > 0).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)

        return {"image": img_t, "label": label_t, "roi": roi_t, "case_id": -1}
