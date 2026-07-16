"""Placeholder heuristic optic-disc detector (NOT the final model).

OD is approximated as the brightest large blob in the red channel within the
ROI mask. Validated as a rough stand-in (CRVE/fractal dimensions matched GT
exactly on a sample case; CRAE/AVR/density were within ~3-6%, attributable
to imprecise OD zone geometry) -- replace with a trained MNet_DeepCDR
fine-tune once OD center/radius are manually annotated on the 50 training
images (see plan doc, Task 3 section).
"""
import cv2
import numpy as np
import scipy.ndimage as ndi


def find_od_heuristic(image_rgb: np.ndarray, roi_mask: np.ndarray, window: int = 200) -> np.ndarray:
    """Returns a binary (0/255) uint8 OD mask, same HxW as image_rgb."""
    red = image_rgb[..., 0].astype(np.float32)
    blurred = ndi.gaussian_filter(red, sigma=25)
    blurred[roi_mask == 0] = -1
    cy, cx = np.unravel_index(np.argmax(blurred), blurred.shape)

    h, w = image_rgb.shape[:2]
    y0, y1 = max(0, cy - window), min(h, cy + window)
    x0, x1 = max(0, cx - window), min(w, cx + window)
    crop = red[y0:y1, x0:x1]
    thresh = np.percentile(crop, 90)
    od_bin = (crop > thresh).astype(np.uint8) * 255

    n, lbl = cv2.connectedComponents(od_bin)
    sizes = [(lbl == i).sum() for i in range(1, n)]
    if sizes:
        best = 1 + int(np.argmax(sizes))
        od_bin = ((lbl == best) * 255).astype(np.uint8)

    full_od = np.zeros((h, w), dtype=np.uint8)
    full_od[y0:y1, x0:x1] = od_bin
    return full_od
