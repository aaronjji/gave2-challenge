"""Post-processing filters for predicted AV masks, targeting a specific
failure mode found via a leakage-free diagnostic against real GT biomarker
labels (2026-07-22): some predicted masks contain large blobby false-positive
regions with no vessel-like shape (not thin/elongated), which distort
density/fractal-dimension/caliber biomarkers far more than the same pixel
count of correctly-shaped thin vessel would -- visually confirmed on g_040
(fold3), where a solid hallucinated region inflated the vein mask ~2.3x over
the true GT pixel count even after the extract_av_masks G&~R filtering.

First attempt filtered by whole-connected-component "effective width" (area /
skeleton length) and only removed ~3.5% of the excess -- the blob was
touching the real thin vessel network, so connected-component analysis lumped
them into one object and the legitimate vessels' skeleton length diluted the
average. Fixed by going local instead of per-component: erode the mask enough
that anything narrower than max_width disappears entirely (thin vessels
vanish, only blob cores survive), dilate the survivors back out to
~their original extent, and subtract that from the original mask. This
removes blobby regions wherever they are, regardless of what they're
topologically connected to.
"""
import numpy as np
from scipy import ndimage


def filter_bloblike_components(binary_mask: np.ndarray, max_width: int = 20) -> np.ndarray:
    """binary_mask: HxW bool or 0/255 uint8. Returns a filtered HxW bool mask
    with locally-blobby (wider than max_width) regions removed, regardless of
    whether they're connected to genuine thin vessel structure."""
    mask_bool = binary_mask > 0
    if not mask_bool.any():
        return mask_bool

    struct = ndimage.generate_binary_structure(2, 1)
    radius = max(max_width // 2, 1)
    eroded = ndimage.binary_erosion(mask_bool, structure=struct, iterations=radius)
    if not eroded.any():
        return mask_bool  # nothing survived erosion -- no blob cores, leave mask untouched

    blob_extent = ndimage.binary_dilation(eroded, structure=struct, iterations=radius + 2)
    return mask_bool & ~blob_extent
