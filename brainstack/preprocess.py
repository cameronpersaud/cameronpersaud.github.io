"""Clean-up for fluorescence slices: background, noise, brightness, tissue mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class PreprocessSettings:
    subtract_background: bool = True
    denoise_sigma: float = 1.0          # Gaussian blur in pixels (0 = off)
    normalize: str = "stack"            # "stack" (keeps real brightness differences) or "slice"
    low_percentile: float = 1.0
    high_percentile: float = 99.97     # high, so a small bright dye deposit isn't clipped
    mask_sensitivity: float = 1.0       # <1 = stricter tissue mask, >1 = looser
    # Channel(s) used to find the brain outline: None = all, an int, or a tuple of ints
    # (e.g. every channel except the dye).
    mask_channel: int | tuple | None = None


def _border_median(x: np.ndarray) -> float:
    return float(np.median(np.concatenate([x[:4].ravel(), x[-4:].ravel(),
                                           x[:, :4].ravel(), x[:, -4:].ravel()])))


def tissue_mask(image: np.ndarray, sensitivity: float = 1.0, channel=None) -> np.ndarray:
    """Boolean mask of tissue (bright) vs. slide background (dark).

    channel=None averages all channels (good default); for best results pick a
    channel that stains ALL tissue, e.g. DAPI / Hoechst / NeuroTrace.
    """
    from skimage.filters import threshold_otsu

    img = image.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if channel is None:
        chans = list(range(img.shape[2]))
    elif isinstance(channel, (int, np.integer)):
        chans = [int(channel)]
    else:
        chans = [int(c) for c in channel if 0 <= int(c) < img.shape[2]] or list(range(img.shape[2]))
    parts = []
    for c in chans:
        x = img[..., c]
        bg = _border_median(x)
        hi = np.percentile(x, 99.9)
        parts.append(np.clip((x - bg) / (hi - bg + 1e-6), 0, 1))
    combined = np.mean(parts, axis=0)
    sm = ndi.gaussian_filter(combined, sigma=max(3.0, combined.shape[0] / 150))
    if sm.max() - sm.min() < 1e-6:
        return np.zeros(sm.shape, dtype=bool)
    thr = threshold_otsu(sm) * 0.6 / max(sensitivity, 1e-3)  # tissue edges are dim; be generous
    mask = sm > thr
    mask = ndi.binary_closing(mask, iterations=3)
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    # Keep the big pieces (a brain cracked in two, both hemispheres) but drop loose
    # fragments and bits of neighbouring sections that often sit at a photo's edge.
    keep = np.where(sizes >= 0.12 * sizes.max())[0] + 1
    mask = np.isin(labels, keep)
    labels, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
        tiny = np.where(sizes < max(16, mask.sum() * 0.001))[0] + 1
        mask[np.isin(labels, tiny)] = False
    return mask


def preprocess_stack(images, settings: PreprocessSettings, progress=None):
    """Returns (processed images scaled 0..1, tissue masks)."""
    n = len(images)
    n_ch = images[0].shape[2]
    masks = [tissue_mask(img, settings.mask_sensitivity, settings.mask_channel) for img in images]

    cleaned = []
    for i, img in enumerate(images):
        out = img.astype(np.float32).copy()
        for c in range(n_ch):
            x = out[..., c]
            if settings.subtract_background:
                outside = x[~masks[i]]
                bg = np.median(outside) if outside.size > 100 else np.percentile(x, 5)
                x = np.clip(x - bg, 0, None)
            if settings.denoise_sigma > 0:
                x = ndi.gaussian_filter(x, settings.denoise_sigma)
            out[..., c] = x
        cleaned.append(out)
        if progress:
            progress(int(60 * (i + 1) / n), f"Cleaning slice {i + 1}/{n}")

    # Brightness normalisation to 0..1
    processed = []
    if settings.normalize == "stack":
        lows, highs = [], []
        for c in range(n_ch):
            sample = np.concatenate(
                [im[..., c][::4, ::4].ravel() for im in cleaned]
            )
            lo, hi = np.percentile(sample, (settings.low_percentile, settings.high_percentile))
            lows.append(lo)
            highs.append(hi)
    for i, im in enumerate(cleaned):
        out = np.empty_like(im)
        for c in range(n_ch):
            if settings.normalize == "slice":
                lo, hi = np.percentile(im[..., c], (settings.low_percentile, settings.high_percentile))
            else:
                lo, hi = lows[c], highs[c]
            out[..., c] = np.clip((im[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
        processed.append(out.astype(np.float16))  # half the memory
        if progress:
            progress(60 + int(40 * (i + 1) / n), f"Normalising slice {i + 1}/{n}")
    return processed, masks
