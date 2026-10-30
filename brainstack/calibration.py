"""Physical calibration: how big is one voxel, in microns?

Each voxel in the 3D model is  (xy_um) x (xy_um) x (z_um):

    xy_um = microns per pixel at the working resolution
    z_um  = distance between consecutive slices (centre to centre)

Three ways to get xy_um, from most to least accurate:

1. MICROSCOPE  - you know microns/pixel for the objective + camera you used.
                 Same setup => identical scale for every brain. Best option.
2. SCALE BAR   - click both ends of a scale bar burned into an image and type
                 its length. Accurate if the bar is in the image.
3. TISSUE SIZE - you know (or estimate) the real height of the tissue, e.g.
                 6,000 um. The software finds the slice with the TALLEST tissue
                 in the stack, measures its height in pixels (rotation-proof
                 caliper measurement, so tilted slices don't inflate it), and computes
                     xy_um = real_height_um / tissue_height_px
                 That ONE number is used for every slice, so smaller slices
                 (e.g. near the front of the brain) correctly come out smaller.
                 It is an estimate: if your real brain is 5,600 um tall, all
                 sizes will be ~7% too big. Use 1 or 2 for publishable numbers.

Note on the "height / 6,000" idea: the pixel size is
6,000 um / (tissue height in pixels) - real distance divided by pixel count -
not the other way round. Z is NOT scaled by the image: it is simply the slice
spacing you enter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import tissue_mask


@dataclass
class Calibration:
    xy_um: float          # microns per WORKING pixel (after downscaling)
    z_um: float           # microns between slices
    method: str           # "microscope" | "scale_bar" | "tissue_size"
    note: str = ""

    @property
    def voxel_volume_um3(self) -> float:
        return self.xy_um * self.xy_um * self.z_um

    @property
    def anisotropy(self) -> float:
        """How many times deeper than wide each voxel is."""
        return self.z_um / self.xy_um

    def describe(self) -> str:
        return (
            f"Each voxel = {self.xy_um:.3g} x {self.xy_um:.3g} x {self.z_um:.3g} µm "
            f"(Z is {self.anisotropy:.2g}x the pixel size)"
        )


def from_microscope(original_um_per_px: float, scale_factor: float, z_um: float) -> Calibration:
    """Known microns/pixel of the ORIGINAL images."""
    if original_um_per_px <= 0:
        raise ValueError("Microns per pixel must be greater than 0.")
    return Calibration(
        xy_um=original_um_per_px * scale_factor,
        z_um=z_um,
        method="microscope",
        note=f"{original_um_per_px:g} µm/px in original images x {scale_factor:.3g} downscale",
    )


def from_scale_bar(p1, p2, bar_length_um: float, z_um: float) -> Calibration:
    """p1, p2 = (x, y) ends of the scale bar in WORKING pixels."""
    dist_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if dist_px < 2:
        raise ValueError("The two scale bar points are too close together.")
    return Calibration(
        xy_um=bar_length_um / dist_px,
        z_um=z_um,
        method="scale_bar",
        note=f"{bar_length_um:g} µm bar = {dist_px:.1f} px",
    )


def feret_diameters_px(mask: np.ndarray) -> tuple[float, float]:
    """(shortest, longest) caliper width of the tissue, in pixels.

    Rotation-independent: a slice mounted at an angle measures the same as an
    upright one (a plain bounding box would not).
    """
    from skimage.measure import find_contours

    if not mask.any():
        return 0.0, 0.0
    pts = np.vstack(find_contours(np.pad(mask, 1).astype(float), 0.5))
    angles = np.deg2rad(np.arange(0, 180, 0.5))
    proj = pts @ np.stack([np.cos(angles), np.sin(angles)])   # (N, n_angles)
    widths = proj.max(axis=0) - proj.min(axis=0)
    return float(widths.min()), float(widths.max())


def main_tissue(mask: np.ndarray) -> np.ndarray:
    """Keep the big tissue pieces; drop loose fragments and debris that would
    otherwise make the tissue look bigger than it is."""
    from scipy import ndimage as ndi

    labels, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.where(sizes >= 0.15 * sizes.max())[0] + 1
    return np.isin(labels, keep)


def tissue_extent_px(image: np.ndarray, axis: str = "height") -> float:
    """axis="height": shortest tissue dimension (dorsal-ventral height of a
    coronal or sagittal slice). axis="width": longest dimension."""
    shortest, longest = feret_diameters_px(main_tissue(tissue_mask(image)))
    return shortest if axis == "height" else longest


def reference_slice(extents) -> int:
    """The slice with the largest tissue, ignoring implausible outliers
    (e.g. a slice whose outline swallowed lots of debris)."""
    e = np.asarray(extents, float)
    ok = e <= 1.25 * np.median(e[e > 0]) if (e > 0).any() else e >= 0
    cand = np.where(ok)[0] if ok.any() else np.arange(len(e))
    return int(cand[np.argmax(e[cand])])


def from_tissue_size(
    images: list[np.ndarray],
    real_size_um: float,
    z_um: float,
    axis: str = "height",
    reference_index: int | None = None,
) -> tuple[Calibration, list[int], int]:
    """Calibrate from the known real height (or width) of the tissue.

    reference_index=None -> use the slice with the largest tissue extent.
    Returns (calibration, per-slice extents in px, reference slice index).
    """
    extents = [tissue_extent_px(img, axis) for img in images]
    if reference_index is None:
        reference_index = reference_slice(extents)
    ref_px = extents[reference_index]
    if ref_px <= 0:
        raise ValueError("Could not find tissue in the reference slice.")
    cal = Calibration(
        xy_um=real_size_um / ref_px,
        z_um=z_um,
        method="tissue_size",
        note=(
            f"Slice {reference_index + 1}: tissue {axis} = {ref_px:.0f} px = {real_size_um:g} µm "
            "(estimate)"
        ),
    )
    return cal, extents, reference_index
