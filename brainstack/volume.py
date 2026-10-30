"""Building the 3D volume from aligned slices, measuring it, and exporting it."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .calibration import Calibration
from .registration import SliceTransform, apply_transform

UM3_PER_MM3 = 1e9
UM2_PER_MM2 = 1e6


@dataclass
class Volume:
    data: np.ndarray                 # (Z, H, W, C) uint8, 0..255
    mask: np.ndarray                 # (Z, H, W) bool tissue mask
    cal: Calibration
    channel_names: list[str]
    channel_colors: list[tuple[int, int, int]] = field(default_factory=list)
    crop_offset: tuple[int, int] = (0, 0)   # (row, col) of this volume's corner on the canvas

    @property
    def shape_zyx(self):
        return self.data.shape[:3]

    def physical_size_um(self) -> tuple[float, float, float]:
        """(width X, height Y, depth Z) of the whole volume box in µm."""
        z, h, w = self.shape_zyx
        return w * self.cal.xy_um, h * self.cal.xy_um, z * self.cal.z_um


def build_volume(processed, masks, transforms: list[SliceTransform], cal: Calibration,
                 channel_names, channel_colors, crop_margin_px: int = 8,
                 progress=None) -> Volume:
    n = len(processed)
    aligned, amasks = [], []
    for i in range(n):
        a = apply_transform(processed[i].astype(np.float32), transforms[i])
        aligned.append((np.clip(a, 0, 1) * 255).astype(np.uint8))
        amasks.append(apply_transform(masks[i], transforms[i], nearest=True))
        if progress:
            progress(int(90 * (i + 1) / n), f"Placing slice {i + 1}/{n}")
    data = np.stack(aligned)            # (Z, H, W, C)
    mask = np.stack(amasks)             # (Z, H, W)

    # Crop to the tissue so the 3D view isn't mostly empty space.
    union = mask.any(axis=0)
    y0 = x0 = 0
    if union.any():
        rows = np.where(union.any(axis=1))[0]
        cols = np.where(union.any(axis=0))[0]
        y0 = max(0, rows[0] - crop_margin_px)
        y1 = min(union.shape[0], rows[-1] + crop_margin_px + 1)
        x0 = max(0, cols[0] - crop_margin_px)
        x1 = min(union.shape[1], cols[-1] + crop_margin_px + 1)
        data = data[:, y0:y1, x0:x1]
        mask = mask[:, y0:y1, x0:x1]

    if progress:
        progress(100, "Volume ready")
    return Volume(np.ascontiguousarray(data), np.ascontiguousarray(mask), cal,
                  list(channel_names), list(channel_colors), crop_offset=(int(y0), int(x0)))


# ---------------------------------------------------------------- measurements
def slice_areas_mm2(vol: Volume) -> np.ndarray:
    px_area = vol.cal.xy_um ** 2
    return vol.mask.reshape(vol.mask.shape[0], -1).sum(axis=1) * px_area / UM2_PER_MM2


def tissue_bounding_size_um(vol: Volume) -> tuple[float, float, float]:
    """Width, height, depth of the tissue itself (not the padded box)."""
    m = vol.mask
    if not m.any():
        return 0.0, 0.0, 0.0
    zs = np.where(m.any(axis=(1, 2)))[0]
    ys = np.where(m.any(axis=(0, 2)))[0]
    xs = np.where(m.any(axis=(0, 1)))[0]
    return (
        (xs[-1] - xs[0] + 1) * vol.cal.xy_um,
        (ys[-1] - ys[0] + 1) * vol.cal.xy_um,
        (zs[-1] - zs[0] + 1) * vol.cal.z_um,
    )


def tissue_volume_mm3(vol: Volume) -> float:
    """Cavalieri estimate: sum of slice areas x slice spacing."""
    return float(vol.mask.sum()) * vol.cal.voxel_volume_um3 / UM3_PER_MM3


def channel_stats(vol: Volume, thresholds: list[int]) -> list[dict]:
    """Per channel: volume of tissue where signal is above the threshold."""
    out = []
    tissue_vox = max(1, int(vol.mask.sum()))
    for c, name in enumerate(vol.channel_names):
        ch = vol.data[..., c]
        pos = (ch >= thresholds[c]) & vol.mask
        n_pos = int(pos.sum())
        out.append({
            "channel": name,
            "threshold": int(thresholds[c]),
            "signal_volume_mm3": n_pos * vol.cal.voxel_volume_um3 / UM3_PER_MM3,
            "percent_of_tissue": 100.0 * n_pos / tissue_vox,
            "mean_intensity_in_tissue": float(ch[vol.mask].mean()) if vol.mask.any() else 0.0,
        })
    return out


def distance_um(p1, p2) -> float:
    return float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p2, float)))


def mask_surface(mask: np.ndarray, cal: Calibration, max_px: int = 300,
                 smooth_iterations: int = 30, sigma=(1.0, 1.0, 1.0)):
    """Surface of a 3D (Z, H, W) mask as a triangle mesh in real µm.

    Uses the same coordinates as the 3D view: X right, Y up (image flipped), Z = slice.
    """
    import pyvista as pv
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes

    step = max(1, int(np.ceil(max(mask.shape[1:]) / max_px)))
    mb = mask[:, ::step, ::step].astype(bool)
    if not mb.any():
        return None
    mb = np.pad(mb, 1)                                 # close the ends of the shape
    sx = cal.xy_um * step

    # Shape-based interpolation: turn every slice into a signed distance map
    # (negative inside, positive outside) and interpolate THOSE between slices.
    # Outlines then morph smoothly from one slice to the next, instead of the
    # flat terraces you get from interpolating a yes/no mask.
    lim = 20.0
    sd = np.empty(mb.shape, np.float32)
    for i, sl in enumerate(mb):
        if not sl.any():
            sd[i] = lim
        else:
            sd[i] = np.clip(ndi.distance_transform_edt(~sl) - ndi.distance_transform_edt(sl),
                            -lim, lim)
    sd = ndi.gaussian_filter(sd, (0, sigma[1], sigma[2]))
    up = int(np.clip(round(cal.z_um / sx), 1, 4))
    dz = cal.z_um
    if up > 1:
        sd = ndi.zoom(sd, (up, 1, 1), order=1)
        # zoom keeps both end samples, so the padded span ((Z+1) gaps) is unchanged
        dz = cal.z_um * (mb.shape[0] - 1) / (sd.shape[0] - 1)
    verts, faces, _, _ = marching_cubes(-sd, 0.0, spacing=(dz, sx, sx))
    h = (mask.shape[1] - 1) * cal.xy_um
    # (z, y, x) -> (x, flipped y, z), undo the padding offset
    pts = np.column_stack([verts[:, 2] - sx, h - (verts[:, 1] - sx), verts[:, 0] - cal.z_um])
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(pts, faces_pv)
    if smooth_iterations:
        mesh = mesh.smooth_taubin(n_iter=smooth_iterations, pass_band=0.05)
    return mesh.compute_normals(auto_orient_normals=True)


def tissue_surface(vol: Volume, smooth_iterations: int = 30, max_px: int = 300):
    """Outer surface of the brain tissue (from the tissue masks)."""
    return mask_surface(vol.mask, vol.cal, max_px, smooth_iterations)


# --------------------------------------------------------------------- export
def export_surface(vol: Volume, path: str) -> None:
    """STL / OBJ / PLY / VTK surface mesh, units = micrometres."""
    tissue_surface(vol).save(path)


def export_tiff(vol: Volume, path: str) -> None:
    """ImageJ/Fiji-compatible stack with real-world spacing embedded."""
    import tifffile

    zcyx = np.moveaxis(vol.data, -1, 1)  # (Z, C, Y, X)
    tifffile.imwrite(
        path,
        zcyx,
        imagej=True,
        resolution=(1.0 / vol.cal.xy_um, 1.0 / vol.cal.xy_um),
        metadata={"spacing": vol.cal.z_um, "unit": "um", "axes": "ZCYX"},
    )


def export_measurements_csv(vol: Volume, thresholds: list[int], path: str,
                            distances: list[float] | None = None) -> None:
    w, h, d = tissue_bounding_size_um(vol)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Summary"])
        wr.writerow(["Calibration method", vol.cal.method, vol.cal.note])
        wr.writerow(["Pixel size XY (um)", f"{vol.cal.xy_um:.4f}"])
        wr.writerow(["Slice spacing Z (um)", f"{vol.cal.z_um:.4f}"])
        wr.writerow(["Number of slices", vol.data.shape[0]])
        wr.writerow(["Tissue width (um)", f"{w:.1f}"])
        wr.writerow(["Tissue height (um)", f"{h:.1f}"])
        wr.writerow(["Tissue depth (um)", f"{d:.1f}"])
        wr.writerow(["Tissue volume (mm3)", f"{tissue_volume_mm3(vol):.4f}"])
        wr.writerow([])
        wr.writerow(["Channel", "Threshold (0-255)", "Signal volume (mm3)",
                     "% of tissue", "Mean intensity in tissue (0-255)"])
        for s in channel_stats(vol, thresholds):
            wr.writerow([s["channel"], s["threshold"], f"{s['signal_volume_mm3']:.4f}",
                         f"{s['percent_of_tissue']:.2f}", f"{s['mean_intensity_in_tissue']:.1f}"])
        wr.writerow([])
        wr.writerow(["Slice", "Z position (um)", "Tissue area (mm2)"])
        for i, a in enumerate(slice_areas_mm2(vol)):
            wr.writerow([i + 1, f"{i * vol.cal.z_um:.1f}", f"{a:.4f}"])
        if distances:
            wr.writerow([])
            wr.writerow(["Measured distance #", "Distance (um)"])
            for i, dd in enumerate(distances):
                wr.writerow([i + 1, f"{dd:.1f}"])
