"""Finding the red dextran dye in 3D and describing where it is.

The dye channel is thresholded in 3D (inside the brain), small specks are
removed, and what remains is the dye "deposit(s)". Everything is reported in
real micrometres, and also relative to the brain (e.g. "62% of the way across").

Coordinate convention (same as the 3D view):
  X = left -> right in the images
  Y = image top -> bottom (reported as distance DOWN from the top of the brain)
  Z = slice 1 -> last slice
Image left/right is not necessarily the animal's left/right - that depends on
how the sections were mounted.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .volume import UM2_PER_MM2, UM3_PER_MM3, Volume, mask_surface

DYE_WORDS = ("dextran", "red", "tomato", "texas", "rhodamine", "alexa 594", "594", "568",
             "555", "cy3", "dye", "mcherry")


def guess_dye_channel(names: list[str], colors: list[tuple[int, int, int]],
                      sample_image: np.ndarray | None = None) -> int:
    """Pick the channel most likely to hold the red dye.

    1. A channel whose name mentions the dye (dextran, red, tomato, 594 ...).
    2. Otherwise the most "spot-like" channel in a sample slice: a dye deposit is
       a small, very bright region, so its brightest pixels stand far above the
       typical tissue level, unlike a stain that covers the whole brain.
    3. Otherwise the reddest display colour.
    """
    for i, n in enumerate(names):
        if any(w in n.lower() for w in DYE_WORDS):
            return i
    if sample_image is not None and sample_image.ndim == 3 and sample_image.shape[2] > 1:
        scores = []
        for c in range(sample_image.shape[2]):
            x = sample_image[..., c].astype(np.float32)[::2, ::2]
            x = x - np.median(x)
            p90, p999 = np.percentile(x, (90, 99.9))
            scores.append(p999 / (abs(p90) + 1e-3 * (abs(p999) + 1)))
        best = int(np.argmax(scores))
        if scores[best] > 2 * sorted(scores)[-2]:
            return best
    redness = [c[0] - max(c[1], c[2]) for c in colors]
    return int(np.argmax(redness))


@dataclass
class DyeSettings:
    channel: int = 0
    threshold: int | None = None      # 0-255; None = automatic ("any detectable dye")
    preset: str = "auto"              # "auto" | "core" | "manual"
    per_slice: bool = True            # auto: judge each slice against its own background
    # The dye channel is the ONLY channel (dye-only images): the "background" is then
    # the tissue's own glow, which varies a lot between regions, so the automatic
    # level sits 4 (not 8) spreads above the typical tissue level.
    dye_only: bool = False
    min_size_um3: float = 0.0         # deposits smaller than this are ignored (0 = auto)
    inside_brain_only: bool = True
    smooth: float = 1.0               # 3D smoothing (voxels) before thresholding


@dataclass
class Deposit:
    label: int
    volume_mm3: float
    centroid_um: tuple[float, float, float]     # (x, y_down, z)
    peak: int
    n_voxels: int


@dataclass
class DyeResult:
    settings: DyeSettings
    threshold: int
    auto_threshold: int
    core_threshold: int
    mask: np.ndarray                   # (Z, H, W) bool
    labels: np.ndarray                 # (Z, H, W) int, 0 = none
    deposits: list[Deposit] = field(default_factory=list)
    slice_thresholds: np.ndarray | None = None   # per-slice levels when per_slice is on

    @property
    def found(self) -> bool:
        return bool(self.deposits)


# ------------------------------------------------------------------ detection
def _smoothed_dye(vol: Volume, s: DyeSettings) -> np.ndarray:
    ch = vol.data[..., s.channel].astype(np.float32)
    if s.smooth > 0:
        # Z voxels are usually much thicker than XY pixels, so smooth less in Z.
        zs = s.smooth * min(1.0, vol.cal.xy_um / vol.cal.z_um)
        ch = ndi.gaussian_filter(ch, (zs, s.smooth, s.smooth))
    return ch


def automatic_threshold(values: np.ndarray, k: float = 8.0) -> int:
    """'Any detectable dye': clearly above the background noise of the dye channel.

    threshold = background median + 8 x robust noise spread, but never below 5%
    of the brightest dye (so a flat, noise-free background isn't called dye).
    """
    v = np.asarray(values, np.float32)
    if v.size == 0 or v.max() <= 0:
        return 255
    med = float(np.median(v))
    spread = float(np.median(np.abs(v - med))) * 1.4826
    spread = max(spread, 1.0)                     # 8-bit data: noise is at least 1 level
    top = float(np.percentile(v, 99.99))
    return int(np.clip(round(max(med + k * spread, 0.05 * top)), 3, 250))


def core_threshold(values: np.ndarray) -> int:
    """'Dense core only': at least half as bright as the brightest dye."""
    v = np.asarray(values, np.float32)
    if v.size == 0:
        return 255
    return int(np.clip(round(0.5 * float(np.percentile(v, 99.99))), 3, 250))


def detect_dye(vol: Volume, s: DyeSettings) -> DyeResult:
    ch = _smoothed_dye(vol, s)
    region = ndi.binary_dilation(vol.mask, iterations=2) if s.inside_brain_only else \
        np.ones(vol.mask.shape, bool)
    sample = ch[region][:: max(1, int(region.sum() // 2_000_000) or 1)]
    k = 4.0 if s.dye_only else 8.0
    auto = automatic_threshold(sample, k)
    core = max(auto, core_threshold(sample))
    if s.preset == "core":
        thr = core
    elif s.preset == "manual" and s.threshold is not None:
        thr = int(s.threshold)
    else:
        thr = auto

    slice_thr = None
    if s.preset == "auto" and s.per_slice:
        # Each slice against its OWN background: slices imaged brighter or dimmer
        # (exposure, staining, section thickness) are then treated fairly, and one
        # very bright slice can't hide the dye in all the others.
        slice_thr = np.full(ch.shape[0], float(auto))
        for z in range(ch.shape[0]):
            vals = ch[z][region[z]]
            if vals.size > 500:
                slice_thr[z] = automatic_threshold(vals[:: max(1, vals.size // 400_000)], k)
        thr = int(round(float(np.median(slice_thr[region.any(axis=(1, 2))])))) \
            if region.any() else auto
        mask = (ch >= slice_thr[:, None, None]) & region
    else:
        mask = (ch >= thr) & region
    voxel_um3 = vol.cal.voxel_volume_um3
    labels, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    deposits: list[Deposit] = []
    if n:
        idx = np.arange(1, n + 1)
        counts = ndi.sum(mask, labels, idx)
        # Default minimum size: 0.5% of the biggest deposit, so stray specks are dropped.
        min_vox = (s.min_size_um3 / voxel_um3) if s.min_size_um3 > 0 else max(4, 0.005 * counts.max())
        keep = idx[counts >= min_vox]
        mask = np.isin(labels, keep)
        labels, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            counts = ndi.sum(mask, labels, idx)
            weights = np.clip(ch, 0, None)
            coms = ndi.center_of_mass(weights, labels, idx)
            peaks = ndi.maximum(vol.data[..., s.channel], labels, idx)
            for k, (c, com, pk) in enumerate(zip(counts, coms, peaks)):
                deposits.append(Deposit(
                    label=k + 1,
                    volume_mm3=float(c) * voxel_um3 / UM3_PER_MM3,
                    centroid_um=_zyx_to_um(vol, com),
                    peak=int(pk),
                    n_voxels=int(c),
                ))
            deposits.sort(key=lambda d: -d.volume_mm3)
    return DyeResult(s, thr, auto, core, mask, labels, deposits, slice_thr)


def _zyx_to_um(vol: Volume, zyx) -> tuple[float, float, float]:
    z, y, x = zyx
    return (float(x) * vol.cal.xy_um, float(y) * vol.cal.xy_um, float(z) * vol.cal.z_um)


# ------------------------------------------------------------------ reporting
def brain_box_vox(vol: Volume):
    m = vol.mask
    zs = np.where(m.any(axis=(1, 2)))[0]
    ys = np.where(m.any(axis=(0, 2)))[0]
    xs = np.where(m.any(axis=(0, 1)))[0]
    return (xs[0], xs[-1]), (ys[0], ys[-1]), (zs[0], zs[-1])


def dye_report(vol: Volume, res: DyeResult) -> dict:
    """Everything a person wants to know about where the dye is."""
    cal = vol.cal
    brain_mm3 = float(vol.mask.sum()) * cal.voxel_volume_um3 / UM3_PER_MM3
    rep = {"found": res.found, "threshold": res.threshold, "auto_threshold": res.auto_threshold,
           "n_deposits": len(res.deposits), "brain_volume_mm3": brain_mm3}
    if not res.found:
        return rep
    m = res.mask
    total = float(m.sum()) * cal.voxel_volume_um3 / UM3_PER_MM3
    weights = np.where(m, vol.data[..., res.settings.channel].astype(np.float32), 0)
    com = ndi.center_of_mass(weights)
    cx, cy, cz = _zyx_to_um(vol, com)
    (bx0, bx1), (by0, by1), (bz0, bz1) = brain_box_vox(vol)
    bx0u, bx1u = bx0 * cal.xy_um, bx1 * cal.xy_um
    by0u, by1u = by0 * cal.xy_um, by1 * cal.xy_um
    bz0u, bz1u = bz0 * cal.z_um, bz1 * cal.z_um

    zs = np.where(m.any(axis=(1, 2)))[0]
    ys = np.where(m.any(axis=(0, 2)))[0]
    xs = np.where(m.any(axis=(0, 1)))[0]

    def pct(v, a, b):
        return 100.0 * (v - a) / (b - a) if b > a else 0.0

    # Brightest point, on lightly smoothed data so one saturated pixel can't win
    sm = ndi.gaussian_filter(vol.data[..., res.settings.channel].astype(np.float32), (0.5, 2, 2))
    peak_idx = np.unravel_index(np.argmax(np.where(m, sm, 0)), m.shape)
    px, py, pz = _zyx_to_um(vol, peak_idx)
    # Slice holding the most dye (total signal), robust to saturation
    per_slice = weights.reshape(m.shape[0], -1).sum(axis=1)

    rep.update({
        "dye_volume_mm3": total,
        "percent_of_brain": 100.0 * total / brain_mm3 if brain_mm3 else 0.0,
        "centre_um": (cx, cy, cz),
        # Each slice is z_um thick, so the brain starts half a slice before the first
        # slice's centre: measure Z from that outer face (keeps front/back symmetric).
        "centre_from_brain_edges_um": (cx - bx0u, cy - by0u, cz - bz0u + cal.z_um / 2),
        "centre_percent_of_brain": (pct(cx, bx0u, bx1u), pct(cy, by0u, by1u),
                                    pct(cz + cal.z_um / 2, bz0u, bz1u + cal.z_um)),
        "brain_size_um": (bx1u - bx0u, by1u - by0u, bz1u - bz0u + cal.z_um),
        "peak_um": (px, py, pz),
        "peak_slice": int(peak_idx[0]) + 1,
        "most_dye_slice": int(np.argmax(per_slice)) + 1,
        "centre_slice": int(round(com[0])) + 1,
        "first_slice": int(zs[0]) + 1,
        "last_slice": int(zs[-1]) + 1,
        "extent_um": ((xs[-1] - xs[0] + 1) * cal.xy_um, (ys[-1] - ys[0] + 1) * cal.xy_um,
                      (zs[-1] - zs[0] + 1) * cal.z_um),
        "slice_areas_mm2": m.reshape(m.shape[0], -1).sum(axis=1) * cal.xy_um ** 2 / UM2_PER_MM2,
        "deposits": res.deposits,
    })
    return rep


def dye_surface(vol: Volume, res: DyeResult, smooth_iterations: int = 20):
    return mask_surface(res.mask, vol.cal, max_px=500, smooth_iterations=smooth_iterations,
                        sigma=(0.7, 0.8, 0.8))


def export_dye_csv(vol: Volume, res: DyeResult, path: str, orientation=None) -> None:
    from .orientation import PLANES, Orientation, anatomical_position

    orientation = orientation or Orientation()
    rep = dye_report(vol, res)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Red dextran dye - location report"])
        w.writerow(["Pixel size XY (um)", f"{vol.cal.xy_um:.4f}"])
        w.writerow(["Slice spacing Z (um)", f"{vol.cal.z_um:.4f}"])
        w.writerow(["Calibration", vol.cal.method, vol.cal.note])
        w.writerow(["Slicing", orientation.describe()])
        w.writerow(["Dye channel", vol.channel_names[res.settings.channel]])
        w.writerow(["Threshold used (0-255)", rep["threshold"], "automatic suggestion",
                    rep["auto_threshold"]])
        w.writerow(["Brain volume (mm3)", f"{rep['brain_volume_mm3']:.4f}"])
        if not rep["found"]:
            w.writerow(["No dye found above the threshold"])
            return
        w.writerow(["Dye volume (mm3)", f"{rep['dye_volume_mm3']:.5f}"])
        w.writerow(["Dye as % of brain volume", f"{rep['percent_of_brain']:.3f}"])
        w.writerow(["Separate deposits", rep["n_deposits"]])
        an = anatomical_position(rep, orientation)
        w.writerow([])
        w.writerow(["Dye centre - anatomical", "um", "% of brain"])
        w.writerow(["Behind the front of the brain", f"{an['from_front_um']:.1f}",
                    f"{an['from_front_pct']:.1f}"])
        w.writerow(["Below the top of the brain", f"{an['from_top_um']:.1f}",
                    f"{an['from_top_pct']:.1f}"])
        w.writerow([f"{an['midline_side'].capitalize()} of the midline",
                    f"{an['midline_offset_um']:.1f}", ""])
        w.writerow([("From the left side" if an["lr_known"] else "From side A (L/R unknown)"),
                    f"{an['from_left_um']:.1f}", f"{an['from_left_pct']:.1f}"])
        w.writerow(["Brain length / height / width (um)", f"{an['brain_length_um']:.1f}",
                    f"{an['brain_height_um']:.1f}", f"{an['brain_width_um']:.1f}"])
        w.writerow([])
        w.writerow(["Image coordinates", "X (image left->right)", "Y (image top->down)",
                    "Z (from slice 1)"])
        w.writerow(["Dye centre (um, volume coordinates)", *[f"{v:.1f}" for v in rep["centre_um"]]])
        w.writerow(["Dye centre from brain edge (um)",
                    *[f"{v:.1f}" for v in rep["centre_from_brain_edges_um"]]])
        w.writerow(["Dye centre as % across brain",
                    *[f"{v:.1f}" for v in rep["centre_percent_of_brain"]]])
        w.writerow(["Brightest point (um)", *[f"{v:.1f}" for v in rep["peak_um"]]])
        w.writerow(["Dye extent / spread (um)", *[f"{v:.1f}" for v in rep["extent_um"]]])
        w.writerow(["Brain size (um)", *[f"{v:.1f}" for v in rep["brain_size_um"]]])
        w.writerow([])
        w.writerow(["Dye appears in slices", f"{rep['first_slice']} to {rep['last_slice']}",
                    "centre slice", rep["centre_slice"], "slice with the most dye",
                    rep["most_dye_slice"]])
        w.writerow([])
        w.writerow(["Deposit", "Volume (mm3)", "Centre X (um)", "Centre Y (um)",
                    "Centre Z (um)", "Peak intensity (0-255)"])
        for d in rep["deposits"]:
            w.writerow([d.label, f"{d.volume_mm3:.5f}", *[f"{v:.1f}" for v in d.centroid_um],
                        d.peak])
        w.writerow([])
        w.writerow(["Slice", "Z (um)", "Dye area (mm2)"])
        for i, a in enumerate(rep["slice_areas_mm2"]):
            w.writerow([i + 1, f"{i * vol.cal.z_um:.1f}", f"{a:.5f}"])
