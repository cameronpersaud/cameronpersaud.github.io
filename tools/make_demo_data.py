"""Generate a synthetic fluorescence brain-slice series for testing BrainStack 3D.

Creates N slices of an ellipsoid "brain" with three channels:
  ch1 (blue, like DAPI)   - nuclei speckle throughout the tissue
  ch2 (green)             - a curved band (like a hippocampus) + cortex layer
  ch3 (red dextran dye)   - an injection site in the right hemisphere: a bright core
                            that fades with distance, plus a faint needle track
                            running down from the top of the brain

Each slice is randomly shifted and rotated (like real mounting), and one slice
is put on the slide almost upside down, to test the aligner.

Usage:  python tools/make_demo_data.py [output_folder] [n_slices]
Ground-truth: tissue is 6,000 µm tall, pixel size 5 µm, slice spacing 100 µm.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import ndimage as ndi

PX_UM = 5.0
Z_UM = 100.0
H, W = 1400, 1800


def make_slice(zf: float, rng: np.random.Generator) -> np.ndarray:
    """zf in [-1, 1] = position along the anterior-posterior axis."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    shrink = math.sqrt(max(0.0, 1 - (zf * 0.95) ** 2))
    # Semi-axes in px. Top half 667 px + flattened bottom 533 px = 1200 px = 6000 µm tall.
    ax, ay = 900 * shrink, 667 * shrink
    # Brain outline: flatter on the bottom (ventral) to break up/down symmetry.
    ny = (yy - cy) / ay
    nx = (xx - cx) / ax
    ny = np.where(ny > 0, ny * 1.25, ny)
    r = np.sqrt(nx ** 2 + ny ** 2)
    tissue = (r < 1).astype(np.float32)
    # Midline notch between hemispheres at the top.
    tissue *= 1 - ((np.abs(xx - cx) < 10 * shrink + 2) & (yy < cy - 0.4 * ay)).astype(np.float32)

    # ch1: nuclei speckle
    speck = rng.random((H, W)).astype(np.float32)
    speck = ndi.gaussian_filter((speck > 0.985).astype(np.float32), 1.5) * 6
    ch1 = tissue * (0.25 + np.clip(speck, 0, 1) * 0.75)

    # ch2: cortex band + curved "hippocampus" on both sides (left slightly bigger)
    cortex = ((r > 0.82) & (r < 0.95)).astype(np.float32)
    hip = np.zeros((H, W), np.float32)
    for side, scale in ((-1, 1.0), (1, 0.8)):
        hx, hy = cx + side * 0.35 * ax, cy - 0.05 * ay
        d = np.sqrt(((xx - hx) / (0.22 * ax + 1)) ** 2 + ((yy - hy) / (0.18 * ay + 1)) ** 2)
        arc = (np.abs(d - 1) < 0.18 * scale) & (yy < hy + 0.1 * ay)
        hip += arc.astype(np.float32)
    ch2 = tissue * (0.08 + 0.7 * cortex + 0.9 * np.clip(hip, 0, 1))

    # ch3: red dextran injection, right hemisphere, centred at zf = 0.1
    ix, iy = cx + 0.45 * 900, cy - 0.25 * 600
    dz = (zf - 0.1) / 0.3
    amp = math.exp(-dz ** 2)                        # fades away from the injection slice
    rad = 110 * math.exp(-0.5 * dz ** 2)
    ch3 = tissue * amp * np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * (rad + 1) ** 2))
    # faint needle track from the top surface down to the injection site
    if abs(zf - 0.1) < 0.08:
        track = (np.abs(xx - ix) < 8) & (yy < iy)
        ch3 = ch3 + tissue * 0.25 * track * math.exp(-((zf - 0.1) / 0.05) ** 2)

    stack = np.stack([ch1, ch2, ch3])
    stack = ndi.gaussian_filter(stack, (0, 1.2, 1.2))
    gain = rng.uniform(0.75, 1.25)                     # staining varies slice to slice
    noise = rng.normal(0, 0.02, stack.shape).astype(np.float32)
    img = np.clip(stack * gain + 0.06 + noise, 0, 1)
    return (img * 4000).astype(np.uint16)


def transform_slice(img: np.ndarray, angle_deg: float, dx: float, dy: float) -> np.ndarray:
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        ch = ndi.rotate(img[c].astype(np.float32), angle_deg, reshape=False, order=1, mode="nearest")
        ch = ndi.shift(ch, (dy, dx), order=1, mode="nearest")
        out[c] = ch.astype(img.dtype)
    return out


def main(folder="demo_slices", n=30):
    import tifffile

    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(7)
    truth = []
    for i in range(n):
        zf = -1 + 2 * (i + 0.5) / n
        img = make_slice(zf, rng)
        ang = float(rng.uniform(-20, 20))
        if i == n // 3:
            ang = 172.0                                 # mounted upside down
        dx, dy = float(rng.uniform(-90, 90)), float(rng.uniform(-70, 70))
        img = transform_slice(img, ang, dx, dy)
        tifffile.imwrite(os.path.join(folder, f"slice_{i + 1:03d}.tif"), img,
                         photometric="minisblack", metadata={"axes": "CYX"})
        truth.append({"slice": i + 1, "angle_deg": ang, "dx": dx, "dy": dy})
        print(f"slice {i + 1}/{n}")
    with open(os.path.join(folder, "ground_truth.json"), "w") as fh:
        json.dump({"pixel_um": PX_UM, "z_um": Z_UM, "tissue_height_um": 6000,
                   "slices": truth}, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_slices",
         int(sys.argv[2]) if len(sys.argv) > 2 else 30)
