"""End-to-end check on synthetic data (python tests/test_pipeline.py <demo_folder>).

Verifies: consistent loading scale, tissue-size calibration accuracy,
alignment quality (mask overlap between neighbouring slices), and measurements.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainstack import calibration, dye, io, preprocess, registration, volume  # noqa: E402


def dice(a, b):
    return 2 * (a & b).sum() / max(1, a.sum() + b.sum())


def main(folder):
    truth = json.load(open(os.path.join(folder, "ground_truth.json")))
    paths = io.find_images(folder)
    t0 = time.time()
    stack = io.load_stack(paths, max_working_size=600)
    print(f"Loaded {stack.n_slices} slices, {stack.n_channels} channels, canvas {stack.canvas_shape}, "
          f"downscale x{stack.scale_factor:.3f}  ({time.time() - t0:.1f}s)")

    true_xy = truth["pixel_um"] * stack.scale_factor
    cal, extents, ref = calibration.from_tissue_size(stack.images, 6000, truth["z_um"])
    err = 100 * (cal.xy_um - true_xy) / true_xy
    print(f"Tissue-size calibration: {cal.xy_um:.3f} µm/px (true {true_xy:.3f}, error {err:+.1f}%), "
          f"reference slice {ref + 1}")
    cal_m = calibration.from_microscope(truth["pixel_um"], stack.scale_factor, truth["z_um"])
    print("Microscope calibration:", cal_m.describe())

    t0 = time.time()
    # Channel 3 is the red dye; find the brain outline from the other channels.
    proc, masks = preprocess.preprocess_stack(stack.images,
                                              preprocess.PreprocessSettings(mask_channel=(0, 1)))
    print(f"Preprocessed in {time.time() - t0:.1f}s")

    before = np.mean([dice(masks[i], masks[i + 1]) for i in range(len(masks) - 1)])
    t0 = time.time()
    align_ch = [p[..., :2].astype(np.float32).mean(axis=-1) for p in proc]  # brain channels only
    tfs = registration.align_stack(align_ch, robust=True)
    t_align = time.time() - t0
    am = [registration.apply_transform(m, t, nearest=True) for m, t in zip(masks, tfs)]
    after = [dice(am[i], am[i + 1]) for i in range(len(am) - 1)]
    print(f"Alignment ({t_align:.1f}s): neighbour mask overlap {before:.3f} -> {np.mean(after):.3f} "
          f"(worst pair {np.min(after):.3f} at slices {np.argmin(after) + 1}-{np.argmin(after) + 2})")

    # Rotation error vs. ground truth, relative to the reference slice
    ref = len(tfs) // 2
    gt = [s["angle_deg"] for s in truth["slices"]]
    rot_err = []
    for i, t in enumerate(tfs):
        expected = gt[i] - gt[ref]             # scipy rotates counter-clockwise; we undo clockwise
        d = (t.angle_deg - expected + 180) % 360 - 180
        rot_err.append(abs(d))
    print(f"Rotation error: mean {np.mean(rot_err):.2f}°, max {np.max(rot_err):.2f}° "
          f"(slice {int(np.argmax(rot_err)) + 1})")

    vol = volume.build_volume(proc, masks, tfs, cal_m, stack.channel_names,
                              [(80, 140, 255), (60, 255, 90), (255, 60, 60)])
    print("Volume shape (Z,Y,X,C):", vol.data.shape)
    w, h, d = volume.tissue_bounding_size_um(vol)
    print(f"Tissue size: {w:.0f} x {h:.0f} x {d:.0f} µm  (expected ~9000 x 6000 x 3000)")
    print(f"Tissue volume: {volume.tissue_volume_mm3(vol):.1f} mm³")
    for s in volume.channel_stats(vol, [60, 60, 60]):
        print("  ", s)

    # ---- dye location vs. ground truth
    t0 = time.time()
    res = dye.detect_dye(vol, dye.DyeSettings(channel=2))
    rep = dye.dye_report(vol, res)
    print(f"Dye ({time.time() - t0:.1f}s): threshold {res.threshold} (core {res.core_threshold}), "
          f"{rep['n_deposits']} deposit(s), {rep['dye_volume_mm3']:.3f} mm³ "
          f"= {rep['percent_of_brain']:.2f}% of brain, slices {rep['first_slice']}-{rep['last_slice']}")
    exp = expected_dye_um(truth, stack, vol, ref)
    got = rep["centre_um"]
    err = np.subtract(got, exp)
    print(f"Dye centre: got ({got[0]:.0f}, {got[1]:.0f}, {got[2]:.0f}) µm, expected "
          f"({exp[0]:.0f}, {exp[1]:.0f}, {exp[2]:.0f}) µm -> error {np.linalg.norm(err):.0f} µm")
    core = dye.dye_report(vol, dye.detect_dye(vol, dye.DyeSettings(channel=2, preset="core")))
    print(f"Dense core only: {core['dye_volume_mm3']:.3f} mm³, slices "
          f"{core['first_slice']}-{core['last_slice']}")
    return vol, tfs, np.mean(after), np.max(rot_err), res, float(np.linalg.norm(err))


def expected_dye_um(truth, stack, vol, ref):
    """Where the synthetic dye centre must land in the volume (µm, y measured down)."""
    from scipy import ndimage as ndi

    from tools_make_demo import H, W  # noqa: F401  (image size of the generator)
    s = truth["slices"][ref]
    # Dye centre in the generator's un-rotated image (see make_demo_data.make_slice)
    px, py = W / 2 + 0.45 * 900, H / 2 - 0.25 * 600
    # Where does scipy's rotate send a point? Measure it rather than assume the sign.
    probe = np.zeros((H // 10, W // 10), np.float32)
    probe[int(py / 10), int(px / 10)] = 1
    rot = ndi.rotate(probe, s["angle_deg"], reshape=False, order=1)
    ry, rx = ndi.center_of_mass(rot)
    rx, ry = rx * 10 + s["dx"], ry * 10 + s["dy"]
    f = stack.scale_factor
    lw, lh = round(W / f), round(H / f)
    ch_, cw_ = stack.canvas_shape
    cx = (rx + 0.5) / f - 0.5 + (cw_ - lw) // 2 - vol.crop_offset[1]
    cy = (ry + 0.5) / f - 0.5 + (ch_ - lh) // 2 - vol.crop_offset[0]
    zf_peak = 0.1  # dye is centred at zf = 0.1 in the generator
    z_idx = ((zf_peak + 1) * len(truth["slices"]) / 2) - 0.5
    return (cx * vol.cal.xy_um, cy * vol.cal.xy_um, z_idx * vol.cal.z_um)


if __name__ == "__main__":
    main(sys.argv[1])
