"""Slice-to-slice alignment (rigid: shift + rotation, plus optional mirror flip).

Strategy ("middle-out"): the middle slice stays put; each slice is aligned to
the already-aligned slice(s) next to it, working outward in both directions.
Starting in the middle halves the distance any alignment error can drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class SliceTransform:
    """Transform in screen terms, about the canvas centre."""

    shift_x: float = 0.0      # pixels, + moves right
    shift_y: float = 0.0      # pixels, + moves down
    angle_deg: float = 0.0    # + rotates clockwise on screen
    flip_h: bool = False      # mirror left-right before rotating

    def copy(self) -> "SliceTransform":
        return SliceTransform(self.shift_x, self.shift_y, self.angle_deg, self.flip_h)


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _center(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([(w - 1) / 2.0, (h - 1) / 2.0])


def to_sitk(t: SliceTransform, shape) -> sitk.Euler2DTransform:
    """Screen transform -> SimpleITK resampling transform (output -> input)."""
    a = math.radians(t.angle_deg)
    s = np.array([t.shift_x, t.shift_y])
    tr = -_rot(-a) @ s
    tx = sitk.Euler2DTransform()
    tx.SetCenter(tuple(_center(shape)))
    tx.SetAngle(-a)
    tx.SetTranslation(tuple(tr))
    return tx


def from_sitk(tx: sitk.Euler2DTransform, shape, flip_h: bool) -> SliceTransform:
    c = _center(shape)
    c2 = np.array(tx.GetCenter())
    theta = tx.GetAngle()
    t2 = np.array(tx.GetTranslation())
    # Re-express about the canvas centre.
    t = _rot(theta) @ (c - c2) + c2 + t2 - c
    a = -theta
    s = -_rot(a) @ t
    return SliceTransform(float(s[0]), float(s[1]), math.degrees(a), flip_h)


def outline_signal_image(img2d: np.ndarray, mask: np.ndarray, outline_weight: float = 0.6):
    """What to align on when there is no separate brain stain (dye-only images):
    mostly the tissue OUTLINE (consistent from slice to slice), plus some signal.
    Intensity alone is dominated by bright dye spots that differ between slices."""
    from scipy import ndimage as ndi

    from .calibration import main_tissue

    shape = ndi.gaussian_filter(main_tissue(mask).astype(np.float32), 3)
    sig = img2d.astype(np.float32)
    sig = sig / max(1e-6, float(np.percentile(sig, 99.5)))
    return outline_weight * shape / max(1e-6, float(shape.max())) + \
        (1 - outline_weight) * np.clip(sig, 0, 1)


def _flip(img: np.ndarray, flip: bool) -> np.ndarray:
    return img[:, ::-1] if flip else img


def apply_transform(img: np.ndarray, t: SliceTransform, nearest: bool = False) -> np.ndarray:
    """Apply to a 2D (H, W) or (H, W, C) array."""
    img = _flip(img, t.flip_h)
    tx = to_sitk(t, img.shape)
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear

    def one(ch2d):
        src = sitk.GetImageFromArray(np.ascontiguousarray(ch2d, dtype=np.float32))
        out = sitk.Resample(src, src, tx, interp, 0.0, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(out)

    if img.ndim == 2:
        out = one(img)
    else:
        out = np.stack([one(img[..., c]) for c in range(img.shape[2])], axis=-1)
    if img.dtype == bool:
        return out > 0.5
    return out.astype(np.float32)


def _as_euler(tx) -> sitk.Euler2DTransform:
    try:
        return sitk.Euler2DTransform(tx)
    except Exception:
        comp = sitk.CompositeTransform(tx)
        return sitk.Euler2DTransform(comp.GetNthTransform(comp.GetNumberOfTransforms() - 1))


def _prep(a: np.ndarray) -> sitk.Image:
    from scipy import ndimage as ndi

    a = ndi.gaussian_filter(a.astype(np.float32), 1.0)
    return sitk.GetImageFromArray(a)


def _coarse_search(fs, ms, init, max_angle: float | None = None):
    """Score starting angles (every 15°, or every 5° within ±max_angle), briefly
    optimise the 4 best (plus upright). Returns [(metric, deg, transform)],
    lower metric = better match."""
    scored = []
    angles = range(-180, 180, 15) if max_angle is None else \
        range(-int(max_angle), int(max_angle) + 1, 5)
    for deg in angles:
        cand = sitk.Euler2DTransform(init)
        cand.SetAngle(math.radians(deg))
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsCorrelation()
        r.SetInterpolator(sitk.sitkLinear)
        r.SetInitialTransform(cand, inPlace=True)
        try:
            scored.append((r.MetricEvaluate(fs, ms), deg, cand))
        except RuntimeError:
            continue
    scored.sort(key=lambda s_: s_[0])
    shortlist = scored[:4]
    if not any(d == 0 for _, d, _ in shortlist):
        shortlist += [s_ for s_ in scored if s_[1] == 0]
    results = []
    for _, deg, cand in shortlist:
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsCorrelation()
        r.SetInterpolator(sitk.sitkLinear)
        r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 60)
        r.SetOptimizerScalesFromPhysicalShift()
        r.SetInitialTransform(cand, inPlace=True)
        try:
            r.Execute(fs, ms)
            if max_angle is not None and abs(math.degrees(cand.GetAngle())) > max_angle + 5:
                continue   # drifted outside the allowed range
            results.append((r.GetMetricValue(), deg, sitk.Euler2DTransform(cand)))
        except RuntimeError:
            continue
    return results


def register_pair(fixed: np.ndarray, moving: np.ndarray, robust: bool = True,
                  flip_h: bool = False, margin: float = 0.04,
                  try_mirror: bool = False, max_angle: float | None = None) -> SliceTransform:
    """Find the transform that lays `moving` (2D) on top of `fixed` (2D).

    `moving` is the RAW (un-flipped) slice; flip_h is applied first.
    robust=True first tries 24 starting rotations, so slices mounted upside
    down or at odd angles still align. try_mirror=True also tries the slice
    mirrored (mounted face-down), e.g. sagittal sections whose nose points the
    other way. max_angle (degrees) limits the search to small turns, for stacks
    where every slice is known to be upright.
    """
    f = _prep(fixed)
    shrink = max(1, int(round(max(fixed.shape) / 96)))
    fs = sitk.Shrink(sitk.SmoothingRecursiveGaussian(f, shrink * 0.7), [shrink, shrink])

    candidates = [flip_h, not flip_h] if (robust and try_mirror) else [flip_h]
    options = []   # (metric, deg, flip, transform, moving image)
    for flip in candidates:
        m = _prep(_flip(moving, flip))
        init = _as_euler(sitk.CenteredTransformInitializer(
            f, m, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
        if robust:
            ms = sitk.Shrink(sitk.SmoothingRecursiveGaussian(m, shrink * 0.7), [shrink, shrink])
            for val, deg, tx in _coarse_search(fs, ms, init, max_angle):
                options.append((val, deg, flip, tx, m))
        else:
            options.append((0.0, 0, flip, init, m))
    if not options:   # every candidate drifted out of range: keep the slice as mounted
        m = _prep(_flip(moving, flip_h))
        options.append((0.0, 0, flip_h, _as_euler(sitk.CenteredTransformInitializer(
            f, m, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)), m))

    options.sort(key=lambda o: o[0])
    best = options[0]
    # Preference order, each taken only if it fits about as well as the best:
    #  1. as mounted (no mirror, no big turn)
    #  2. upright: a small turn, mirrored or not. Sections are usually laid down
    #     top-side-up but often face-down (= mirrored); 180-degree turns, which put
    #     the brain upside down, are less common and easy to confuse with mirroring.
    #  3. whatever fits best.
    default = [o for o in options if o[2] == flip_h and o[1] == 0]
    upright = [o for o in options if abs(o[1]) <= 45]
    if default and default[0][0] <= best[0] + margin:
        best = default[0]
    elif upright and upright[0][0] <= best[0] + margin:
        best = upright[0]
    _, _, flip_h, init, m = best

    # Fine alignment. Work on images of at most ~400 px (physical coordinates are
    # preserved by Shrink, so the result still applies to the full image).
    pre = max(1, int(math.ceil(max(fixed.shape) / 400)))
    if pre > 1:
        f = sitk.Shrink(sitk.SmoothingRecursiveGaussian(f, pre * 0.5), [pre, pre])
        m = sitk.Shrink(sitk.SmoothingRecursiveGaussian(m, pre * 0.5), [pre, pre])
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsCorrelation()            # same stain on both slices -> correlation is robust
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.3, seed=42)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    r.SetInitialTransform(init, inPlace=False)
    out = _as_euler(r.Execute(f, m))
    if max_angle is not None and abs(math.degrees(out.GetAngle())) > max_angle + 5:
        out = _as_euler(init)   # fine step wandered off: keep the coarse (upright) result
    return from_sitk(out, fixed.shape, flip_h)


def align_stack(images2d: list[np.ndarray], flips: list[bool] | None = None,
                reference_index: int | None = None, robust: bool = True,
                progress=None, cancel=None, try_mirror: bool = False,
                max_angle: float | None = None) -> list[SliceTransform]:
    """Align every slice; returns one SliceTransform per slice."""
    n = len(images2d)
    flips = flips or [False] * n
    ref = n // 2 if reference_index is None else reference_index
    transforms: list[SliceTransform | None] = [None] * n
    transforms[ref] = SliceTransform(flip_h=flips[ref])
    aligned: dict[int, np.ndarray] = {ref: apply_transform(images2d[ref], transforms[ref])}

    order = [(i, i - 1) for i in range(ref + 1, n)] + [(i, i + 1) for i in range(ref - 1, -1, -1)]
    for k, (i, nb) in enumerate(order):
        if cancel and cancel():
            break
        # Target = average of up to two already-aligned neighbours (less jitter).
        step = nb - i
        nbs = [j for j in (nb, nb + step) if j in aligned]
        target = np.mean([aligned[j] for j in nbs], axis=0)
        transforms[i] = register_pair(target, images2d[i], robust=robust, flip_h=flips[i],
                                      try_mirror=try_mirror, max_angle=max_angle)
        aligned[i] = apply_transform(images2d[i], transforms[i])
        if progress:
            progress(int(100 * (k + 1) / max(1, len(order))), f"Aligned slice {i + 1}/{n}")
    return [t if t is not None else SliceTransform(flip_h=flips[j]) for j, t in enumerate(transforms)]
