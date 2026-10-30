"""Loading slice images at a single, consistent working resolution.

Key idea for consistent sizing: every image in a stack is downscaled by the
SAME factor (chosen from the largest image), so one pixel means the same
physical distance in every slice. Images are then padded onto a common canvas
so they can be shifted/rotated during alignment without being cut off.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def natural_key(path: str):
    """Sort 'slice_2' before 'slice_10'."""
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def find_images(folder: str) -> list[str]:
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith(".")
    ]
    return sorted(files, key=natural_key)


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Normalise any common microscope layout to (H, W, C) float32."""
    arr = np.asarray(arr)
    # Collapse extra leading dims (e.g. a z-stack per file) by max projection.
    while arr.ndim > 3:
        arr = arr.max(axis=0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 8:
            arr = arr[:, :, :3]  # RGB(A) -> drop alpha
        elif arr.shape[0] <= 8 and arr.shape[-1] > 8:
            arr = np.moveaxis(arr, 0, -1)  # (C, H, W) -> (H, W, C)
    return arr.astype(np.float32)


def read_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(path)
    else:
        from PIL import Image

        arr = np.array(Image.open(path))
    hwc = _to_hwc(arr)
    # A colour (RGB) image whose channels are identical is really grayscale.
    if hwc.shape[2] == 3 and np.allclose(hwc[..., 0], hwc[..., 1]) and np.allclose(
        hwc[..., 0], hwc[..., 2]
    ):
        hwc = hwc[..., :1]
    return hwc


def read_shape(path: str) -> tuple[int, int]:
    """(H, W) without decoding the full image where possible."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(path) as tf:
                shape = tf.series[0].shape
            hwc_shape = _to_hwc(np.empty(shape, dtype=np.uint8)).shape
            return hwc_shape[0], hwc_shape[1]
        from PIL import Image

        with Image.open(path) as im:
            return im.height, im.width
    except Exception:
        a = read_image(path)
        return a.shape[0], a.shape[1]


@dataclass
class SliceStack:
    """All slices at working resolution on a common padded canvas."""

    paths: list[str]
    images: list[np.ndarray]          # each (H, W, C) float16, same shape
    scale_factor: float               # working pixel = scale_factor original pixels
    original_shapes: list[tuple[int, int]]
    channel_names: list[str] = field(default_factory=list)
    empty_slices: list[int] = field(default_factory=list)      # indices with ~no signal
    dropped_channels: list[str] = field(default_factory=list)  # channels that were blank

    @property
    def n_slices(self) -> int:
        return len(self.images)

    @property
    def n_channels(self) -> int:
        return self.images[0].shape[2] if self.images else 0

    @property
    def canvas_shape(self) -> tuple[int, int]:
        return self.images[0].shape[:2]


RGB_NAMES = ["Red", "Green", "Blue"]
NAMED_COLORS = {"red": (255, 50, 50), "green": (70, 255, 110), "blue": (90, 150, 255)}


def channel_colors_for(names: list[str]) -> list[tuple[int, int, int]]:
    """Colours that match channel names where possible (Red -> red, ...)."""
    defaults = default_channel_colors(len(names))
    return [NAMED_COLORS.get(n.lower(), defaults[i]) for i, n in enumerate(names)]


def _signal(img: np.ndarray) -> np.ndarray:
    """Per-channel signal strength: bright end minus typical level."""
    sub = img[::4, ::4].reshape(-1, img.shape[2]).astype(np.float32)
    return np.percentile(sub, 99.9, axis=0) - np.median(sub, axis=0)


def prune_channels(images: list[np.ndarray], names: list[str]):
    """Drop channels that are blank in most images, and find images with no signal.

    E.g. a red-only dye photo saved as RGB has empty green/blue channels; using
    those as "brain" channels would build the brain outline from noise.
    Returns (images, names, dropped_names, empty_slice_indices).
    """
    S = np.array([_signal(im) for im in images])            # (n_images, n_channels)
    rel = S / np.maximum(S.max(axis=1, keepdims=True), 1e-6)
    # A channel is blank only if it has real signal in (almost) no image. A dye that
    # shows up in just some slices must be kept.
    need = max(2, int(np.ceil(0.1 * len(images))))
    keep = [c for c in range(S.shape[1]) if (rel[:, c] >= 0.25).sum() >= need]
    if not keep:
        keep = list(range(S.shape[1]))
    dropped = [names[c] for c in range(S.shape[1]) if c not in keep]
    if dropped:
        images = [im[..., keep] for im in images]
        names = [names[c] for c in keep]
    strength = S[:, keep].max(axis=1)
    empty = [i for i, v in enumerate(strength) if v < 0.2 * np.median(strength)]
    return images, names, dropped, empty


def default_channel_names(n: int) -> list[str]:
    if n == 1:
        return ["Signal"]
    return [f"Channel {i + 1}" for i in range(n)]


DEFAULT_COLORS = [(255, 70, 70), (70, 255, 110), (90, 150, 255),
                  (255, 80, 255), (80, 240, 255), (255, 230, 80)]


def default_channel_colors(n: int) -> list[tuple[int, int, int]]:
    if n == 1:
        return [(110, 255, 140)]
    return [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]


def load_stack(
    paths: list[str],
    max_working_size: int | None = 1024,
    pad_fraction: float = 0.15,
    progress=None,
) -> SliceStack:
    """Load images, downscale them all by ONE common factor, pad to one canvas.

    max_working_size: longest side (pixels) of the largest image after
    downscaling. None = keep full resolution (needs lots of memory).
    """
    from skimage.transform import rescale

    if not paths:
        raise ValueError("No images selected.")

    shapes = [read_shape(p) for p in paths]
    largest = max(max(s) for s in shapes)
    factor = 1.0
    if max_working_size and largest > max_working_size:
        factor = largest / float(max_working_size)

    loaded = []
    n_ch = None
    for i, p in enumerate(paths):
        img = read_image(p)
        if n_ch is None:
            n_ch = img.shape[2]
        elif img.shape[2] != n_ch:
            raise ValueError(
                f"'{os.path.basename(p)}' has {img.shape[2]} channel(s) but earlier "
                f"images have {n_ch}. All slices must have the same channels."
            )
        if factor > 1.0:
            img = rescale(
                img, 1.0 / factor, channel_axis=2, anti_aliasing=True, preserve_range=True
            ).astype(np.float32)
        loaded.append(img)
        if progress:
            progress(int(100 * (i + 1) / len(paths)), f"Loaded {os.path.basename(p)}")

    names = default_channel_names(n_ch)
    if n_ch in (3, 4) and all(p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) for p in paths):
        names = RGB_NAMES[:3]
    dropped, empty = [], []
    if len(loaded) >= 2:
        loaded, names, dropped, empty = prune_channels(loaded, names)

    max_h = max(a.shape[0] for a in loaded)
    max_w = max(a.shape[1] for a in loaded)
    # Square canvas with margin so rotation never clips the tissue.
    side = int(np.ceil(max(max_h, max_w) * (1 + 2 * pad_fraction)))
    canvas_h = canvas_w = side
    padded = []
    for a in loaded:
        # Pad with the image's own background level so the padding edge doesn't
        # look like a real structure. Use the 30th percentile of the whole image:
        # slide background usually covers most of the photo, and this ignores
        # black margins that stitched/cropped images often have along their edges
        # (padding with those made the whole photo rectangle look like tissue).
        bg = np.percentile(a[::4, ::4].reshape(-1, a.shape[2]), 30, axis=0).astype(np.float32)
        # float16 halves memory; plenty of precision for 8/12/16-bit microscope data.
        out = np.empty((canvas_h, canvas_w, a.shape[2]), dtype=np.float16)
        out[:] = bg
        y0 = (canvas_h - a.shape[0]) // 2
        x0 = (canvas_w - a.shape[1]) // 2
        out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = np.clip(a, 0, 65000)
        padded.append(out)

    return SliceStack(
        paths=list(paths),
        images=padded,
        scale_factor=factor,
        original_shapes=shapes,
        channel_names=names,
        empty_slices=empty,
        dropped_channels=dropped,
    )


def estimate_memory_gb(n_slices: int, largest_px: int, n_channels: int,
                       max_working_size: int | None, pad_fraction: float = 0.15) -> float:
    side = largest_px if not max_working_size else min(largest_px, max_working_size)
    side = side * (1 + 2 * pad_fraction)
    # raw (float16) + processed (float16) + 3D volume (uint8) + working copies
    per_voxel = n_channels * (2 + 2 + 1) + 2
    return n_slices * side * side * per_voxel / 1e9


def read_pixel_size_um(path: str) -> float | None:
    """Best-effort: pixel size stored in the file by the microscope / Fiji.

    Supports ImageJ/Fiji TIFFs and OME-TIFFs. Returns None if not found.
    """
    if not path.lower().endswith((".tif", ".tiff")):
        return None
    try:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            if tf.ome_metadata:
                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
                unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
                if m:
                    v = float(m.group(1))
                    u = unit.group(1) if unit else "µm"
                    factor = {"nm": 1e-3, "mm": 1e3, "m": 1e6}.get(u, 1.0)
                    return v * factor
            page = tf.pages[0]
            tag = page.tags.get("XResolution")
            ij = tf.imagej_metadata or {}
            unit = str(ij.get("unit", "")).lower()
            if tag is not None:
                num, den = tag.value
                px_per_unit = num / den if den else 0
                if px_per_unit > 0:
                    if unit in ("micron", "um", "µm", "\\u00b5m", "microns"):
                        return 1.0 / px_per_unit
                    res_unit = page.tags.get("ResolutionUnit")
                    ru = int(res_unit.value) if res_unit is not None else 2
                    if ru == 3 and not ij:          # pixels per centimetre
                        return 1e4 / px_per_unit
    except Exception:
        return None
    return None
