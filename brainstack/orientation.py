"""Anatomical orientation: which way were the slices cut, and which way is up?

The app always keeps slices in a standard order:
    coronal     slice 1 = front (rostral)   -> last = back (caudal)
    sagittal    slice 1 = animal's left     -> last = animal's right
    transverse  slice 1 = top (dorsal)      -> last = bottom (ventral)

From the cutting plane plus two questions about how the images look ("is the
top of the image the top of the brain?", "which way does the nose point?")
we know which anatomical direction each image axis points in. The 3D view is
then turned so the brain sits upright with its nose forward, and positions are
reported in anatomical words.

Anatomical frame used everywhere (right-handed, like RAS):
    X: animal's left  -> right
    Y: back (caudal)  -> front (rostral)
    Z: bottom (ventral) -> top (dorsal)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLANES = {
    "coronal": {
        "label": "Coronal",
        "desc": "Cut front-to-back, like slicing a loaf of bread from the nose towards "
                "the cerebellum. Each slice shows both hemispheres side by side.",
        "ends": ("front (rostral)", "back (caudal)"),
        "orders": ("Front → back  (rostral → caudal)", "Back → front  (caudal → rostral)"),
    },
    "sagittal": {
        "label": "Sagittal",
        "desc": "Cut side-to-side, parallel to the midline. Each slice shows the brain "
                "in profile: nose at one end, cerebellum at the other.",
        "ends": ("left side", "right side"),
        "orders": ("Left → right", "Right → left"),
    },
    "transverse": {
        "label": "Transverse / horizontal",
        "desc": "Cut top-to-bottom, parallel to the table. Each slice is a view looking "
                "down on the brain, with both hemispheres and nose-to-cerebellum.",
        "ends": ("top (dorsal)", "bottom (ventral)"),
        "orders": ("Top → bottom  (dorsal → ventral)", "Bottom → top  (ventral → dorsal)"),
    },
}

_X = np.array([1, 0, 0])   # left -> right
_Y = np.array([0, 1, 0])   # back -> front
_Z = np.array([0, 0, 1])   # bottom -> top


@dataclass
class Orientation:
    plane: str = "coronal"
    top_is_dorsal: bool = True     # coronal & sagittal: top of the image = top of the brain
    right_is: str = "right"        # coronal & transverse: image right = animal's "right" | "left" | "unknown"
    nose_points: str = "left"      # sagittal: front of the brain points "left" | "right" in the images
    front_points: str = "up"       # transverse: front of the brain points "up" | "down" in the images

    # ------------------------------------------------------------------
    @property
    def ends(self) -> tuple[str, str]:
        """(what slice 1 is, what the last slice is) in the standard order."""
        return PLANES[self.plane]["ends"]

    def matrix(self) -> np.ndarray:
        """3x3 matrix whose columns give the anatomical direction of the 3D view's
        axes: image right, image up, and increasing slice number."""
        sr = -1 if self.right_is == "left" else 1
        up = _Z if self.top_is_dorsal else -_Z
        if self.plane == "coronal":
            cols = (sr * _X, up, -_Y)              # slices run front -> back
        elif self.plane == "sagittal":
            cols = (-_Y if self.nose_points == "left" else _Y, up, _X)   # left -> right
        elif self.plane == "transverse":
            cols = (sr * _X, _Y if self.front_points == "up" else -_Y, -_Z)  # top -> bottom
        else:
            raise ValueError(self.plane)
        return np.column_stack(cols).astype(float)

    def lr_known(self) -> bool:
        return self.plane == "sagittal" or self.right_is != "unknown"

    def axis_titles(self) -> tuple[str, str, str]:
        lr = "Left -> Right (um)" if self.lr_known() else "Side A -> Side B (um, L/R unknown)"
        return lr, "Back -> Front (um)", "Bottom -> Top (um)"

    def describe(self) -> str:
        a, b = self.ends
        return f"{PLANES[self.plane]['label']} slices, slice 1 = {a}, last = {b}"


# ---------------------------------------------------------------- 3D placement
def world_transform(orient: Orientation, view_extent) -> tuple[np.ndarray, np.ndarray]:
    """(M, t) so that world = M @ view + t puts the box [0, extent] into the
    anatomical frame with its minimum corner at the origin."""
    M = orient.matrix()
    ex, ey, ez = view_extent
    corners = np.array([[x, y, z] for x in (0, ex) for y in (0, ey) for z in (0, ez)], float)
    t = -(corners @ M.T).min(axis=0)
    return M, t


def to_world(p, M, t):
    return np.asarray(p, float) @ M.T + t


def to_view(p, M, t):
    return (np.asarray(p, float) - t) @ M      # M is orthonormal: inverse = transpose


def vtk_matrix(M, t):
    from vtkmodules.vtkCommonMath import vtkMatrix4x4

    m = vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, float(M[i, j]))
        m.SetElement(i, 3, float(t[i]))
    return m


# ---------------------------------------------------------------- reporting
def anatomical_position(rep: dict, orient: Orientation) -> dict | None:
    """Turn the dye report's image-based numbers into anatomical ones.

    Uses: centre_from_brain_edges_um = (from left of image, down from top of
    image, from slice 1) and brain_size_um = (W, H, D).
    """
    if not rep.get("found"):
        return None
    x_left, y_down, z_first = rep["centre_from_brain_edges_um"]
    sx, sy, sz = rep["brain_size_um"]
    d = np.array([x_left, sy - y_down, z_first])      # along image right, image up, slice+
    size = np.array([sx, sy, sz])
    M = orient.matrix()
    pos, length = np.zeros(3), np.zeros(3)
    for a in range(3):                                # anatomical X, Y, Z
        k = int(np.argmax(np.abs(M[a])))
        s = M[a, k]
        pos[a] = d[k] if s > 0 else size[k] - d[k]    # distance from the negative end
        length[a] = size[k]

    ml, ap, dv = pos
    ml_len, ap_len, dv_len = length
    from_front = ap_len - ap
    from_top = dv_len - dv
    off = ml - ml_len / 2
    side = ("right" if off > 0 else "left") if orient.lr_known() else ("side B" if off > 0 else "side A")
    return {
        "from_front_um": from_front, "from_front_pct": 100 * from_front / ap_len if ap_len else 0,
        "from_top_um": from_top, "from_top_pct": 100 * from_top / dv_len if dv_len else 0,
        "from_left_um": ml, "from_left_pct": 100 * ml / ml_len if ml_len else 0,
        "midline_offset_um": abs(off), "midline_side": side,
        "brain_length_um": ap_len, "brain_height_um": dv_len, "brain_width_um": ml_len,
        "lr_known": orient.lr_known(),
    }
