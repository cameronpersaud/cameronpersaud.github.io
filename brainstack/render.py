"""3D scene building with PyVista/VTK (no Qt here, so it can also run headless).

View modes, all in real micrometre coordinates:
  dye    - the main view: the red dye (solid shape and/or glow showing how
           concentrated it is) inside a see-through brain overlay
  planes - every aligned slice drawn as a see-through image at its Z height
           (the "3D plot" view: you see the actual photos stacked in space)
  points - bright pixels shown as a 3D scatter plot
  volume - true volume rendering: a solid, rotatable reconstruction
  surface- smooth outer surface of the tissue (also exportable as STL)
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from .orientation import Orientation, to_view, to_world, vtk_matrix, world_transform
from .volume import Volume


@dataclass
class ViewSettings:
    mode: str = "dye"                    # "dye" | "planes" | "points" | "volume" | "surface"
    visible: list[bool] = field(default_factory=list)
    threshold: int = 20                  # 0-255; dimmer pixels are hidden
    opacity: float = 0.6                 # 0-1 overall opacity
    brightness: float = 1.5              # display gain
    z_scale: float = 1.0                 # visual Z exaggeration (measurements stay true)
    tissue_only: bool = True             # hide everything outside the tissue mask
    show_grid: bool = True
    point_size: float = 3.0
    max_points: int = 400_000
    display_max_px: int = 400            # downsample XY for interactive speed
    background: str = "#101418"
    # --- dye view
    dye_channel: int = 0
    brain_style: str = "surface"         # "surface" | "glow" | "slices" | "none"
    brain_opacity: float = 0.15
    brain_color: str = "#c9d6e3"
    dye_style: str = "solid"             # "solid" | "glow" | "both"
    dye_opacity: float = 1.0
    dye_color: tuple = (255, 40, 40)
    show_centre: bool = True


def _display_step(vol: Volume, max_px: int) -> int:
    h, w = vol.data.shape[1:3]
    return max(1, int(math.ceil(max(h, w) / max_px)))


def composite_rgb(vol: Volume, vs: ViewSettings, step: int) -> np.ndarray:
    """(Z, h, w, 3) float 0..1: visible channels tinted by their colour and added."""
    sub = vol.data[:, ::step, ::step, :].astype(np.float32) / 255.0
    rgb = np.zeros(sub.shape[:3] + (3,), dtype=np.float32)
    for c in range(sub.shape[3]):
        if vs.visible and not vs.visible[c]:
            continue
        col = np.asarray(vol.channel_colors[c], dtype=np.float32) / 255.0
        rgb += sub[..., c, None] * col
    rgb = np.clip(rgb * vs.brightness, 0, 1)
    if vs.tissue_only:
        rgb *= vol.mask[:, ::step, ::step, None]
    return rgb


def _alpha(rgb: np.ndarray, vs: ViewSettings) -> np.ndarray:
    lum = rgb.max(axis=-1)
    thr = vs.threshold / 255.0
    a = np.clip((lum - thr) / max(1e-3, 1 - thr), 0, 1)
    return a


def scene_extent_um(vol: Volume, z_scale: float = 1.0):
    z, h, w = vol.data.shape[:3]
    return w * vol.cal.xy_um, h * vol.cal.xy_um, max(1, z - 1) * vol.cal.z_um * z_scale


def _planes(vol: Volume, vs: ViewSettings, step: int):
    rgb = composite_rgb(vol, vs, step)
    alpha = _alpha(rgb, vs)
    alpha = np.clip(alpha * 1.4, 0, 1) * vs.opacity
    sx = vol.cal.xy_um * step
    meshes = []
    for i in range(rgb.shape[0]):
        rgba = np.concatenate([rgb[i], alpha[i, ..., None]], axis=-1)[::-1]  # flip: image top = +Y
        rgba8 = (rgba * 255).astype(np.uint8)
        h, w = rgba8.shape[:2]
        grid = pv.ImageData(dimensions=(w, h, 1), spacing=(sx, sx, 1.0),
                            origin=(0.0, 0.0, i * vol.cal.z_um * vs.z_scale))
        grid.point_data["rgba"] = rgba8.reshape(-1, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meshes.append(grid.extract_surface())
    merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
    return merged


def _points(vol: Volume, vs: ViewSettings, step: int):
    rgb = composite_rgb(vol, vs, step)
    lum = rgb.max(axis=-1)
    sel = lum > max(vs.threshold, 1) / 255.0
    idx = np.argwhere(sel)  # (N, 3): z, y, x
    if len(idx) == 0:
        return None
    if len(idx) > vs.max_points:
        rng = np.random.default_rng(0)
        # Prefer brighter points when thinning.
        w = lum[sel] + 1e-3
        keep = rng.choice(len(idx), vs.max_points, replace=False, p=w / w.sum())
        idx = idx[keep]
    h = rgb.shape[1]
    sx = vol.cal.xy_um * step
    pts = np.column_stack([
        idx[:, 2] * sx,
        (h - 1 - idx[:, 1]) * sx,
        idx[:, 0] * vol.cal.z_um * vs.z_scale,
    ]).astype(np.float32)
    colors = (rgb[idx[:, 0], idx[:, 1], idx[:, 2]] * 255).astype(np.uint8)
    cloud = pv.PolyData(pts)
    cloud.point_data["rgb"] = colors
    return cloud


def _rgba_volume_actor(rgb: np.ndarray, alpha: np.ndarray, sx: float, sz: float,
                       opacity: float, haze: bool = False):
    """vtkVolume from per-voxel colour (Z, h, w, 3) and opacity (Z, h, w), both 0..1."""
    from scipy import ndimage as ndi
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    from vtkmodules.vtkRenderingCore import vtkVolume, vtkVolumeProperty
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

    # Spread colour into the invisible voxels next to visible ones. Otherwise the
    # renderer blends edge colours with black and edges look dark/sooty.
    empty = alpha <= 0
    if empty.any() and not haze:
        grown = np.stack([ndi.grey_dilation(rgb[..., c], size=(3, 5, 5)) for c in range(3)], -1)
        rgb = np.where(empty[..., None], grown, rgb)
    # opacity 1 = solid where alpha is 1; a haze is a small fraction of that.
    scale = (0.045 * opacity ** 0.75) if haze else opacity
    # Slices far apart compared with the pixel size render as blocky slabs. Add
    # interpolated layers between slices (display only; measurements untouched).
    k = int(np.clip(round(sz / sx / 2), 1, 4))
    if k > 1 and rgb.shape[0] > 1:
        z0 = rgb.shape[0]
        rgb = ndi.zoom(rgb, (k, 1, 1, 1), order=1)
        alpha = ndi.zoom(alpha, (k, 1, 1), order=1)
        sz = sz * (z0 - 1) / (rgb.shape[0] - 1)
    rgba = np.concatenate([rgb, (alpha * scale)[..., None]], axis=-1)[:, ::-1]  # flip Y
    rgba8 = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    z, h, w = rgba8.shape[:3]
    grid = pv.ImageData(dimensions=(w, h, z), spacing=(sx, sx, sz))
    grid.point_data["rgba"] = rgba8.reshape(-1, 4)  # C-order (z,y,x) => x fastest, as VTK expects
    grid.point_data.active_scalars_name = "rgba"

    # Identity transfer function: each voxel's own alpha IS its opacity. (Several
    # VTK GPU paths take opacity straight from the 4th component, so controlling
    # translucency through the alpha values themselves behaves the same everywhere.)
    otf = vtkPiecewiseFunction()
    otf.AddPoint(0, 0.0)
    otf.AddPoint(255, 1.0)
    prop = vtkVolumeProperty()
    prop.IndependentComponentsOff()               # use the 4 components as RGBA
    prop.SetScalarOpacity(otf)
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOff()
    prop.SetScalarOpacityUnitDistance(sx)

    mapper = vtkSmartVolumeMapper()
    mapper.SetInputData(grid)
    mapper.SetBlendModeToComposite()
    actor = vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    return actor


def _volume_prop(vol: Volume, vs: ViewSettings, step: int):
    """All visible channels as a true multi-colour volume."""
    rgb = composite_rgb(vol, vs, step)
    return _rgba_volume_actor(rgb, _alpha(rgb, vs), vol.cal.xy_um * step,
                              vol.cal.z_um * vs.z_scale, vs.opacity)


# ------------------------------------------------------------------ dye view
def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _brain_channels(vol: Volume, vs: ViewSettings) -> list[int]:
    chans = [c for c in range(vol.data.shape[3])
             if c != vs.dye_channel and (not vs.visible or vs.visible[c])]
    return chans or [c for c in range(vol.data.shape[3]) if c != vs.dye_channel] or [0]


def _brain_overlay(plotter, vol: Volume, vs: ViewSettings, step: int, props: list):
    if vs.brain_style == "none":
        return
    if vs.brain_style == "surface":
        mesh = getattr(vol, "_surface_cache", None)
        if mesh is None:
            from .volume import tissue_surface

            mesh = tissue_surface(vol)
            vol._surface_cache = mesh
        if vs.z_scale != 1:
            mesh = mesh.scale((1, 1, vs.z_scale), inplace=False)
        props.append(plotter.add_mesh(mesh, color=vs.brain_color, opacity=vs.brain_opacity,
                                      smooth_shading=True, specular=0.2, ambient=0.25,
                                      name="brain", pickable=True))
        return
    # "glow" / "slices": the brain's own fluorescence, in a neutral tint
    chans = _brain_channels(vol, vs)
    sub = vol.data[:, ::step, ::step][..., chans].astype(np.float32) / 255.0
    lum = np.clip(sub.max(axis=-1) * vs.brightness, 0, 1)
    if vs.tissue_only:
        lum *= vol.mask[:, ::step, ::step]
    tint = _hex_to_rgb01(vs.brain_color)
    sx = vol.cal.xy_um * step
    if vs.brain_style == "glow":
        # Constant light tint; how much brain tissue is there = how hazy it looks.
        rgb = np.broadcast_to(tint, lum.shape + (3,)).astype(np.float32)
        alpha = np.clip((lum - vs.threshold / 255.0) * 1.5, 0, 1)
        actor = _rgba_volume_actor(rgb, alpha, sx, vol.cal.z_um * vs.z_scale, vs.brain_opacity,
                                   haze=True)
        plotter.renderer.AddVolume(actor)
        props.append(actor)
    else:  # slices
        rgb = lum[..., None] * tint
        alpha = np.clip((lum - vs.threshold / 255.0) * 1.5, 0, 1) * vs.brain_opacity
        meshes = []
        for i in range(rgb.shape[0]):
            rgba8 = (np.concatenate([rgb[i], alpha[i, ..., None]], -1)[::-1] * 255).astype(np.uint8)
            h, w = rgba8.shape[:2]
            g = pv.ImageData(dimensions=(w, h, 1), spacing=(sx, sx, 1.0),
                             origin=(0.0, 0.0, i * vol.cal.z_um * vs.z_scale))
            g.point_data["rgba"] = rgba8.reshape(-1, 4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meshes.append(g.extract_surface())
        merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
        props.append(plotter.add_mesh(merged, scalars="rgba", rgba=True, lighting=False,
                                      show_scalar_bar=False, name="brain"))


def _dye_layer(plotter, vol: Volume, vs: ViewSettings, step: int, props: list, dye_result):
    if dye_result is None or not dye_result.found:
        return
    col = np.asarray(vs.dye_color, np.float32) / 255.0
    if vs.dye_style in ("solid", "both"):
        mesh = getattr(dye_result, "_surface_cache", None)
        if mesh is None:
            from .dye import dye_surface

            mesh = dye_surface(vol, dye_result)
            dye_result._surface_cache = mesh
        if mesh is not None:
            if vs.z_scale != 1:
                mesh = mesh.scale((1, 1, vs.z_scale), inplace=False)
            op = vs.dye_opacity if vs.dye_style == "solid" else min(vs.dye_opacity, 0.35)
            props.append(plotter.add_mesh(mesh, color=tuple(col), opacity=op,
                                          smooth_shading=True, specular=0.5, specular_power=20,
                                          name="dye_solid", pickable=True))
    if vs.dye_style in ("glow", "both"):
        ch = vol.data[:, ::step, ::step, vs.dye_channel].astype(np.float32) / 255.0
        thr = dye_result.threshold / 255.0
        region = vol.mask[:, ::step, ::step] if dye_result.settings.inside_brain_only else 1
        # Brightness = concentration: dim dye is faint and see-through, the core is solid.
        level = np.clip((ch - thr * 0.6) / max(1e-3, ch.max() - thr * 0.6), 0, 1) * region
        # Colour ramps from the dye colour (dilute) towards white-hot (concentrated),
        # and dilute dye is see-through so you can look into the deposit.
        hot = level[..., None] ** 2 * 0.7
        rgb = np.clip(col * (1 - hot) + hot, 0, 1)
        alpha = level ** 1.5 * 0.5
        actor = _rgba_volume_actor(rgb, alpha, vol.cal.xy_um * step,
                                   vol.cal.z_um * vs.z_scale, max(0.15, vs.dye_opacity))
        plotter.renderer.AddVolume(actor)
        props.append(actor)


def dye_centre_view_coords(vol: Volume, centre_um, z_scale: float = 1.0):
    """Report coords (x, y down, z) -> 3D view coords (x, y up, z * z_scale)."""
    x, y_down, z = centre_um
    h = (vol.data.shape[1] - 1) * vol.cal.xy_um
    return (x, h - y_down, z * z_scale)


def _centre_marker(plotter, vol: Volume, vs: ViewSettings, dye_result, props: list,
                   M=np.eye(3), t=np.zeros(3)):
    if not (vs.show_centre and dye_result is not None and dye_result.found):
        return
    from .dye import dye_report

    rep = getattr(dye_result, "_report_cache", None)
    if rep is None:
        rep = dye_report(vol, dye_result)
        dye_result._report_cache = rep
    c = dye_centre_view_coords(vol, rep["centre_um"], vs.z_scale)
    size = 0.012 * max(vol.data.shape[1:3]) * vol.cal.xy_um
    props.append(plotter.add_mesh(pv.Sphere(radius=size, center=c), color="#ffd23f",
                                  name="dye_centre", pickable=False))
    plotter.add_point_labels([to_world(c, M, t)], ["dye centre"], font_size=12,
                             text_color="#ffd23f",
                             shape_opacity=0.0, show_points=False, name="dye_centre_lbl",
                             always_visible=True, pickable=False)


def render_scene(plotter: pv.Plotter, vol: Volume, vs: ViewSettings, dye_result=None,
                 orientation: Orientation | None = None) -> list:
    """Clear the plotter and draw the chosen view. Returns the props drawn.

    Everything is built in "view" coordinates (image right, image up, slice
    number) and then turned into the anatomical frame given by `orientation`
    (left->right, back->front, bottom->top), so the brain sits upright.
    """
    plotter.clear_actors()
    plotter.remove_bounds_axes()
    plotter.set_background(vs.background)
    step = _display_step(vol, vs.display_max_px)
    props = []

    view_ext = scene_extent_um(vol, vs.z_scale)
    if orientation is not None:
        M, t = world_transform(orientation, view_ext)
    else:
        M, t = np.eye(3), np.zeros(3)
    plotter._bs_transform = (M, t)

    if vs.mode == "dye":
        # Opaque dye first, then the see-through brain around it.
        _dye_layer(plotter, vol, vs, step, props, dye_result)
        _brain_overlay(plotter, vol, vs, step, props)
        _centre_marker(plotter, vol, vs, dye_result, props, M, t)
    elif vs.mode == "planes":
        mesh = _planes(vol, vs, step)
        props.append(plotter.add_mesh(mesh, scalars="rgba", rgba=True, show_scalar_bar=False,
                                      lighting=False, pickable=True, name="planes"))
    elif vs.mode == "points":
        cloud = _points(vol, vs, step)
        if cloud is not None:
            props.append(plotter.add_mesh(cloud, scalars="rgb", rgb=True, point_size=vs.point_size,
                                          style="points", opacity=max(0.05, vs.opacity),
                                          show_scalar_bar=False, lighting=False, name="points"))
    elif vs.mode == "volume":
        actor = _volume_prop(vol, vs, step)
        plotter.renderer.AddVolume(actor)
        props.append(actor)
    elif vs.mode == "surface":
        mesh = getattr(vol, "_surface_cache", None)
        if mesh is None:
            from .volume import tissue_surface

            mesh = tissue_surface(vol)
            vol._surface_cache = mesh
        if vs.z_scale != 1:
            mesh = mesh.scale((1, 1, vs.z_scale), inplace=False)
        props.append(plotter.add_mesh(mesh, color="#e8b4b8", opacity=max(0.05, vs.opacity),
                                      smooth_shading=True, specular=0.3, name="surface"))
    else:
        raise ValueError(f"Unknown view mode {vs.mode}")

    if orientation is not None:
        m4 = vtk_matrix(M, t)
        for p in props:
            p.SetUserMatrix(m4)

    if vs.show_grid:
        true_ext = np.array(scene_extent_um(vol, 1.0))
        world_ext = np.abs(M) @ np.asarray(view_ext)
        ranges, titles = [], list(orientation.axis_titles() if orientation is not None
                                  else ("X (um)", "Y (um)", "Z (um)"))
        for a in range(3):
            k = int(np.argmax(np.abs(M[a])))
            ranges += [0.0, float(true_ext[k])]
            if k == 2 and vs.z_scale != 1:
                titles[a] = titles[a].replace("(um", f"(um, drawn {vs.z_scale:g}x")
        # (3D text fonts lack the µ symbol, so axis titles use "um")
        plotter.show_bounds(
            bounds=(0, world_ext[0], 0, world_ext[1], 0, world_ext[2]),
            axes_ranges=ranges,
            xtitle=titles[0], ytitle=titles[1], ztitle=titles[2],
            grid="back", location="outer", color="#9aa4ae", fmt="%.0f",
            font_size=10,
        )
    return props


def picked_to_um(point, z_scale: float, transform=None):
    """Convert a picked world point back to true micrometres (view frame)."""
    if transform is not None:
        point = to_view(point, *transform)
    x, y, z = point
    return (x, y, z / z_scale if z_scale else z)


# Camera presets in the anatomical frame: (direction camera looks FROM, up vector)
CAMERAS = {
    "front": ((0, 1, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "side": ((-1, 0, 0), (0, 0, 1)),
    "3d": ((-1.0, 1.3, 0.9), (0, 0, 1)),
}
