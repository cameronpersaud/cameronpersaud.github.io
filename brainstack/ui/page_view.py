"""Step 5: where is the dye? Interactive 3D view, dye report, measurements, export."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from .. import dye as D
from .. import volume as V
from ..orientation import PLANES, anatomical_position, to_world
from ..render import CAMERAS, ViewSettings, dye_centre_view_coords, picked_to_um, render_scene
from .page_import import Page
from .widgets import ColorButton, hint_label

MODES = [
    ("dye", "Dye inside see-through brain  (recommended)"),
    ("planes", "Stacked images  (3D plot of your slices)"),
    ("points", "Point cloud  (3D scatter of bright pixels)"),
    ("volume", "Solid volume  (all channels)"),
    ("surface", "Brain surface only"),
]
BRAIN_STYLES = [("surface", "See-through brain surface"),
                ("glow", "Brain fluorescence haze"),
                ("slices", "Faint slice images"),
                ("none", "Hidden")]
DYE_STYLES = [("solid", "Solid 3D shape"),
              ("glow", "Glow  (brighter = more dye)"),
              ("both", "Both")]
DETAIL = [("Low (fastest)", 250), ("Medium", 400), ("High", 600), ("Very high (slow)", 900)]


def _slider(lo, hi, val):
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _combo(items, idx=0):
    c = QComboBox()
    c.addItems([t for _, t in items])
    c.setCurrentIndex(idx)
    return c


class ViewPage(Page):
    def __init__(self, win):
        super().__init__(win)
        self.plotter = None
        self.props = []
        self._clip_plane = None
        self._picks: list[tuple[float, float, float]] = []
        self.distances: list[float] = []
        self._vol_shown = None
        self._dye_pending = False
        self._dye_key = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 10, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        left = QVBoxLayout(host)
        left.setContentsMargins(0, 0, 8, 0)
        scroll.setWidget(host)
        outer.addWidget(scroll)

        # ---- dye report (the answer, at the top)
        rbox = QGroupBox("Where is the dye?")
        rl = QVBoxLayout(rbox)
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.report)
        jrow = QHBoxLayout()
        self.jump_btn = QPushButton("Zoom to dye")
        self.jump_btn.clicked.connect(self._zoom_to_dye)
        jrow.addWidget(self.jump_btn)
        rl.addLayout(jrow)
        left.addWidget(rbox)

        # ---- dye detection
        dbox = QGroupBox("Dye detection")
        df = QFormLayout(dbox)
        self.preset_group = QButtonGroup(self)
        self.r_auto = QRadioButton("All detectable dye  (automatic)")
        self.r_core = QRadioButton("Dense core only  (≥ half of brightest)")
        self.r_manual = QRadioButton("Set the level myself:")
        for i, r in enumerate((self.r_auto, self.r_core, self.r_manual)):
            self.preset_group.addButton(r, i)
            df.addRow(r)
        self.r_auto.setChecked(True)
        mrow = QHBoxLayout()
        self.dye_thr = _slider(3, 250, 40)
        self.dye_thr_lbl = QLabel("40")
        self.dye_thr_lbl.setMinimumWidth(28)
        mrow.addWidget(self.dye_thr, 1)
        mrow.addWidget(self.dye_thr_lbl)
        df.addRow("Dye level:", mrow)
        self.inside = QCheckBox("Only count dye inside the brain")
        self.inside.setChecked(True)
        df.addRow(self.inside)
        df.addRow(hint_label(
            "<b>All detectable</b> = clearly above the background noise of the red channel "
            "(includes faint spread). <b>Dense core</b> = the injection centre. Drag the "
            "level to see how the shape changes."))
        self.preset_group.idToggled.connect(lambda i, on: on and self._dye_settings_changed())
        self.dye_thr.valueChanged.connect(self._manual_level_moved)
        self.inside.toggled.connect(self._dye_settings_changed)
        left.addWidget(dbox)

        # ---- dye + brain look
        lbox = QGroupBox("Dye && brain overlay")
        lf = QFormLayout(lbox)
        self.dye_style = _combo(DYE_STYLES)
        self.dye_color = ColorButton((255, 40, 40))
        drow = QHBoxLayout()
        drow.addWidget(self.dye_style, 1)
        drow.addWidget(self.dye_color)
        lf.addRow("Show dye as:", drow)
        self.dye_opa = _slider(10, 100, 100)
        lf.addRow("Dye opacity:", self.dye_opa)
        self.brain_style = _combo(BRAIN_STYLES)
        self.brain_color = ColorButton((201, 214, 227))
        brow = QHBoxLayout()
        brow.addWidget(self.brain_style, 1)
        brow.addWidget(self.brain_color)
        lf.addRow("Brain:", brow)
        self.brain_opa = _slider(2, 80, 15)
        self.brain_opa_lbl = QLabel("15%")
        orow = QHBoxLayout()
        orow.addWidget(self.brain_opa, 1)
        orow.addWidget(self.brain_opa_lbl)
        lf.addRow("Brain opacity:", orow)
        self.show_centre = QCheckBox("Mark the centre of the dye")
        self.show_centre.setChecked(True)
        lf.addRow(self.show_centre)
        lf.addRow(hint_label("Lower <b>brain opacity</b> = more transparent brain, so the "
                             "dye stands out. The brain and dye come from the same aligned "
                             "slices, so they are always in register."))
        for w in (self.dye_style, self.brain_style):
            w.currentIndexChanged.connect(self._schedule)
        for w in (self.dye_opa, self.brain_opa):
            w.valueChanged.connect(self._schedule)
        self.brain_opa.valueChanged.connect(lambda v: self.brain_opa_lbl.setText(f"{v}%"))
        self.dye_color.colorChanged.connect(lambda *_: self._schedule())
        self.brain_color.colorChanged.connect(lambda *_: self._schedule())
        self.show_centre.toggled.connect(self._schedule)
        left.addWidget(lbox)

        # ---- view mode & camera
        mbox = QGroupBox("View")
        ml = QVBoxLayout(mbox)
        self.mode_group = QButtonGroup(self)
        for i, (_, label) in enumerate(MODES):
            r = QRadioButton(label)
            self.mode_group.addButton(r, i)
            ml.addWidget(r)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idToggled.connect(lambda i, on: on and self._mode_changed())
        cam = QGridLayout()
        for k, (text, key, tip) in enumerate((
                ("Front", "front", "Look at the brain from the front (nose towards you)"),
                ("Top", "top", "Look down on the brain from above"),
                ("Side", "side", "Look at the brain from the animal's left side"),
                ("3D angle", "3d", "Front-left, from above"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, key=key: self._camera(key))
            cam.addWidget(b, 0, k)
        ml.addLayout(cam)
        ml.addWidget(hint_label("Left-drag = rotate · right-drag or wheel = zoom · "
                                "middle-drag / Shift+drag = pan"))
        left.addWidget(mbox)

        # ---- channels (for the other views)
        self.ch_box = QGroupBox("Channels")
        self.ch_layout = QGridLayout(self.ch_box)
        left.addWidget(self.ch_box)
        self._ch_checks: list[QCheckBox] = []

        # ---- display
        self.disp_box = QGroupBox("Display")
        dfm = QFormLayout(self.disp_box)
        self.thr = _slider(0, 200, 20)
        self.opa = _slider(5, 100, 60)
        self.bright = _slider(50, 400, 150)
        self.zstr = _slider(10, 100, 10)
        self.zlbl = QLabel("1.0×")
        zrow = QHBoxLayout()
        zrow.addWidget(self.zstr, 1)
        zrow.addWidget(self.zlbl)
        self.detail = QComboBox()
        self.detail.addItems([d[0] for d in DETAIL])
        self.detail.setCurrentIndex(1)
        self.thr_row_lbl = QLabel("Hide dim pixels:")
        self.opa_row_lbl = QLabel("Opacity:")
        dfm.addRow(self.thr_row_lbl, self.thr)
        dfm.addRow(self.opa_row_lbl, self.opa)
        dfm.addRow("Brightness:", self.bright)
        dfm.addRow("Stretch Z:", zrow)
        dfm.addRow("Detail:", self.detail)
        self.tissue_only = QCheckBox("Only show tissue (hide slide background)")
        self.tissue_only.setChecked(True)
        self.grid = QCheckBox("Show axes in micrometres")
        self.grid.setChecked(True)
        dfm.addRow(self.tissue_only)
        dfm.addRow(self.grid)
        dfm.addRow(hint_label("<b>Stretch Z</b> only changes the picture (handy for thin "
                              "stacks); all measurements stay in true units."))
        self.clip_btn = QPushButton("Cut-away plane (look inside)")
        self.clip_btn.setCheckable(True)
        self.clip_btn.toggled.connect(self._toggle_clip)
        dfm.addRow(self.clip_btn)
        for s in (self.thr, self.opa, self.bright, self.zstr):
            s.valueChanged.connect(self._schedule)
        self.zstr.valueChanged.connect(lambda v: self.zlbl.setText(f"{v / 10:.1f}×"))
        self.detail.currentIndexChanged.connect(self._schedule)
        self.tissue_only.toggled.connect(self._schedule)
        self.grid.toggled.connect(self._schedule)
        left.addWidget(self.disp_box)

        # ---- general measurements
        qbox = QGroupBox("Other measurements")
        ql = QVBoxLayout(qbox)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Count signal brighter than:"))
        self.sig_thr = QSpinBox()
        self.sig_thr.setRange(1, 255)
        self.sig_thr.setValue(60)
        self.sig_thr.setSuffix(" / 255")
        self.sig_thr.valueChanged.connect(self._update_measurements)
        trow.addWidget(self.sig_thr)
        ql.addLayout(trow)
        self.meas = QLabel("")
        self.meas.setWordWrap(True)
        self.meas.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ql.addWidget(self.meas)
        self.dist_btn = QPushButton("Measure a distance")
        self.dist_btn.setCheckable(True)
        self.dist_btn.toggled.connect(self._toggle_distance)
        ql.addWidget(self.dist_btn)
        self.dist_lbl = QLabel("")
        self.dist_lbl.setWordWrap(True)
        ql.addWidget(self.dist_lbl)
        clr = QPushButton("Clear distances")
        clr.clicked.connect(self._clear_distances)
        ql.addWidget(clr)
        left.addWidget(qbox)

        # ---- export
        ebox = QGroupBox("Save / export")
        eg = QGridLayout(ebox)
        for k, (text, fn) in enumerate((
                ("Dye report (CSV)", self._save_dye_csv),
                ("Dye 3D shape (STL)", self._save_dye_stl),
                ("Picture (PNG)", self._save_png),
                ("Brain surface (STL)", self._save_stl),
                ("All measurements (CSV)", self._save_csv),
                ("3D stack (TIFF for Fiji)", self._save_tiff))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            eg.addWidget(b, k // 2, k % 2)
        left.addWidget(ebox)
        left.addStretch(1)

        # ---- 3D canvas
        self.canvas_holder = QVBoxLayout()
        outer.addLayout(self.canvas_holder, 1)
        self._timer = QTimer(self, singleShot=True, interval=250)
        self._timer.timeout.connect(self.render)
        self._dye_timer = QTimer(self, singleShot=True, interval=400)
        self._dye_timer.timeout.connect(self._run_dye_detection)

    # ---------------------------------------------------------------- lifecycle
    def _ensure_plotter(self):
        if self.plotter is None:
            from pyvistaqt import QtInteractor

            self.plotter = QtInteractor(self)
            self.canvas_holder.addWidget(self.plotter.interactor)

    def _build_channel_controls(self):
        while self.ch_layout.count():
            w = self.ch_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        vol = self.state.vol
        self._ch_checks = []
        for i, name in enumerate(vol.channel_names):
            label = name + ("  (dye)" if i == self.state.dye_channel else "")
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._schedule)
            btn = ColorButton(vol.channel_colors[i])
            btn.colorChanged.connect(lambda rgb, i=i: self._recolor(i, rgb))
            self.ch_layout.addWidget(cb, i, 0)
            self.ch_layout.addWidget(btn, i, 1)
            self._ch_checks.append(cb)

    def _recolor(self, i, rgb):
        self.state.vol.channel_colors[i] = rgb
        self.state.channel_colors[i] = rgb
        self._schedule()

    def on_enter(self):
        self._ensure_plotter()
        vol = self.state.vol
        if vol is None:
            return
        first_time = self._vol_shown is None
        if vol is not self._vol_shown:
            self._build_channel_controls()
            self._vol_shown = vol
            self.distances = []
            self.dist_lbl.setText("")
            self.state.dye_result = None
            self._dye_key = None
            col = self.state.channel_colors[self.state.dye_channel]
            self.dye_color.set_color(col if col[0] > max(col[1], col[2]) else (255, 40, 40))
        self._mode_changed(render=False)
        self._update_measurements()
        if self.state.dye_result is None or self._dye_key != self._current_dye_key():
            self._run_dye_detection(then_camera=first_time)
        else:
            self.render()
            if first_time:
                self._camera("3d")

    # ---------------------------------------------------------------- dye detection
    def dye_settings(self) -> D.DyeSettings:
        preset = ("auto", "core", "manual")[self.preset_group.checkedId()]
        st = self.state
        dye_only = st.brain_channels() == [st.dye_channel]   # no separate brain stain
        return D.DyeSettings(channel=st.dye_channel,
                             threshold=self.dye_thr.value() if preset == "manual" else None,
                             preset=preset, inside_brain_only=self.inside.isChecked(),
                             dye_only=dye_only)

    def _current_dye_key(self):
        s = self.dye_settings()
        return (id(self.state.vol), s.channel, s.preset, s.threshold, s.inside_brain_only)

    def _manual_level_moved(self, v):
        self.dye_thr_lbl.setText(str(v))
        if not self.r_manual.isChecked():
            self.r_manual.setChecked(True)  # triggers detection itself
        else:
            self._dye_settings_changed()

    def _dye_settings_changed(self, *_):
        self._dye_timer.start()

    def _run_dye_detection(self, then_camera=False):
        vol = self.state.vol
        if vol is None:
            return
        if self.win.worker is not None:
            self._dye_pending = True
            return
        s = self.dye_settings()
        key = self._current_dye_key()

        def task(progress, cancel):
            progress(10, "Finding the dye in 3D…")
            res = D.detect_dye(vol, s)
            progress(60, "Measuring the dye…")
            res._report_cache = D.dye_report(vol, res)
            progress(80, "Building the dye shape…")
            res._surface_cache = D.dye_surface(vol, res) if res.found else None
            return res

        def done(res):
            self.state.dye_result = res
            self._dye_key = key
            # Keep the manual slider in sync with what's being used.
            self.dye_thr.blockSignals(True)
            self.dye_thr.setValue(res.threshold)
            self.dye_thr.blockSignals(False)
            self.dye_thr_lbl.setText(str(res.threshold))
            self._update_report()
            self.render()
            if then_camera:
                self._camera("3d")
            if self._dye_pending:
                self._dye_pending = False
                self._run_dye_detection()

        self.win.run_task(task, done, "Finding the dye…")

    def _update_report(self):
        res = self.state.dye_result
        vol = self.state.vol
        if res is None:
            self.report.setText("")
            return
        rep = res._report_cache
        ch_name = vol.channel_names[res.settings.channel]
        if not rep["found"]:
            self.report.setText(
                f"<span style='color:#f0b35a'><b>No dye found</b> in “{ch_name}” at level "
                f"{res.threshold}.</span><br>Try <b>Set the level myself</b> and drag it "
                f"lower, or check the dye channel in step 3.")
            return
        orient = self.state.orientation
        an = anatomical_position(rep, orient)
        ex, ey, ez = rep["extent_um"]
        dep = rep["n_deposits"]
        dep_txt = "1 deposit" if dep == 1 else f"{dep} separate deposits"
        big = res.deposits[0]
        extra = ""
        if dep > 1:
            extra = (f"<br><span style='color:#9aa5b1'>Largest deposit: "
                     f"{big.volume_mm3:.3f} mm³ ({100 * big.volume_mm3 / rep['dye_volume_mm3']:.0f}% "
                     f"of the dye)</span>")
        side_word = an["midline_side"]
        lr_note = "" if an["lr_known"] else (" <span style='color:#f0b35a'>(left/right not "
                                              "known – set it in step 1)</span>")
        first, last = orient.ends
        self.report.setText(
            f"<b style='color:#ff6b6b'>Dye volume: {rep['dye_volume_mm3']:.3f} mm³</b>"
            f" &nbsp;({rep['percent_of_brain']:.2f}% of brain) · {dep_txt}{extra}<br><br>"
            f"<b>Centre of the dye:</b>"
            f"<table width=100%>"
            f"<tr><td>Behind the front</td><td align=right>{an['from_front_um']:,.0f} µm</td>"
            f"<td align=right>{an['from_front_pct']:.0f}% of length</td></tr>"
            f"<tr><td>Below the top</td><td align=right>{an['from_top_um']:,.0f} µm</td>"
            f"<td align=right>{an['from_top_pct']:.0f}% of height</td></tr>"
            f"<tr><td>{side_word.capitalize()} of midline</td>"
            f"<td align=right>{an['midline_offset_um']:,.0f} µm</td>"
            f"<td align=right>{an['from_left_pct']:.0f}% from {'left' if an['lr_known'] else 'side A'}"
            f"</td></tr></table>{lr_note}"
            f"<br><b>Found in slices {rep['first_slice']}–{rep['last_slice']}</b> "
            f"(centre: slice {rep['centre_slice']}, most dye: slice {rep['most_dye_slice']})<br>"
            f"<span style='color:#9aa5b1'>{PLANES[orient.plane]['label']} slices: slice 1 = "
            f"{first}, slice {self.state.vol.data.shape[0]} = {last}.</span><br>"
            f"<b>Spread:</b> {ex:,.0f} × {ey:,.0f} µm in each slice, over {ez:,.0f} µm of slices<br>"
            f"<span style='color:#9aa5b1'>Channel “{ch_name}”, level {res.threshold}/255 "
            f"(auto {res.auto_threshold}, core {res.core_threshold}). Distances are measured "
            f"from the edges of the brain tissue; the midline is taken as the middle of the "
            f"brain's width.</span>")

    def _zoom_to_dye(self):
        res = self.state.dye_result
        if self.plotter is None or res is None or not res.found:
            return
        zs = self.zstr.value() / 10.0
        c = dye_centre_view_coords(self.state.vol, res._report_cache["centre_um"], zs)
        tf = getattr(self.plotter, "_bs_transform", None)
        if tf is not None:
            c = tuple(to_world(c, *tf))
        ex, ey, ez = res._report_cache["extent_um"]
        r = 1.6 * max(ex, ey, ez * zs)
        cam = self.plotter.camera
        direction = np.subtract(cam.position, cam.focal_point)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        cam.focal_point = c
        cam.position = tuple(np.add(c, direction * r * 2.2))
        self.plotter.render()

    # ---------------------------------------------------------------- rendering
    def _mode_changed(self, render=True):
        dye_mode = MODES[self.mode_group.checkedId()][0] == "dye"
        for w in (self.thr, self.thr_row_lbl, self.opa, self.opa_row_lbl):
            w.setVisible(not dye_mode)
        self.ch_box.setVisible(not dye_mode)
        if render:
            self._schedule()

    def settings(self) -> ViewSettings:
        return ViewSettings(
            mode=MODES[self.mode_group.checkedId()][0],
            visible=[c.isChecked() for c in self._ch_checks],
            threshold=self.thr.value(),
            opacity=self.opa.value() / 100.0,
            brightness=self.bright.value() / 100.0,
            z_scale=self.zstr.value() / 10.0,
            tissue_only=self.tissue_only.isChecked(),
            show_grid=self.grid.isChecked(),
            display_max_px=DETAIL[self.detail.currentIndex()][1],
            dye_channel=self.state.dye_channel,
            brain_style=BRAIN_STYLES[self.brain_style.currentIndex()][0],
            brain_opacity=self.brain_opa.value() / 100.0,
            brain_color="#%02x%02x%02x" % self.brain_color.rgb,
            dye_style=DYE_STYLES[self.dye_style.currentIndex()][0],
            dye_opacity=self.dye_opa.value() / 100.0,
            dye_color=self.dye_color.rgb,
            show_centre=self.show_centre.isChecked(),
        )

    def _schedule(self, *_):
        self._timer.start()

    def render(self):
        if self.plotter is None or self.state.vol is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cam = self.plotter.camera_position if self.props else None
            self.props = render_scene(self.plotter, self.state.vol, self.settings(),
                                      self.state.dye_result, self.state.orientation)
            if cam is not None:
                self.plotter.camera_position = cam
            else:
                self.plotter.reset_camera()
            self._apply_clip()
            self.plotter.render()
        finally:
            QApplication.restoreOverrideCursor()

    def _camera(self, key):
        if self.plotter is None:
            return
        direction, up = CAMERAS[key]
        self.plotter.view_vector(direction, viewup=up)
        self.plotter.reset_camera()
        self.plotter.render()

    # ---------------------------------------------------------------- cut-away
    def _toggle_clip(self, on):
        if self.plotter is None:
            return
        if on:
            self.plotter.add_plane_widget(self._plane_moved, normal="z",
                                          interaction_event="always", color="#ffd23f",
                                          outline_opacity=0.2)
        else:
            self.plotter.clear_plane_widgets()
            self._clip_plane = None
            self._apply_clip()
            self.plotter.render()

    def _plane_moved(self, normal, origin):
        from vtkmodules.vtkCommonDataModel import vtkPlane

        plane = vtkPlane()
        plane.SetNormal(*normal)
        plane.SetOrigin(*origin)
        self._clip_plane = plane
        self._apply_clip()

    def _apply_clip(self):
        for prop in self.props:
            mapper = prop.GetMapper()
            if mapper is None:
                continue
            mapper.RemoveAllClippingPlanes()
            if self._clip_plane is not None:
                mapper.AddClippingPlane(self._clip_plane)

    # ---------------------------------------------------------------- measurements
    def _update_measurements(self, *_):
        vol = self.state.vol
        if vol is None:
            return
        w, h, d = V.tissue_bounding_size_um(vol)
        tv = V.tissue_volume_mm3(vol)
        thr = self.sig_thr.value()
        stats = V.channel_stats(vol, [thr] * len(vol.channel_names))
        rows = "".join(
            f"<tr><td>{s['channel']}</td><td align=right>{s['signal_volume_mm3']:.3f} mm³</td>"
            f"<td align=right>{s['percent_of_tissue']:.1f}%</td></tr>" for s in stats)
        self.meas.setText(
            f"<b>Brain size</b> (W × H × D):<br>{w:,.0f} × {h:,.0f} × {d:,.0f} µm<br>"
            f"<b>Brain volume:</b> {tv:.3f} mm³<br>"
            f"<span style='color:#9aa5b1'>{vol.data.shape[0]} slices · "
            f"{vol.cal.describe()}</span><br><br>"
            f"<b>Signal above threshold</b><table width=100%>{rows}</table>")

    def _toggle_distance(self, on):
        if self.plotter is None:
            return
        self._picks = []
        if on:
            self.plotter.enable_point_picking(
                callback=self._on_pick, picker="volume", left_clicking=False,
                show_message="Right-click two points to measure the distance between them",
                show_point=True, color="#ffd23f", point_size=12, font_size=12)
        else:
            self.plotter.disable_picking()

    def _on_pick(self, point):
        if point is None:
            return
        zs = self.zstr.value() / 10.0
        self._picks.append(tuple(point))
        if len(self._picks) == 2:
            import pyvista as pv

            p1, p2 = self._picks
            tf = getattr(self.plotter, "_bs_transform", None)
            d = V.distance_um(picked_to_um(p1, zs, tf), picked_to_um(p2, zs, tf))
            self.distances.append(d)
            k = len(self.distances)
            self.plotter.add_mesh(pv.Line(p1, p2), color="#ffd23f", line_width=3,
                                  name=f"distance{k}", pickable=False)
            self.plotter.add_point_labels([np.add(p1, p2) / 2], [f"{d:,.0f} µm"],
                                          font_size=14, text_color="#ffd23f",
                                          shape_opacity=0.4, name=f"distance_lbl{k}",
                                          pickable=False)
            self._picks = []
            self._show_distances()

    def _show_distances(self):
        self.dist_lbl.setText("<br>".join(f"Distance {i + 1}: <b>{d:,.1f} µm</b>"
                                          for i, d in enumerate(self.distances)))

    def _clear_distances(self):
        self.distances = []
        self._picks = []
        self._show_distances()
        self.render()

    # ---------------------------------------------------------------- export
    def _ask_path(self, title, default, filt):
        start = os.path.join(os.path.dirname(self.state.paths[0]), default)
        path, _ = QFileDialog.getSaveFileName(self, title, start, filt)
        return path

    def _need_dye(self) -> bool:
        res = self.state.dye_result
        if res is None or not res.found:
            self.win.warn("No dye has been detected yet, so there is nothing to export.")
            return False
        return True

    def _save_dye_csv(self):
        if self.state.dye_result is None:
            return
        path = self._ask_path("Save dye report", "dye_location_report.csv", "CSV (*.csv)")
        if path:
            D.export_dye_csv(self.state.vol, self.state.dye_result, path,
                             self.state.orientation)
            self.win.status.setText(f"Saved {path}")

    def _save_dye_stl(self):
        if not self._need_dye():
            return
        path = self._ask_path("Save dye 3D shape", "dye_shape.stl",
                              "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)")
        if path:
            res = self.state.dye_result
            mesh = res._surface_cache or D.dye_surface(self.state.vol, res)
            mesh.save(path)
            self.win.status.setText(f"Saved {path} (units: micrometres, same coordinates "
                                    f"as the brain surface)")

    def _save_csv(self):
        path = self._ask_path("Save measurements", "measurements.csv", "CSV (*.csv)")
        if path:
            vol = self.state.vol
            V.export_measurements_csv(vol, [self.sig_thr.value()] * len(vol.channel_names),
                                      path, self.distances)
            self.win.status.setText(f"Saved {path}")

    def _save_tiff(self):
        path = self._ask_path("Save aligned 3D stack", "aligned_stack.tif", "TIFF (*.tif)")
        if path:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                V.export_tiff(self.state.vol, path)
            finally:
                QApplication.restoreOverrideCursor()
            self.win.status.setText(f"Saved {path} (opens in Fiji/ImageJ with correct scale)")

    def _save_stl(self):
        path = self._ask_path("Save brain surface", "brain_surface.stl",
                              "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)")
        if path:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                V.export_surface(self.state.vol, path)
            finally:
                QApplication.restoreOverrideCursor()
            self.win.status.setText(f"Saved {path} (units: micrometres)")

    def _save_png(self):
        path = self._ask_path("Save picture", "dye_in_brain_3d.png", "PNG (*.png)")
        if path and self.plotter is not None:
            self.plotter.screenshot(path, scale=2)
            self.win.status.setText(f"Saved {path}")
