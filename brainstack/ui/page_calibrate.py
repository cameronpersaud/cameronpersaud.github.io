"""Step 2: physical scale (pixel size and slice spacing)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import calibration, io
from ..preprocess import tissue_mask
from .page_import import Page
from .widgets import ImageView, composite, downsample_for_display, hint_label, mask_outline


def _spin(lo, hi, val, dec=3, suffix=" µm", step=1.0):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setSingleStep(step)
    s.setMinimumWidth(140)
    return s


class CalibratePage(Page):
    def __init__(self, win):
        super().__init__(win)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 10, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        form_host = QWidget()
        left = QVBoxLayout(form_host)
        left.setContentsMargins(0, 0, 8, 0)
        scroll.setWidget(form_host)
        scroll.setMinimumWidth(470)

        # ---- Z spacing
        zbox = QGroupBox("Distance between slices (Z)")
        zl = QFormLayout(zbox)
        self.z = _spin(0.01, 100000, 10.0, dec=2, step=1.0)
        self.z.valueChanged.connect(self._update_result)
        zl.addRow("Slice spacing:", self.z)
        zl.addRow(hint_label(
            "Centre-to-centre distance from one image to the next. If you cut 10 µm sections "
            "but only imaged every 4th one, enter <b>40</b>."))
        left.addWidget(zbox)

        # ---- XY pixel size
        xybox = QGroupBox("Pixel size (X / Y)")
        xl = QVBoxLayout(xybox)
        self.group = QButtonGroup(self)
        self.r_micro = QRadioButton("I know the microscope's microns per pixel  (most accurate)")
        self.r_bar = QRadioButton("Measure a scale bar in one of the images")
        self.r_tissue = QRadioButton("Estimate from the real size of the tissue  (e.g. 6,000 µm tall)")
        for i, r in enumerate((self.r_micro, self.r_bar, self.r_tissue)):
            self.group.addButton(r, i)
            xl.addWidget(r)
        self.r_tissue.setChecked(True)
        self.methods = QStackedWidget()
        xl.addWidget(self.methods)
        self.group.idToggled.connect(self._method_changed)

        # method 0: microscope
        m0 = QWidget()
        f0 = QFormLayout(m0)
        self.um_px = _spin(0.0001, 1000, 1.0, dec=4, suffix=" µm / pixel", step=0.05)
        self.um_px.valueChanged.connect(self._update_result)
        f0.addRow("Original images:", self.um_px)
        self.detect_btn = QPushButton("Read from image file")
        self.detect_btn.clicked.connect(self._detect_px)
        f0.addRow("", self.detect_btn)
        f0.addRow(hint_label(
            "Found in the microscope software, the image metadata, or the objective's "
            "calibration sheet. Using the same microscope settings for every brain gives "
            "every brain exactly the same scale."))
        self.methods.addWidget(m0)

        # method 1: scale bar
        m1 = QWidget()
        f1 = QFormLayout(m1)
        self.bar_len = _spin(0.01, 100000, 1000.0, dec=2, step=50)
        self.bar_len.valueChanged.connect(self._update_result)
        f1.addRow("Scale bar length:", self.bar_len)
        self.bar_slice = QComboBox()
        self.bar_slice.currentIndexChanged.connect(self._show_bar_slice)
        f1.addRow("Image:", self.bar_slice)
        self.bar_px = QLabel("Click both ends of the scale bar in the image on the right →")
        self.bar_px.setWordWrap(True)
        f1.addRow(self.bar_px)
        self.methods.addWidget(m1)

        # method 2: tissue size
        m2 = QWidget()
        f2 = QFormLayout(m2)
        self.real_size = _spin(1, 1_000_000, 6000.0, dec=0, step=100)
        self.real_size.valueChanged.connect(self._update_result)
        f2.addRow("Real tissue size:", self.real_size)
        self.dim = QComboBox()
        self.dim.addItems(["Height  (shortest dimension, dorsal–ventral)",
                           "Width  (longest dimension)"])
        self.dim.currentIndexChanged.connect(lambda *_: self._invalidate_tissue())
        f2.addRow("This is the:", self.dim)
        self.ref = QComboBox()
        self.ref.currentIndexChanged.connect(lambda *_: self._invalidate_tissue())
        f2.addRow("Measured on:", self.ref)
        self.measure_btn = QPushButton("Measure tissue", objectName="primary")
        self.measure_btn.clicked.connect(self._measure_tissue)
        f2.addRow("", self.measure_btn)
        self.tissue_txt = QLabel("")
        self.tissue_txt.setWordWrap(True)
        f2.addRow(self.tissue_txt)
        f2.addRow(hint_label(
            "<b>How this keeps sizes consistent:</b> the slice with the largest tissue is found "
            "automatically and set to your real size. That <i>one</i> pixel size is then used "
            "for every slice, so smaller slices (e.g. near the front of the brain) correctly "
            "come out smaller. The measurement is rotation-proof, so tilted slices don't skew "
            "it. Because brains vary, treat results as an estimate — if you can, use the "
            "microscope value or a scale bar instead."))
        self.methods.addWidget(m2)
        left.addWidget(xybox)

        # ---- result
        rbox = QGroupBox("Result")
        rl = QVBoxLayout(rbox)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("font-size: 14px;")
        rl.addWidget(self.result)
        left.addWidget(rbox)
        left.addStretch(1)
        outer.addWidget(scroll, 2)

        # ---- right: image
        pbox = QGroupBox("Image")
        pl = QVBoxLayout(pbox)
        self.view = ImageView("")
        self.view.clicked.connect(self._bar_click)
        pl.addWidget(self.view, 1)
        self.view_caption = QLabel("")
        self.view_caption.setProperty("hint", True)
        self.view_caption.setWordWrap(True)
        pl.addWidget(self.view_caption)
        outer.addWidget(pbox, 3)

        self._bar_pts: list[tuple[float, float]] = []
        self._disp_step = 1
        self._tissue_result = None   # (Calibration, extents, ref_index)
        self.methods.setCurrentIndex(2)

    # ----------------------------------------------------------------
    def on_enter(self):
        st = self.state.stack
        n = st.n_slices
        names = [f"Slice {i + 1}" for i in range(n)]
        for combo, extra in ((self.bar_slice, []), (self.ref, ["Largest slice (automatic)"])):
            combo.blockSignals(True)
            cur = combo.currentIndex()
            combo.clear()
            combo.addItems(extra + names)
            combo.setCurrentIndex(cur if 0 <= cur < combo.count() else 0)
            combo.blockSignals(False)
        # What the shortest / longest in-plane dimension means depends on the cut.
        names = {
            "coronal": ("Height  (shortest: top to bottom)", "Width  (longest: left to right)"),
            "sagittal": ("Height  (shortest: top to bottom)", "Length  (longest: front to back)"),
            "transverse": ("Width  (shortest: left to right)", "Length  (longest: front to back)"),
        }[self.state.orientation.plane]
        for i, t in enumerate(names):
            self.dim.setItemText(i, t)
        px = io.read_pixel_size_um(self.state.paths[0])
        self.detect_btn.setEnabled(px is not None)
        self.detect_btn.setText(f"Use value from file ({px:.4g} µm/px)" if px
                                else "No pixel size stored in the files")
        self._method_changed(self.group.checkedId(), True)
        self._update_result()

    def _method_changed(self, idx, checked=True):
        if not checked:
            return
        self.methods.setCurrentIndex(idx)
        if idx == 1:
            self._show_bar_slice()
        elif idx == 2:
            self._show_tissue_preview()
        else:
            self._show_slice(self.state.stack.n_slices // 2,
                             "Middle slice of your stack.")
        self._update_result()

    def _show_slice(self, i, caption, outline=None):
        img = self.state.stack.images[i]
        disp, step = downsample_for_display(img, 800)
        rgb = composite(disp, self.state.channel_colors)
        if outline is not None:
            o = mask_outline(outline[::step, ::step])
            rgb[o] = (255, 210, 63)
        self._disp_step = step
        self.view.set_image(rgb)
        self.view_caption.setText(caption)

    # ---- scale bar
    def _show_bar_slice(self, *_):
        if self.state.stack is None:
            return
        self._bar_pts = []
        self.view.set_points([])
        self._show_slice(max(0, self.bar_slice.currentIndex()),
                         "Click one end of the scale bar, then the other. Click again to restart.")
        self._update_result()

    def _bar_click(self, x, y):
        if self.methods.currentIndex() != 1:
            return
        if len(self._bar_pts) >= 2:
            self._bar_pts = []
        self._bar_pts.append((x, y))
        self.view.set_points(self._bar_pts, draw_line=True)
        if len(self._bar_pts) == 2:
            (x1, y1), (x2, y2) = self._bar_pts
            d = np.hypot(x2 - x1, y2 - y1) * self._disp_step
            self.bar_px.setText(f"Scale bar measured: <b>{d:.1f} pixels</b> long.")
        self._update_result()

    # ---- microscope
    def _detect_px(self):
        px = io.read_pixel_size_um(self.state.paths[0])
        if px:
            self.um_px.setValue(px)

    # ---- tissue size
    def _invalidate_tissue(self):
        self._tissue_result = None
        self.tissue_txt.setText("Press <b>Measure tissue</b>.")
        self._update_result()

    def _show_tissue_preview(self):
        if self._tissue_result is None:
            i = max(0, self.ref.currentIndex() - 1) if self.ref.currentIndex() > 0 \
                else self.state.stack.n_slices // 2
            self._show_slice(i, "Press “Measure tissue” to find the tissue outline.")
        else:
            _, _, ref = self._tissue_result
            mask = tissue_mask(self.state.stack.images[ref])
            self._show_slice(ref, f"Reference: slice {ref + 1}. Yellow = detected tissue edge.",
                             outline=mask)
        self.view.set_points([])

    def _measure_tissue(self):
        st = self.state.stack
        axis = "height" if self.dim.currentIndex() == 0 else "width"
        ref = None if self.ref.currentIndex() <= 0 else self.ref.currentIndex() - 1

        def task(progress, cancel):
            ext = []
            for i, img in enumerate(st.images):
                ext.append(calibration.tissue_extent_px(img, axis))
                progress(int(100 * (i + 1) / st.n_slices), f"Measuring slice {i + 1}")
            r = calibration.reference_slice(ext) if ref is None else ref
            return ext, r

        def done(res):
            ext, r = res
            if ext[r] <= 0:
                self.win.warn("No tissue was found in the reference slice.")
                return
            cal = calibration.Calibration(
                xy_um=self.real_size.value() / ext[r], z_um=self.z.value(),
                method="tissue_size")
            self._tissue_result = (cal, ext, r)
            smallest = int(np.argmin(ext))
            self.tissue_txt.setText(
                f"Slice {r + 1}: tissue {axis} = <b>{ext[r]:.0f} px</b>.<br>"
                f"Range across slices: {min(ext):.0f} px (slice {smallest + 1}) to "
                f"{max(ext):.0f} px.")
            self._show_tissue_preview()
            self._update_result()

        self.win.run_task(task, done, "Measuring tissue…")

    # ----------------------------------------------------------------
    def build_calibration(self) -> calibration.Calibration:
        method = self.group.checkedId()
        z = self.z.value()
        st = self.state.stack
        if method == 0:
            return calibration.from_microscope(self.um_px.value(), st.scale_factor, z)
        if method == 1:
            if len(self._bar_pts) != 2:
                raise ValueError("Click both ends of the scale bar first.")
            p1 = [v * self._disp_step for v in self._bar_pts[0]]
            p2 = [v * self._disp_step for v in self._bar_pts[1]]
            return calibration.from_scale_bar(p1, p2, self.bar_len.value(), z)
        if self._tissue_result is None:
            raise ValueError("Press “Measure tissue” first.")
        cal, ext, r = self._tissue_result
        axis = "height" if self.dim.currentIndex() == 0 else "width"
        return calibration.Calibration(
            xy_um=self.real_size.value() / ext[r], z_um=z, method="tissue_size",
            note=f"Slice {r + 1}: tissue {axis} = {ext[r]:.0f} px = "
                 f"{self.real_size.value():g} µm (estimate)")

    def _update_result(self, *_):
        if self.state.stack is None:
            return
        try:
            cal = self.build_calibration()
        except ValueError as e:
            self.result.setText(f"<span style='color:#f0b35a'>{e}</span>")
            return
        st = self.state.stack
        h, w = st.canvas_shape
        orig_px = cal.xy_um / st.scale_factor
        self.result.setText(
            f"Each 3D pixel (voxel) is <b>{cal.xy_um:.3g} × {cal.xy_um:.3g} × {cal.z_um:.3g} µm</b>"
            f"<br><span style='color:#9aa5b1'>= {orig_px:.4g} µm per original pixel × "
            f"{st.scale_factor:.3g} (downscaling), and {cal.z_um:g} µm between slices.<br>"
            f"Whole stack: {st.n_slices} slices ≈ {st.n_slices * cal.z_um / 1000:.2f} mm deep. "
            f"Slices are drawn {cal.anisotropy:.2g}× further apart than pixels, so the 3D "
            f"model has true proportions.</span>")

    def leave(self, proceed):
        try:
            self.state.cal = self.build_calibration()
        except ValueError as e:
            self.win.warn(str(e))
            return
        if self.state.vol is not None:
            self.state.vol.cal = self.state.cal  # scale change doesn't need re-alignment
            self.state.vol._surface_cache = None
        proceed()
