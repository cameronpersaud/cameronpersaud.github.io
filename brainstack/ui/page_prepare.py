"""Step 3: choose the dye channel, name/colour channels, clean up the images."""
from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from ..dye import guess_dye_channel
from ..preprocess import PreprocessSettings, preprocess_stack
from .page_import import Page
from .widgets import ColorButton, ImageView, composite, downsample_for_display, hint_label, mask_outline


class PreparePage(Page):
    def __init__(self, win):
        super().__init__(win)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 10, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        left = QVBoxLayout(host)
        left.setContentsMargins(0, 0, 8, 0)
        scroll.setWidget(host)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(470)
        host.setMaximumWidth(455)

        dbox = QGroupBox("Red dextran dye")
        df = QFormLayout(dbox)
        self.dye_ch = QComboBox()
        self.dye_ch.currentIndexChanged.connect(self._dye_channel_changed)
        df.addRow("The dye is in:", self.dye_ch)
        df.addRow(hint_label(
            "The dye is shown in red. Every other channel is treated as the <b>brain</b>: "
            "it is used to find the brain outline and to line the slices up, so the dye "
            "doesn't pull the alignment around."))
        left.addWidget(dbox)

        self.ch_box = QGroupBox("Fluorescence channels")
        self.ch_grid = QGridLayout(self.ch_box)
        left.addWidget(self.ch_box)

        cbox = QGroupBox("Clean-up")
        f = QFormLayout(cbox)
        self.bg = QCheckBox("Subtract slide background")
        self.bg.setChecked(True)
        f.addRow(self.bg)
        self.denoise = QDoubleSpinBox()
        self.denoise.setRange(0, 10)
        self.denoise.setSingleStep(0.5)
        self.denoise.setValue(1.0)
        self.denoise.setSuffix(" px")
        f.addRow("Smooth noise:", self.denoise)
        self.norm = QComboBox()
        self.norm.addItems(["Same scale for whole stack", "Equalise each slice"])
        f.addRow("Brightness:", self.norm)
        f.addRow(hint_label(
            "Keep <b>whole stack</b> for dye work: it keeps real differences in how much dye "
            "each slice has. <b>Equalise</b> only if staining is very uneven between slices."))
        left.addWidget(cbox)

        mbox = QGroupBox("Tissue detection")
        fm = QFormLayout(mbox)
        self.mask_ch = QComboBox()
        fm.addRow("Detect tissue using:", self.mask_ch)
        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(40, 250)
        self.sens.setValue(100)
        fm.addRow("Sensitivity:", self.sens)
        fm.addRow(hint_label(
            "The yellow outline should hug the brain; it becomes the see-through brain in "
            "3D. A channel that stains everything works best (e.g. DAPI / Hoechst / "
            "NeuroTrace). Slide sensitivity right if tissue is missed, left if background "
            "is included."))
        left.addWidget(mbox)
        left.addStretch(1)
        outer.addWidget(scroll)

        # ---- preview
        right = QVBoxLayout()
        pbox = QGroupBox("Preview")
        pl = QVBoxLayout(pbox)
        views = QHBoxLayout()
        col_a, col_b = QVBoxLayout(), QVBoxLayout()
        la, lb = QLabel("Original"), QLabel("Cleaned  (yellow = brain outline, red = dye)")
        la.setProperty("hint", True)
        lb.setProperty("hint", True)
        self.before = ImageView("")
        self.after = ImageView("")
        col_a.addWidget(la)
        col_a.addWidget(self.before, 1)
        col_b.addWidget(lb)
        col_b.addWidget(self.after, 1)
        views.addLayout(col_a)
        views.addLayout(col_b)
        pl.addLayout(views, 1)
        srow = QHBoxLayout()
        self.slice_lbl = QLabel("Slice 1")
        self.slice = QSlider(Qt.Horizontal)
        self.slice.valueChanged.connect(self._schedule)
        srow.addWidget(self.slice_lbl)
        srow.addWidget(self.slice, 1)
        pl.addLayout(srow)
        right.addWidget(pbox, 1)
        outer.addLayout(right, 1)

        self._timer = QTimer(self, singleShot=True, interval=200)
        self._timer.timeout.connect(self._update_preview)
        for w in (self.bg, ):
            w.toggled.connect(self._schedule)
        self.denoise.valueChanged.connect(self._schedule)
        self.norm.currentIndexChanged.connect(self._schedule)
        self.mask_ch.currentIndexChanged.connect(self._schedule)
        self.sens.valueChanged.connect(self._schedule)
        self._name_edits: list[QLineEdit] = []
        self._color_btns: list[ColorButton] = []
        self._show_checks: list[QCheckBox] = []

    # ----------------------------------------------------------------
    def _refresh_dye_combo(self):
        self.dye_ch.blockSignals(True)
        self.dye_ch.clear()
        self.dye_ch.addItems(self.state.channel_names)
        self.dye_ch.setCurrentIndex(self.state.dye_channel)
        self.dye_ch.blockSignals(False)

    def _dye_channel_changed(self, idx):
        st = self.state
        if idx < 0 or idx == st.dye_channel and st.channel_colors[idx][0] > 200:
            return
        old = st.dye_channel
        st.dye_channel = idx
        st.dye_result = None
        self._make_dye_red(old)
        for i, b in enumerate(self._color_btns):
            b.set_color(st.channel_colors[i])
        if st.vol is not None:
            st.vol.channel_colors = list(st.channel_colors)
        self._schedule()

    def _make_dye_red(self, previous_dye: int | None = None):
        """Show the dye in red; give red-coloured brain channels another colour."""
        st = self.state
        cols = st.channel_colors
        red = (255, 50, 50)
        old_dye_col = cols[st.dye_channel]
        for i, c in enumerate(cols):
            if i != st.dye_channel and c[0] > 180 and c[1] < 120 and c[2] < 120:
                cols[i] = old_dye_col if old_dye_col[0] < 180 else (90, 150, 255)
        cols[st.dye_channel] = red

    def _build_channel_rows(self):
        while self.ch_grid.count():
            w = self.ch_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        st = self.state
        self._name_edits, self._color_btns, self._show_checks = [], [], []
        hdr = ["Show", "Name", "Colour"]
        for c, h in enumerate(hdr):
            lab = QLabel(h)
            lab.setProperty("hint", True)
            self.ch_grid.addWidget(lab, 0, c)
        for i, name in enumerate(st.channel_names):
            show = QCheckBox()
            show.setChecked(True)
            show.toggled.connect(self._schedule)
            edit = QLineEdit(name)
            edit.textChanged.connect(self._names_changed)
            btn = ColorButton(st.channel_colors[i])
            btn.colorChanged.connect(lambda rgb, i=i: self._color_changed(i, rgb))
            self.ch_grid.addWidget(show, i + 1, 0)
            self.ch_grid.addWidget(edit, i + 1, 1)
            self.ch_grid.addWidget(btn, i + 1, 2)
            self._show_checks.append(show)
            self._name_edits.append(edit)
            self._color_btns.append(btn)
        tip = hint_label("Rename channels to your stains (e.g. DAPI, Dextran) and pick the "
                         "colours you want to see them in.")
        self.ch_grid.addWidget(tip, len(st.channel_names) + 1, 0, 1, 3)
        self._refresh_mask_combo()

    def _refresh_mask_combo(self):
        cur = self.mask_ch.currentIndex()
        self.mask_ch.blockSignals(True)
        self.mask_ch.clear()
        self.mask_ch.addItem("Brain channels (all except the dye)")
        self.mask_ch.addItem("All channels combined")
        self.mask_ch.addItems(self.state.channel_names)
        self.mask_ch.setCurrentIndex(cur if 0 <= cur < self.mask_ch.count() else 0)
        self.mask_ch.blockSignals(False)

    def _names_changed(self):
        self.state.channel_names = [e.text() or f"Channel {i + 1}"
                                    for i, e in enumerate(self._name_edits)]
        self._refresh_mask_combo()
        self._refresh_dye_combo()
        if self.state.vol is not None:
            self.state.vol.channel_names = list(self.state.channel_names)

    def _color_changed(self, i, rgb):
        self.state.channel_colors[i] = rgb
        if self.state.vol is not None:
            self.state.vol.channel_colors = list(self.state.channel_colors)
        self._schedule()

    def settings(self) -> PreprocessSettings:
        mc = self.mask_ch.currentIndex()
        return PreprocessSettings(
            subtract_background=self.bg.isChecked(),
            denoise_sigma=self.denoise.value(),
            normalize="stack" if self.norm.currentIndex() == 0 else "slice",
            mask_sensitivity=self.sens.value() / 100.0,
            mask_channel=(tuple(self.state.brain_channels()) if mc <= 0
                          else None if mc == 1 else mc - 2),
        )

    def on_enter(self):
        st = self.state
        if len(self._name_edits) != len(st.channel_names) or st.pre_key is None:
            if st.pre_key is None:
                mid = st.stack.images[st.stack.n_slices // 2]
                st.dye_channel = guess_dye_channel(st.channel_names, st.channel_colors, mid)
                self._make_dye_red()
            self._build_channel_rows()
            self._refresh_dye_combo()
        n = st.stack.n_slices
        self.slice.blockSignals(True)
        self.slice.setRange(0, n - 1)
        if self.slice.value() == 0:
            self.slice.setValue(n // 2)
        self.slice.blockSignals(False)
        self._update_preview()

    def _schedule(self, *_):
        self._timer.start()

    def _update_preview(self):
        st = self.state
        if st.stack is None:
            return
        i = self.slice.value()
        self.slice_lbl.setText(f"Slice {i + 1} / {st.stack.n_slices}")
        visible = [c.isChecked() for c in self._show_checks] or None
        raw, step = downsample_for_display(st.stack.images[i], 600)
        self.before.set_image(composite(raw, st.channel_colors, visible))
        s = self.settings()
        if s.normalize == "stack":
            s.normalize = "slice"  # preview of a single slice can't see the whole stack
        proc, masks = preprocess_stack([raw.astype("float32")], s)
        rgb = composite(proc[0].astype("float32"), st.channel_colors, visible, normalize=False)
        rgb[mask_outline(masks[0])] = (255, 210, 63)
        self.after.set_image(rgb)

    def leave(self, proceed):
        st = self.state
        s = self.settings()
        key = (st.loaded_key, tuple(sorted(asdict(s).items())))
        if st.proc is not None and st.pre_key == key:
            proceed()
            return

        def task(progress, cancel):
            return preprocess_stack(st.stack.images, s, progress=progress)

        def done(res):
            st.proc, st.masks = res
            st.pre = s
            st.pre_key = key
            st.vol = None  # needs rebuilding; alignment transforms are kept
            proceed()

        self.win.run_task(task, done, "Cleaning up slices…")
