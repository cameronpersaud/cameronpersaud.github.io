"""Step 4: align slices automatically, inspect, and fix by hand."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout,
)

from ..registration import (SliceTransform, align_stack, apply_transform, outline_signal_image,
                            register_pair)
from ..volume import build_volume
from .page_import import Page
from .widgets import ImageView, auto_contrast, hint_label


class AlignPage(Page):
    def __init__(self, win):
        super().__init__(win)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 10, 0, 0)

        # ---- left: overlay viewer
        vbox = QGroupBox("Overlay check")
        vl = QVBoxLayout(vbox)
        legend = hint_label(
            "<span style='color:#ff5cff'>■ Magenta</span> = neighbouring slice &nbsp; "
            "<span style='color:#5cff7a'>■ Green</span> = this slice &nbsp; "
            "<b>White/grey = they line up</b>. Coloured fringes mean the slice is off.")
        vl.addWidget(legend)
        self.view = ImageView("")
        vl.addWidget(self.view, 1)
        srow = QHBoxLayout()
        self.slice_lbl = QLabel("Slice 1")
        self.slice_lbl.setMinimumWidth(110)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slice_changed)
        self.compare = QComboBox()
        self.compare.addItems(["Compare with previous slice", "Compare with next slice"])
        self.compare.currentIndexChanged.connect(self._redraw)
        srow.addWidget(self.slice_lbl)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.compare)
        vl.addLayout(srow)
        outer.addWidget(vbox, 3)

        # ---- right: controls
        right = QVBoxLayout()
        abox = QGroupBox("Automatic alignment")
        af = QFormLayout(abox)
        self.align_ch = QComboBox()
        self.align_ch.currentIndexChanged.connect(self._prepare_images)
        af.addRow("Align using:", self.align_ch)
        self.ref = QSpinBox()
        af.addRow("Keep fixed:", self.ref)
        self.upright = QCheckBox("All slices are upright (top of the brain at the top)")
        self.upright.setChecked(True)
        self.upright.setToolTip("Only small turns are tried (up to 15°). Untick if some slices "
                                "were mounted upside down or sideways.")
        af.addRow(self.upright)
        self.mirror = QCheckBox("Check for mirrored slices (mounted face-down)")
        self.mirror.setChecked(True)
        af.addRow(self.mirror)
        self.robust = QCheckBox("Thorough search (slower, more reliable)")
        self.robust.setChecked(True)
        af.addRow(self.robust)
        self.auto_btn = QPushButton("Auto-align all slices", objectName="primary")
        self.auto_btn.clicked.connect(self._auto_align)
        af.addRow(self.auto_btn)
        self.skip_btn = QPushButton("My images are already aligned")
        self.skip_btn.clicked.connect(self._skip)
        af.addRow(self.skip_btn)
        self.status = QLabel("Not aligned yet.")
        self.status.setWordWrap(True)
        af.addRow(self.status)
        right.addWidget(abox)

        mbox = QGroupBox("Fix this slice by hand")
        mf = QFormLayout(mbox)
        self.sx = QDoubleSpinBox()
        self.sy = QDoubleSpinBox()
        for s in (self.sx, self.sy):
            s.setRange(-5000, 5000)
            s.setDecimals(1)
            s.setSingleStep(1.0)
            s.setSuffix(" px")
        self.rot = QDoubleSpinBox()
        self.rot.setRange(-180, 180)
        self.rot.setDecimals(1)
        self.rot.setSingleStep(0.5)
        self.rot.setWrapping(True)
        self.rot.setSuffix(" °")
        self.flip = QCheckBox("Mirror (slice mounted face-down)")
        mf.addRow("Move right:", self.sx)
        mf.addRow("Move down:", self.sy)
        mf.addRow("Rotate clockwise:", self.rot)
        mf.addRow(self.flip)
        for w in (self.sx, self.sy, self.rot):
            w.valueChanged.connect(self._manual_changed)
        self.flip.toggled.connect(self._manual_changed)
        b1 = QPushButton("Re-align just this slice")
        b1.clicked.connect(self._realign_one)
        b2 = QPushButton("Reset this slice")
        b2.clicked.connect(self._reset_one)
        row = QHBoxLayout()
        row.addWidget(b1)
        row.addWidget(b2)
        mf.addRow(row)
        mf.addRow(hint_label(
            "Tip: click in a box and use the ↑/↓ keys to nudge. If a slice is mirrored, tick "
            "<b>Mirror</b> and press <b>Re-align just this slice</b>."))
        right.addWidget(mbox)

        sbox = QGroupBox("Straighten the whole stack")
        sl = QVBoxLayout(sbox)
        self.straight_hint = hint_label("")
        sl.addWidget(self.straight_hint)
        srow2 = QHBoxLayout()
        for text, deg in (("⟲ 90°", -90), ("⟲ 5°", -5), ("⟳ 5°", 5), ("⟳ 90°", 90)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=deg: self._rotate_all(d))
            srow2.addWidget(b)
        sl.addLayout(srow2)
        right.addWidget(sbox)
        right.addStretch(1)
        outer.addLayout(right, 2)

        self._imgs: list[np.ndarray] = []       # full working-res 2D alignment images
        self._disp: list[np.ndarray] = []       # downsampled for the overlay
        self._step = 1
        self._cache: dict[int, np.ndarray] = {}
        self._img_key = None
        self._align_imgs: list[np.ndarray] = []
        self.aligned_once = False

    # ---------------------------------------------------------------- setup
    def on_enter(self):
        st = self.state
        n = st.stack.n_slices
        cur = self.align_ch.currentIndex()
        self.align_ch.blockSignals(True)
        self.align_ch.clear()
        self.align_ch.addItem("Tissue outline + signal  (best with no brain stain)", ("outline", None))
        self.align_ch.addItem("Brain channels (all except the dye)", ("brain", None))
        self.align_ch.addItem("All channels combined", ("all", None))
        for c, name in enumerate(st.channel_names):
            self.align_ch.addItem(name, ("ch", c))
        if cur < 0:
            # Dye-only images: nothing but the dye channel, so its intensity is mostly
            # dye spots that change from slice to slice -> use the tissue outline.
            cur = 0 if len(st.brain_channels()) == 1 and st.brain_channels()[0] == st.dye_channel \
                else 1
        self.align_ch.setCurrentIndex(cur if 0 <= cur < self.align_ch.count() else 0)
        self.align_ch.blockSignals(False)
        self.ref.setRange(1, n)
        self.ref.setPrefix("Slice ")
        if self.ref.value() <= 1:
            self.ref.setValue(n // 2 + 1)
        if st.transforms is None or len(st.transforms) != n:
            st.transforms = [SliceTransform() for _ in range(n)]
            self.aligned_once = False
            self.status.setText("Not aligned yet. Press <b>Auto-align all slices</b>.")
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.blockSignals(False)
        o = st.orientation
        if o.plane == "transverse":
            want = f"the front of the brain points <b>{o.front_points}</b>"
        else:
            want = "the top of the brain points <b>{}</b>".format("up" if o.top_is_dorsal else "down")
        self.straight_hint.setText(
            f"After aligning, rotate everything so {want}, as you told step 1. That makes "
            f"the 3D model sit upright. (Rotation is in the image plane; screen clockwise.)")
        self._prepare_images()

    def _prepare_images(self, *_):
        st = self.state
        if st.proc is None:
            return
        mode, ch = self.align_ch.currentData() or ("brain", None)
        brain = st.brain_channels()
        key = (st.pre_key, mode, ch, tuple(brain))
        if key != self._img_key:
            # What the overlay shows (signal) and what the aligner matches can differ.
            signal = [p[..., brain].astype(np.float32).mean(axis=-1) for p in st.proc]
            if mode == "outline":
                self._imgs = signal
                self._align_imgs = [outline_signal_image(s_, m) for s_, m in zip(signal, st.masks)]
            elif mode == "all":
                self._imgs = [p.astype(np.float32).mean(axis=-1) for p in st.proc]
                self._align_imgs = self._imgs
            elif mode == "ch":
                self._imgs = [p[..., ch].astype(np.float32) for p in st.proc]
                self._align_imgs = self._imgs
            else:
                self._imgs = signal
                self._align_imgs = signal
            self._step = max(1, int(np.ceil(max(self._imgs[0].shape) / 560)))
            self._disp = [auto_contrast(im[::self._step, ::self._step]) for im in self._imgs]
            self._cache.clear()
            self._img_key = key
        self._slice_changed(self.slider.value())

    # ---------------------------------------------------------------- drawing
    def _disp_transform(self, t: SliceTransform) -> SliceTransform:
        return SliceTransform(t.shift_x / self._step, t.shift_y / self._step, t.angle_deg, t.flip_h)

    def _aligned_disp(self, i):
        if i not in self._cache:
            self._cache[i] = apply_transform(self._disp[i],
                                             self._disp_transform(self.state.transforms[i]))
        return self._cache[i]

    def _neighbour(self, i):
        n = len(self._disp)
        want_prev = self.compare.currentIndex() == 0
        j = i - 1 if want_prev else i + 1
        if j < 0 or j >= n:
            j = i + 1 if want_prev else i - 1
        return j if 0 <= j < n else None

    def _redraw(self, *_):
        if not self._disp:
            return
        i = self.slider.value()
        cur = self._aligned_disp(i)
        j = self._neighbour(i)
        nb = self._aligned_disp(j) if j is not None else np.zeros_like(cur)
        rgb = np.stack([nb, cur, nb], axis=-1)
        self.view.set_image((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        self.slice_lbl.setText(f"Slice {i + 1} / {len(self._disp)}"
                               + (f"  vs {j + 1}" if j is not None else ""))

    def _slice_changed(self, i):
        t = self.state.transforms[i]
        for w, v in ((self.sx, t.shift_x), (self.sy, t.shift_y), (self.rot, t.angle_deg)):
            w.blockSignals(True)
            w.setValue(((v + 180) % 360) - 180 if w is self.rot else v)
            w.blockSignals(False)
        self.flip.blockSignals(True)
        self.flip.setChecked(t.flip_h)
        self.flip.blockSignals(False)
        self._redraw()

    def _manual_changed(self, *_):
        i = self.slider.value()
        self.state.transforms[i] = SliceTransform(self.sx.value(), self.sy.value(),
                                                  self.rot.value(), self.flip.isChecked())
        self._cache.pop(i, None)
        self.state.vol = None
        self._redraw()

    def _rotate_all(self, deg: float):
        """Turn every slice by the same angle about the canvas centre.

        Composes with each slice's own transform: new angle = old + deg and the
        shift is rotated along, so slices stay aligned with each other.
        """
        import math

        a = math.radians(deg)
        c, s_ = math.cos(a), math.sin(a)
        st = self.state
        for i, t in enumerate(st.transforms):
            sx = c * t.shift_x - s_ * t.shift_y
            sy = s_ * t.shift_x + c * t.shift_y
            st.transforms[i] = SliceTransform(sx, sy, t.angle_deg + deg, t.flip_h)
        st.vol = None
        self._cache.clear()
        self._slice_changed(self.slider.value())

    def _reset_one(self):
        i = self.slider.value()
        self.state.transforms[i] = SliceTransform()
        self._cache.pop(i, None)
        self.state.vol = None
        self._slice_changed(i)

    # ---------------------------------------------------------------- alignment
    def _auto_align(self, then=None):
        st = self.state
        flips = [t.flip_h for t in st.transforms]
        imgs = self._align_imgs
        ref = self.ref.value() - 1
        robust = self.robust.isChecked()
        try_mirror = self.mirror.isChecked()
        max_angle = 15 if self.upright.isChecked() else None

        def task(progress, cancel):
            return align_stack(imgs, flips=flips, reference_index=ref, robust=robust,
                               progress=progress, cancel=cancel, try_mirror=try_mirror,
                               max_angle=max_angle)

        def done(tfs):
            st.transforms = tfs
            st.vol = None
            self._cache.clear()
            self.aligned_once = True
            self.status.setText(
                "✓ Aligned. Scrub through the slices below and fix any that look off. "
                "The first and last slices are the most likely to need help.")
            self._slice_changed(self.slider.value())
            if callable(then):
                then()

        self.win.run_task(task, done, "Aligning slices…")

    def _skip(self):
        st = self.state
        st.transforms = [SliceTransform() for _ in st.transforms]
        st.vol = None
        self._cache.clear()
        self.aligned_once = True
        self.status.setText("Using the images as they are (no alignment).")
        self._slice_changed(self.slider.value())

    def _realign_one(self):
        st = self.state
        i = self.slider.value()
        j = self._neighbour(i)
        if j is None:
            return
        fixed = apply_transform(self._align_imgs[j], st.transforms[j])
        moving = self._align_imgs[i]
        flip = self.flip.isChecked()   # the user's mirror choice is respected, not re-guessed
        robust = self.robust.isChecked()
        max_angle = 15 if self.upright.isChecked() else None

        def task(progress, cancel):
            return register_pair(fixed, moving, robust=robust, flip_h=flip, max_angle=max_angle)

        def done(t):
            st.transforms[i] = t
            st.vol = None
            self._cache.pop(i, None)
            self._slice_changed(i)

        self.win.run_task(task, done, f"Re-aligning slice {i + 1}…")

    # ---------------------------------------------------------------- next
    def leave(self, proceed):
        st = self.state

        def build():
            if st.vol is not None:
                proceed()
                return

            def task(progress, cancel):
                return build_volume(st.proc, st.masks, st.transforms, st.cal,
                                    st.channel_names, st.channel_colors, progress=progress)

            def done(vol):
                st.vol = vol
                proceed()

            self.win.run_task(task, done, "Building the 3D model…")

        if not self.aligned_once:
            ans = QMessageBox.question(
                self, "Align first?",
                "The slices haven't been aligned yet.\n\nRun automatic alignment now?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Yes:
                self._auto_align(then=build)
                return
        build()
