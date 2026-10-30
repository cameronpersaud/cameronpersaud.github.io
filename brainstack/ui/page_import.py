"""Step 1: choose and order the slice images."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from .. import io
from .orientation_widget import OrientationPanel
from .widgets import ImageView, composite, downsample_for_display, hint_label

RESOLUTIONS = [
    ("Fast preview  (512 px)", 512),
    ("Balanced  (1024 px)  — recommended", 1024),
    ("Detailed  (2048 px)", 2048),
    ("Full resolution  (needs a lot of memory)", None),
]


class Page(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.state = win.state

    def on_enter(self):
        pass

    def leave(self, proceed):
        proceed()


class ImportPage(Page):
    def __init__(self, win):
        super().__init__(win)
        self.setAcceptDrops(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 0)

        # ---- left column: file list
        left = QVBoxLayout()
        box = QGroupBox("Slices in order (top = slice 1)")
        bl = QVBoxLayout(box)
        bl.addWidget(hint_label(
            "Add images with the buttons or <b>drag files/folders here</b>. "
            "Drag rows to reorder. Supported: TIFF (incl. multi-channel), PNG, JPG."))
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentRowChanged.connect(self._preview)
        self.list.model().rowsMoved.connect(self._rows_moved)
        bl.addWidget(self.list, 1)
        row1 = QHBoxLayout()
        add_files = QPushButton("Add images…", objectName="primary")
        add_folder = QPushButton("Add folder…")
        add_files.clicked.connect(self._add_files)
        add_folder.clicked.connect(self._add_folder)
        row1.addWidget(add_files)
        row1.addWidget(add_folder)
        bl.addLayout(row1)
        row2 = QHBoxLayout()
        for text, fn in (("Sort by name", self._sort), ("Remove selected", self._remove),
                         ("Clear all", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row2.addWidget(b)
        bl.addLayout(row2)
        left.addWidget(box, 1)

        res_box = QGroupBox("Working resolution")
        rl = QVBoxLayout(res_box)
        self.res = QComboBox()
        for label, _ in RESOLUTIONS:
            self.res.addItem(label)
        self.res.setCurrentIndex(1)
        self.res.currentIndexChanged.connect(self._update_info)
        rl.addWidget(self.res)
        rl.addWidget(hint_label(
            "All slices are scaled by the <b>same</b> factor so a pixel means the same distance "
            "in every slice. Higher = more detail but slower and more memory."))
        self.mem = QLabel("")
        rl.addWidget(self.mem)
        left.addWidget(res_box)
        lay.addLayout(left, 2)

        # ---- right column: cutting plane + preview
        right = QVBoxLayout()
        self.orient = OrientationPanel()
        self.orient.orderFlipped.connect(self._reverse)
        self.orient.changed.connect(self._orientation_changed)
        right.addWidget(self.orient)
        pbox = QGroupBox("Preview")
        pl = QVBoxLayout(pbox)
        self.view = ImageView("Select a slice to preview it")
        pl.addWidget(self.view, 1)
        self.info = QLabel("")
        self.info.setProperty("hint", True)
        pl.addWidget(self.info)
        right.addWidget(pbox, 1)
        lay.addLayout(right, 3)
        self.state.orientation = self.orient.orientation()

        self._shapes_cache: dict[str, tuple[int, int]] = {}

    # ---------------------------------------------------------------- file ops
    def _set_paths(self, paths):
        self.state.paths = list(dict.fromkeys(paths))  # de-duplicate, keep order
        self.list.clear()
        for p in self.state.paths:
            self.list.addItem(os.path.basename(p))
            self.list.item(self.list.count() - 1).setToolTip(p)
        self._refresh_labels()
        self._update_info()
        if self.state.paths and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)

    def _refresh_labels(self):
        """Mark the two ends of the stack with their anatomical position."""
        first, last = self.state.orientation.ends
        n = self.list.count()
        for i in range(n):
            item = self.list.item(i)
            name = os.path.basename(item.toolTip())
            if i == 0:
                name += f"     ◀ slice 1 · {first}"
            elif i == n - 1:
                name += f"     ◀ slice {n} · {last}"
            item.setText(name)

    def _rows_moved(self, *args):
        self._sync_paths()
        self._refresh_labels()

    def _orientation_changed(self):
        self.state.orientation = self.orient.orientation()
        self._refresh_labels()   # the 3D view picks up the new orientation when shown again

    def _sync_paths(self, *args):
        """After a drag-reorder, read the new order back from the list."""
        self.state.paths = [self.list.item(i).toolTip() for i in range(self.list.count())]

    def _ordered(self, files):
        files = sorted(files, key=io.natural_key)
        return files[::-1] if self.orient.files_reversed() else files

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose slice images", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)")
        if files:
            self._set_paths(self.state.paths + self._ordered(files))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of slice images")
        if folder:
            self._add_from_folder(folder)

    def _add_from_folder(self, folder):
        found = io.find_images(folder)
        if not found:
            self.win.warn("No images (TIFF/PNG/JPG) found in that folder.")
            return
        self._set_paths(self.state.paths + self._ordered(found))

    def _sort(self):
        paths = sorted(self.state.paths, key=io.natural_key)
        if self.orient.files_reversed():
            paths.reverse()   # files are numbered the other way: put them in anatomical order
        self._set_paths(paths)

    def _reverse(self):
        self._sync_paths()
        self._set_paths(list(reversed(self.state.paths)))

    def _remove(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        paths = list(self.state.paths)
        for r in rows:
            paths.pop(r)
        self._set_paths(paths)

    def _clear(self):
        self._set_paths([])
        self.view.set_image(None)
        self.view.setText("Select a slice to preview it")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                paths += io.find_images(p)
            elif p.lower().endswith(io.IMAGE_EXTENSIONS):
                paths.append(p)
        if paths:
            self._set_paths(self.state.paths + self._ordered(paths))

    # ---------------------------------------------------------------- preview / info
    def _preview(self, row):
        if row < 0 or row >= len(self.state.paths):
            return
        path = self.state.paths[row]
        try:
            img = io.read_image(path)
        except Exception as e:  # noqa: BLE001
            self.info.setText(f"Could not read this file: {e}")
            return
        disp, _ = downsample_for_display(img, 900)
        colors = io.default_channel_colors(img.shape[2])
        self.view.set_image(composite(disp, colors))
        self._shapes_cache[path] = img.shape[:2]
        px = io.read_pixel_size_um(path)
        px_txt = f" · pixel size in file: {px:.4g} µm" if px else ""
        self.info.setText(
            f"Slice {row + 1} of {len(self.state.paths)} · {os.path.basename(path)} · "
            f"{img.shape[1]} × {img.shape[0]} px · {img.shape[2]} channel(s){px_txt}")

    def _update_info(self):
        n = len(self.state.paths)
        if n == 0:
            self.mem.setText("")
            return
        sample = self.state.paths[0]
        shape = self._shapes_cache.get(sample)
        if shape is None:
            try:
                shape = io.read_shape(sample)
                self._shapes_cache[sample] = shape
            except Exception:  # noqa: BLE001
                return
        size = RESOLUTIONS[self.res.currentIndex()][1]
        gb = io.estimate_memory_gb(n, max(shape), 3, size)
        warn = "  ⚠ may be too much for this computer" if gb > 8 else ""
        self.mem.setText(f"{n} slices · estimated memory ≈ {gb:.1f} GB{warn}")

    def _flip_loaded(self, key):
        st = self.state
        stk = st.stack
        stk.images.reverse()
        stk.paths = list(st.paths)
        stk.original_shapes.reverse()
        for name in ("proc", "masks", "transforms"):
            lst = getattr(st, name)
            if lst is not None:
                lst.reverse()
        if st.pre_key is not None:
            st.pre_key = (key,) + tuple(st.pre_key[1:])
        st.loaded_key = key
        st.vol = None          # rebuilt from the flipped slices when needed
        st.dye_result = None

    # ---------------------------------------------------------------- navigation
    def leave(self, proceed):
        st = self.state
        self._sync_paths()
        if len(st.paths) < 2:
            self.win.warn("Please add at least two slice images.")
            return
        st.orientation = self.orient.orientation()
        size = RESOLUTIONS[self.res.currentIndex()][1]
        key = (tuple(st.paths), size)
        if st.stack is not None and st.loaded_key == key:
            proceed()
            return
        if (st.stack is not None and st.loaded_key is not None
                and st.loaded_key[1] == size and tuple(st.paths) == st.loaded_key[0][::-1]):
            # Same slices, opposite order: flip everything in place, nothing to redo.
            self._flip_loaded(key)
            proceed()
            return

        def task(progress, cancel):
            return io.load_stack(st.paths, max_working_size=size, progress=progress)

        def done(stack):
            if stack.empty_slices:
                names = "\n".join("  • " + os.path.basename(stack.paths[i])
                                   for i in stack.empty_slices)
                chans = " / ".join(stack.channel_names)
                ans = QMessageBox.question(
                    self, "Images without signal",
                    f"These images have almost no signal in the {chans} channel that the "
                    f"other images use (e.g. a DAPI-only or blank image):\n\n{names}\n\n"
                    "Leave them out? Recommended: if they stay, the brain outline and "
                    "alignment will treat them as empty slices.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if ans == QMessageBox.Yes:
                    self._set_paths([p for i, p in enumerate(stack.paths)
                                     if i not in stack.empty_slices])
                    self.leave(proceed)      # reload without them
                    return
            st.stack = stack
            st.loaded_key = key
            st.max_working_size = size
            st.reset_after_load()
            st.channel_names = list(stack.channel_names)
            st.channel_colors = io.channel_colors_for(stack.channel_names)
            if stack.dropped_channels:
                self.win.status.setText(
                    f"Ignoring blank channel(s): {', '.join(stack.dropped_channels)} "
                    f"(no real signal in these images).")
            proceed()

        self.win.run_task(task, done, "Loading slices…")
