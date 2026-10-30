"""'How were the slices cut?' panel for step 1, with a little explanatory diagram."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QRadioButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from ..orientation import PLANES, Orientation

PLANE_KEYS = ["coronal", "sagittal", "transverse"]


class PlaneDiagram(QWidget):
    """Cartoon of a rodent brain showing where the cuts go and where slice 1 is."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plane = "coronal"
        self.setMinimumSize(250, 130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_plane(self, plane):
        self.plane = plane
        self.update()

    # -- drawing helpers
    def _side_brain(self, r: QRectF) -> QPainterPath:
        """Brain seen from the side: nose (olfactory bulb) left, cerebellum right."""
        p = QPainterPath()
        cx, cy, w, h = r.center().x(), r.center().y(), r.width(), r.height()
        p.addEllipse(QPointF(cx - 0.02 * w, cy), 0.36 * w, 0.40 * h)            # cerebrum
        p.addEllipse(QPointF(cx - 0.40 * w, cy + 0.05 * h), 0.09 * w, 0.13 * h)  # olfactory bulb
        p.addEllipse(QPointF(cx + 0.33 * w, cy + 0.12 * h), 0.13 * w, 0.24 * h)  # cerebellum
        return p.simplified()

    def _top_brain(self, r: QRectF) -> QPainterPath:
        """Brain seen from above: nose pointing up, two hemispheres."""
        p = QPainterPath()
        cx, cy, w, h = r.center().x(), r.center().y(), r.width(), r.height()
        p.addEllipse(QPointF(cx, cy + 0.02 * h), 0.27 * w, 0.36 * h)
        p.addEllipse(QPointF(cx, cy - 0.40 * h), 0.08 * w, 0.09 * h)
        p.addEllipse(QPointF(cx, cy + 0.40 * h), 0.18 * w, 0.10 * h)
        return p.simplified()

    def _arrow(self, qp, a: QPointF, b: QPointF, color):
        qp.setPen(QPen(color, 2))
        qp.drawLine(a, b)
        d = b - a
        n = (d.x() ** 2 + d.y() ** 2) ** 0.5 or 1
        ux, uy = d.x() / n, d.y() / n
        tip = QPolygonF([b, b - QPointF(ux * 9 - uy * 5, uy * 9 + ux * 5),
                         b - QPointF(ux * 9 + uy * 5, uy * 9 - ux * 5)])
        qp.setBrush(color)
        qp.drawPolygon(tip)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        brain_col = QColor("#c9d6e3")
        cut_col = QColor("#ff6b6b")
        txt = QColor("#9aa5b1")
        f = QFont()
        f.setPointSize(8)
        qp.setFont(f)

        if self.plane == "sagittal":
            r = QRectF(W / 2 - 80, 8, 160, H - 22)
            brain = self._top_brain(r)
            label = "seen from above (nose up)"
        else:
            r = QRectF(W / 2 - 115, 14, 230, H - 34)
            brain = self._side_brain(r)
            label = "seen from the side (nose left)"
        qp.setPen(QPen(brain_col, 1.5))
        qp.setBrush(QColor(201, 214, 227, 40))
        qp.drawPath(brain)

        # cut lines, clipped to the brain
        qp.save()
        qp.setClipPath(brain)
        qp.setPen(QPen(cut_col, 1.4, Qt.DashLine))
        br = brain.boundingRect()
        n = 9
        for i in range(n):
            if self.plane == "transverse":
                y = br.top() + (i + 0.5) * br.height() / n
                qp.drawLine(QPointF(br.left(), y), QPointF(br.right(), y))
            else:
                x = br.left() + (i + 0.5) * br.width() / n
                qp.drawLine(QPointF(x, br.top()), QPointF(x, br.bottom()))
        qp.restore()

        # direction arrow: slice 1 -> last slice
        if self.plane == "transverse":
            a, b = QPointF(br.right() + 14, br.top() + 4), QPointF(br.right() + 14, br.bottom() - 4)
            self._arrow(qp, a, b, cut_col)
            qp.setPen(txt)
            qp.drawText(QPointF(br.right() + 20, br.top() + 10), "slice 1")
            qp.drawText(QPointF(br.right() + 20, br.bottom()), "last")
        else:
            y = br.bottom() + 9
            a, b = QPointF(br.left() + 4, y), QPointF(br.right() - 4, y)
            self._arrow(qp, a, b, cut_col)
            qp.setPen(txt)
            qp.drawText(QPointF(br.left() - 4, y + 12), "slice 1")
            qp.drawText(QPointF(br.right() - 18, y + 12), "last")
        qp.setPen(txt)
        qp.drawText(QPointF(6, 12), label)
        qp.end()


class OrientationPanel(QGroupBox):
    """Asks how the slices were cut. Emits changed() and orderFlipped()."""

    changed = Signal()
    orderFlipped = Signal()

    def __init__(self, parent=None):
        super().__init__("How were the slices cut?", parent)
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        radios = QVBoxLayout()
        self.group = QButtonGroup(self)
        for i, key in enumerate(PLANE_KEYS):
            r = QRadioButton(PLANES[key]["label"])
            r.setToolTip(PLANES[key]["desc"])
            self.group.addButton(r, i)
            radios.addWidget(r)
        self.group.button(0).setChecked(True)
        radios.addStretch(1)
        top.addLayout(radios)
        self.diagram = PlaneDiagram()
        top.addWidget(self.diagram, 1)
        lay.addLayout(top)
        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setProperty("hint", True)
        lay.addWidget(self.desc)

        self.form = QFormLayout()
        self.order = QComboBox()
        self.form.addRow("My files are numbered:", self.order)
        self.top_is = QComboBox()
        self.top_is.addItems(["the top of the brain (dorsal)", "the bottom of the brain (ventral)"])
        self.form.addRow("Top of each image is:", self.top_is)
        self.nose = QComboBox()
        self.nose.addItems(["left", "right"])
        self.form.addRow("In the images the nose points:", self.nose)
        self.front = QComboBox()
        self.front.addItems(["up", "down"])
        self.form.addRow("In the images the nose points:", self.front)
        self.right_is = QComboBox()
        self.right_is.addItems(["the animal's right", "the animal's left", "not sure"])
        self.form.addRow("Right side of each image is:", self.right_is)
        lay.addLayout(self.form)

        self._order_idx = 0
        self.group.idToggled.connect(lambda i, on: on and self._plane_changed())
        self.order.currentIndexChanged.connect(self._order_changed)
        for c in (self.top_is, self.nose, self.front, self.right_is):
            c.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self._plane_changed(emit=False)

    # ------------------------------------------------------------------
    def plane(self) -> str:
        return PLANE_KEYS[max(0, self.group.checkedId())]

    def orientation(self) -> Orientation:
        return Orientation(
            plane=self.plane(),
            top_is_dorsal=self.top_is.currentIndex() == 0,
            right_is=("right", "left", "unknown")[self.right_is.currentIndex()],
            nose_points=("left", "right")[self.nose.currentIndex()],
            front_points=("up", "down")[self.front.currentIndex()],
        )

    def files_reversed(self) -> bool:
        return self.order.currentIndex() == 1

    def _plane_changed(self, emit=True):
        plane = self.plane()
        self.diagram.set_plane(plane)
        self.desc.setText(PLANES[plane]["desc"])
        self.order.blockSignals(True)
        idx = self.order.currentIndex()
        self.order.clear()
        self.order.addItems(PLANES[plane]["orders"])
        self.order.setCurrentIndex(max(0, idx))
        self.order.blockSignals(False)
        self.form.setRowVisible(self.top_is, plane in ("coronal", "sagittal"))
        self.form.setRowVisible(self.nose, plane == "sagittal")
        self.form.setRowVisible(self.front, plane == "transverse")
        self.form.setRowVisible(self.right_is, plane in ("coronal", "transverse"))
        if emit:
            self.changed.emit()

    def _order_changed(self, idx):
        if idx < 0 or idx == self._order_idx:
            return
        self._order_idx = idx
        self.orderFlipped.emit()
        self.changed.emit()
