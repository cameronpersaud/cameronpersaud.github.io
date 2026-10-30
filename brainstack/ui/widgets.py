"""Reusable Qt pieces: image viewer, background worker, helpers."""
from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QColorDialog, QLabel, QPushButton, QSizePolicy

# ------------------------------------------------------------------ helpers

def to_qimage(rgb: np.ndarray) -> QImage:
    """(H, W, 3) uint8 -> QImage (copied, so the array can be freed)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def auto_contrast(x: np.ndarray, lo=1.0, hi=99.7) -> np.ndarray:
    x = x.astype(np.float32)
    a, b = np.percentile(x[::2, ::2], (lo, hi))
    # Don't blow a channel that is only background noise up to full brightness:
    # insist on a minimum range relative to the data's likely bit depth.
    top = float(x.max())
    full = 1.0 if top <= 1.0 else 255.0 if top <= 255 else 4095.0 if top <= 4095 else 65535.0
    b = max(b, a + 0.03 * full)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def composite(img: np.ndarray, colors, visible=None, normalize: bool = True) -> np.ndarray:
    """Multi-channel image -> coloured RGB uint8 for display."""
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    rgb = np.zeros(img.shape[:2] + (3,), np.float32)
    for c in range(img.shape[2]):
        if visible is not None and not visible[c]:
            continue
        ch = auto_contrast(img[..., c]) if normalize else np.clip(img[..., c], 0, 1)
        rgb += ch[..., None] * (np.asarray(colors[c], np.float32) / 255.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def downsample_for_display(img: np.ndarray, max_px: int = 700) -> tuple[np.ndarray, int]:
    step = max(1, int(np.ceil(max(img.shape[:2]) / max_px)))
    return img[::step, ::step], step


def mask_outline(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    return mask & ~ndi.binary_erosion(mask, iterations=2)


# ------------------------------------------------------------------ widgets

class ImageView(QLabel):
    """Shows an RGB image scaled to fit; reports clicks in image pixel coords."""

    clicked = Signal(float, float)

    def __init__(self, placeholder: str = "No image", parent=None):
        super().__init__(parent)
        self._img: QImage | None = None
        self._points: list[tuple[float, float]] = []
        self._draw_line = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0d1014; color:#7d8794; border-radius:6px;")
        self.setText(placeholder)

    def set_image(self, rgb: np.ndarray | None):
        self._img = None if rgb is None else to_qimage(rgb)
        self.update()

    def set_points(self, pts, draw_line=False):
        self._points = list(pts)
        self._draw_line = draw_line
        self.update()

    def sizeHint(self):
        return QSize(480, 400)

    def _target_rect(self) -> QRectF | None:
        if self._img is None:
            return None
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, ev):
        if self._img is None:
            return super().paintEvent(ev)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d1014"))
        r = self._target_rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, self._img)
        if self._points:
            sx = r.width() / self._img.width()
            pts = [QPointF(r.x() + x * sx, r.y() + y * sx) for x, y in self._points]
            pen = QPen(QColor("#ffd23f"), 2)
            p.setPen(pen)
            if self._draw_line and len(pts) == 2:
                p.drawLine(pts[0], pts[1])
            for q in pts:
                p.drawEllipse(q, 5, 5)
        p.end()

    def mousePressEvent(self, ev):
        r = self._target_rect()
        if r is None or ev.button() != Qt.LeftButton:
            return
        pos = ev.position()
        if not r.contains(pos):
            return
        s = r.width() / self._img.width()
        self.clicked.emit((pos.x() - r.x()) / s, (pos.y() - r.y()) / s)


class ColorButton(QPushButton):
    colorChanged = Signal(tuple)

    def __init__(self, rgb=(255, 255, 255), parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 22)
        self.set_color(rgb)
        self.clicked.connect(self._pick)

    def set_color(self, rgb):
        self.rgb = tuple(int(v) for v in rgb)
        self.setStyleSheet(
            f"background: rgb{self.rgb}; border: 1px solid #555; border-radius: 4px;"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(*self.rgb), self, "Channel colour")
        if c.isValid():
            self.set_color((c.red(), c.green(), c.blue()))
            self.colorChanged.emit(self.rgb)


class Worker(QThread):
    """Runs fn(progress=..., cancel=...) off the UI thread."""

    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda v, msg="": self.progress.emit(int(v), msg),
                          cancel=lambda: self._cancel, **self.kwargs)
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001 - show any failure to the user
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


def hint_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.RichText)
    lab.setProperty("hint", True)
    return lab
