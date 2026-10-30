"""Main window: a five-step guided workflow."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from .state import AppState
from .widgets import Worker

STYLE = """
QWidget { font-size: 13px; }
QMainWindow, QWidget#central { background: #161a1f; }
QLabel { color: #dfe4ea; }
QLabel[hint="true"] { color: #9aa5b1; }
QLabel#stepTitle { font-size: 20px; font-weight: 600; color: #ffffff; }
QLabel#stepHint { color: #aab4bf; font-size: 13px; }
QGroupBox { color: #dfe4ea; border: 1px solid #2c333b; border-radius: 8px; margin-top: 14px;
            padding: 10px 8px 8px 8px; background: #1b2026; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8fd3ff;
                   font-weight: 600; }
QListWidget#steps { background: #11151a; border: none; color: #c9d1d9; font-size: 14px;
                    padding-top: 8px; outline: 0; }
QListWidget#steps::item { padding: 12px 14px; border-left: 3px solid transparent; }
QListWidget#steps::item:selected { background: #1f2933; color: #ffffff;
                                   border-left: 3px solid #4cc2ff; }
QListWidget#steps::item:disabled { color: #4a535d; }
QPushButton { background: #2a323b; color: #e6edf3; border: 1px solid #39424c;
              border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #333d47; }
QPushButton:disabled { color: #5d6670; background: #20262c; }
QPushButton#primary { background: #1f8fff; border-color: #1f8fff; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3a9dff; }
QPushButton:checked { background: #1f6fb8; border-color: #4cc2ff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #0f1317; color: #e6edf3;
              border: 1px solid #39424c; border-radius: 5px; padding: 4px 6px; }
QListWidget { background: #0f1317; color: #e6edf3; border: 1px solid #2c333b; border-radius: 6px; }
QListWidget::item:selected { background: #1f6fb8; }
QCheckBox, QRadioButton { color: #dfe4ea; }
QProgressBar { background: #0f1317; border: 1px solid #2c333b; border-radius: 5px;
               color: white; text-align: center; height: 16px; }
QProgressBar::chunk { background: #1f8fff; border-radius: 4px; }
QScrollArea { border: none; background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #39424c; border-radius: 2px; }
QSlider::handle:horizontal { background: #4cc2ff; width: 14px; margin: -6px 0; border-radius: 7px; }
"""

STEPS = [
    ("1  Import slices", "Add your slice images and put them in order (first slice at the top)."),
    ("2  Set the scale", "Tell the software how big a pixel is and how far apart the slices are."),
    ("3  Dye & channels", "Tell the software which channel holds the red dextran dye, and clean up "
                          "the images."),
    ("4  Align slices", "Line up every slice with its neighbours. Check the overlay and fix any by hand."),
    ("5  Dye in 3D", "See where the dye sits inside the brain, measure it, and export."),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from .page_align import AlignPage
        from .page_calibrate import CalibratePage
        from .page_import import ImportPage
        from .page_prepare import PreparePage
        from .page_view import ViewPage

        self.setWindowTitle(f"BrainStack 3D  —  map red dextran dye in 3D  (v{__version__})")
        self.resize(1400, 880)
        self.state = AppState()
        self.worker: Worker | None = None
        self.reached = 0  # furthest step unlocked

        central = QWidget(objectName="central")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- left: step list
        side = QFrame()
        side.setStyleSheet("background:#11151a;")
        side.setFixedWidth(230)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 18, 0, 12)
        brand = QLabel("  BrainStack 3D")
        brand.setStyleSheet("font-size:18px; font-weight:700; color:#4cc2ff; padding:0 8px 10px 8px;")
        sl.addWidget(brand)
        self.step_list = QListWidget(objectName="steps")
        for name, _ in STEPS:
            self.step_list.addItem(QListWidgetItem(name))
        self.step_list.currentRowChanged.connect(self._step_clicked)
        sl.addWidget(self.step_list, 1)
        tip = QLabel("Tip: you can jump back to any\nearlier step at any time.")
        tip.setProperty("hint", True)
        tip.setStyleSheet("padding: 8px 14px; color:#6b7580; font-size:12px;")
        sl.addWidget(tip)
        root.addWidget(side)

        # ---- right: header, page, footer
        main = QVBoxLayout()
        main.setContentsMargins(22, 16, 22, 12)
        self.title = QLabel(objectName="stepTitle")
        self.hint = QLabel(objectName="stepHint")
        self.hint.setWordWrap(True)
        main.addWidget(self.title)
        main.addWidget(self.hint)
        self.pages = QStackedWidget()
        self.page_objs = [ImportPage(self), CalibratePage(self), PreparePage(self),
                          AlignPage(self), ViewPage(self)]
        for p in self.page_objs:
            self.pages.addWidget(p)
            # Let long option names shrink instead of pushing panels wider than the window.
            for combo in p.findChildren(QComboBox):
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
        main.addWidget(self.pages, 1)

        footer = QHBoxLayout()
        self.status = QLabel("")
        self.status.setProperty("hint", True)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(260)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.back_btn = QPushButton("◀  Back")
        self.next_btn = QPushButton("Next  ▶", objectName="primary")
        self.back_btn.clicked.connect(self.go_back)
        self.next_btn.clicked.connect(self.go_next)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.progress)
        footer.addWidget(self.cancel_btn)
        footer.addSpacing(12)
        footer.addWidget(self.back_btn)
        footer.addWidget(self.next_btn)
        main.addLayout(footer)
        root.addLayout(main, 1)

        self._show_step(0)

    # ------------------------------------------------------------ navigation
    def current(self) -> int:
        return self.pages.currentIndex()

    def _refresh_step_list(self):
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled if i <= self.reached else Qt.NoItemFlags
            item.setFlags(flags)

    def _show_step(self, i: int):
        self.reached = max(self.reached, i)
        self.pages.setCurrentIndex(i)
        self.title.setText(STEPS[i][0][3:])
        self.hint.setText(STEPS[i][1])
        self.step_list.blockSignals(True)
        self._refresh_step_list()
        self.step_list.setCurrentRow(i)
        self.step_list.blockSignals(False)
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setVisible(i < len(STEPS) - 1)
        self.page_objs[i].on_enter()

    def _step_clicked(self, row: int):
        if row < 0 or row == self.current():
            return
        if self.worker is not None or row > self.reached:
            self.step_list.blockSignals(True)
            self.step_list.setCurrentRow(self.current())
            self.step_list.blockSignals(False)
            return
        if row > self.current():
            # Moving forward through the list = pressing Next until we get there.
            self.go_next(target=row)
        else:
            self._show_step(row)

    def go_back(self):
        if self.current() > 0 and self.worker is None:
            self._show_step(self.current() - 1)

    def go_next(self, target: int | None = None):
        if self.worker is not None:
            return
        i = self.current()
        if i >= len(STEPS) - 1:
            return

        def proceed():
            nxt = i + 1
            if target is not None and target > nxt and nxt <= self.reached:
                self._show_step(nxt)
                self.go_next(target)
            else:
                self._show_step(nxt)

        self.page_objs[i].leave(proceed)

    # ------------------------------------------------------------ background tasks
    def run_task(self, fn, on_done, label: str, *args, **kwargs):
        """Run fn in the background with progress shown in the footer."""
        w = Worker(fn, *args, **kwargs)
        self.worker = w
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_btn.show()
        self.status.setText(label)
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)

        def finish():
            # The result arrives just BEFORE the thread's run() returns. Wait for the
            # thread to really stop before letting go of it: destroying a QThread that
            # is still running makes Qt abort the whole app
            # ("QThread: Destroyed while thread is still running").
            w.wait()
            self.worker = None
            self.progress.hide()
            self.cancel_btn.hide()
            self.next_btn.setEnabled(True)
            self.back_btn.setEnabled(self.current() > 0)

        def ok(res):
            cancelled = w._cancel
            finish()
            if cancelled:
                self.status.setText("Cancelled.")
                return
            self.status.setText("")
            on_done(res)

        def bad(msg):
            finish()
            self.status.setText("Something went wrong.")
            QMessageBox.critical(self, "BrainStack 3D", msg)

        w.progress.connect(lambda v, m: (self.progress.setValue(v),
                                         self.status.setText(m or label)))
        w.done.connect(ok)
        w.failed.connect(bad)
        w.start()

    def closeEvent(self, event):
        """Closing mid-task: ask the task to stop and wait for it, instead of crashing."""
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Finishing the current task before closing…")
            self.worker.wait()
        super().closeEvent(event)

    def _cancel_task(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling…")

    def info(self, text: str, title="BrainStack 3D"):
        QMessageBox.information(self, title, text)

    def warn(self, text: str, title="BrainStack 3D"):
        QMessageBox.warning(self, title, text)


def apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#161a1f"))
    pal.setColor(QPalette.WindowText, QColor("#dfe4ea"))
    pal.setColor(QPalette.Base, QColor("#0f1317"))
    pal.setColor(QPalette.AlternateBase, QColor("#1b2026"))
    pal.setColor(QPalette.Text, QColor("#e6edf3"))
    pal.setColor(QPalette.Button, QColor("#2a323b"))
    pal.setColor(QPalette.ButtonText, QColor("#e6edf3"))
    pal.setColor(QPalette.Highlight, QColor("#1f6fb8"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#1b2026"))
    pal.setColor(QPalette.ToolTipText, QColor("#e6edf3"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)
