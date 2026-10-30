"""BrainStack 3D — start the app with:  python main.py"""
import os
import sys


def main():
    # pyvistaqt needs to know we use PySide6 before Qt is imported anywhere.
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication

    from brainstack.ui.main_window import MainWindow, apply_dark_palette

    app = QApplication(sys.argv)
    app.setApplicationName("BrainStack 3D")
    apply_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
