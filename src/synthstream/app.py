"""Desktop application entry point."""

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    """Minimal application shell used to validate the project setup."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SynthStream")
        self.setCentralWidget(QLabel("SynthStream — project setup complete"))
        self.resize(640, 360)


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process application, creating it when necessary."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication(list(argv) if argv is not None else [])


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the SynthStream desktop application."""
    app = create_application(sys.argv if argv is None else argv)
    window = MainWindow()
    window.show()
    return app.exec()
