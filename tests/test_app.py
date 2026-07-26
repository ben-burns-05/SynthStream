from pytestqt.qtbot import QtBot

from synthstream import __version__
from synthstream.app import MainWindow, create_application


def test_package_has_version() -> None:
    assert __version__ == "0.1.0"


def test_create_application_reuses_qapplication(qapp: object) -> None:
    assert create_application(["synthstream"]) is qapp


def test_main_window_launches(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    assert window.windowTitle() == "SynthStream"
    assert window.centralWidget().text() == "SynthStream — project setup complete"
    assert window.isVisible()

