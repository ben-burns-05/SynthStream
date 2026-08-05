"""Desktop application entry point and production-engine controls."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Event, Thread
from typing import Any, cast

import sounddevice as sd  # type: ignore[import-untyped]
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from synthstream.audio import DuplexAudioBackend, SoundDeviceDuplexBackend
from synthstream.live import LiveVoicebankEngine
from synthstream.voicebank import Voicebank
from synthstream.voicebank import load_voicebank as load_voicebank_data

DeviceValue = int | str | None
DeviceProvider = Callable[[], tuple[tuple[tuple[str, DeviceValue], ...], ...]]
BackendFactory = Callable[[DeviceValue, DeviceValue], DuplexAudioBackend]


class _MainPanel(QWidget):
    """Control panel that retains the original launch smoke-test text API."""

    def text(self) -> str:
        """Return the launch message used by the original application test."""
        return "SynthStream — project setup complete"


class MainWindow(QMainWindow):
    """Basic GUI for selecting a bank/devices and running the live engine."""

    def __init__(
        self,
        *,
        bank: Voicebank | None = None,
        voicebank_root: str | Path | None = None,
        backend: DuplexAudioBackend | None = None,
        backend_factory: BackendFactory | None = None,
        device_provider: DeviceProvider | None = None,
        use_direct_ipa: bool = True,
        engine_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SynthStream")
        self.resize(760, 520)
        self._bank = bank
        self._backend = backend
        self._backend_factory = backend_factory or (
            lambda input_device, output_device: SoundDeviceDuplexBackend(
                input_device=input_device,
                output_device=output_device,
            )
        )
        self._device_provider = device_provider or _default_device_provider
        self._use_direct_ipa = use_direct_ipa
        self._engine_kwargs: dict[str, Any] = dict(engine_kwargs or {})
        self._engine: LiveVoicebankEngine | None = None
        self._startup_thread: Thread | None = None
        self._startup_cancel = Event()
        self._startup_error: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.refresh_status)

        panel = _MainPanel()
        self.setCentralWidget(panel)
        root_layout = QVBoxLayout(panel)

        title = QLabel("SynthStream live voicebank conversion")
        title.setObjectName("title_label")
        root_layout.addWidget(title)

        bank_row = QHBoxLayout()
        self.voicebank_path_edit = QLineEdit()
        self.voicebank_path_edit.setReadOnly(True)
        self.voicebank_path_edit.setPlaceholderText("Choose a voicebank folder")
        self.voicebank_button = QPushButton("Select voicebank…")
        self.voicebank_button.clicked.connect(self._choose_voicebank)
        bank_row.addWidget(self.voicebank_path_edit, stretch=1)
        bank_row.addWidget(self.voicebank_button)
        root_layout.addLayout(bank_row)

        self.input_device_combo = QComboBox()
        self.output_device_combo = QComboBox()
        self.errors_label = QLabel("Audio errors: none")
        self.refresh_devices()
        form = QFormLayout()
        form.addRow("Input device", self.input_device_combo)
        form.addRow("Output device", self.output_device_combo)
        root_layout.addLayout(form)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start_conversion)
        self.stop_button.clicked.connect(self.stop_conversion)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        root_layout.addLayout(controls)

        self.status_label = QLabel("Select a voicebank to begin.")
        self.units_label = QLabel("Voicebank units: —")
        self.input_level_label = QLabel("Input buffer: 0 samples")
        self.output_level_label = QLabel("Output buffer: 0 samples")
        self.committed_label = QLabel("Committed sections: 0")
        self.latency_label = QLabel("Estimated processing time: 0.000 s")
        diagnostics = QFormLayout()
        diagnostics.addRow("Status", self.status_label)
        diagnostics.addRow("Voicebank", self.units_label)
        diagnostics.addRow("Input", self.input_level_label)
        diagnostics.addRow("Output", self.output_level_label)
        diagnostics.addRow("Matching", self.committed_label)
        diagnostics.addRow("Processing", self.latency_label)
        diagnostics.addRow("Audio", self.errors_label)
        root_layout.addLayout(diagnostics)
        root_layout.addStretch(1)

        if bank is not None:
            self.set_voicebank(bank, voicebank_root)
        elif voicebank_root is not None:
            self.load_voicebank(voicebank_root)

    @property
    def engine(self) -> LiveVoicebankEngine | None:
        """Return the currently configured production engine, if any."""
        return self._engine

    @property
    def voicebank(self) -> Voicebank | None:
        """Return the loaded voicebank."""
        return self._bank

    @property
    def is_converting(self) -> bool:
        """Whether the live transport is currently running."""
        return self._engine is not None and self._engine.is_running

    @property
    def is_starting(self) -> bool:
        """Whether direct-IPA assets are still being prepared."""
        return self._startup_thread is not None

    def set_voicebank(self, bank: Voicebank, root: str | Path | None = None) -> None:
        """Install an already loaded bank, primarily useful for integrations/tests."""
        if self.is_converting or self.is_starting:
            self.stop_conversion()
        self._bank = bank
        self._engine = None
        display_path = Path(root) if root is not None else bank.root
        self.voicebank_path_edit.setText(str(display_path))
        self.units_label.setText(f"Voicebank units: {len(bank.units)}")
        self.status_label.setText(f"Loaded {len(bank.units)} voicebank units.")

    def load_voicebank(self, root: str | Path) -> Voicebank:
        """Load a complete UTAU-style bank and update the GUI state."""
        bank = load_voicebank_data(root)
        self.set_voicebank(bank, root)
        return bank

    def refresh_devices(self) -> None:
        """Refresh selectable input and output devices without opening them."""
        try:
            input_devices, output_devices = self._device_provider()
        except Exception as error:  # pragma: no cover - hardware-dependent
            input_devices = (("System default", None),)
            output_devices = (("System default", None),)
            self.errors_label.setText(f"Audio errors: {error}")
        self._fill_device_combo(self.input_device_combo, input_devices)
        self._fill_device_combo(self.output_device_combo, output_devices)

    def start_conversion(self) -> None:
        """Start the same live engine used by the programmatic API."""
        if self.is_converting or self.is_starting:
            return
        if self._bank is None:
            self.status_label.setText("Select a voicebank before starting.")
            return
        input_device = self.input_device_combo.currentData()
        output_device = self.output_device_combo.currentData()
        if self._backend is not None:
            selected_backend = self._backend
        else:
            selected_backend = self._backend_factory(
                cast(DeviceValue, input_device), cast(DeviceValue, output_device)
            )
        try:
            self._engine = LiveVoicebankEngine(
                self._bank,
                selected_backend,
                use_direct_ipa=self._use_direct_ipa,
                **self._engine_kwargs,
            )
        except Exception as error:
            self._engine = None
            self.status_label.setText(f"Unable to start: {error}")
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._startup_error = None
        self._startup_cancel.clear()
        if self._use_direct_ipa:
            self.status_label.setText(
                "Loading direct IPA model; first use may download approximately 1.3 GB."
            )
            self._startup_thread = Thread(
                target=self._prepare_engine,
                args=(self._engine,),
                name="synthstream-gui-startup",
                daemon=True,
            )
            self._startup_thread.start()
        else:
            self._activate_engine()
        self._timer.start()

    def stop_conversion(self) -> None:
        """Flush committed audio and stop the selected transport cleanly."""
        if self.is_starting:
            self._startup_cancel.set()
            self._startup_thread = None
            self._engine = None
            self._timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Conversion stopped.")
            return
        engine = self._engine
        if engine is not None and engine.is_running:
            engine.stop(flush=True)
        self._timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Conversion stopped.")
        self.refresh_status()

    def refresh_status(self) -> None:
        """Refresh diagnostics from the live engine and transport buffers."""
        startup_thread = self._startup_thread
        if startup_thread is not None:
            if startup_thread.is_alive():
                self.status_label.setText(
                    "Loading direct IPA model; first use may download approximately 1.3 GB."
                )
                return
            self._startup_thread = None
            if self._startup_cancel.is_set():
                return
            if self._startup_error is not None:
                self._engine = None
                self._timer.stop()
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.status_label.setText(
                    f"Unable to prepare direct IPA model: {self._startup_error}"
                )
                return
            self._activate_engine()
        engine = self._engine
        if engine is None:
            return
        statistics = engine.statistics
        stream_stats = engine.stream.statistics
        self.input_level_label.setText(
            f"Input buffer: {engine.stream.input_buffer.available_samples} samples"
        )
        self.output_level_label.setText(
            f"Output buffer: {engine.stream.output_buffer.available_samples} samples"
        )
        self.committed_label.setText(
            f"Committed sections: {statistics.committed_segments} · rendered: "
            f"{statistics.rendered_output_samples} samples"
        )
        self.latency_label.setText(
            f"Estimated processing time: {statistics.processing_seconds:.3f} s"
        )
        error_text = statistics.worker_error
        if error_text is None:
            error_count = (
                stream_stats.output_underflow_samples + stream_stats.input_overflow_samples
            )
            self.errors_label.setText(f"Audio errors: {error_count} samples")
        else:
            self.errors_label.setText(f"Audio errors: {error_text}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_converting or self.is_starting:
            self.stop_conversion()
        event.accept()

    def _prepare_engine(self, engine: LiveVoicebankEngine | None) -> None:
        if engine is None:
            self._startup_error = "engine was not constructed"
            return
        try:
            engine.prepare_direct_ipa()
        except Exception as error:  # pragma: no cover - model/network dependent
            self._startup_error = str(error)

    def _activate_engine(self) -> None:
        engine = self._engine
        if engine is None:
            return
        try:
            engine.start(background=True)
        except Exception as error:
            self._engine = None
            self._timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText(f"Unable to start: {error}")
            return
        self.status_label.setText("Converting live microphone audio.")

    def _choose_voicebank(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "Select voicebank folder")
        if root:
            try:
                self.load_voicebank(root)
            except Exception as error:
                self.status_label.setText(f"Unable to load voicebank: {error}")

    @staticmethod
    def _fill_device_combo(
        combo: QComboBox,
        devices: Sequence[tuple[str, DeviceValue]],
    ) -> None:
        combo.clear()
        for name, value in devices:
            combo.addItem(name, value)


def _default_device_provider() -> tuple[tuple[tuple[str, DeviceValue], ...], ...]:
    """Return device choices, falling back safely on systems without audio hardware."""
    try:
        devices = sd.query_devices()
    except Exception:  # pragma: no cover - hardware-dependent
        return ((('System default', None),), (('System default', None),))
    inputs: list[tuple[str, DeviceValue]] = []
    outputs: list[tuple[str, DeviceValue]] = []
    for index, raw_info in enumerate(devices):
        info = cast(Mapping[str, Any], raw_info)
        name = str(info.get("name", f"Device {index}"))
        if int(info.get("max_input_channels", 0)) > 0:
            inputs.append((name, index))
        if int(info.get("max_output_channels", 0)) > 0:
            outputs.append((name, index))
    if not inputs:
        inputs.append(("System default", None))
    if not outputs:
        outputs.append(("System default", None))
    return tuple(inputs), tuple(outputs)


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
