from pathlib import Path
from threading import Event

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from pytestqt.qtbot import QtBot

from synthstream import __version__
from synthstream.app import MainWindow, create_application
from synthstream.audio import FakeDuplexAudioBackend
from synthstream.live import LiveVoicebankEngine
from synthstream.live import engine as live_engine_module
from synthstream.offline.voicebank_phonemizer import VoicebankCapability, VoicebankProfile
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


def _make_bank(root: Path) -> None:
    time = np.arange(round(SAMPLE_RATE * 0.5), dtype=np.float32) / SAMPLE_RATE
    audio = np.concatenate(
        (
            0.3 * np.sin(2 * np.pi * 220 * time[: round(SAMPLE_RATE * 0.1)]),
            0.3 * np.sin(2 * np.pi * 330 * time[: round(SAMPLE_RATE * 0.15)]),
            0.3 * np.sin(2 * np.pi * 440 * time),
        )
    ).astype(np.float32)
    sf.write(root / "a.wav", audio, SAMPLE_RATE)
    (root / "oto.ini").write_text("a.wav=a,0,200,0,50,20\n", encoding="utf-8")


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


def test_gui_requires_a_voicebank_before_starting(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    window.start_conversion()

    assert not window.is_converting
    assert window.status_label.text() == "Select a voicebank before starting."


def test_gui_device_discovery_failure_keeps_system_defaults(qtbot: QtBot) -> None:
    def broken_provider() -> tuple[tuple[tuple[str, int | str | None], ...], ...]:
        raise RuntimeError("device query failed")

    window = MainWindow(device_provider=broken_provider)
    qtbot.addWidget(window)

    assert window.input_device_combo.currentText() == "System default"
    assert window.output_device_combo.currentText() == "System default"
    assert "device query failed" in window.errors_label.text()


def test_gui_waits_for_direct_model_before_starting_transport(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch,
) -> None:
    _make_bank(tmp_path)
    capability = VoicebankCapability(
        VoicebankProfile("test", "english-cvvc", {}, frozenset()),
        0.95,
        1.0,
        (),
        "test profile",
    )
    monkeypatch.setattr(live_engine_module, "detect_voicebank_profile", lambda bank: capability)
    preparation_started = Event()
    release_preparation = Event()

    def slow_prepare(self: LiveVoicebankEngine) -> None:
        preparation_started.set()
        release_preparation.wait(2.0)

    monkeypatch.setattr(LiveVoicebankEngine, "prepare_direct_ipa", slow_prepare)
    backend = FakeDuplexAudioBackend()
    window = MainWindow(
        bank=load_voicebank(tmp_path, use_cache=False),
        backend=backend,
        engine_kwargs={"buffer_duration_seconds": 2.0},
    )
    qtbot.addWidget(window)

    window.start_conversion()
    qtbot.waitUntil(preparation_started.is_set, timeout=2_000)
    assert window.is_starting
    assert not window.is_converting
    assert "Loading direct IPA model" in window.status_label.text()

    release_preparation.set()
    qtbot.waitUntil(lambda: window.is_converting, timeout=2_000)
    window.stop_conversion()
    assert not window.is_converting


def test_gui_controls_production_engine_end_to_end(tmp_path: Path, qtbot: QtBot) -> None:
    _make_bank(tmp_path)
    backend = FakeDuplexAudioBackend()
    window = MainWindow(
        bank=load_voicebank(tmp_path, use_cache=False),
        backend=backend,
        use_direct_ipa=False,
        engine_kwargs={
            "analysis_chunk_seconds": 0.1,
            "buffer_duration_seconds": 3.0,
        },
    )
    qtbot.addWidget(window)
    window.start_conversion()
    assert window.is_converting

    source_time = np.arange(round(SAMPLE_RATE * 0.5), dtype=np.float32) / SAMPLE_RATE
    source = (0.3 * np.sin(2 * np.pi * 330 * source_time)).astype(np.float32)
    backend.feed(source)
    qtbot.waitUntil(
        lambda: window.engine is not None and window.engine.statistics.feature_chunks_processed > 0,
        timeout=10_000,
    )
    window.stop_conversion()

    assert not window.is_converting
    assert window.engine is not None
    assert window.engine.statistics.committed_segments > 0
    assert window.engine.statistics.rendered_output_samples > 0
    queued = window.engine.stream.output_buffer.read(
        window.engine.stream.output_buffer.available_samples
    )
    assert float(np.max(np.abs(queued))) > 0.01
    assert window.status_label.text() == "Conversion stopped."
    window.refresh_status()
    assert "Committed sections:" in window.committed_label.text()
    assert window.errors_label.text().startswith("Audio errors:")


def test_gui_direct_aiko_voicebank_end_to_end(qtbot: QtBot) -> None:
    project_root = Path(__file__).parents[1]
    bank_path = project_root / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    human_path = project_root / "tests" / "fixtures" / "human" / "voices_sentence.wav"
    if not bank_path.is_dir():
        import pytest

        pytest.skip("local Aiko development voicebank is not installed")

    human, sample_rate = sf.read(human_path, dtype="float32", always_2d=True)
    assert sample_rate == SAMPLE_RATE
    backend = FakeDuplexAudioBackend()
    window = MainWindow(
        bank=load_voicebank(bank_path),
        backend=backend,
        engine_kwargs={
            "analysis_chunk_seconds": 0.2,
            "direct_update_seconds": 0.4,
            "buffer_duration_seconds": 5.0,
        },
    )
    qtbot.addWidget(window)

    window.start_conversion()
    qtbot.waitUntil(lambda: window.is_converting, timeout=60_000)
    assert "Converting" in window.status_label.text()
    backend.feed(np.mean(human, axis=1))
    qtbot.waitUntil(
        lambda: window.engine is not None
        and window.engine.statistics.direct_ipa_updates >= 1,
        timeout=60_000,
    )
    window.stop_conversion()

    assert window.engine is not None
    statistics = window.engine.statistics
    assert statistics.committed_segments >= 20
    assert statistics.rendered_output_samples > 0
    queued = window.engine.stream.output_buffer.read(
        window.engine.stream.output_buffer.available_samples
    )
    assert len(queued) > 0
    assert float(np.max(np.abs(queued))) > 0.01
    assert statistics.worker_error is None
