import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import AnalysisConfig
from synthstream.audio import FakeDuplexAudioBackend
from synthstream.live import LiveVoicebankEngine
from synthstream.offline.direct_phonemes import DetectedPhone
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).parents[2]


def _tone(frequency: float, duration: float) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.35 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _make_bank(root: Path) -> None:
    audio = np.concatenate((_tone(220, 0.05), _tone(330, 0.15), _tone(440, 0.3)))
    sf.write(root / "a.wav", audio, SAMPLE_RATE)
    (root / "oto.ini").write_text("a.wav=a,0,200,0,50,20\n", encoding="utf-8")


def test_live_engine_moves_fake_microphone_through_decode_and_render(
    tmp_path: Path,
) -> None:
    _make_bank(tmp_path)
    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine(
        load_voicebank(tmp_path, use_cache=False),
        backend,
        analysis_chunk_seconds=0.1,
        buffer_duration_seconds=2.0,
        use_direct_ipa=False,
    )

    source = np.concatenate((_tone(220, 0.15), _tone(330, 0.15), _tone(440, 0.2)))
    engine.start(background=False)
    backend.feed(source)
    assert engine.process_available() > 0
    engine.flush()

    queued = engine.stream.output_buffer.read(engine.stream.output_buffer.available_samples)
    engine.stop()

    statistics = engine.statistics
    assert statistics.input_blocks_processed > 0
    assert statistics.feature_chunks_processed >= 1
    assert statistics.feature_frames_processed >= 1
    assert statistics.committed_segments >= 1
    assert statistics.rendered_output_samples == len(queued)
    assert len(queued) > 0
    assert np.all(np.isfinite(queued))
    assert float(np.max(np.abs(queued))) > 0.01
    assert statistics.worker_error is None


def test_live_engine_flush_is_idempotent_after_final_decoder_commit(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine(
        load_voicebank(tmp_path, use_cache=False),
        backend,
        analysis_chunk_seconds=0.1,
        buffer_duration_seconds=2.0,
        use_direct_ipa=False,
    )

    engine.start(background=False)
    backend.feed(_tone(220, 0.3))
    engine.process_available()
    engine.flush()
    first_statistics = engine.statistics
    engine.flush()
    second_statistics = engine.statistics
    engine.stop()

    assert second_statistics.committed_segments == first_statistics.committed_segments
    assert second_statistics.rendered_output_samples == first_statistics.rendered_output_samples
    assert second_statistics.worker_error is None


def test_live_engine_can_run_background_worker_with_fake_backend(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine(
        load_voicebank(tmp_path, use_cache=False),
        backend,
        analysis_chunk_seconds=0.1,
        buffer_duration_seconds=2.0,
        use_direct_ipa=False,
    )

    engine.start(background=True)
    backend.feed(np.concatenate((_tone(220, 0.2), _tone(330, 0.2))))
    engine.stop(flush=True)

    assert engine.statistics.worker_error is None
    assert engine.statistics.feature_chunks_processed >= 1


def test_live_engine_requires_matching_stream_and_analysis_rates(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    with pytest.raises(ValueError, match="sample rates"):
        LiveVoicebankEngine(
            load_voicebank(tmp_path, use_cache=False),
            FakeDuplexAudioBackend(),
            sample_rate=SAMPLE_RATE,
            analysis_config=AnalysisConfig(sample_rate=22_050),
            use_direct_ipa=False,
        )


def test_live_engine_requires_supported_profile_for_direct_mode(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    with pytest.raises(ValueError, match="supported English voicebank profile"):
        LiveVoicebankEngine(load_voicebank(tmp_path, use_cache=False), FakeDuplexAudioBackend())


def test_live_engine_uses_direct_ipa_for_real_aiko_voicebank() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")
    human, sample_rate = sf.read(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_sentence.wav",
        dtype="float32",
        always_2d=True,
    )
    assert sample_rate == SAMPLE_RATE

    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine.from_voicebank(
        bank_path,
        backend,
        analysis_chunk_seconds=0.2,
        direct_update_seconds=0.4,
        buffer_duration_seconds=5.0,
    )
    engine.start(background=False)
    # Exercise the low-level signal range common to USB/wireless microphones.
    backend.feed(np.mean(human, axis=1) * 0.01)
    engine.process_available()
    engine.flush()
    queued = engine.stream.output_buffer.read(engine.stream.output_buffer.available_samples)
    engine.stop()

    statistics = engine.statistics
    assert statistics.direct_ipa_updates >= 1
    assert statistics.committed_segments >= 20
    assert len(queued) > 0
    assert float(np.max(np.abs(queued))) > 0.01
    assert statistics.worker_error is None


def test_live_direct_endpoint_commits_a_short_final_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine.from_voicebank(
        bank_path,
        backend,
        analysis_chunk_seconds=0.2,
        direct_update_seconds=0.4,
        direct_endpoint_seconds=0.2,
        buffer_duration_seconds=2.0,
    )
    assert engine.direct_recognizer is not None

    def fake_recognize(
        samples: np.ndarray, sample_rate: int
    ) -> tuple[DetectedPhone, ...]:
        del samples, sample_rate
        return (
            DetectedPhone("d", 0.05, 0.1, 0.9),
            DetectedPhone("i", 0.1, 0.2, 0.9),
        )

    monkeypatch.setattr(engine.direct_recognizer, "recognize", fake_recognize)
    engine.start(background=False)
    source = np.concatenate((_tone(220, 0.2), np.zeros(round(SAMPLE_RATE * 0.4))))
    backend.feed(source)
    engine.process_available()
    engine.stop()

    statistics = engine.statistics
    assert statistics.direct_ipa_updates >= 1
    assert statistics.committed_segments > 0
    assert statistics.rendered_output_samples > 0


def test_live_real_aiko_survives_long_silence_and_recognizes_followup_speech() -> None:
    """Exercise the production callback/worker path across long silent gaps."""
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    human, sample_rate = sf.read(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_sentence.wav",
        dtype="float32",
        always_2d=True,
    )
    assert sample_rate == SAMPLE_RATE
    speech = np.mean(human, axis=1)

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)

    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine.from_voicebank(
        bank_path,
        backend,
        analysis_chunk_seconds=0.2,
        direct_update_seconds=0.4,
        buffer_duration_seconds=5.0,
    )
    engine.prepare_direct_ipa()
    engine.start(background=True)

    def feed_realtime(samples: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(samples), 320):
            outputs.append(backend.feed(samples[start : start + 320]))
            time.sleep(0.02)
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    try:
        feed_realtime(silence(2.0))
        quiet_statistics = engine.statistics
        first_window = np.concatenate(
            (feed_realtime(speech), feed_realtime(silence(4.0)))
        )
        first_statistics = engine.statistics
        second_window = np.concatenate(
            (feed_realtime(speech), feed_realtime(silence(3.0)))
        )
    finally:
        engine.stop(flush=True)

    final_statistics = engine.statistics
    transport_statistics = engine.stream.statistics
    assert quiet_statistics.direct_ipa_updates == 0
    assert quiet_statistics.detected_phones == 0
    assert first_statistics.direct_ipa_updates > quiet_statistics.direct_ipa_updates
    assert first_statistics.detected_phones > 0
    assert first_statistics.committed_segments > 20
    assert first_statistics.rendered_output_samples > 0
    assert float(np.max(np.abs(first_window))) > 0.01
    assert final_statistics.direct_ipa_updates > first_statistics.direct_ipa_updates
    assert final_statistics.detected_phones > first_statistics.detected_phones
    assert final_statistics.committed_segments > first_statistics.committed_segments
    assert float(np.max(np.abs(second_window))) > 0.01
    assert transport_statistics.input_overflow_samples == 0
    assert final_statistics.worker_error is None
