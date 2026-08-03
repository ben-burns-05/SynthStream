from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import AnalysisConfig
from synthstream.audio import FakeDuplexAudioBackend
from synthstream.live import LiveVoicebankEngine
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


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


def test_live_engine_can_run_background_worker_with_fake_backend(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine(
        load_voicebank(tmp_path, use_cache=False),
        backend,
        analysis_chunk_seconds=0.1,
        buffer_duration_seconds=2.0,
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
        )
