from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.audio import FakeDuplexAudioBackend, RealtimeAudioStream
from synthstream.rendering import VoicebankRenderer
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]
STREAM_SAMPLE_RATE = 44_100


def test_recorded_human_input_and_real_voicebank_output_cross_production_buffers() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "TETO-English-150401"
    if not bank_path.is_dir():
        pytest.skip("local TETO English voicebank is not installed")
    human_path = PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_excerpt.wav"
    human, human_rate = sf.read(human_path, dtype="float32")
    human_at_stream_rate = np.asarray(
        resample_poly(human, STREAM_SAMPLE_RATE, human_rate), dtype=np.float32
    )
    bank = load_voicebank(bank_path)
    unit = next(item for item in bank.units if item.alias == "- a")
    rendered = VoicebankRenderer(output_gain=0.8).render_unit(
        unit,
        duration_seconds=len(human_at_stream_rate) / STREAM_SAMPLE_RATE,
        pitch_ratio=1.1,
    )
    backend = FakeDuplexAudioBackend()
    stream = RealtimeAudioStream(
        backend,
        sample_rate=STREAM_SAMPLE_RATE,
        block_size=441,
        buffer_duration_seconds=0.75,
    )
    stream.write_output(rendered.samples)

    stream.start()
    callback_output = backend.feed(human_at_stream_rate)
    stream.stop()

    captured_input = stream.read_input(len(human_at_stream_rate))
    captured_output = backend.take_captured_output()
    assert np.array_equal(captured_input, human_at_stream_rate)
    assert np.array_equal(callback_output, rendered.samples)
    assert np.array_equal(captured_output, rendered.samples)
    assert np.sqrt(np.mean(np.square(captured_output))) > 0.01
    assert stream.statistics.input_overflow_samples == 0
    assert stream.statistics.output_overflow_samples == 0
    assert stream.statistics.output_underflow_samples == 0
    assert stream.statistics.callback_statuses == ()
