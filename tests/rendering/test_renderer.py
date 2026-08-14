from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from synthstream.audio import WavFileSink
from synthstream.pitch import PitchTransfer
from synthstream.rendering import (
    AliasEvent,
    VoicebankRenderer,
    allocate_alias_section_durations,
    rebalance_section_durations,
)
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


def _make_sine_bank(root: Path, frequency: float = 220.0) -> None:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    waveform = (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(root / "a.wav", waveform, SAMPLE_RATE, subtype="PCM_16")
    (root / "oto.ini").write_text("a.wav=a,100,200,100,50,20\n", encoding="utf-8")


def _dominant_frequency(samples: npt.NDArray[np.float32], sample_rate: int) -> float:
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.argmax(spectrum)])


def _tone(frequency: float, duration: float = 0.5) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_renders_real_voicebank_region_at_requested_duration_and_pitch(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]

    result = VoicebankRenderer().render_unit(unit, duration_seconds=0.6, pitch_ratio=2.0)

    assert len(result.samples) == 9_600
    assert result.target_duration_seconds == pytest.approx(0.6)
    assert result.source_duration_seconds == pytest.approx(0.8)
    assert result.stretch_ratio == pytest.approx(0.75)
    assert result.pitch_ratio == 2.0
    assert _dominant_frequency(result.samples, result.sample_rate) == pytest.approx(440, abs=8)
    assert np.all(np.isfinite(result.samples))
    assert np.max(np.abs(result.samples)) <= 1.0
    assert np.sqrt(np.mean(np.square(result.samples))) > 0.1


def test_can_render_an_individual_metadata_section(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]
    section = unit.sections[0]

    result = VoicebankRenderer().render_section(unit, section, duration_seconds=0.1)

    assert len(result.samples) == 1_600
    assert result.source_duration_seconds == pytest.approx(0.05)
    assert result.unit_id == unit.id


def test_estimates_recorded_section_pitch_and_caches_it(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]
    renderer = VoicebankRenderer()

    assert unit.source_pitch_hz == pytest.approx(220, abs=0.1)
    first = renderer.estimate_section_pitch_hz(unit, unit.sections[0])
    second = renderer.estimate_section_pitch_hz(unit, unit.sections[0])

    assert first == pytest.approx(220, abs=0.1)
    assert second == first


def test_pitch_transfer_uses_one_alias_reference_pitch(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]

    ratio = PitchTransfer().ratio_for_alias(
        _tone(440.0, duration=0.4),
        SAMPLE_RATE,
        unit,
    )

    assert ratio == pytest.approx(2.0)
    assert unit.pitch_reference_section.kind == "sustain"
    assert unit.section_at(0) is unit.sections[0]


def test_source_pitch_is_persisted_in_voicebank_cache(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    first = load_voicebank(tmp_path)
    second = load_voicebank(tmp_path)

    assert second.cache_hit
    assert second.units[0].source_pitch_hz == first.units[0].source_pitch_hz
    assert second.units[0].source_pitch_hz == pytest.approx(220, abs=0.1)


def test_section_duration_rebalance_assigns_extra_time_to_sustain(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]

    durations = rebalance_section_durations(
        unit.sections,
        (4_000, 4_000, 4_000),
        sample_rate=SAMPLE_RATE,
    )

    assert sum(durations) == 12_000
    assert durations[0] == 800
    assert durations[1] == 2_400
    assert durations[2] == 8_800


def test_alias_event_allocates_only_sustain_time(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]
    event = AliasEvent(unit.id, unit.alias, 0.0, 0.75)

    durations = allocate_alias_section_durations(event, unit, timebase_hz=SAMPLE_RATE)

    assert durations == (800, 2_400, 8_800)


def test_result_writes_to_wav_output_sink(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]
    result = VoicebankRenderer().render_unit(unit, duration_seconds=0.25)
    output_path = tmp_path / "output" / "rendered.wav"

    result.send_to(WavFileSink(output_path))
    samples, sample_rate = sf.read(output_path, dtype="float32")

    assert sample_rate == SAMPLE_RATE
    assert len(samples) == 4_000
    assert np.max(np.abs(samples)) > 0.1


@dataclass
class RecordingSink:
    samples: npt.NDArray[np.float32] | None = None
    sample_rate: int | None = None

    def write(self, samples: npt.NDArray[np.float32], sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate


def test_result_uses_generic_output_abstraction(tmp_path: Path) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]
    result = VoicebankRenderer().render_unit(unit, duration_seconds=0.1)
    sink = RecordingSink()

    result.send_to(sink)

    assert sink.samples is result.samples
    assert sink.sample_rate == SAMPLE_RATE


@pytest.mark.parametrize(
    ("duration", "pitch"),
    [(0.0, 1.0), (float("nan"), 1.0), (0.1, 0.0), (0.1, float("inf"))],
)
def test_rejects_invalid_transform_parameters(
    tmp_path: Path, duration: float, pitch: float
) -> None:
    _make_sine_bank(tmp_path)
    unit = load_voicebank(tmp_path, use_cache=False).units[0]

    with pytest.raises(ValueError):
        VoicebankRenderer().render_unit(unit, duration_seconds=duration, pitch_ratio=pitch)
