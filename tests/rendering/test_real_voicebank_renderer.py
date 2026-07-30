from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import AnalysisConfig, FeatureExtractor
from synthstream.audio import WavFileSink
from synthstream.rendering import VoicebankRenderer
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("directory_name", "alias"),
    [
        ("CZloid VCCV 2015", "a"),
        ("Kikyuune Aiko RockLoud CVVC EN", "a"),
        ("TETO-English-150401", "- a"),
        ("TETO-namerakaongen211125", None),
    ],
)
def test_real_voicebank_unit_renders_at_independent_duration_and_pitch(
    tmp_path: Path, directory_name: str, alias: str | None
) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / directory_name
    if not bank_path.is_dir():
        pytest.skip(f"local development voicebank is not installed: {directory_name}")
    bank = load_voicebank(bank_path)
    unit = (
        bank.units[0]
        if alias is None
        else next(item for item in bank.units if item.alias == alias)
    )
    target_duration = max(0.15, min(0.8, unit.duration_seconds * 1.25))
    renderer = VoicebankRenderer(output_gain=0.8)

    original_pitch = renderer.render_unit(
        unit, duration_seconds=target_duration, pitch_ratio=1.0
    )
    shifted_pitch = renderer.render_unit(
        unit, duration_seconds=target_duration, pitch_ratio=1.25
    )
    output_path = tmp_path / f"{directory_name}.wav"
    shifted_pitch.send_to(WavFileSink(output_path))
    written, written_rate = sf.read(output_path, dtype="float32")
    extractor = FeatureExtractor(AnalysisConfig(sample_rate=unit.sample_rate))
    original_f0 = float(np.nanmedian(extractor.analyze(original_pitch.samples).f0_hz))
    shifted_f0 = float(np.nanmedian(extractor.analyze(shifted_pitch.samples).f0_hz))

    assert len(shifted_pitch.samples) == round(target_duration * unit.sample_rate)
    assert shifted_pitch.target_duration_seconds == pytest.approx(target_duration, abs=1e-5)
    assert shifted_f0 / original_f0 == pytest.approx(1.25, abs=0.03)
    assert written_rate == unit.sample_rate
    assert len(written) == len(shifted_pitch.samples)
    assert np.all(np.isfinite(written))
    assert np.sqrt(np.mean(np.square(written))) > 0.01
    assert np.max(np.abs(written)) <= 1.0


def test_real_stereo_voicebank_entry_downmixes_and_renders(tmp_path: Path) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko voicebank is not installed")
    bank = load_voicebank(bank_path)
    stereo_unit = next(unit for unit in bank.units if unit.channel_count == 2)

    result = VoicebankRenderer().render_unit(
        stereo_unit, duration_seconds=0.25, pitch_ratio=1.0
    )

    assert result.samples.ndim == 1
    assert len(result.samples) == round(0.25 * stereo_unit.sample_rate)
    assert np.all(np.isfinite(result.samples))
