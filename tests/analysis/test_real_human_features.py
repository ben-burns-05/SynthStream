from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample  # type: ignore[import-untyped]

from synthstream.analysis import FeatureExtractor

PROJECT_ROOT = Path(__file__).parents[2]


def _recorded_human() -> np.ndarray:
    samples, sample_rate = sf.read(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_excerpt.wav",
        dtype="float32",
    )
    assert sample_rate == 16_000
    return np.asarray(samples, dtype=np.float32)


def test_recorded_human_speech_produces_plausible_complete_features() -> None:
    features = FeatureExtractor().analyze(_recorded_human())
    voiced_f0 = features.f0_hz[features.voiced]

    assert features.frame_count == 49
    assert features.recognition_features.shape == (49, 85)
    assert np.all(np.isfinite(features.recognition_features))
    assert 0.5 < np.mean(features.voiced) < 1.0
    assert len(voiced_f0) > 30
    assert 80 < np.median(voiced_f0) < 300
    assert np.all((voiced_f0 >= 60) & (voiced_f0 <= 500))
    assert np.max(features.spectral_flux) > 0.5
    assert np.ptp(features.rms_energy) > 0.2
    assert np.max(features.periodicity) > 0.9


def test_recorded_speech_features_resist_microphone_gain_change() -> None:
    human = _recorded_human()
    extractor = FeatureExtractor()
    normal = extractor.analyze(human)
    quiet = extractor.analyze(human * 0.1)

    assert np.mean(np.abs(normal.log_mel - quiet.log_mel)) < 1e-4
    assert np.mean(normal.voiced == quiet.voiced) > 0.95
    assert np.nanmedian(quiet.f0_hz) / np.nanmedian(normal.f0_hz) == pytest.approx(
        1.0, abs=0.03
    )


def test_recorded_speech_f0_tracks_known_pitch_resampling() -> None:
    human = _recorded_human()
    pitch_ratio = 1.25
    shifted = np.asarray(resample(human, round(len(human) / pitch_ratio)), dtype=np.float32)
    extractor = FeatureExtractor()
    original = extractor.analyze(human)
    transformed = extractor.analyze(shifted)
    measured_ratio = np.nanmedian(transformed.f0_hz) / np.nanmedian(original.f0_hz)

    assert measured_ratio == pytest.approx(pitch_ratio, rel=0.06)


def test_real_recording_level_silence_is_not_voiced() -> None:
    silence = np.zeros_like(_recorded_human())

    features = FeatureExtractor().analyze(silence)

    assert not np.any(features.voiced)
    assert np.all(np.isnan(features.f0_hz))
    assert np.all(features.periodicity == 0)
