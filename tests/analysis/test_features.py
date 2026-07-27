import numpy as np
import pytest

from synthstream.analysis import AnalysisConfig, FeatureExtractor

SAMPLE_RATE = 16_000


def _tone(frequency: float, duration: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_extracts_aligned_frame_features_and_matcher_matrix() -> None:
    features = FeatureExtractor().analyze(_tone(220))

    assert features.frame_count == 49
    assert features.frame_times_seconds[0] == pytest.approx(0.0125)
    assert features.log_mel.shape == (49, 40)
    assert features.delta_mel.shape == (49, 40)
    assert features.recognition_features.shape == (49, 85)
    assert features.normalized_energy.shape == (49,)
    assert np.all(np.isfinite(features.recognition_features))


def test_yin_pitch_tracks_tone_and_keeps_f0_out_of_recognition_features() -> None:
    extractor = FeatureExtractor()
    low = extractor.analyze(_tone(220))
    high = extractor.analyze(_tone(440))

    assert np.mean(low.voiced) > 0.9
    assert np.nanmedian(low.f0_hz) == pytest.approx(220, abs=2)
    assert np.nanmedian(high.f0_hz) == pytest.approx(440, abs=4)
    assert np.mean(low.periodicity) > 0.8
    assert low.recognition_features.shape == high.recognition_features.shape


def test_silence_is_unvoiced_and_has_no_f0() -> None:
    features = FeatureExtractor().analyze(np.zeros(SAMPLE_RATE // 4, dtype=np.float32))

    assert not np.any(features.voiced)
    assert np.all(np.isnan(features.f0_hz))
    assert np.all(features.periodicity == 0)
    assert np.all(features.rms_energy < 1e-5)


def test_noise_is_flatter_and_less_periodic_than_tone() -> None:
    generator = np.random.default_rng(42)
    noise = generator.normal(0, 0.2, SAMPLE_RATE // 2).astype(np.float32)
    extractor = FeatureExtractor()
    tone_features = extractor.analyze(_tone(220))
    noise_features = extractor.analyze(noise)

    assert np.median(noise_features.spectral_flatness) > np.median(
        tone_features.spectral_flatness
    )
    assert np.median(noise_features.periodicity) < np.median(tone_features.periodicity)


def test_energy_and_flux_detect_an_amplitude_transition() -> None:
    quiet = _tone(220, duration=0.25, amplitude=0.02)
    loud = _tone(220, duration=0.25, amplitude=0.8)
    features = FeatureExtractor().analyze(np.concatenate((quiet, loud)))
    transition = len(features.delta_energy) // 2

    assert np.mean(features.normalized_energy[: transition - 2]) < -0.8
    assert np.mean(features.normalized_energy[transition + 2 :]) > 0.8
    assert np.max(features.delta_energy[transition - 2 : transition + 3]) > 0.5
    assert np.max(features.spectral_flux[transition - 2 : transition + 3]) > 0.01


def test_spectral_normalization_reduces_absolute_level_difference() -> None:
    extractor = FeatureExtractor()
    quiet = extractor.analyze(_tone(220, amplitude=0.05))
    loud = extractor.analyze(_tone(220, amplitude=0.8))

    assert np.mean(np.abs(quiet.log_mel - loud.log_mel)) < 0.15


def test_accepts_stereo_and_short_audio() -> None:
    mono = _tone(220, duration=0.01)
    stereo = np.column_stack((mono, mono * 0.5))

    features = FeatureExtractor().analyze(stereo)

    assert features.frame_count == 1
    assert features.log_mel.shape == (1, 40)


def test_empty_audio_returns_well_shaped_empty_features() -> None:
    features = FeatureExtractor().analyze(np.empty(0, dtype=np.float32))

    assert features.frame_count == 0
    assert features.log_mel.shape == (0, 40)
    assert features.recognition_features.shape == (0, 85)


@pytest.mark.parametrize(
    "configuration",
    [
        {"sample_rate": 0},
        {"window_ms": 0},
        {"mel_bands": 2},
        {"maximum_frequency_hz": 9_000},
        {"minimum_f0_hz": 500, "maximum_f0_hz": 100},
        {"yin_threshold": 1.0},
        {"voicing_threshold": -0.1},
    ],
)
def test_rejects_invalid_analysis_configuration(configuration: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AnalysisConfig(**configuration)


def test_rejects_audio_with_invalid_shape() -> None:
    with pytest.raises(ValueError):
        FeatureExtractor().analyze(np.zeros((10, 0), dtype=np.float32))

