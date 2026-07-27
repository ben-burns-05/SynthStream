"""Speaker-normalized frame features and lightweight pitch analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configurable frame, spectrum, and pitch settings."""

    sample_rate: int = 16_000
    window_ms: float = 25.0
    hop_ms: float = 10.0
    mel_bands: int = 40
    minimum_frequency_hz: float = 50.0
    maximum_frequency_hz: float = 7_600.0
    minimum_f0_hz: float = 60.0
    maximum_f0_hz: float = 500.0
    yin_threshold: float = 0.15
    voicing_threshold: float = 0.55
    energy_floor_db: float = -60.0

    def __post_init__(self) -> None:
        nyquist = self.sample_rate / 2
        if self.sample_rate < 1 or self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("sample rate, window, and hop must be positive")
        if self.mel_bands < 3:
            raise ValueError("mel_bands must be at least 3")
        if not 0 <= self.minimum_frequency_hz < self.maximum_frequency_hz <= nyquist:
            raise ValueError("mel frequency range must lie between zero and Nyquist")
        if not 0 < self.minimum_f0_hz < self.maximum_f0_hz < nyquist:
            raise ValueError("F0 range must lie between zero and Nyquist")
        if not 0 < self.yin_threshold < 1 or not 0 <= self.voicing_threshold <= 1:
            raise ValueError("pitch thresholds must lie between zero and one")

    @property
    def window_samples(self) -> int:
        return max(1, round(self.window_ms * self.sample_rate / 1000))

    @property
    def hop_samples(self) -> int:
        return max(1, round(self.hop_ms * self.sample_rate / 1000))


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Aligned recognition and prosody features for human-audio frames."""

    frame_times_seconds: FloatArray
    log_mel: FloatArray
    delta_mel: FloatArray
    normalized_energy: FloatArray
    delta_energy: FloatArray
    spectral_flux: FloatArray
    spectral_flatness: FloatArray
    periodicity: FloatArray
    f0_hz: FloatArray
    voiced: BoolArray
    rms_energy: FloatArray

    @property
    def frame_count(self) -> int:
        return len(self.frame_times_seconds)

    @property
    def recognition_features(self) -> FloatArray:
        """Return the matcher-facing features, deliberately excluding absolute F0."""
        scalar_features = np.column_stack(
            (
                self.normalized_energy,
                self.delta_energy,
                self.spectral_flux,
                self.spectral_flatness,
                self.periodicity,
            )
        )
        return np.concatenate((self.log_mel, self.delta_mel, scalar_features), axis=1).astype(
            np.float32, copy=False
        )


class FeatureExtractor:
    """Extract matching features and separate F0/prosody from human audio."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()
        self._window = np.hanning(self.config.window_samples).astype(np.float32)
        self._fft_size = _next_power_of_two(self.config.window_samples)
        self._mel_filters = _mel_filterbank(self.config, self._fft_size)

    def analyze(self, samples: npt.ArrayLike) -> FeatureBatch:
        """Analyze mono or channel-last audio at the configured sample rate."""
        mono = _as_mono(samples)
        frames, starts = _frame_audio(
            mono, self.config.window_samples, self.config.hop_samples
        )
        if len(frames) == 0:
            return _empty_batch(self.config.mel_bands)

        windowed = frames * self._window
        spectrum = np.fft.rfft(windowed, n=self._fft_size, axis=1)
        power = np.square(np.abs(spectrum)).astype(np.float32)
        mel_power = np.maximum(power @ self._mel_filters.T, 1e-10)
        log_mel = np.log(mel_power).astype(np.float32)
        log_mel -= np.mean(log_mel, axis=1, keepdims=True)
        log_mel /= max(float(np.std(log_mel)), 1e-5)
        delta_mel = _delta(log_mel)

        rms_energy = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12).astype(np.float32)
        energy_db = (20 * np.log10(np.maximum(rms_energy, 1e-8))).astype(np.float32)
        normalized_energy = _standardize(energy_db)
        delta_energy = _delta(normalized_energy)
        spectral_flux = _spectral_flux(power)
        spectral_flatness = _spectral_flatness(power)
        periodicity, f0_hz, voiced = self._pitch_features(frames, energy_db)
        frame_times = (
            (starts + self.config.window_samples / 2) / self.config.sample_rate
        ).astype(np.float32)
        return FeatureBatch(
            frame_times,
            log_mel,
            delta_mel,
            normalized_energy,
            delta_energy,
            spectral_flux,
            spectral_flatness,
            periodicity,
            f0_hz,
            voiced,
            rms_energy,
        )

    def _pitch_features(
        self, frames: FloatArray, energy_db: FloatArray
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        periodicity = np.zeros(len(frames), dtype=np.float32)
        f0_hz = np.full(len(frames), np.nan, dtype=np.float32)
        voiced = np.zeros(len(frames), dtype=np.bool_)
        for index, frame in enumerate(frames):
            confidence, frequency = _yin_pitch(frame, self.config)
            periodicity[index] = confidence
            is_voiced = (
                confidence >= self.config.voicing_threshold
                and energy_db[index] >= self.config.energy_floor_db
            )
            voiced[index] = is_voiced
            if is_voiced:
                f0_hz[index] = frequency
        return periodicity, f0_hz, voiced


def _as_mono(samples: npt.ArrayLike) -> FloatArray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2 and audio.shape[1] > 0:
        return np.asarray(np.mean(audio, axis=1), dtype=np.float32)
    raise ValueError("audio must be mono or a non-empty channel-last matrix")


def _frame_audio(
    samples: FloatArray, window_samples: int, hop_samples: int
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    if len(samples) == 0:
        return np.empty((0, window_samples), dtype=np.float32), np.empty(0, dtype=np.int64)
    frame_count = max(1, 1 + int(np.ceil((len(samples) - window_samples) / hop_samples)))
    total_samples = (frame_count - 1) * hop_samples + window_samples
    padded = np.pad(samples, (0, max(0, total_samples - len(samples))))
    starts = np.arange(frame_count, dtype=np.int64) * hop_samples
    frames = np.lib.stride_tricks.sliding_window_view(padded, window_samples)[::hop_samples]
    return np.asarray(frames[:frame_count], dtype=np.float32), starts


def _mel_filterbank(config: AnalysisConfig, fft_size: int) -> FloatArray:
    frequencies = np.fft.rfftfreq(fft_size, 1 / config.sample_rate)
    low_mel = _hz_to_mel(config.minimum_frequency_hz)
    high_mel = _hz_to_mel(config.maximum_frequency_hz)
    points = _mel_to_hz(np.linspace(low_mel, high_mel, config.mel_bands + 2))
    filters = np.zeros((config.mel_bands, len(frequencies)), dtype=np.float32)
    point_triples = zip(points, points[1:], points[2:], strict=False)
    for index, (left, center, right) in enumerate(point_triples):
        filters[index] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-9),
                (right - frequencies) / max(right - center, 1e-9),
            ),
        )
    filters /= np.maximum(np.sum(filters, axis=1, keepdims=True), 1e-9)
    return filters


def _hz_to_mel(frequency: float) -> float:
    return float(2595 * np.log10(1 + frequency / 700))


def _mel_to_hz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 700 * (np.power(10, mel / 2595) - 1)


def _delta(features: FloatArray) -> FloatArray:
    if len(features) < 2:
        return np.zeros_like(features, dtype=np.float32)
    return np.asarray(np.gradient(features, axis=0), dtype=np.float32)


def _standardize(values: FloatArray) -> FloatArray:
    standard_deviation = float(np.std(values))
    if standard_deviation < 1e-5:
        return np.zeros_like(values)
    return ((values - np.mean(values)) / standard_deviation).astype(np.float32)


def _spectral_flux(power: FloatArray) -> FloatArray:
    magnitude = np.sqrt(power)
    magnitude /= np.maximum(np.linalg.norm(magnitude, axis=1, keepdims=True), 1e-8)
    difference = np.maximum(np.diff(magnitude, axis=0, prepend=magnitude[:1]), 0)
    return np.asarray(np.sqrt(np.sum(np.square(difference), axis=1)), dtype=np.float32)


def _spectral_flatness(power: FloatArray) -> FloatArray:
    usable = np.maximum(power[:, 1:], 1e-12)
    geometric_mean = np.exp(np.mean(np.log(usable), axis=1))
    arithmetic_mean = np.mean(usable, axis=1)
    return np.asarray(
        geometric_mean / np.maximum(arithmetic_mean, 1e-12), dtype=np.float32
    )


def _yin_pitch(frame: FloatArray, config: AnalysisConfig) -> tuple[float, float]:
    centered = frame - np.mean(frame)
    minimum_lag = max(2, int(config.sample_rate / config.maximum_f0_hz))
    maximum_lag = min(len(frame) - 2, int(config.sample_rate / config.minimum_f0_hz))
    if maximum_lag <= minimum_lag or float(np.max(np.abs(centered))) < 1e-7:
        return 0.0, float("nan")

    difference = np.zeros(maximum_lag + 1, dtype=np.float64)
    for lag in range(1, maximum_lag + 1):
        delta = centered[:-lag] - centered[lag:]
        difference[lag] = np.dot(delta, delta)
    cumulative = np.cumsum(difference[1:])
    cmnd = np.ones_like(difference)
    cmnd[1:] = difference[1:] * np.arange(1, maximum_lag + 1) / np.maximum(cumulative, 1e-12)

    lag = minimum_lag + int(np.argmin(cmnd[minimum_lag : maximum_lag + 1]))
    for candidate in range(minimum_lag, maximum_lag):
        if cmnd[candidate] < config.yin_threshold and cmnd[candidate] <= cmnd[candidate + 1]:
            lag = candidate
            break
    refined_lag = _parabolic_minimum(cmnd, lag)
    confidence = float(np.clip(1 - cmnd[lag], 0, 1))
    return confidence, config.sample_rate / refined_lag


def _parabolic_minimum(values: npt.NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _empty_batch(mel_bands: int) -> FeatureBatch:
    empty = np.empty(0, dtype=np.float32)
    matrix = np.empty((0, mel_bands), dtype=np.float32)
    return FeatureBatch(
        empty,
        matrix,
        matrix.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        np.empty(0, dtype=np.bool_),
        empty.copy(),
    )
