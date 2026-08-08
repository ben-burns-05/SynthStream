"""Worker-driven live speech-to-voicebank audio conversion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.analysis import AnalysisConfig, FeatureExtractor
from synthstream.audio import DuplexAudioBackend, RealtimeAudioStream
from synthstream.decoding import DecodedSegment, DecoderConfig, SegmentalBeamDecoder
from synthstream.matching import MatchWeights, SectionFeatureIndex, SectionMatcher
from synthstream.offline.direct_phonemes import (
    DetectedPhone,
    DirectAliasPlanner,
    DirectIPARecognizer,
    DirectPlannedAlias,
)
from synthstream.offline.recognizer import _phone_aware_section_boundaries
from synthstream.offline.voicebank_phonemizer import detect_voicebank_profile
from synthstream.rendering import BufferedOverlapComposer, VoicebankRenderer
from synthstream.voicebank import Voicebank, VoicebankUnit, load_voicebank

AudioSamples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LiveEngineStatistics:
    """Observable work and failure counters for one live engine."""

    input_blocks_processed: int
    feature_chunks_processed: int
    feature_frames_processed: int
    committed_segments: int
    rendered_output_samples: int
    processing_seconds: float
    direct_ipa_updates: int
    detected_phones: int
    planned_aliases: int
    input_peak: float
    input_rms: float
    worker_error: str | None


class LiveVoicebankEngine:
    """Connect microphone transport, analysis, decoding, and rendering.

    The PortAudio callback only moves bounded sample blocks through
    ``RealtimeAudioStream``.  A worker (or the caller of ``process_available``)
    performs feature extraction, streaming beam decoding, and voicebank section
    rendering outside the callback thread.
    """

    def __init__(
        self,
        bank: Voicebank,
        backend: DuplexAudioBackend,
        *,
        sample_rate: int = 16_000,
        block_size: int = 320,
        analysis_chunk_seconds: float = 0.1,
        buffer_duration_seconds: float = 1.0,
        analysis_config: AnalysisConfig | None = None,
        match_weights: MatchWeights | None = None,
        decoder_config: DecoderConfig | None = None,
        lookahead_frames: int = 3,
        pitch_ratio: float = 1.0,
        output_gain: float = 1.0,
        use_direct_ipa: bool = True,
        direct_window_seconds: float = 1.2,
        direct_update_seconds: float = 0.4,
        direct_commit_lag_seconds: float = 0.4,
        direct_endpoint_seconds: float = 0.24,
        direct_silence_rms_threshold: float = 0.002,
    ) -> None:
        config = analysis_config or AnalysisConfig(sample_rate=sample_rate)
        if config.sample_rate != sample_rate:
            raise ValueError("live stream and analysis sample rates must match")
        if not math.isfinite(analysis_chunk_seconds) or analysis_chunk_seconds <= 0:
            raise ValueError("analysis_chunk_seconds must be finite and positive")
        if not math.isfinite(pitch_ratio) or pitch_ratio <= 0:
            raise ValueError("pitch_ratio must be finite and positive")
        if not math.isfinite(output_gain) or output_gain < 0:
            raise ValueError("output_gain must be finite and non-negative")
        for value, name in (
            (direct_window_seconds, "direct_window_seconds"),
            (direct_update_seconds, "direct_update_seconds"),
            (direct_commit_lag_seconds, "direct_commit_lag_seconds"),
            (direct_endpoint_seconds, "direct_endpoint_seconds"),
            (direct_silence_rms_threshold, "direct_silence_rms_threshold"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.bank = bank
        self.extractor = FeatureExtractor(config)
        self.use_direct_ipa = use_direct_ipa
        self.feature_index: SectionFeatureIndex | None
        self.decoder: SegmentalBeamDecoder | None
        self.streaming_decoder = None
        self.direct_recognizer: DirectIPARecognizer | None = None
        self.direct_planner: DirectAliasPlanner | None = None
        if use_direct_ipa:
            capability = detect_voicebank_profile(bank)
            if not capability.supported or capability.profile is None:
                raise ValueError(
                    "live direct-IPA mode requires a supported English voicebank profile"
                )
            self.direct_recognizer = DirectIPARecognizer()
            self.direct_planner = DirectAliasPlanner(bank, capability.profile)
            self.feature_index = None
            self.decoder = None
        else:
            self.feature_index = SectionFeatureIndex.build(bank, self.extractor)
            self.decoder = SegmentalBeamDecoder(
                SectionMatcher(self.feature_index, match_weights), decoder_config
            )
            self.streaming_decoder = self.decoder.stream(lookahead_frames=lookahead_frames)
        self.renderer = VoicebankRenderer(output_gain=output_gain)
        self.stream = RealtimeAudioStream(
            backend,
            sample_rate=sample_rate,
            block_size=block_size,
            buffer_duration_seconds=buffer_duration_seconds,
        )
        self.pitch_ratio = pitch_ratio
        self.analysis_chunk_samples = max(1, round(analysis_chunk_seconds * sample_rate))
        self.direct_window_samples = max(1, round(direct_window_seconds * sample_rate))
        self.direct_update_samples = max(1, round(direct_update_seconds * sample_rate))
        self.direct_commit_lag_seconds = direct_commit_lag_seconds
        self.direct_endpoint_samples = max(1, round(direct_endpoint_seconds * sample_rate))
        self.direct_silence_rms_threshold = direct_silence_rms_threshold
        self._units = {unit.id: unit for unit in bank.units}
        self._overlap_composer = BufferedOverlapComposer(sample_rate, staging_seconds=0.12)
        self._scheduled_output_samples = 0
        self._previous_rendered_unit_id: str | None = None
        self._onset_stretch_by_unit: dict[str, float] = {}
        self._pending_input = np.empty(0, dtype=np.float32)
        self._direct_audio = np.empty(0, dtype=np.float32)
        self._direct_total_samples = 0
        self._direct_samples_since_update = 0
        self._direct_emitted_seconds = 0.0
        self._direct_quiet_samples = 0
        self._direct_speech_seen = False
        self._emitted_output_samples = 0
        self._input_blocks_processed = 0
        self._feature_chunks_processed = 0
        self._feature_frames_processed = 0
        self._committed_segments = 0
        self._rendered_output_samples = 0
        self._processing_seconds = 0.0
        self._direct_ipa_updates = 0
        self._detected_phones = 0
        self._planned_aliases = 0
        self._input_peak = 0.0
        self._input_rms = 0.0
        self._worker_error: str | None = None
        self._statistics_lock = Lock()
        self._worker_stop = Event()
        self._worker: Thread | None = None

    @classmethod
    def from_voicebank(
        cls,
        voicebank_root: str | Path,
        backend: DuplexAudioBackend,
        **kwargs: object,
    ) -> LiveVoicebankEngine:
        """Load a voicebank and construct its live conversion engine."""
        return cls(load_voicebank(voicebank_root), backend, **kwargs)  # type: ignore[arg-type]

    @property
    def is_running(self) -> bool:
        return self.stream.is_running

    def prepare_direct_ipa(self) -> None:
        """Load direct-IPA assets before starting the audio transport."""
        if not self.use_direct_ipa or self.direct_recognizer is None:
            return
        self.direct_recognizer.warmup()

    @property
    def statistics(self) -> LiveEngineStatistics:
        with self._statistics_lock:
            return LiveEngineStatistics(
                self._input_blocks_processed,
                self._feature_chunks_processed,
                self._feature_frames_processed,
                self._committed_segments,
                self._rendered_output_samples,
                self._processing_seconds,
                self._direct_ipa_updates,
                self._detected_phones,
                self._planned_aliases,
                self._input_peak,
                self._input_rms,
                self._worker_error,
            )

    def start(self, *, background: bool = True) -> None:
        """Start the duplex transport and optionally its processing worker."""
        if self.is_running:
            raise RuntimeError("live engine is already running")
        if self.streaming_decoder is not None:
            self.streaming_decoder.reset()
        self._pending_input = np.empty(0, dtype=np.float32)
        self._direct_audio = np.empty(0, dtype=np.float32)
        self._direct_total_samples = 0
        self._direct_samples_since_update = 0
        self._direct_emitted_seconds = 0.0
        self._direct_quiet_samples = 0
        self._direct_speech_seen = False
        self._emitted_output_samples = 0
        self._scheduled_output_samples = 0
        self._previous_rendered_unit_id = None
        self._onset_stretch_by_unit.clear()
        self._overlap_composer.reset()
        self.stream.input_buffer.clear()
        self.stream.output_buffer.clear()
        self._worker_stop.clear()
        self._worker_error = None
        self.stream.start()
        if background:
            self._worker = Thread(target=self._worker_loop, name="synthstream-live", daemon=True)
            self._worker.start()

    def stop(self, *, flush: bool = False) -> None:
        """Stop processing and release the duplex audio transport."""
        self._worker_stop.set()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.join(timeout=2.0)
        if flush:
            self.process_available()
            self.flush()
        self.stream.stop()

    def process_available(self, *, max_blocks: int | None = None) -> int:
        """Process queued microphone blocks outside the audio callback."""
        if max_blocks is not None and max_blocks < 1:
            raise ValueError("max_blocks must be positive or None")
        processed = 0
        while self.stream.input_buffer.available_samples >= self.stream.block_size:
            if max_blocks is not None and processed >= max_blocks:
                break
            block = self.stream.read_input(self.stream.block_size)
            self._input_blocks_processed += 1
            self._pending_input = np.concatenate((self._pending_input, block))
            self._process_complete_chunks()
            processed += 1
        return processed

    def flush(self) -> None:
        """Analyze pending samples and commit/render the complete current path."""
        if len(self._pending_input):
            pending = self._pending_input
            self._pending_input = np.empty(0, dtype=np.float32)
            self._process_feature_chunk(pending)
        if self.use_direct_ipa:
            self._process_direct_update(final=True)
        elif (
            self.streaming_decoder is not None
            and self.streaming_decoder.frames_processed
            and not self.streaming_decoder.finished
        ):
            update = self.streaming_decoder.finish()
            self._consume_committed(update.committed_segments)
        self._flush_overlap_output()

    def _worker_loop(self) -> None:
        wait_seconds = self.stream.block_size / self.stream.sample_rate / 2
        while not self._worker_stop.is_set():
            try:
                processed = self.process_available(max_blocks=8)
            except Exception as error:  # pragma: no cover - defensive worker boundary
                with self._statistics_lock:
                    self._worker_error = str(error)
                self._worker_stop.set()
                return
            if not processed:
                self._worker_stop.wait(wait_seconds)

    def _process_complete_chunks(self) -> None:
        while len(self._pending_input) >= self.analysis_chunk_samples:
            chunk = self._pending_input[: self.analysis_chunk_samples]
            self._pending_input = self._pending_input[self.analysis_chunk_samples :]
            self._process_feature_chunk(chunk)

    def _process_feature_chunk(self, samples: AudioSamples) -> None:
        started = time.perf_counter()
        self._input_peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        self._input_rms = (
            float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
            if len(samples)
            else 0.0
        )
        if self.use_direct_ipa:
            self._direct_audio = np.concatenate((self._direct_audio, samples))
            self._direct_total_samples += len(samples)
            if len(self._direct_audio) > self.direct_window_samples:
                self._direct_audio = self._direct_audio[-self.direct_window_samples :]
            self._direct_samples_since_update += len(samples)
            self._process_direct_update()
            if self._input_rms >= self.direct_silence_rms_threshold:
                self._direct_speech_seen = True
                self._direct_quiet_samples = 0
            elif self._direct_speech_seen:
                self._direct_quiet_samples += len(samples)
                if self._direct_quiet_samples >= self.direct_endpoint_samples:
                    self._process_direct_update(final=True)
                    self._direct_quiet_samples = 0
                    self._direct_speech_seen = False
            feature_count = 0
        else:
            if self.streaming_decoder is None:
                raise RuntimeError("streaming acoustic decoder is unavailable")
            features = self.extractor.analyze(samples)
            update = self.streaming_decoder.push(features)
            self._consume_committed(update.committed_segments)
            feature_count = features.frame_count
        with self._statistics_lock:
            self._feature_chunks_processed += 1
            self._feature_frames_processed += feature_count
            self._processing_seconds += time.perf_counter() - started

    def _process_direct_update(self, *, final: bool = False) -> None:
        if self.direct_recognizer is None or self.direct_planner is None:
            raise RuntimeError("direct IPA frontend is unavailable")
        if not final and self._direct_samples_since_update < self.direct_update_samples:
            return
        if len(self._direct_audio) < 1:
            return
        total_samples = self._direct_total_samples
        window_start_samples = total_samples - len(self._direct_audio)
        window = self._direct_audio
        actual_window_seconds = len(window) / self.extractor.config.sample_rate
        minimum_context_samples = round(0.8 * self.extractor.config.sample_rate)
        recognition_window = window
        if len(recognition_window) < minimum_context_samples:
            recognition_window = np.pad(
                recognition_window,
                (0, minimum_context_samples - len(recognition_window)),
            ).astype(np.float32, copy=False)
        phones = self.direct_recognizer.recognize(
            recognition_window,
            self.extractor.config.sample_rate,
        )
        phones = tuple(
            phone
            for phone in phones
            if phone.start_seconds < actual_window_seconds + 1e-6
        )
        self._detected_phones += len(phones)
        offset_seconds = window_start_samples / self.extractor.config.sample_rate
        shifted_phones = tuple(
            replace(
                phone,
                start_seconds=phone.start_seconds + offset_seconds,
                end_seconds=phone.end_seconds + offset_seconds,
            )
            for phone in phones
        )
        recognition = self.direct_planner.plan(shifted_phones)
        self._planned_aliases += len(recognition.aliases)
        window_end_seconds = total_samples / self.extractor.config.sample_rate
        cutoff = window_end_seconds if final else max(
            0.0, window_end_seconds - self.direct_commit_lag_seconds
        )
        segments = self._direct_segments(recognition.aliases, shifted_phones, cutoff)
        self._consume_committed(segments)
        self._direct_samples_since_update = 0
        with self._statistics_lock:
            self._direct_ipa_updates += 1

    def _direct_segments(
        self,
        aliases: tuple[DirectPlannedAlias, ...],
        phones: tuple[DetectedPhone, ...],
        cutoff_seconds: float,
    ) -> tuple[DecodedSegment, ...]:
        segments: list[DecodedSegment] = []
        hop_seconds = self.extractor.config.hop_samples / self.extractor.config.sample_rate
        for alias in aliases:
            if alias.end_seconds > cutoff_seconds + 1e-6:
                continue
            effective_start_seconds = max(alias.start_seconds, self._direct_emitted_seconds)
            if alias.end_seconds <= effective_start_seconds + 1e-6:
                continue
            start_frame = max(0, round(effective_start_seconds / hop_seconds))
            end_frame = max(start_frame + 1, round(alias.end_seconds / hop_seconds))
            boundaries = _phone_aware_section_boundaries(
                alias,
                phones,
                start_frame,
                end_frame,
                hop_seconds,
            )
            confidence = max(alias.confidence, 1e-6)
            section_cost = -math.log(confidence) / len(alias.unit.sections)
            for section_index, (section_start, section_end) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True)
            ):
                if section_end <= section_start:
                    continue
                nominal_frames = max(
                    1,
                    round(alias.unit.sections[section_index].duration_seconds / hop_seconds),
                )
                segments.append(
                    DecodedSegment(
                        alias.unit.id,
                        alias.alias,
                        section_index,
                        alias.unit.sections[section_index].kind,
                        section_start,
                        section_end,
                        (section_end - section_start) / nominal_frames,
                        section_cost,
                        0.0,
                        0.0,
                        section_cost,
                    )
                )
            self._direct_emitted_seconds = max(self._direct_emitted_seconds, alias.end_seconds)
        return tuple(segments)

    def _consume_committed(self, segments: tuple[DecodedSegment, ...]) -> None:
        for segment in segments:
            start_sample = round(
                segment.start_frame
                * self.extractor.config.hop_samples
                * self.stream.sample_rate
                / self.extractor.config.sample_rate
            )
            if start_sample > self._scheduled_output_samples:
                self._write_released(
                    self._overlap_composer.append(
                        np.zeros(start_sample - self._scheduled_output_samples, dtype=np.float32),
                        overlap_samples=0,
                    )
                )
                self._scheduled_output_samples = start_sample
                self._previous_rendered_unit_id = None

            target_samples = max(
                1,
                round(
                    (segment.end_frame - segment.start_frame)
                    * self.extractor.config.hop_samples
                    * self.stream.sample_rate
                    / self.extractor.config.sample_rate
                ),
            )
            if segment.is_silence:
                rendered = np.zeros(target_samples, dtype=np.float32)
                overlap_samples = 0
                self._previous_rendered_unit_id = None
            else:
                if segment.unit_id is None or segment.section_index is None:
                    raise ValueError("voiced segment is missing voicebank identity")
                unit = self._units.get(segment.unit_id)
                if unit is None or not 0 <= segment.section_index < len(unit.sections):
                    raise ValueError(
                        "committed segment references an unavailable voicebank section"
                    )
                overlap_samples = self._calculate_overlap_samples(unit, segment, target_samples)
                if segment.section_kind == "onset":
                    self._onset_stretch_by_unit[unit.id] = max(segment.stretch_ratio, 1e-6)
                result = self.renderer.render_section(
                    unit,
                    unit.sections[segment.section_index],
                    duration_seconds=(target_samples + overlap_samples) / self.stream.sample_rate,
                    pitch_ratio=self.pitch_ratio,
                )
                rendered = _resample(result.samples, result.sample_rate, self.stream.sample_rate)
                rendered = _fit_length(rendered, target_samples + overlap_samples)
                self._previous_rendered_unit_id = unit.id
            self._write_released(
                self._overlap_composer.append(rendered, overlap_samples=overlap_samples)
            )
            self._scheduled_output_samples = max(
                self._scheduled_output_samples,
                start_sample + target_samples,
            )
            self._emitted_output_samples = self._scheduled_output_samples
            with self._statistics_lock:
                self._committed_segments += 1

    def _calculate_overlap_samples(
        self,
        unit: VoicebankUnit,
        segment: DecodedSegment,
        target_samples: int,
    ) -> int:
        unit_id = unit.id
        if self._previous_rendered_unit_id in (None, unit_id):
            return 0
        overlap_ms = max(0.0, unit.overlap_ms)
        if overlap_ms <= 0:
            return 0
        stretch = self._onset_stretch_by_unit.get(unit_id, 1.0)
        if segment.section_kind == "onset":
            stretch = max(segment.stretch_ratio, 1e-6)
        return min(
            target_samples,
            max(0, round(overlap_ms * self.stream.sample_rate / 1000.0 * stretch)),
        )

    def _write_released(self, samples: AudioSamples) -> None:
        if not len(samples):
            return
        self.stream.write_output(samples)
        with self._statistics_lock:
            self._rendered_output_samples += len(samples)

    def _flush_overlap_output(self) -> None:
        self._write_released(self._overlap_composer.flush())


def _resample(samples: AudioSamples, source_rate: int, target_rate: int) -> AudioSamples:
    if source_rate == target_rate:
        return samples
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(samples, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _fit_length(samples: AudioSamples, target_samples: int) -> AudioSamples:
    if len(samples) == target_samples:
        return samples
    if len(samples) > target_samples:
        return np.asarray(samples[:target_samples], dtype=np.float32)
    result = np.zeros(target_samples, dtype=np.float32)
    result[: len(samples)] = samples
    return result
