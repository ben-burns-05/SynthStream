"""Worker-driven live speech-to-voicebank audio conversion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import numpy.typing as npt

from synthstream.analysis import (
    AnalysisConfig,
    FeatureExtractor,
)
from synthstream.audio import DuplexAudioBackend, RealtimeAudioStream
from synthstream.decoding import DecodedSegment, DecoderConfig, SegmentalBeamDecoder
from synthstream.matching import MatchWeights, SectionFeatureIndex, SectionMatcher
from synthstream.offline.direct_phonemes import (
    DIRECT_SAMPLE_RATE,
    DirectAliasPlanner,
    DirectIPARecognizer,
    DirectPhoneticRecognition,
    DirectPlannedAlias,
    stable_phone_indices,
)
from synthstream.offline.voicebank_phonemizer import detect_voicebank_profile
from synthstream.pitch import PitchTransfer
from synthstream.rendering import (
    AliasEvent,
    RenderSegment,
    VoicebankRenderer,
    VoicebankRenderScheduler,
    allocate_alias_section_durations,
)
from synthstream.voicebank import Voicebank, load_voicebank

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
    input_peak_max: float
    input_clipped_samples: int


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
        buffer_duration_seconds: float = 4.0,
        analysis_config: AnalysisConfig | None = None,
        match_weights: MatchWeights | None = None,
        decoder_config: DecoderConfig | None = None,
        lookahead_frames: int = 3,
        pitch_ratio: float = 1.0,
        track_pitch: bool = True,
        output_gain: float = 1.0,
        use_direct_ipa: bool = True,
        direct_window_seconds: float = 1.2,
        direct_update_seconds: float = 0.8,
        direct_commit_lag_seconds: float = 0.4,
        direct_endpoint_seconds: float = 0.24,
        direct_silence_rms_threshold: float = 0.00005,
        direct_utterance_seconds: float = 6.0,
        direct_context_padding_seconds: float = 0.2,
        diagnostic_input_seconds: float = 10.0,
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
            (direct_utterance_seconds, "direct_utterance_seconds"),
            (direct_context_padding_seconds, "direct_context_padding_seconds"),
            (diagnostic_input_seconds, "diagnostic_input_seconds"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.bank = bank
        self.extractor = FeatureExtractor(config)
        self.use_direct_ipa = use_direct_ipa
        if use_direct_ipa and sample_rate != DIRECT_SAMPLE_RATE:
            raise ValueError(
                f"live direct-IPA mode requires {DIRECT_SAMPLE_RATE} Hz audio"
            )
        self.track_pitch = track_pitch
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
        self.pitch_transfer = PitchTransfer()
        self.pitch_ratio = pitch_ratio
        self.analysis_chunk_samples = max(1, round(analysis_chunk_seconds * sample_rate))
        self.direct_window_samples = max(1, round(direct_window_seconds * sample_rate))
        self.direct_update_samples = max(1, round(direct_update_seconds * sample_rate))
        self.direct_commit_lag_seconds = direct_commit_lag_seconds
        self.direct_endpoint_samples = max(1, round(direct_endpoint_seconds * sample_rate))
        self.direct_silence_rms_threshold = direct_silence_rms_threshold
        self.direct_utterance_samples = max(1, round(direct_utterance_seconds * sample_rate))
        self.direct_context_padding_samples = max(
            1, round(direct_context_padding_seconds * sample_rate)
        )
        self.diagnostic_input_samples = max(1, round(diagnostic_input_seconds * sample_rate))
        self._render_scheduler = VoicebankRenderScheduler(
            bank,
            self.renderer,
            sample_rate,
            staging_seconds=0.12,
        )
        # Give the worker a short bounded window to enqueue the first voiced
        # event.  Subsequent timeline gaps are queued explicitly by the live
        # scheduler, so the callback is never asked to infer transport timing.
        self.stream = RealtimeAudioStream(
            backend,
            sample_rate=sample_rate,
            block_size=block_size,
            buffer_duration_seconds=buffer_duration_seconds,
            startup_buffer_seconds=0.12,
        )
        self._pending_input = np.empty(0, dtype=np.float32)
        self._direct_audio = np.empty(0, dtype=np.float32)
        self._direct_total_samples = 0
        self._direct_samples_since_update = 0
        self._direct_emitted_seconds = 0.0
        self._direct_utterance_audio = np.empty(0, dtype=np.float32)
        self._direct_utterance_start_samples = 0
        self._direct_previous_recognition: DirectPhoneticRecognition | None = None
        self._direct_previous_cutoff_seconds = 0.0
        self._diagnostic_input_audio = np.empty(0, dtype=np.float32)
        self._direct_quiet_samples = 0
        self._direct_speech_seen = False
        self._live_output_started = False
        self._live_timeline_origin_samples: int | None = None
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
        self._input_peak_max = 0.0
        self._input_clipped_samples = 0
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
                self._input_peak_max,
                self._input_clipped_samples,
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
        self._direct_utterance_audio = np.empty(0, dtype=np.float32)
        self._direct_utterance_start_samples = 0
        self._direct_previous_recognition = None
        self._direct_previous_cutoff_seconds = 0.0
        self._diagnostic_input_audio = np.empty(0, dtype=np.float32)
        self._direct_quiet_samples = 0
        self._direct_speech_seen = False
        self._live_output_started = False
        self._live_timeline_origin_samples = None
        self._emitted_output_samples = 0
        self._render_scheduler.reset()
        self.stream.input_buffer.clear()
        self.stream.output_buffer.clear()
        self._worker_stop.clear()
        self._worker_error = None
        self._input_peak = 0.0
        self._input_rms = 0.0
        self._input_peak_max = 0.0
        self._input_clipped_samples = 0
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
            block_peak = float(np.max(np.abs(block))) if len(block) else 0.0
            clipped_samples = int(np.count_nonzero(np.abs(block) >= 0.98))
            with self._statistics_lock:
                self._input_peak_max = max(self._input_peak_max, block_peak)
                self._input_clipped_samples += clipped_samples
            self._append_diagnostic_input(block)
            self._pending_input = np.concatenate((self._pending_input, block))
            self._process_complete_chunks()
            processed += 1
        return processed

    def recent_input_audio(self) -> AudioSamples:
        """Return the most recent captured microphone samples for diagnostics."""
        with self._statistics_lock:
            return self._diagnostic_input_audio.copy()

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
        self._write_released(self._render_scheduler.flush())

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
            endpoint_reached = False
            if self._input_rms >= self.direct_silence_rms_threshold:
                if not self._direct_speech_seen:
                    self._direct_utterance_audio = np.empty(0, dtype=np.float32)
                    self._direct_utterance_start_samples = (
                        self._direct_total_samples - len(samples)
                    )
                self._append_direct_utterance(samples)
                self._direct_speech_seen = True
                self._direct_quiet_samples = 0
            elif self._direct_speech_seen:
                self._append_direct_utterance(samples)
                self._direct_quiet_samples += len(samples)
                if self._direct_quiet_samples >= self.direct_endpoint_samples:
                    self._process_direct_update(final=True)
                    self._direct_utterance_audio = np.empty(0, dtype=np.float32)
                    self._direct_quiet_samples = 0
                    self._direct_speech_seen = False
                    endpoint_reached = True
            if (
                self._live_output_started
                and self._live_timeline_origin_samples is not None
                and self._input_rms < self.direct_silence_rms_threshold
            ):
                self.stream.allow_silence_until(
                    max(
                        0,
                        self._direct_total_samples
                        - self._live_timeline_origin_samples,
                    )
                )
            if not endpoint_reached:
                self._process_direct_update()
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

    def _append_direct_utterance(self, samples: AudioSamples) -> None:
        """Retain a bounded speech utterance for endpoint re-decoding."""
        self._direct_utterance_audio = np.concatenate(
            (self._direct_utterance_audio, samples)
        )
        excess = len(self._direct_utterance_audio) - self.direct_utterance_samples
        if excess > 0:
            self._direct_utterance_audio = self._direct_utterance_audio[excess:]
            self._direct_utterance_start_samples += excess

    def _append_diagnostic_input(self, samples: AudioSamples) -> None:
        with self._statistics_lock:
            self._diagnostic_input_audio = np.concatenate(
                (self._diagnostic_input_audio, samples)
            )[-self.diagnostic_input_samples :]

    def _process_direct_update(self, *, final: bool = False) -> None:
        if self.direct_recognizer is None or self.direct_planner is None:
            raise RuntimeError("direct IPA frontend is unavailable")
        if not final and self._direct_samples_since_update < self.direct_update_samples:
            return
        if not final and not self._direct_speech_seen:
            return
        if len(self._direct_audio) < 1:
            return
        total_samples = self._direct_total_samples
        window_start_samples = total_samples - len(self._direct_audio)
        window = self._direct_audio
        context_left_samples = 0
        context_right_samples = 0
        if final and len(self._direct_utterance_audio):
            window = self._direct_utterance_audio
            window_start_samples = self._direct_utterance_start_samples
            context_right_samples = self.direct_context_padding_samples
            window = np.pad(
                window,
                (0, context_right_samples),
            ).astype(np.float32, copy=False)
        actual_window_seconds = len(window) / self.extractor.config.sample_rate
        valid_start_seconds = context_left_samples / self.extractor.config.sample_rate
        valid_end_seconds = actual_window_seconds - (
            context_right_samples / self.extractor.config.sample_rate
        )
        phones = self.direct_recognizer.recognize(
            window,
            self.extractor.config.sample_rate,
        )
        phones = tuple(
            replace(
                phone,
                start_seconds=max(phone.start_seconds, valid_start_seconds),
                end_seconds=min(phone.end_seconds, valid_end_seconds),
            )
            for phone in phones
            if phone.start_seconds < valid_end_seconds + 1e-6
            and phone.end_seconds > valid_start_seconds
        )
        if context_left_samples:
            phones = tuple(
                replace(
                    phone,
                    start_seconds=phone.start_seconds - valid_start_seconds,
                    end_seconds=phone.end_seconds - valid_start_seconds,
                )
                for phone in phones
            )
        self._detected_phones += len(phones)
        offset_seconds = (
            self._direct_utterance_start_samples
            if final and len(self._direct_utterance_audio)
            else window_start_samples
        ) / self.extractor.config.sample_rate
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
        window_start_seconds = window_start_samples / self.extractor.config.sample_rate
        aliases_to_commit = recognition.aliases
        if not final:
            # Keep the newest hypothesis mutable.  The current prefix must agree
            # with the preceding window before its aliases can be emitted; an
            # older prefix is retired only after it has crossed both windows'
            # safe frontiers.
            previous = self._direct_previous_recognition
            stable_indices = (
                stable_phone_indices(previous.phones, recognition.phones)
                if previous is not None
                else ()
            )
            stable_index_set = set(stable_indices)
            stable_aliases = tuple(
                alias
                for alias in recognition.aliases
                if alias.phone_indices
                and all(index in stable_index_set for index in alias.phone_indices)
            )
            retired_aliases: tuple[DirectPlannedAlias, ...] = ()
            if previous is not None:
                retired_aliases = tuple(
                    alias
                    for alias in previous.aliases
                    if alias.end_seconds <= self._direct_previous_cutoff_seconds + 1e-6
                    and alias.end_seconds <= window_start_seconds + 1e-6
                )
            aliases_to_commit = tuple(
                sorted(
                    (*retired_aliases, *stable_aliases),
                    key=lambda alias: (alias.start_seconds, alias.end_seconds),
                )
            )
            self._direct_previous_recognition = recognition
            self._direct_previous_cutoff_seconds = cutoff
        segments = self._direct_segments(
            aliases_to_commit,
            cutoff,
            pitch_audio=window,
            pitch_audio_start_samples=window_start_samples,
        )
        self._consume_committed(segments)
        self._direct_samples_since_update = 0
        if final:
            self._direct_previous_recognition = None
            self._direct_previous_cutoff_seconds = 0.0
        with self._statistics_lock:
            self._direct_ipa_updates += 1

    def _direct_segments(
        self,
        aliases: tuple[DirectPlannedAlias, ...],
        cutoff_seconds: float,
        *,
        pitch_audio: AudioSamples | None = None,
        pitch_audio_start_samples: int = 0,
    ) -> tuple[DecodedSegment, ...]:
        segments: list[DecodedSegment] = []
        hop_seconds = self.extractor.config.hop_samples / self.extractor.config.sample_rate
        for alias in aliases:
            if alias.end_seconds > cutoff_seconds + 1e-6:
                continue
            effective_start_seconds = max(alias.start_seconds, self._direct_emitted_seconds)
            if alias.end_seconds <= effective_start_seconds + 1e-6:
                continue
            if self.track_pitch and pitch_audio is not None:
                alias_start_sample = round(
                    effective_start_seconds * self.extractor.config.sample_rate
                ) - pitch_audio_start_samples
                alias_end_sample = round(
                    alias.end_seconds * self.extractor.config.sample_rate
                ) - pitch_audio_start_samples
                alias_start_sample = max(0, min(len(pitch_audio), alias_start_sample))
                alias_end_sample = max(
                    alias_start_sample, min(len(pitch_audio), alias_end_sample)
                )
            start_frame = max(0, round(effective_start_seconds / hop_seconds))
            end_frame = max(start_frame + 1, round(alias.end_seconds / hop_seconds))
            pitch_ratio = 1.0
            if self.track_pitch and pitch_audio is not None:
                pitch_ratio = self.pitch_transfer.ratio_for_alias(
                    pitch_audio[alias_start_sample:alias_end_sample],
                    self.extractor.config.sample_rate,
                    alias.unit,
                )
            event = AliasEvent(
                unit_id=alias.unit.id,
                alias=alias.alias,
                start_seconds=start_frame * hop_seconds,
                end_seconds=end_frame * hop_seconds,
                confidence=max(alias.confidence, 1e-6),
                pitch_ratio=pitch_ratio,
            )
            section_frames = allocate_alias_section_durations(
                event,
                alias.unit,
                timebase_hz=1.0 / hop_seconds,
            )
            boundaries = [start_frame]
            for frame_count in section_frames:
                boundaries.append(boundaries[-1] + frame_count)
            confidence = max(alias.confidence, 1e-6)
            section_cost = -math.log(confidence) / len(alias.unit.sections)
            for section_index, (section_start, section_end) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True)
            ):
                if section_end <= section_start:
                    continue
                nominal_frames = max(
                    1,
                    round(alias.unit.section_at(section_index).duration_seconds / hop_seconds),
                )
                segments.append(
                    DecodedSegment(
                        alias.unit.id,
                        alias.alias,
                        section_index,
                        alias.unit.section_at(section_index).kind,
                        section_start,
                        section_end,
                        (section_end - section_start) / nominal_frames,
                        section_cost,
                        0.0,
                        0.0,
                        section_cost,
                        event.pitch_ratio,
                    )
                )
            self._direct_emitted_seconds = max(self._direct_emitted_seconds, alias.end_seconds)
        return tuple(segments)

    def _consume_committed(self, segments: tuple[DecodedSegment, ...]) -> None:
        """Render complete aliases as single events when available."""
        index = 0
        while index < len(segments):
            group_end = self._complete_alias_group_end(segments, index)
            if group_end > index + 1:
                self._consume_alias_group(segments[index:group_end])
                index = group_end
            else:
                self._consume_section_segments((segments[index],))
                index += 1

    def _complete_alias_group_end(
        self,
        segments: tuple[DecodedSegment, ...],
        start: int,
    ) -> int:
        first = segments[start]
        if first.unit_id is None or first.section_index != 0:
            return start + 1
        unit = self._render_scheduler.units.get(first.unit_id)
        if unit is None:
            return start + 1
        end = start + 1
        while end < len(segments):
            candidate = segments[end]
            previous = segments[end - 1]
            if (
                candidate.unit_id != first.unit_id
                or candidate.alias != first.alias
                or candidate.section_index != end - start
                or previous.section_index is None
                or candidate.section_index != previous.section_index + 1
            ):
                break
            end += 1
        return end if end - start == len(unit.sections) else start + 1

    def _consume_alias_group(self, segments: tuple[DecodedSegment, ...]) -> None:
        first = segments[0]
        last = segments[-1]
        hop_seconds = self.extractor.config.hop_samples / self.extractor.config.sample_rate
        start_seconds = first.start_frame * hop_seconds
        end_seconds = last.end_frame * hop_seconds
        start_sample = round(start_seconds * self.stream.sample_rate)
        end_sample = round(end_seconds * self.stream.sample_rate)
        if self._live_timeline_origin_samples is None:
            self._live_timeline_origin_samples = start_sample
        include_leading_gap = self._live_output_started
        if self._live_output_started:
            playback_frontier = (
                self._live_timeline_origin_samples + self.stream.output_clock_samples
            )
            if end_sample <= playback_frontier:
                with self._statistics_lock:
                    self._committed_segments += len(segments)
                return
            if start_sample < playback_frontier:
                start_seconds = playback_frontier / self.stream.sample_rate
                include_leading_gap = False
        event = AliasEvent(
            unit_id=first.unit_id,
            alias=first.alias or first.unit_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            confidence=1.0,
            pitch_ratio=self.pitch_ratio * first.pitch_ratio,
        )
        leading_gap_limit: int | None = None
        if include_leading_gap:
            target_samples = max(
                0,
                end_sample - round(start_seconds * self.stream.sample_rate),
            )
            reserve_samples = max(
                self.stream.block_size,
                round(0.12 * self.stream.sample_rate),
                target_samples,
            )
            free_samples = (
                self.stream.output_buffer.capacity_samples
                - self.stream.output_buffer.available_samples
            )
            leading_gap_limit = min(
                max(0, free_samples - reserve_samples),
                round(0.25 * self.stream.sample_rate),
            )
        result = self._render_scheduler.append_alias(
            event,
            include_leading_gap=include_leading_gap,
            leading_gap_limit_samples=leading_gap_limit,
        )
        self._write_released(result.released)
        if result.rendered:
            self._live_output_started = True
        self._emitted_output_samples = self._render_scheduler.scheduled_samples
        with self._statistics_lock:
            self._committed_segments += len(segments)

    def _consume_section_segments(self, segments: tuple[DecodedSegment, ...]) -> None:
        for segment in segments:
            hop_seconds = self.extractor.config.hop_samples / self.extractor.config.sample_rate
            start_seconds = segment.start_frame * hop_seconds
            end_seconds = segment.end_frame * hop_seconds
            start_sample = round(start_seconds * self.stream.sample_rate)
            end_sample = round(end_seconds * self.stream.sample_rate)
            if self._live_timeline_origin_samples is None:
                self._live_timeline_origin_samples = start_sample
            include_leading_gap = self._live_output_started
            if self._live_output_started:
                # The callback is the authoritative output clock.  A worker
                # update can arrive after the device has already played past
                # its source interval (most visibly after long silence).  Do
                # not enqueue that stale interval or replay its old gap.
                playback_frontier = (
                    self._live_timeline_origin_samples
                    + self.stream.output_clock_samples
                )
                if end_sample <= playback_frontier:
                    include_leading_gap = False
                    start_seconds = end_seconds
                elif start_sample < playback_frontier:
                    start_seconds = playback_frontier / self.stream.sample_rate
                    include_leading_gap = False
            render_segment = RenderSegment(
                unit_id=segment.unit_id,
                alias=segment.alias,
                section_index=segment.section_index,
                section_kind=segment.section_kind,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                pitch_ratio=self.pitch_ratio * segment.pitch_ratio,
            )
            leading_gap_limit: int | None = None
            if include_leading_gap:
                target_samples = max(0, end_sample - round(start_seconds * self.stream.sample_rate))
                reserve_samples = max(
                    self.stream.block_size,
                    round(0.12 * self.stream.sample_rate),
                    target_samples,
                )
                free_samples = (
                    self.stream.output_buffer.capacity_samples
                    - self.stream.output_buffer.available_samples
                )
                # Long gaps are normally already being emitted by the device
                # while the recognizer catches up.  Queue only a short future
                # gap; retaining a multi-second gap would fill the bounded
                # ring and then burst the following voiced sections.
                leading_gap_limit = min(
                    max(0, free_samples - reserve_samples),
                    round(0.25 * self.stream.sample_rate),
                )
            result = self._render_scheduler.append(
                render_segment,
                include_leading_gap=include_leading_gap,
                leading_gap_limit_samples=leading_gap_limit,
            )
            self._write_released(result.released)
            if result.rendered:
                self._live_output_started = True
            self._emitted_output_samples = self._render_scheduler.scheduled_samples
            with self._statistics_lock:
                self._committed_segments += 1

    def _write_released(self, samples: AudioSamples) -> None:
        if not len(samples):
            return
        self.stream.write_output(samples)
        with self._statistics_lock:
            self._rendered_output_samples += len(samples)
