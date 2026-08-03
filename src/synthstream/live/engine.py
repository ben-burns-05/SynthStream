"""Worker-driven live speech-to-voicebank audio conversion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.analysis import AnalysisConfig, FeatureExtractor
from synthstream.audio import DuplexAudioBackend, RealtimeAudioStream
from synthstream.decoding import DecodedSegment, DecoderConfig, SegmentalBeamDecoder
from synthstream.matching import MatchWeights, SectionFeatureIndex, SectionMatcher
from synthstream.rendering import VoicebankRenderer
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

        self.bank = bank
        self.extractor = FeatureExtractor(config)
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
        self._units = {unit.id: unit for unit in bank.units}
        self._pending_input = np.empty(0, dtype=np.float32)
        self._emitted_output_samples = 0
        self._input_blocks_processed = 0
        self._feature_chunks_processed = 0
        self._feature_frames_processed = 0
        self._committed_segments = 0
        self._rendered_output_samples = 0
        self._processing_seconds = 0.0
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
                self._worker_error,
            )

    def start(self, *, background: bool = True) -> None:
        """Start the duplex transport and optionally its processing worker."""
        if self.is_running:
            raise RuntimeError("live engine is already running")
        self.streaming_decoder.reset()
        self._pending_input = np.empty(0, dtype=np.float32)
        self._emitted_output_samples = 0
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
        if self.streaming_decoder.frames_processed:
            update = self.streaming_decoder.finish()
            self._consume_committed(update.committed_segments)

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
        features = self.extractor.analyze(samples)
        update = self.streaming_decoder.push(features)
        self._consume_committed(update.committed_segments)
        with self._statistics_lock:
            self._feature_chunks_processed += 1
            self._feature_frames_processed += features.frame_count
            self._processing_seconds += time.perf_counter() - started

    def _consume_committed(self, segments: tuple[DecodedSegment, ...]) -> None:
        for segment in segments:
            start_sample = round(
                segment.start_frame
                * self.extractor.config.hop_samples
                * self.stream.sample_rate
                / self.extractor.config.sample_rate
            )
            if start_sample > self._emitted_output_samples:
                self.stream.write_output(
                    np.zeros(start_sample - self._emitted_output_samples, dtype=np.float32)
                )
                self._emitted_output_samples = start_sample

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
            else:
                if segment.unit_id is None or segment.section_index is None:
                    raise ValueError("voiced segment is missing voicebank identity")
                unit = self._units.get(segment.unit_id)
                if unit is None or not 0 <= segment.section_index < len(unit.sections):
                    raise ValueError(
                        "committed segment references an unavailable voicebank section"
                    )
                result = self.renderer.render_section(
                    unit,
                    unit.sections[segment.section_index],
                    duration_seconds=(segment.end_frame - segment.start_frame)
                    * self.extractor.config.hop_samples
                    / self.extractor.config.sample_rate,
                    pitch_ratio=self.pitch_ratio,
                )
                rendered = _resample(result.samples, result.sample_rate, self.stream.sample_rate)
                rendered = _fit_length(rendered, target_samples)
            self.stream.write_output(rendered)
            self._emitted_output_samples += len(rendered)
            with self._statistics_lock:
                self._committed_segments += 1
                self._rendered_output_samples += len(rendered)


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
