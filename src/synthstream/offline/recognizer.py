"""Production human-WAV to voicebank-section timeline workflow."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.analysis import AnalysisConfig, FeatureExtractor
from synthstream.decoding import (
    DecodedPath,
    DecodedSegment,
    DecoderConfig,
    DecodeResult,
    SegmentalBeamDecoder,
)
from synthstream.matching import MatchWeights, SectionFeatureIndex, SectionMatcher
from synthstream.offline.direct_phonemes import (
    DIRECT_SAMPLE_RATE,
    DirectAliasPlanner,
    DirectIPARecognizer,
    canonicalize_direct_audio,
)
from synthstream.offline.voicebank_phonemizer import detect_voicebank_profile
from synthstream.rendering import AliasEvent, allocate_alias_section_durations
from synthstream.voicebank import Voicebank, load_voicebank


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    """Machine-readable timing and score details for one decoded segment."""

    unit_id: str | None
    alias: str | None
    section_index: int | None
    section_kind: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    stretch_ratio: float
    silence: bool
    acoustic_cost: float
    duration_cost: float
    transition_cost: float
    total_cost: float
    pitch_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class RecognitionTimeline:
    """Offline recognition result suitable for inspection or JSON export."""

    source_wav: Path
    voicebank_root: Path
    sample_rate: int
    hop_seconds: float
    input_duration_seconds: float
    path_cost: float
    segments: tuple[TimelineSegment, ...]
    decode_result: DecodeResult
    recognition_mode: str = "acoustic-segmental"
    transcript: str | None = None
    unmapped_words: tuple[str, ...] = ()
    detected_phonemes: tuple[str, ...] = ()
    unmapped_phonemes: tuple[str, ...] = ()
    voicebank_profile: str | None = None
    voicebank_profile_confidence: float = 0.0
    alias_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-facing timeline representation."""
        return {
            "format": "synthstream-timeline",
            "version": 1,
            "source_wav": str(self.source_wav),
            "voicebank_root": str(self.voicebank_root),
            "sample_rate": self.sample_rate,
            "hop_seconds": self.hop_seconds,
            "input_duration_seconds": self.input_duration_seconds,
            "path_cost": self.path_cost,
            "recognition_mode": self.recognition_mode,
            "transcript": self.transcript,
            "unmapped_words": list(self.unmapped_words),
            "detected_phonemes": list(self.detected_phonemes),
            "unmapped_phonemes": list(self.unmapped_phonemes),
            "voicebank_profile": self.voicebank_profile,
            "voicebank_profile_confidence": self.voicebank_profile_confidence,
            "alias_coverage": self.alias_coverage,
            "segments": [asdict(segment) for segment in self.segments],
            "diagnostics": {
                "frames_processed": self.decode_result.frames_processed,
                "hypotheses_evaluated": self.decode_result.hypotheses_evaluated,
                "segment_scores_evaluated": self.decode_result.segment_scores_evaluated,
                "hypotheses_pruned": self.decode_result.hypotheses_pruned,
                "alternative_final_paths": len(self.decode_result.alternatives),
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return output_path


class OfflineRecognizer:
    """Compose the production loader, analyzer, matcher, and decoder."""

    def __init__(
        self,
        bank: Voicebank,
        *,
        analysis_config: AnalysisConfig | None = None,
        match_weights: MatchWeights | None = None,
        decoder_config: DecoderConfig | None = None,
        use_feature_cache: bool = True,
    ) -> None:
        self.bank = bank
        self.extractor = FeatureExtractor(analysis_config)
        self.phonemizer_capability = detect_voicebank_profile(bank)
        if self.phonemizer_capability.supported:
            self.feature_index = None
            self.matcher = None
            self.decoder = None
        else:
            self.feature_index = SectionFeatureIndex.build(
                bank, self.extractor, use_cache=use_feature_cache
            )
            self.matcher = SectionMatcher(self.feature_index, match_weights)
            self.decoder = SegmentalBeamDecoder(self.matcher, decoder_config)

    @classmethod
    def from_voicebank(
        cls,
        voicebank_root: str | Path,
        *,
        analysis_config: AnalysisConfig | None = None,
        match_weights: MatchWeights | None = None,
        decoder_config: DecoderConfig | None = None,
        use_cache: bool = True,
    ) -> OfflineRecognizer:
        bank = load_voicebank(voicebank_root, use_cache=use_cache)
        return cls(
            bank,
            analysis_config=analysis_config,
            match_weights=match_weights,
            decoder_config=decoder_config,
            use_feature_cache=use_cache,
        )

    def recognize(self, wav_path: str | Path) -> RecognitionTimeline:
        """Run a human WAV through the complete production recognition path."""
        source_path = Path(wav_path).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"Human WAV does not exist: {source_path}")
        try:
            waveform, source_rate = sf.read(
                source_path, dtype="float32", always_2d=True
            )
        except (RuntimeError, sf.LibsndfileError) as error:
            raise ValueError(f"Cannot read human WAV: {source_path}") from error
        if not len(waveform) or source_rate < 1:
            raise ValueError(f"Human WAV is empty: {source_path}")

        mono = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
        input_duration = len(mono) / source_rate
        if self.phonemizer_capability.supported:
            direct_mono = canonicalize_direct_audio(waveform, source_rate)
            return self._recognize_direct(
                source_path,
                direct_mono,
                DIRECT_SAMPLE_RATE,
                input_duration,
            )
        target_rate = self.extractor.config.sample_rate
        if source_rate != target_rate:
            divisor = math.gcd(source_rate, target_rate)
            mono = np.asarray(
                resample_poly(mono, target_rate // divisor, source_rate // divisor),
                dtype=np.float32,
            )
        features = self.extractor.analyze(mono)
        if self.decoder is None:
            raise RuntimeError("acoustic decoder is unavailable for this mapped voicebank")
        decode_result = self.decoder.decode(features)
        hop_seconds = self.extractor.config.hop_samples / target_rate
        timeline_segments = tuple(
            _timeline_segment(
                segment,
                hop_seconds,
                input_duration,
                features.frame_count,
            )
            for segment in decode_result.best_path.segments
        )
        return RecognitionTimeline(
            source_path,
            self.bank.root,
            target_rate,
            hop_seconds,
            input_duration,
            decode_result.best_path.total_cost,
            timeline_segments,
            decode_result,
            voicebank_profile=None,
        )

    def _recognize_direct(
        self,
        source_path: Path,
        mono: np.ndarray,
        sample_rate: int,
        input_duration: float,
    ) -> RecognitionTimeline:
        phones = DirectIPARecognizer().recognize(mono, sample_rate)
        profile = self.phonemizer_capability.profile
        if profile is None:
            raise RuntimeError("direct phoneme profile is unavailable")
        recognition = DirectAliasPlanner(self.bank, profile).plan(phones)
        if not recognition.aliases:
            raise RuntimeError("English frontend recognized no mappable voicebank aliases")
        hop_seconds = self.extractor.config.hop_samples / sample_rate
        frame_count = max(1, math.ceil(input_duration / hop_seconds))
        active_end_frame = _estimate_active_end_frame(mono, sample_rate, hop_seconds, frame_count)
        segments: list[DecodedSegment] = []

        first_start = recognition.aliases[0].start_seconds
        if first_start >= hop_seconds:
            segments.append(_silence_segment(0, round(first_start / hop_seconds)))
        for planned in recognition.aliases:
            start_frame = max(0, round(planned.start_seconds / hop_seconds))
            end_frame = min(
                active_end_frame,
                max(start_frame + 1, round(planned.end_seconds / hop_seconds)),
            )
            if end_frame <= start_frame:
                continue
            if segments and start_frame > segments[-1].end_frame:
                segments.append(_silence_segment(segments[-1].end_frame, start_frame))
            event = AliasEvent(
                unit_id=planned.unit.id,
                alias=planned.alias,
                start_seconds=start_frame * hop_seconds,
                end_seconds=end_frame * hop_seconds,
                confidence=max(planned.confidence, 1e-6),
            )
            section_frames = allocate_alias_section_durations(
                event,
                planned.unit,
                timebase_hz=1.0 / hop_seconds,
            )
            boundaries = [start_frame]
            for frame_count_for_section in section_frames:
                boundaries.append(boundaries[-1] + frame_count_for_section)
            section_cost = -math.log(event.confidence) / len(planned.unit.sections)
            for section_index, (section, section_start, section_end) in enumerate(
                zip(planned.unit.sections, boundaries[:-1], boundaries[1:], strict=True)
            ):
                if section_end <= section_start:
                    continue
                nominal_frames = max(1, round(section.duration_seconds / hop_seconds))
                segments.append(
                    DecodedSegment(
                        planned.unit.id,
                        planned.alias,
                        section_index,
                        section.kind,
                        section_start,
                        section_end,
                        (section_end - section_start) / nominal_frames,
                        section_cost,
                        0.0,
                        0.0,
                        section_cost,
                    )
                )
        final_end = segments[-1].end_frame
        if final_end < frame_count:
            segments.append(_silence_segment(final_end, frame_count))
        decoded = tuple(segments)
        path_cost = sum(segment.total_cost for segment in decoded)
        path = DecodedPath(decoded, path_cost)
        decode_result = DecodeResult(path, (), frame_count, 0, 0, 0)
        timeline_segments = tuple(
            _timeline_segment(segment, hop_seconds, input_duration, frame_count)
            for segment in decoded
        )
        return RecognitionTimeline(
            source_wav=source_path,
            voicebank_root=self.bank.root,
            sample_rate=sample_rate,
            hop_seconds=hop_seconds,
            input_duration_seconds=input_duration,
            path_cost=path_cost,
            segments=timeline_segments,
            decode_result=decode_result,
            recognition_mode=f"wav2vec2-ipa-ctc-{profile.name}",
            detected_phonemes=tuple(phone.ipa for phone in recognition.phones),
            unmapped_phonemes=recognition.unmapped_phones,
            voicebank_profile=profile.name,
            voicebank_profile_confidence=self.phonemizer_capability.confidence,
            alias_coverage=self.phonemizer_capability.alias_coverage,
        )


def recognize_wav(
    human_wav: str | Path,
    voicebank_root: str | Path,
    *,
    output_json: str | Path | None = None,
    output_wav: str | Path | None = None,
    output_sample_rate: int | None = None,
    pitch_ratio: float = 1.0,
    track_pitch: bool = True,
    output_gain: float = 1.0,
    analysis_config: AnalysisConfig | None = None,
    match_weights: MatchWeights | None = None,
    decoder_config: DecoderConfig | None = None,
    use_cache: bool = True,
) -> RecognitionTimeline:
    """Convenience entry point for one complete offline recognition run."""
    recognizer = OfflineRecognizer.from_voicebank(
        voicebank_root,
        analysis_config=analysis_config,
        match_weights=match_weights,
        decoder_config=decoder_config,
        use_cache=use_cache,
    )
    timeline = recognizer.recognize(human_wav)
    if output_json is not None:
        timeline.write_json(output_json)
    if output_wav is not None:
        from synthstream.offline.synthesis import synthesize_timeline

        result = synthesize_timeline(
            timeline,
            recognizer.bank,
            output_sample_rate=output_sample_rate,
            pitch_ratio=pitch_ratio,
            track_pitch=track_pitch,
            output_gain=output_gain,
        )
        result.write_wav(output_wav)
    return timeline


def _timeline_segment(
    segment: DecodedSegment,
    hop_seconds: float,
    input_duration: float,
    frame_count: int,
) -> TimelineSegment:
    start_seconds = segment.start_frame * hop_seconds
    end_seconds = segment.end_frame * hop_seconds
    if segment.end_frame == frame_count:
        end_seconds = input_duration
    end_seconds = min(end_seconds, input_duration)
    return TimelineSegment(
        segment.unit_id,
        segment.alias,
        segment.section_index,
        segment.section_kind,
        start_seconds,
        end_seconds,
        end_seconds - start_seconds,
        segment.stretch_ratio,
        segment.is_silence,
        segment.acoustic_cost,
        segment.duration_cost,
        segment.transition_cost,
        segment.total_cost,
        segment.pitch_ratio,
    )


def _silence_segment(start_frame: int, end_frame: int) -> DecodedSegment:
    return DecodedSegment(
        None, None, None, "silence", start_frame, end_frame, 1.0, 0.0, 0.0, 0.0, 0.0
    )


def _estimate_active_end_frame(
    samples: np.ndarray,
    sample_rate: int,
    hop_seconds: float,
    frame_count: int,
) -> int:
    """Trim CTC's terminal phone extension when the waveform has fallen silent."""
    window = max(1, round(sample_rate * 0.02))
    rms = np.sqrt(
        np.add.reduceat(samples.astype(np.float64) ** 2, np.arange(0, len(samples), window))
        / np.maximum(1, np.minimum(window, len(samples) - np.arange(0, len(samples), window)))
    )
    if not len(rms) or float(np.max(rms)) <= 0:
        return frame_count
    threshold = max(0.008, float(np.max(rms)) * 0.08)
    active = np.flatnonzero(rms >= threshold)
    if not len(active):
        return frame_count
    end_seconds = min(len(samples) / sample_rate, (int(active[-1]) + 1) * window / sample_rate)
    end_seconds = min(len(samples) / sample_rate, math.ceil(end_seconds / 0.05) * 0.05)
    return min(frame_count, max(1, math.ceil(end_seconds / hop_seconds)))
