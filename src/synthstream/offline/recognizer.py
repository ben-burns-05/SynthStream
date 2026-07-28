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
    DecodedSegment,
    DecoderConfig,
    DecodeResult,
    SegmentalBeamDecoder,
)
from synthstream.matching import MatchWeights, SectionFeatureIndex, SectionMatcher
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
        target_rate = self.extractor.config.sample_rate
        if source_rate != target_rate:
            divisor = math.gcd(source_rate, target_rate)
            mono = np.asarray(
                resample_poly(mono, target_rate // divisor, source_rate // divisor),
                dtype=np.float32,
            )
        features = self.extractor.analyze(mono)
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
        )


def recognize_wav(
    human_wav: str | Path,
    voicebank_root: str | Path,
    *,
    output_json: str | Path | None = None,
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
    )
