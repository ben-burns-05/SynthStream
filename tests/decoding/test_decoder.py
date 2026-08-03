from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import AnalysisConfig, FeatureBatch, FeatureExtractor
from synthstream.decoding import (
    DecoderConfig,
    SegmentalBeamDecoder,
    StreamingSegmentalBeamDecoder,
)
from synthstream.matching import SectionFeatureIndex, SectionMatcher, SectionMatchScore
from synthstream.matching.index import SectionTemplate
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


def _tone(frequency: float, duration: float) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _piecewise_unit(frequencies: tuple[float, float, float]) -> np.ndarray:
    return np.concatenate(
        (_tone(frequencies[0], 0.05), _tone(frequencies[1], 0.15), _tone(frequencies[2], 0.3))
    )


def _make_sequence_bank(root: Path) -> None:
    sf.write(root / "a.wav", _piecewise_unit((220, 330, 440)), SAMPLE_RATE)
    sf.write(root / "b.wav", _piecewise_unit((550, 660, 770)), SAMPLE_RATE)
    (root / "oto.ini").write_text(
        "a.wav=a,0,200,0,50,20\n"
        "b.wav=b,0,200,0,50,20\n",
        encoding="utf-8",
    )


def _real_decoder(root: Path) -> tuple[SegmentalBeamDecoder, FeatureExtractor]:
    extractor = FeatureExtractor()
    bank = load_voicebank(root, use_cache=False)
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    return SegmentalBeamDecoder(SectionMatcher(index)), extractor


def test_decodes_ordered_sections_with_joint_boundaries(tmp_path: Path) -> None:
    _make_sequence_bank(tmp_path)
    decoder, extractor = _real_decoder(tmp_path)
    human = extractor.analyze(_piecewise_unit((220, 330, 440)))

    result = decoder.decode(human)
    voiced_segments = [segment for segment in result.best_path.segments if not segment.is_silence]

    assert [segment.alias for segment in voiced_segments] == ["a", "a", "a"]
    assert [segment.section_index for segment in voiced_segments] == [0, 1, 2]
    assert [segment.section_kind for segment in voiced_segments] == [
        "onset",
        "transition",
        "sustain",
    ]
    assert voiced_segments[0].start_frame == 0
    assert voiced_segments[-1].end_frame == human.frame_count
    assert all(segment.end_frame > segment.start_frame for segment in voiced_segments)
    assert result.segment_scores_evaluated < result.hypotheses_evaluated


def test_silence_participates_in_same_search_before_voice(tmp_path: Path) -> None:
    _make_sequence_bank(tmp_path)
    decoder, extractor = _real_decoder(tmp_path)
    audio = np.concatenate(
        (
            np.zeros(round(0.2 * SAMPLE_RATE), dtype=np.float32),
            _piecewise_unit((220, 330, 440)),
        )
    )
    human = extractor.analyze(audio)

    result = decoder.decode(human)

    assert result.best_path.segments[0].is_silence
    assert result.best_path.segments[0].start_frame == 0
    assert any(segment.alias == "a" for segment in result.best_path.segments)
    assert result.best_path.segments[-1].end_frame == human.frame_count


def _blank_features(frame_count: int) -> FeatureBatch:
    vector = np.zeros(frame_count, dtype=np.float32)
    matrix = np.zeros((frame_count, 40), dtype=np.float32)
    return FeatureBatch(
        np.arange(frame_count, dtype=np.float32) * 0.01,
        matrix,
        matrix.copy(),
        vector,
        vector.copy(),
        vector.copy(),
        vector.copy(),
        vector.copy(),
        np.full(frame_count, np.nan, dtype=np.float32),
        np.zeros(frame_count, dtype=np.bool_),
        np.ones(frame_count, dtype=np.float32),
    )


def _ambiguous_index() -> SectionFeatureIndex:
    templates = []
    for alias in ("a", "b"):
        for section_index, (kind, frames) in enumerate((("onset", 2), ("sustain", 3))):
            features = np.zeros((frames, 85), dtype=np.float32)
            templates.append(
                SectionTemplate(
                    alias,
                    alias,
                    section_index,
                    kind,
                    frames * 0.01,
                    frames,
                    1.0,
                    1.0,
                    0.0,
                    features,
                    features[0],
                )
            )
    return SectionFeatureIndex(tuple(templates), AnalysisConfig(), "test")


class FutureEvidenceMatcher:
    def __init__(self) -> None:
        self.index = _ambiguous_index()

    def score_candidate(
        self,
        template: SectionTemplate,
        human: FeatureBatch,
        start_frame: int,
        end_frame: int,
    ) -> SectionMatchScore | None:
        return self.score_candidate_features(
            template, human.recognition_features, start_frame, end_frame
        )

    def score_candidate_features(
        self,
        template: SectionTemplate,
        human_features: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> SectionMatchScore | None:
        del human_features
        if end_frame - start_frame != template.nominal_frame_count:
            return None
        if template.section_kind == "onset":
            acoustic = 0.0 if template.alias == "a" else 1.0
        else:
            acoustic = 10.0 if template.alias == "a" else 0.0
        return SectionMatchScore(
            template,
            start_frame,
            end_frame,
            1.0,
            acoustic,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            acoustic,
            0.0,
            acoustic,
        )


def test_future_evidence_can_overturn_initially_best_unit() -> None:
    matcher = FutureEvidenceMatcher()
    decoder = SegmentalBeamDecoder(
        matcher,  # type: ignore[arg-type]
        DecoderConfig(maximum_hypotheses=32, beam_threshold=20),
    )

    result = decoder.decode(_blank_features(5))

    assert [segment.alias for segment in result.best_path.segments] == ["b", "b"]
    assert result.best_path.total_cost == pytest.approx(1.0)
    assert any(
        [segment.alias for segment in path.segments] == ["a", "a"]
        for path in result.alternatives
    )


def test_narrow_beam_reports_pruning() -> None:
    decoder = SegmentalBeamDecoder(
        FutureEvidenceMatcher(),  # type: ignore[arg-type]
        DecoderConfig(maximum_hypotheses=1, beam_threshold=0.1),
    )

    result = decoder.decode(_blank_features(5))

    assert result.hypotheses_pruned > 0
    assert len(result.alternatives) == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_hypotheses": 0},
        {"beam_threshold": -1},
        {"silence_energy_scale_db": 0},
        {"minimum_silence_frames": 0},
        {"minimum_silence_frames": 3, "maximum_silence_frames": 2},
        {"start_candidate_limit": 0},
    ],
)
def test_rejects_invalid_decoder_configuration(settings: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DecoderConfig(**settings)


def test_rejects_empty_feature_batch() -> None:
    decoder = SegmentalBeamDecoder(FutureEvidenceMatcher())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty"):
        decoder.decode(_blank_features(0))


def _slice_features(features: FeatureBatch, start: int, end: int) -> FeatureBatch:
    return FeatureBatch(
        features.frame_times_seconds[start:end],
        features.log_mel[start:end],
        features.delta_mel[start:end],
        features.normalized_energy[start:end],
        features.delta_energy[start:end],
        features.spectral_flux[start:end],
        features.spectral_flatness[start:end],
        features.periodicity[start:end],
        features.f0_hz[start:end],
        features.voiced[start:end],
        features.rms_energy[start:end],
    )


def test_streaming_decoder_matches_batch_and_emits_fixed_lag_commits(
    tmp_path: Path,
) -> None:
    _make_sequence_bank(tmp_path)
    decoder, extractor = _real_decoder(tmp_path)
    features = extractor.analyze(_piecewise_unit((220, 330, 440)))
    expected = decoder.decode(features).best_path

    stream = decoder.stream(lookahead_frames=2)
    first = stream.push(_slice_features(features, 0, 4))
    second = stream.push(_slice_features(features, 4, 8))
    final = stream.push(_slice_features(features, 8, features.frame_count), final=True)

    assert first.committed_until_frame <= 2
    assert second.frames_processed == 8
    assert final.provisional_path.segments == expected.segments
    committed = first.committed_segments + second.committed_segments + final.committed_segments
    assert committed == expected.segments
    assert final.committed_path.segments == expected.segments


def test_streaming_future_evidence_stays_uncommitted_until_final() -> None:
    decoder = StreamingSegmentalBeamDecoder(
        FutureEvidenceMatcher(),  # type: ignore[arg-type]
        config=DecoderConfig(maximum_hypotheses=32, beam_threshold=20),
        lookahead_frames=2,
    )

    early = decoder.push(_blank_features(2))
    assert early.committed_segments == ()

    final = decoder.push(_blank_features(3), final=True)
    assert [segment.alias for segment in final.provisional_path.segments] == ["b", "b"]
    assert [segment.alias for segment in final.committed_segments] == ["b", "b"]


@pytest.mark.parametrize("lookahead_frames", [-1])
def test_streaming_decoder_rejects_invalid_lookahead(lookahead_frames: int) -> None:
    with pytest.raises(ValueError, match="lookahead"):
        StreamingSegmentalBeamDecoder(FutureEvidenceMatcher(), lookahead_frames=lookahead_frames)  # type: ignore[arg-type]
