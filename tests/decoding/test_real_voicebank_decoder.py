from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import FeatureExtractor
from synthstream.decoding import DecoderConfig, SegmentalBeamDecoder
from synthstream.matching import SectionFeatureIndex, SectionMatcher
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]


def test_recorded_speech_decodes_through_real_bank_with_silence_and_continuation() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    recorded, sample_rate = sf.read(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_excerpt.wav",
        dtype="float32",
        always_2d=True,
    )
    assert sample_rate == 16_000
    audio = np.concatenate(
        (np.zeros(round(0.15 * sample_rate), dtype=np.float32), np.mean(recorded, axis=1))
    )

    extractor = FeatureExtractor()
    bank = load_voicebank(bank_path)
    index = SectionFeatureIndex.build(bank, extractor)
    decoder = SegmentalBeamDecoder(
        SectionMatcher(index),
        DecoderConfig(
            maximum_hypotheses=32,
            beam_threshold=15,
            start_candidate_limit=16,
        ),
    )

    result = decoder.decode(extractor.analyze(audio))
    segments = result.best_path.segments
    voiced = [segment for segment in segments if not segment.is_silence]

    assert len(index.templates) == 4_651
    assert segments[0].is_silence
    assert 12 <= segments[0].end_frame <= 17
    assert [segment.section_index for segment in voiced] == [0, 1]
    assert [segment.section_kind for segment in voiced] == ["onset", "transition"]
    assert voiced[0].unit_id == voiced[1].unit_id
    assert voiced[0].end_frame == voiced[1].start_frame
    assert voiced[1].transition_cost == 0
    assert segments[-1].end_frame == result.frames_processed
    assert result.alternatives
    assert result.hypotheses_pruned > 0
    assert result.segment_scores_evaluated > 20_000
