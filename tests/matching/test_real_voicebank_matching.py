from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import FeatureExtractor
from synthstream.matching import SectionFeatureIndex, SectionMatcher
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]


def test_recorded_human_interval_scores_complete_real_voicebank() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    samples, sample_rate = sf.read(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_excerpt.wav",
        dtype="float32",
        always_2d=True,
    )
    assert sample_rate == 16_000

    extractor = FeatureExtractor()
    human = extractor.analyze(np.mean(samples, axis=1))
    bank = load_voicebank(bank_path)
    index = SectionFeatureIndex.build(bank, extractor)
    matcher = SectionMatcher(index)

    # Forced alignment places the vowel in "had" at 0.15--0.22 seconds in
    # this excerpt. Feature timestamps are window centres, hence frames 14:21.
    ranked = matcher.match_interval(human, 14, 21)

    assert len(index.templates) == sum(len(unit.sections) for unit in bank.units) == 4_651
    assert len(ranked) > 2_000
    # Aiko's own guide maps the vowel in "cat" to @. The raw spectral matcher
    # keeps a compatible contextual unit reachable, but does not rank it well;
    # semantic recognition is covered by the phone-aware offline regression.
    assert any(score.template.alias == "@d" for score in ranked)
