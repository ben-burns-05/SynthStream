from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.analysis import FeatureExtractor
from synthstream.matching import (
    MatchWeights,
    SectionFeatureIndex,
    SectionMatcher,
    uniformly_resample,
)
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


def _sine(frequency: float, duration: float = 0.5) -> np.ndarray:
    time = np.arange(round(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _make_bank(root: Path) -> None:
    sf.write(root / "a.wav", _sine(220), SAMPLE_RATE)
    sf.write(root / "i.wav", _sine(660), SAMPLE_RATE)
    (root / "oto.ini").write_text(
        "a.wav=a,0,200,0,50,20\n"
        "i.wav=i,0,200,0,50,20\n",
        encoding="utf-8",
    )


def test_precomputes_every_section_in_complete_bank(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)

    index = SectionFeatureIndex.build(bank, use_cache=False)

    assert len(index.templates) == sum(len(unit.sections) for unit in bank.units) == 6
    assert {template.alias for template in index.templates} == {"a", "i"}
    assert {template.section_kind for template in index.templates} == {
        "onset",
        "transition",
        "sustain",
    }
    assert all(template.features.shape[1] == 85 for template in index.templates)


def test_feature_index_cache_avoids_reanalysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    first = SectionFeatureIndex.build(bank, extractor)

    def fail_if_called(samples: np.ndarray) -> None:
        pytest.fail(f"cache hit reanalyzed {len(samples)} audio samples")

    monkeypatch.setattr(extractor, "analyze", fail_if_called)
    second = SectionFeatureIndex.build(bank, extractor)

    assert not first.cache_hit
    assert second.cache_hit
    assert len(second.templates) == len(first.templates)
    assert np.array_equal(second.templates[0].features, first.templates[0].features)


def test_uniform_warp_uses_linear_constant_rate_mapping() -> None:
    trajectory = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)

    stretched = uniformly_resample(trajectory, 5)

    assert np.array_equal(stretched[:, 0], [0, 5, 10, 15, 20])
    assert np.array_equal(uniformly_resample(trajectory, 3), trajectory)


def test_matching_section_beats_spectrally_different_section(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    matcher = SectionMatcher(index)
    human = extractor.analyze(_sine(220, 0.3))
    templates = {
        (template.alias, template.section_kind): template for template in index.templates
    }

    matching = matcher.score_candidate(templates[("a", "sustain")], human, 0, human.frame_count)
    different = matcher.score_candidate(templates[("i", "sustain")], human, 0, human.frame_count)

    assert matching is not None
    assert different is not None
    assert matching.acoustic_cost < different.acoustic_cost
    assert matching.total_cost < different.total_cost
    assert matching.stretch_ratio == pytest.approx(29 / 30)
    assert matching.total_cost == pytest.approx(
        matching.acoustic_cost + matching.duration_cost
    )


def test_interval_search_scores_full_bank_and_exposes_components(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    matcher = SectionMatcher(index)
    human = extractor.analyze(_sine(220, 0.3))

    scores = matcher.match_interval(human, 0, human.frame_count)

    assert scores
    assert scores[0].template.alias == "a"
    assert scores[0].template.section_kind == "sustain"
    assert scores == tuple(sorted(scores, key=lambda score: score.total_cost))
    assert all(score.mel_cost >= 0 for score in scores)
    assert all(score.periodicity_cost >= 0 for score in scores)


def test_duration_constraints_reject_implausible_stretch(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    matcher = SectionMatcher(index)
    human = extractor.analyze(_sine(220, 0.4))
    onset = next(template for template in index.templates if template.section_kind == "onset")

    assert matcher.score_candidate(onset, human, 0, human.frame_count) is None


def test_vectorized_start_ranking_can_return_every_entry(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    human = extractor.analyze(_sine(660, 0.1))

    all_ranked = index.rank_start_candidates(human.recognition_features[0])
    shortlist = index.rank_start_candidates(human.recognition_features[0], limit=2)

    assert len(all_ranked) == len(index.templates)
    assert shortlist == all_ranked[:2]
    assert {template.id for template in all_ranked} == {
        template.id for template in index.templates
    }


def test_entries_throughout_large_voicebank_remain_selectable(tmp_path: Path) -> None:
    sf.write(tmp_path / "shared.wav", _sine(330, 0.1), SAMPLE_RATE)
    entries = [f"shared.wav=unit-{index:03d},0,0,0,0,0" for index in range(64)]
    (tmp_path / "oto.ini").write_text("\n".join(entries), encoding="utf-8")
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    matcher = SectionMatcher(index)
    human = extractor.analyze(_sine(330, 0.1))

    scores = matcher.match_interval(human, 0, human.frame_count)

    assert len(bank.units) == 64
    assert len(index.templates) == 64
    assert len(scores) == 64
    assert any(score.template.alias == "unit-063" for score in scores)


def test_matcher_validates_weights_intervals_and_limits(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    extractor = FeatureExtractor()
    index = SectionFeatureIndex.build(bank, extractor, use_cache=False)
    matcher = SectionMatcher(index)
    human = extractor.analyze(_sine(220, 0.1))

    with pytest.raises(ValueError):
        MatchWeights(mel=-1)
    with pytest.raises(ValueError):
        matcher.score_candidate(index.templates[0], human, 2, 2)
    with pytest.raises(ValueError):
        matcher.match_interval(human, 0, human.frame_count, limit=0)
    with pytest.raises(ValueError):
        index.rank_start_candidates(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        index.rank_start_candidates(human.recognition_features[0], limit=0)
    with pytest.raises(ValueError):
        uniformly_resample(np.empty((0, 2), dtype=np.float32), 2)
