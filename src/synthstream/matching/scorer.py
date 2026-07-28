"""Uniform-warp acoustic and duration scoring for voicebank sections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from synthstream.analysis import FeatureBatch
from synthstream.matching.index import FloatArray, SectionFeatureIndex, SectionTemplate


@dataclass(frozen=True, slots=True)
class MatchWeights:
    """Inspectable acoustic cost weights."""

    mel: float = 1.0
    delta_mel: float = 0.5
    energy: float = 0.2
    delta_energy: float = 0.3
    spectral_flux: float = 0.3
    spectral_flatness: float = 0.4
    periodicity: float = 0.4

    def __post_init__(self) -> None:
        if any(value < 0 or not math.isfinite(value) for value in self.values):
            raise ValueError("match weights must be finite and non-negative")

    @property
    def values(self) -> tuple[float, ...]:
        return (
            self.mel,
            self.delta_mel,
            self.energy,
            self.delta_energy,
            self.spectral_flux,
            self.spectral_flatness,
            self.periodicity,
        )


@dataclass(frozen=True, slots=True)
class SectionMatchScore:
    """Detailed score for one section and one human frame interval."""

    template: SectionTemplate
    start_frame: int
    end_frame: int
    stretch_ratio: float
    mel_cost: float
    delta_mel_cost: float
    energy_cost: float
    delta_energy_cost: float
    spectral_flux_cost: float
    spectral_flatness_cost: float
    periodicity_cost: float
    acoustic_cost: float
    duration_cost: float
    total_cost: float


class SectionMatcher:
    """Score uniformly warped templates against arbitrary human intervals."""

    def __init__(
        self, index: SectionFeatureIndex, weights: MatchWeights | None = None
    ) -> None:
        self.index = index
        self.weights = weights or MatchWeights()

    def score_candidate(
        self,
        template: SectionTemplate,
        human: FeatureBatch,
        start_frame: int,
        end_frame: int,
    ) -> SectionMatchScore | None:
        """Return a component score, or ``None`` outside duration constraints."""
        if not 0 <= start_frame < end_frame <= human.frame_count:
            raise ValueError("candidate frame interval is invalid")
        return self.score_candidate_features(
            template, human.recognition_features, start_frame, end_frame
        )

    def score_candidate_features(
        self,
        template: SectionTemplate,
        human_features: FloatArray,
        start_frame: int,
        end_frame: int,
    ) -> SectionMatchScore | None:
        """Score against an already combined matrix for high-volume decoder use."""
        if (
            human_features.ndim != 2
            or human_features.shape[1] != template.features.shape[1]
            or not 0 <= start_frame < end_frame <= len(human_features)
        ):
            raise ValueError("candidate feature matrix or frame interval is invalid")
        duration_frames = end_frame - start_frame
        stretch_ratio = duration_frames / template.nominal_frame_count
        if not template.minimum_stretch <= stretch_ratio <= template.maximum_stretch:
            return None

        candidate = human_features[start_frame:end_frame]
        warped = uniformly_resample(template.features, duration_frames)
        mel_bands = self.index.config.mel_bands
        component_costs = _component_costs(warped, candidate, mel_bands)
        weighted = tuple(
            cost * weight for cost, weight in zip(component_costs, self.weights.values, strict=True)
        )
        acoustic_cost = float(sum(weighted))
        duration_cost = template.duration_penalty_strength * math.log(stretch_ratio) ** 2
        total_cost = acoustic_cost + duration_cost
        return SectionMatchScore(
            template,
            start_frame,
            end_frame,
            stretch_ratio,
            *component_costs,
            acoustic_cost,
            duration_cost,
            total_cost,
        )

    def match_interval(
        self,
        human: FeatureBatch,
        start_frame: int,
        end_frame: int,
        *,
        templates: tuple[SectionTemplate, ...] | None = None,
        limit: int | None = None,
    ) -> tuple[SectionMatchScore, ...]:
        """Detailed-score all supplied sections (the complete bank by default)."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive or None")
        if not 0 <= start_frame < end_frame <= human.frame_count:
            raise ValueError("candidate frame interval is invalid")
        candidates = templates if templates is not None else self.index.templates
        human_features = human.recognition_features
        scores = tuple(
            score
            for template in candidates
            if (
                score := self.score_candidate_features(
                    template, human_features, start_frame, end_frame
                )
            )
            is not None
        )
        ranked = tuple(sorted(scores, key=lambda score: score.total_cost))
        return ranked if limit is None else ranked[:limit]


def uniformly_resample(features: FloatArray, frame_count: int) -> FloatArray:
    """Linearly resample a trajectory using one constant warp rate."""
    if features.ndim != 2 or len(features) < 1 or frame_count < 1:
        raise ValueError("features and target frame count must be non-empty")
    if len(features) == frame_count:
        return features.copy()
    source_positions = np.linspace(0.0, 1.0, len(features))
    target_positions = np.linspace(0.0, 1.0, frame_count)
    result = np.empty((frame_count, features.shape[1]), dtype=np.float32)
    for column in range(features.shape[1]):
        result[:, column] = np.interp(
            target_positions, source_positions, features[:, column]
        )
    return result


def _component_costs(
    template: FloatArray, candidate: FloatArray, mel_bands: int
) -> tuple[float, float, float, float, float, float, float]:
    squared = np.square(template - candidate)
    scalar_start = mel_bands * 2
    slices: tuple[slice | int, ...] = (
        slice(0, mel_bands),
        slice(mel_bands, scalar_start),
        scalar_start,
        scalar_start + 1,
        scalar_start + 2,
        scalar_start + 3,
        scalar_start + 4,
    )
    return tuple(float(np.mean(squared[:, component])) for component in slices)  # type: ignore[return-value]
