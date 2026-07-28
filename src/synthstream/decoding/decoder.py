"""Offline segmental beam search with path history and integrated silence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from synthstream.analysis import FeatureBatch
from synthstream.matching import SectionMatcher, SectionMatchScore, SectionTemplate
from synthstream.matching.index import FloatArray


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Search, transition, and silence parameters."""

    maximum_hypotheses: int = 64
    beam_threshold: float = 25.0
    unit_switch_cost: float = 1.0
    silence_entry_cost: float = 0.4
    silence_exit_cost: float = 0.4
    silence_energy_threshold_db: float = -48.0
    silence_energy_scale_db: float = 18.0
    silence_periodicity_weight: float = 0.5
    minimum_silence_frames: int = 1
    maximum_silence_frames: int | None = None
    start_candidate_limit: int | None = None

    def __post_init__(self) -> None:
        finite_nonnegative = (
            self.beam_threshold,
            self.unit_switch_cost,
            self.silence_entry_cost,
            self.silence_exit_cost,
            self.silence_periodicity_weight,
        )
        if self.maximum_hypotheses < 1 or any(
            value < 0 or not math.isfinite(value) for value in finite_nonnegative
        ):
            raise ValueError("beam sizes and costs must be finite and non-negative")
        if self.silence_energy_scale_db <= 0 or self.minimum_silence_frames < 1:
            raise ValueError("silence scale and minimum duration must be positive")
        if (
            self.maximum_silence_frames is not None
            and self.maximum_silence_frames < self.minimum_silence_frames
        ):
            raise ValueError("maximum silence duration is smaller than the minimum")
        if self.start_candidate_limit is not None and self.start_candidate_limit < 1:
            raise ValueError("start_candidate_limit must be positive or None")


@dataclass(frozen=True, slots=True)
class DecodedSegment:
    """One selected section or silence interval."""

    unit_id: str | None
    alias: str | None
    section_index: int | None
    section_kind: str
    start_frame: int
    end_frame: int
    stretch_ratio: float
    acoustic_cost: float
    duration_cost: float
    transition_cost: float
    total_cost: float

    @property
    def is_silence(self) -> bool:
        return self.unit_id is None


@dataclass(frozen=True, slots=True)
class DecodedPath:
    """A recoverable decoder path ordered from earliest to latest segment."""

    segments: tuple[DecodedSegment, ...]
    total_cost: float


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Winning and alternative final paths plus search diagnostics."""

    best_path: DecodedPath
    alternatives: tuple[DecodedPath, ...]
    frames_processed: int
    hypotheses_evaluated: int
    segment_scores_evaluated: int
    hypotheses_pruned: int


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    segments: tuple[DecodedSegment, ...]
    total_cost: float
    last_template: SectionTemplate | None
    last_is_silence: bool


class SegmentalBeamDecoder:
    """Jointly search section identity, boundaries, duration, transitions, and silence."""

    def __init__(
        self, matcher: SectionMatcher, config: DecoderConfig | None = None
    ) -> None:
        self.matcher = matcher
        self.config = config or DecoderConfig()
        templates = matcher.index.templates
        self._units: dict[str, tuple[SectionTemplate, ...]] = {}
        for template in templates:
            self._units.setdefault(template.unit_id, ())
            self._units[template.unit_id] += (template,)
        self._units = {
            unit_id: tuple(sorted(unit_templates, key=lambda item: item.section_index))
            for unit_id, unit_templates in self._units.items()
        }
        self._starts = tuple(unit_templates[0] for unit_templates in self._units.values())
        self._next = self._build_within_unit_transitions()

    def decode(self, human: FeatureBatch) -> DecodeResult:
        """Decode a complete feature batch and return traceback-ready paths."""
        if human.frame_count < 1:
            raise ValueError("cannot decode an empty feature batch")
        beams: list[tuple[_Hypothesis, ...]] = [()] * (human.frame_count + 1)
        score_cache: dict[tuple[str, int, int], SectionMatchScore | None] = {}
        human_features = human.recognition_features
        hypotheses_evaluated = 0
        hypotheses_pruned = 0

        for end_frame in range(1, human.frame_count + 1):
            candidates: list[_Hypothesis] = []
            for template in self._start_templates(human, 0):
                hypothesis = self._extend_section(
                    None,
                    template,
                    human_features,
                    0,
                    end_frame,
                    0.0,
                    score_cache,
                )
                hypotheses_evaluated += 1
                if hypothesis is not None:
                    candidates.append(hypothesis)
            silence = self._extend_silence(None, human, 0, end_frame, 0.0)
            hypotheses_evaluated += 1
            if silence is not None:
                candidates.append(silence)

            for start_frame in range(1, end_frame):
                for previous in beams[start_frame]:
                    for template, transition_cost in self._successors(
                        previous, human, start_frame
                    ):
                        hypothesis = self._extend_section(
                            previous,
                            template,
                            human_features,
                            start_frame,
                            end_frame,
                            transition_cost,
                            score_cache,
                        )
                        hypotheses_evaluated += 1
                        if hypothesis is not None:
                            candidates.append(hypothesis)
                    if self._can_enter_silence(previous):
                        hypothesis = self._extend_silence(
                            previous,
                            human,
                            start_frame,
                            end_frame,
                            self.config.silence_entry_cost,
                        )
                        hypotheses_evaluated += 1
                        if hypothesis is not None:
                            candidates.append(hypothesis)

            beams[end_frame], pruned = self._prune(candidates)
            hypotheses_pruned += pruned

        final_hypotheses = beams[human.frame_count]
        if not final_hypotheses:
            raise RuntimeError("decoder found no valid path through the audio")
        paths = tuple(
            DecodedPath(hypothesis.segments, hypothesis.total_cost)
            for hypothesis in final_hypotheses
        )
        return DecodeResult(
            paths[0],
            paths[1:],
            human.frame_count,
            hypotheses_evaluated,
            len(score_cache),
            hypotheses_pruned,
        )

    def _build_within_unit_transitions(self) -> dict[str, tuple[SectionTemplate, ...]]:
        transitions: dict[str, tuple[SectionTemplate, ...]] = {}
        for templates in self._units.values():
            for current, following in zip(templates, templates[1:], strict=False):
                transitions[current.id] = (following,)
            transitions[templates[-1].id] = ()
        return transitions

    def _start_templates(
        self, human: FeatureBatch, frame: int
    ) -> tuple[SectionTemplate, ...]:
        limit = self.config.start_candidate_limit
        if limit is None or limit >= len(self._starts):
            return self._starts
        query = human.recognition_features[frame]
        distances = tuple(
            float(np.mean(np.square(template.start_signature - query)))
            for template in self._starts
        )
        order = np.argsort(distances, kind="stable")[:limit]
        return tuple(self._starts[int(index)] for index in order)

    def _successors(
        self, previous: _Hypothesis, human: FeatureBatch, frame: int
    ) -> tuple[tuple[SectionTemplate, float], ...]:
        if previous.last_is_silence:
            return tuple(
                (template, self.config.silence_exit_cost)
                for template in self._start_templates(human, frame)
            )
        if previous.last_template is None:
            return ()
        within_unit = self._next[previous.last_template.id]
        if within_unit:
            return tuple((template, 0.0) for template in within_unit)
        return tuple(
            (template, self.config.unit_switch_cost)
            for template in self._start_templates(human, frame)
        )

    def _can_enter_silence(self, previous: _Hypothesis) -> bool:
        return (
            not previous.last_is_silence
            and previous.last_template is not None
            and not self._next[previous.last_template.id]
        )

    def _extend_section(
        self,
        previous: _Hypothesis | None,
        template: SectionTemplate,
        human_features: FloatArray,
        start_frame: int,
        end_frame: int,
        transition_cost: float,
        cache: dict[tuple[str, int, int], SectionMatchScore | None],
    ) -> _Hypothesis | None:
        key = (template.id, start_frame, end_frame)
        if key not in cache:
            cache[key] = self.matcher.score_candidate_features(
                template, human_features, start_frame, end_frame
            )
        score = cache[key]
        if score is None:
            return None
        segment_cost = score.total_cost + transition_cost
        segment = DecodedSegment(
            template.unit_id,
            template.alias,
            template.section_index,
            template.section_kind,
            start_frame,
            end_frame,
            score.stretch_ratio,
            score.acoustic_cost,
            score.duration_cost,
            transition_cost,
            segment_cost,
        )
        return self._append(previous, segment, template, False)

    def _extend_silence(
        self,
        previous: _Hypothesis | None,
        human: FeatureBatch,
        start_frame: int,
        end_frame: int,
        transition_cost: float,
    ) -> _Hypothesis | None:
        duration = end_frame - start_frame
        if duration < self.config.minimum_silence_frames or (
            self.config.maximum_silence_frames is not None
            and duration > self.config.maximum_silence_frames
        ):
            return None
        rms = human.rms_energy[start_frame:end_frame]
        energy_db = 20 * np.log10(np.maximum(rms, 1e-8))
        excess = np.maximum(
            0.0,
            (energy_db - self.config.silence_energy_threshold_db)
            / self.config.silence_energy_scale_db,
        )
        acoustic_cost = float(
            np.mean(np.square(excess))
            + self.config.silence_periodicity_weight
            * np.mean(np.square(human.periodicity[start_frame:end_frame]))
        )
        segment_cost = acoustic_cost + transition_cost
        segment = DecodedSegment(
            None,
            None,
            None,
            "silence",
            start_frame,
            end_frame,
            1.0,
            acoustic_cost,
            0.0,
            transition_cost,
            segment_cost,
        )
        return self._append(previous, segment, None, True)

    @staticmethod
    def _append(
        previous: _Hypothesis | None,
        segment: DecodedSegment,
        template: SectionTemplate | None,
        is_silence: bool,
    ) -> _Hypothesis:
        segments = (() if previous is None else previous.segments) + (segment,)
        previous_cost = 0.0 if previous is None else previous.total_cost
        return _Hypothesis(
            segments,
            previous_cost + segment.total_cost,
            template,
            is_silence,
        )

    def _prune(
        self, candidates: list[_Hypothesis]
    ) -> tuple[tuple[_Hypothesis, ...], int]:
        if not candidates:
            return (), 0
        candidates.sort(key=lambda hypothesis: hypothesis.total_cost)
        best_cost = candidates[0].total_cost
        within_beam = [
            hypothesis
            for hypothesis in candidates
            if hypothesis.total_cost <= best_cost + self.config.beam_threshold
        ]
        kept = tuple(within_beam[: self.config.maximum_hypotheses])
        return kept, len(candidates) - len(kept)
