"""Offline segmental beam search with path history and integrated silence."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

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
        self._start_signatures = np.stack(
            [template.start_signature for template in self._starts]
        )
        self._next = self._build_within_unit_transitions()

    def stream(
        self,
        *,
        lookahead_frames: int = 3,
        candidate_limit: int | None = 2,
        maximum_hypotheses: int | None = 4,
        beam_threshold: float | None = 5.0,
        max_start_lookback_frames: int | None = 80,
        max_segment_frames: int | None = 100,
    ) -> StreamingSegmentalBeamDecoder:
        """Create an incremental decoder with fixed-lag commitment.

        The returned decoder retains the active beam between feature chunks.  Its
        provisional path can change as new frames arrive, while committed segments
        trail the input by ``lookahead_frames`` frames.  Streaming defaults use a
        smaller beam and bounded start history than exhaustive offline decoding;
        pass ``None`` for those overrides when exact offline search is preferred.
        """
        config = replace(
            self.config,
            start_candidate_limit=(
                self.config.start_candidate_limit
                if candidate_limit is None
                else candidate_limit
            ),
            maximum_hypotheses=(
                self.config.maximum_hypotheses
                if maximum_hypotheses is None
                else maximum_hypotheses
            ),
            beam_threshold=(
                self.config.beam_threshold if beam_threshold is None else beam_threshold
            ),
        )
        streaming_decoder = SegmentalBeamDecoder(self.matcher, config)
        return StreamingSegmentalBeamDecoder(
            streaming_decoder,
            lookahead_frames=lookahead_frames,
            max_start_lookback_frames=max_start_lookback_frames,
            max_segment_frames=max_segment_frames,
        )

    def decode(self, human: FeatureBatch) -> DecodeResult:
        """Decode a complete feature batch and return traceback-ready paths."""
        if human.frame_count < 1:
            raise ValueError("cannot decode an empty feature batch")
        beams: list[tuple[_Hypothesis, ...]] = [()] * (human.frame_count + 1)
        score_cache: dict[tuple[str, int, int], SectionMatchScore | None] = {}
        start_cache: dict[int, tuple[SectionTemplate, ...]] = {}
        human_features = human.recognition_features
        hypotheses_evaluated = 0
        hypotheses_pruned = 0

        for end_frame in range(1, human.frame_count + 1):
            candidates: list[_Hypothesis] = []
            for template in self._start_templates(
                human, 0, start_cache, human_features
            ):
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
                        previous, human, start_frame, start_cache, human_features
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
        self,
        human: FeatureBatch,
        frame: int,
        cache: dict[int, tuple[SectionTemplate, ...]] | None = None,
        human_features: FloatArray | None = None,
    ) -> tuple[SectionTemplate, ...]:
        limit = self.config.start_candidate_limit
        if limit is None or limit >= len(self._starts):
            return self._starts
        if cache is not None and frame in cache:
            return cache[frame]
        features = human.recognition_features if human_features is None else human_features
        query = features[frame]
        distances = np.mean(np.square(self._start_signatures - query), axis=1)
        order = np.argsort(distances, kind="stable")[:limit]
        result = tuple(self._starts[int(index)] for index in order)
        if cache is not None:
            cache[frame] = result
        return result

    def _successors(
        self,
        previous: _Hypothesis,
        human: FeatureBatch,
        frame: int,
        cache: dict[int, tuple[SectionTemplate, ...]] | None = None,
        human_features: FloatArray | None = None,
    ) -> tuple[tuple[SectionTemplate, float], ...]:
        if previous.last_is_silence:
            return tuple(
                (template, self.config.silence_exit_cost)
                for template in self._start_templates(human, frame, cache, human_features)
            )
        if previous.last_template is None:
            return ()
        within_unit = self._next[previous.last_template.id]
        if within_unit:
            return tuple((template, 0.0) for template in within_unit)
        return tuple(
            (template, self.config.unit_switch_cost)
            for template in self._start_templates(human, frame, cache, human_features)
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
        acoustic_cost: float | None = None,
    ) -> _Hypothesis | None:
        duration = end_frame - start_frame
        if duration < self.config.minimum_silence_frames or (
            self.config.maximum_silence_frames is not None
            and duration > self.config.maximum_silence_frames
        ):
            return None
        if acoustic_cost is None:
            rms = human.rms_energy[start_frame:end_frame]
            energy_db = 20 * np.log10(np.maximum(rms, 1e-8))
            excess = np.maximum(
                0.0,
                (energy_db - self.config.silence_energy_threshold_db)
                / self.config.silence_energy_scale_db,
            )
            # Silence is a frame-wise state. Accumulate its evidence so that
            # swallowing additional voiced frames becomes progressively more
            # expensive; averaging here made an arbitrarily long silence compete
            # as a single cheap segment against several voicebank sections.
            acoustic_cost = float(
                np.sum(
                    np.square(excess)
                    + self.config.silence_periodicity_weight
                    * np.square(human.periodicity[start_frame:end_frame])
                )
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


@dataclass(frozen=True, slots=True)
class StreamingDecodeResult:
    """One update from an incremental decoder.

    ``committed_segments`` contains only the newly safe segments from this
    update.  ``provisional_path`` is the current best path and may be revised
    by later feature chunks.  ``decode_result`` retains the full beam
    diagnostics for observability.
    """

    decode_result: DecodeResult
    committed_segments: tuple[DecodedSegment, ...]
    committed_path: DecodedPath
    committed_until_frame: int
    lookahead_frames: int
    is_final: bool

    @property
    def provisional_path(self) -> DecodedPath:
        """Return the best path before fixed-lag commitment is applied."""
        return self.decode_result.best_path

    @property
    def frames_processed(self) -> int:
        """Number of feature frames represented by this update."""
        return self.decode_result.frames_processed


class StreamingSegmentalBeamDecoder:
    """Incremental segmental decoding with delayed fixed-lag commitment.

    Feature batches are appended in callback-sized chunks.  Only newly reached
    end frames are searched, while the existing beam and score cache are kept
    alive.  Before finalization, a segment is emitted only when it is shared by
    every surviving hypothesis and ends before the configured lookahead lag.
    """

    def __init__(
        self,
        decoder: SegmentalBeamDecoder | SectionMatcher,
        config: DecoderConfig | None = None,
        *,
        lookahead_frames: int = 3,
        max_start_lookback_frames: int | None = None,
        max_segment_frames: int | None = None,
    ) -> None:
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames must be non-negative")
        if max_start_lookback_frames is not None and max_start_lookback_frames < 1:
            raise ValueError("max_start_lookback_frames must be positive or None")
        if max_segment_frames is not None and max_segment_frames < 1:
            raise ValueError("max_segment_frames must be positive or None")
        self.decoder = (
            decoder
            if isinstance(decoder, SegmentalBeamDecoder)
            else SegmentalBeamDecoder(decoder, config)
        )
        self.lookahead_frames = lookahead_frames
        self.max_start_lookback_frames = max_start_lookback_frames
        self.max_segment_frames = max_segment_frames
        self._features: FeatureBatch | None = None
        self._recognition_features: FloatArray | None = None
        self._feature_accumulator = _FeatureAccumulator()
        self._beams: list[tuple[_Hypothesis, ...]] = [()]
        self._score_cache: dict[tuple[str, int, int], SectionMatchScore | None] = {}
        self._start_cache: dict[int, tuple[SectionTemplate, ...]] = {}
        self._silence_prefix = [0.0]
        self._hypotheses_evaluated = 0
        self._hypotheses_pruned = 0
        self._committed_segments: tuple[DecodedSegment, ...] = ()
        self._finished = False

    @property
    def frames_processed(self) -> int:
        """Number of feature frames accepted so far."""
        return 0 if self._features is None else self._features.frame_count

    @property
    def finished(self) -> bool:
        """Whether ``finish`` or a final push has closed this stream."""
        return self._finished

    @property
    def committed_path(self) -> DecodedPath:
        """The path already safe for downstream synthesis."""
        return _path_from_segments(self._committed_segments)

    def push(self, human: FeatureBatch, *, final: bool = False) -> StreamingDecodeResult:
        """Append a feature chunk and return its provisional/committed paths."""
        if self._finished:
            raise RuntimeError("streaming decoder has already been finalized")
        if human.frame_count < 1:
            raise ValueError("cannot append an empty feature chunk")

        previous_count = self.frames_processed
        self._features = self._feature_accumulator.append(human)
        self._recognition_features = self._features.recognition_features
        self._append_silence_costs(human)
        new_count = self.frames_processed
        self._beams.extend([()] * (new_count - previous_count))
        assert self._features is not None

        for end_frame in range(previous_count + 1, new_count + 1):
            self._decode_end_frame(end_frame)

        result = self._make_result(final=final)
        if final:
            self._finished = True
        return result

    def finish(self) -> StreamingDecodeResult:
        """Finalize the current stream and commit the complete winning path."""
        if self._finished:
            raise RuntimeError("streaming decoder has already been finalized")
        if self.frames_processed < 1:
            raise ValueError("cannot finalize an empty feature stream")
        result = self._make_result(final=True)
        self._finished = True
        return result

    def reset(self) -> None:
        """Discard buffered features, hypotheses, and commitment state."""
        self._features = None
        self._recognition_features = None
        self._feature_accumulator.reset()
        self._beams = [()]
        self._score_cache.clear()
        self._start_cache.clear()
        self._silence_prefix = [0.0]
        self._hypotheses_evaluated = 0
        self._hypotheses_pruned = 0
        self._committed_segments = ()
        self._finished = False

    def _append_silence_costs(self, chunk: FeatureBatch) -> None:
        config = self.decoder.config
        energy_db = 20 * np.log10(np.maximum(chunk.rms_energy, 1e-8))
        excess = np.maximum(
            0.0,
            (energy_db - config.silence_energy_threshold_db)
            / config.silence_energy_scale_db,
        )
        frame_costs = np.square(excess) + config.silence_periodicity_weight * np.square(
            chunk.periodicity
        )
        self._silence_prefix.extend(
            float(value) for value in np.cumsum(frame_costs, dtype=np.float64)
            + self._silence_prefix[-1]
        )

    def _silence_cost(self, start_frame: int, end_frame: int) -> float:
        return self._silence_prefix[end_frame] - self._silence_prefix[start_frame]

    def _decode_end_frame(self, end_frame: int) -> None:
        assert self._features is not None
        assert self._recognition_features is not None
        candidates: list[_Hypothesis] = []
        for template in self.decoder._start_templates(
            self._features, 0, self._start_cache, self._recognition_features
        ):
            if self.max_segment_frames is not None and end_frame > self.max_segment_frames:
                continue
            hypothesis = self.decoder._extend_section(
                None,
                template,
                self._recognition_features,
                0,
                end_frame,
                0.0,
                self._score_cache,
            )
            self._hypotheses_evaluated += 1
            if hypothesis is not None:
                candidates.append(hypothesis)
        silence = self.decoder._extend_silence(
            None,
            self._features,
            0,
            end_frame,
            0.0,
            self._silence_cost(0, end_frame),
        )
        self._hypotheses_evaluated += 1
        if silence is not None:
            candidates.append(silence)

        start_floor = 1
        if self.max_start_lookback_frames is not None:
            start_floor = max(1, end_frame - self.max_start_lookback_frames)
        for start_frame in range(start_floor, end_frame):
            for previous in self._beams[start_frame]:
                for template, transition_cost in self.decoder._successors(
                    previous,
                    self._features,
                    start_frame,
                    self._start_cache,
                    self._recognition_features,
                ):
                    if (
                        self.max_segment_frames is not None
                        and end_frame - start_frame > self.max_segment_frames
                    ):
                        continue
                    hypothesis = self.decoder._extend_section(
                        previous,
                        template,
                        self._recognition_features,
                        start_frame,
                        end_frame,
                        transition_cost,
                        self._score_cache,
                    )
                    self._hypotheses_evaluated += 1
                    if hypothesis is not None:
                        candidates.append(hypothesis)
                if self.decoder._can_enter_silence(previous):
                    hypothesis = self.decoder._extend_silence(
                        previous,
                        self._features,
                        start_frame,
                        end_frame,
                        self.decoder.config.silence_entry_cost,
                        self._silence_cost(start_frame, end_frame),
                    )
                    self._hypotheses_evaluated += 1
                    if hypothesis is not None:
                        candidates.append(hypothesis)

        self._beams[end_frame], pruned = self.decoder._prune(candidates)
        self._hypotheses_pruned += pruned

    def _make_result(self, *, final: bool) -> StreamingDecodeResult:
        final_hypotheses = self._beams[self.frames_processed]
        if not final_hypotheses:
            raise RuntimeError("decoder found no valid path through the audio")
        paths = tuple(
            DecodedPath(hypothesis.segments, hypothesis.total_cost)
            for hypothesis in final_hypotheses
        )
        decode_result = DecodeResult(
            paths[0],
            paths[1:],
            self.frames_processed,
            self._hypotheses_evaluated,
            len(self._score_cache),
            self._hypotheses_pruned,
        )

        if final:
            safe_segments = paths[0].segments
        else:
            shared = _shared_prefix(path.segments for path in paths)
            cutoff = max(0, self.frames_processed - self.lookahead_frames)
            safe_segments = tuple(
                segment for segment in shared if segment.end_frame <= cutoff
            )

        # A committed segment is immutable to downstream synthesis.  If beam
        # pruning later changes an already emitted prefix, retain the emitted
        # prefix and wait for a final result rather than emitting a contradictory
        # replacement in the middle of a stream.
        committed_count = len(self._committed_segments)
        if safe_segments[:committed_count] != self._committed_segments:
            safe_segments = self._committed_segments
        newly_committed = safe_segments[committed_count:]
        self._committed_segments = safe_segments
        return StreamingDecodeResult(
            decode_result,
            newly_committed,
            _path_from_segments(self._committed_segments),
            (self._committed_segments[-1].end_frame if self._committed_segments else 0),
            self.lookahead_frames,
            final,
        )


class _FeatureAccumulator:
    """Amortized feature storage for callback-sized streaming updates."""

    _fields = (
        "frame_times_seconds",
        "log_mel",
        "delta_mel",
        "normalized_energy",
        "delta_energy",
        "spectral_flux",
        "spectral_flatness",
        "periodicity",
        "f0_hz",
        "voiced",
        "rms_energy",
    )

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._capacity = 0
        self._count = 0

    def append(self, incoming: FeatureBatch) -> FeatureBatch:
        if incoming.frame_count < 1:
            raise ValueError("cannot append an empty feature chunk")
        if not self._arrays:
            self._capacity = max(16, incoming.frame_count)
            self._arrays = {
                name: np.empty(
                    (self._capacity, *getattr(incoming, name).shape[1:]),
                    dtype=getattr(incoming, name).dtype,
                )
                for name in self._fields
            }
        elif self._count + incoming.frame_count > self._capacity:
            new_capacity = max(
                self._capacity * 2,
                self._count + incoming.frame_count,
            )
            self._arrays = {
                name: _grow_array(array, new_capacity)
                for name, array in self._arrays.items()
            }
            self._capacity = new_capacity

        start = self._count
        end = start + incoming.frame_count
        for name in self._fields:
            self._arrays[name][start:end] = getattr(incoming, name)
        self._count = end
        return FeatureBatch(*(self._arrays[name][:end] for name in self._fields))

    def reset(self) -> None:
        self._arrays.clear()
        self._capacity = 0
        self._count = 0


def _grow_array(array: np.ndarray, capacity: int) -> np.ndarray:
    grown = np.empty((capacity, *array.shape[1:]), dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def _shared_prefix(
    paths: Iterable[tuple[DecodedSegment, ...]],
) -> tuple[DecodedSegment, ...]:
    path_iter = iter(paths)
    try:
        prefix = list(next(path_iter))
    except StopIteration:
        return ()
    for path in path_iter:
        length = 0
        for left, right in zip(prefix, path, strict=False):
            if left != right:
                break
            length += 1
        del prefix[length:]
        if not prefix:
            break
    return tuple(prefix)


def _path_from_segments(segments: tuple[DecodedSegment, ...]) -> DecodedPath:
    return DecodedPath(segments, sum(segment.total_cost for segment in segments))
