"""Direct audio-to-IPA recognition and inventory-aware alias planning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly  # type: ignore[import-untyped]
from transformers import AutoFeatureExtractor, AutoModelForCTC

from synthstream.offline.voicebank_phonemizer import (
    VoicebankProfile,
    detect_voicebank_profile,
    resolve_alias_candidates,
)
from synthstream.voicebank import Voicebank, VoicebankUnit

FloatArray = npt.NDArray[np.float32]
_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DIRECT_SAMPLE_RATE = 16_000
DIRECT_MIN_CONTEXT_SECONDS = 0.8


@dataclass(frozen=True, slots=True)
class DetectedPhone:
    ipa: str
    start_seconds: float
    end_seconds: float
    confidence: float
    # Alternative CTC labels at the phone's strongest frame.
    alternatives: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class DirectPlannedAlias:
    """A real voicebank unit anchored to direct acoustic phone detections."""

    alias: str
    unit: VoicebankUnit
    phone_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DirectPhoneticRecognition:
    phones: tuple[DetectedPhone, ...]
    aliases: tuple[DirectPlannedAlias, ...]
    unmapped_phones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectRecognitionWindow:
    """Prepared model input plus the portion that contains real audio."""

    samples: FloatArray
    valid_start_seconds: float
    valid_end_seconds: float


def canonicalize_direct_audio(
    samples: npt.ArrayLike,
    sample_rate: int,
) -> FloatArray:
    """Convert direct-IPA input to the model's mono 16 kHz waveform format."""
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[1] < 1:
            raise ValueError("audio must contain at least one channel")
        waveform = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
    elif waveform.ndim != 1:
        raise ValueError("audio must be mono or a channel-last matrix")
    if not len(waveform):
        raise ValueError("audio must be non-empty")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("audio must contain only finite samples")
    if sample_rate == DIRECT_SAMPLE_RATE:
        return waveform.copy()
    divisor = math.gcd(sample_rate, DIRECT_SAMPLE_RATE)
    resampled = resample_poly(
        waveform,
        DIRECT_SAMPLE_RATE // divisor,
        sample_rate // divisor,
    )
    return np.asarray(resampled, dtype=np.float32)


def prepare_direct_window(
    samples: FloatArray,
    sample_rate: int,
    *,
    valid_start_seconds: float = 0.0,
    valid_end_seconds: float | None = None,
    minimum_context_seconds: float = DIRECT_MIN_CONTEXT_SECONDS,
) -> DirectRecognitionWindow:
    """Canonicalize and right-pad one direct-IPA recognition window.

    Live and offline recognition must feed the model the same kind of input.  The
    model always receives a finite mono ``float32`` waveform, padded to a small
    minimum context when necessary.  ``valid_*`` describes the unpadded audio;
    callers can therefore discard phones created from right padding or context.
    """
    if sample_rate != DIRECT_SAMPLE_RATE:
        raise ValueError(
            f"direct IPA recognizer requires {DIRECT_SAMPLE_RATE} Hz audio"
        )
    if not math.isfinite(minimum_context_seconds) or minimum_context_seconds <= 0:
        raise ValueError("minimum_context_seconds must be finite and positive")
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("direct IPA recognizer requires a non-empty mono waveform")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("direct IPA recognizer input must contain only finite samples")

    duration_seconds = len(waveform) / sample_rate
    valid_end = duration_seconds if valid_end_seconds is None else valid_end_seconds
    if (
        not math.isfinite(valid_start_seconds)
        or not math.isfinite(valid_end)
        or valid_start_seconds < 0
        or valid_end <= valid_start_seconds
        or valid_end > duration_seconds + 1e-9
    ):
        raise ValueError("valid audio range must be inside the supplied waveform")

    minimum_samples = max(1, round(minimum_context_seconds * sample_rate))
    if len(waveform) < minimum_samples:
        waveform = np.pad(waveform, (0, minimum_samples - len(waveform)))
        waveform = np.asarray(waveform, dtype=np.float32)
    return DirectRecognitionWindow(waveform, valid_start_seconds, valid_end)


def stable_phone_indices(
    previous: tuple[DetectedPhone, ...],
    current: tuple[DetectedPhone, ...],
    *,
    match_tolerance_seconds: float = 0.16,
    boundary_tolerance_seconds: float = 0.08,
) -> tuple[int, ...]:
    """Return current-phone indices supported by the preceding hypothesis.

    The alignment is ordered and one-to-one, so repeated phones cannot all
    collapse onto the first matching label.  A match is considered stable only
    when both its label and estimated boundaries remain close; the looser anchor
    tolerance lets the decoder align a phone before making the stricter commit
    decision.
    """
    if not previous or not current:
        return ()
    if (
        not math.isfinite(match_tolerance_seconds)
        or match_tolerance_seconds <= 0
        or not math.isfinite(boundary_tolerance_seconds)
        or boundary_tolerance_seconds <= 0
    ):
        raise ValueError("phone alignment tolerances must be finite and positive")

    def match_score(left: DetectedPhone, right: DetectedPhone) -> tuple[float, bool] | None:
        if left.ipa != right.ipa:
            return None
        left_anchor = (left.start_seconds + left.end_seconds) / 2
        right_anchor = (right.start_seconds + right.end_seconds) / 2
        anchor_error = abs(left_anchor - right_anchor)
        duration = max(
            left.end_seconds - left.start_seconds,
            right.end_seconds - right.start_seconds,
        )
        anchor_limit = max(match_tolerance_seconds, duration * 0.75)
        if anchor_error > anchor_limit:
            return None
        boundary_error = max(
            abs(left.start_seconds - right.start_seconds),
            abs(left.end_seconds - right.end_seconds),
        )
        boundary_limit = max(boundary_tolerance_seconds, duration * 0.5)
        stable = boundary_error <= boundary_limit
        # Matching labels is more important than tiny timing differences, while
        # still preferring the closest ordered candidate for repeated phones.
        score = 1.0 - min(anchor_error / anchor_limit, 1.0) * 0.25
        return score, stable

    rows = len(previous)
    columns = len(current)
    scores = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    matches: list[list[tuple[int, int] | None]] = [
        [None] * (columns + 1) for _ in range(rows + 1)
    ]
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best_score = max(scores[row - 1][column], scores[row][column - 1])
            best_match: tuple[int, int] | None = None
            candidate = match_score(previous[row - 1], current[column - 1])
            if candidate is not None:
                candidate_score, _ = candidate
                diagonal_score = scores[row - 1][column - 1] + candidate_score
                if diagonal_score >= best_score:
                    best_score = diagonal_score
                    best_match = (row - 1, column - 1)
            scores[row][column] = best_score
            matches[row][column] = best_match

    aligned: list[tuple[int, int]] = []
    row, column = rows, columns
    while row and column:
        match = matches[row][column]
        if match is not None:
            aligned.append(match)
            row -= 1
            column -= 1
        elif scores[row - 1][column] >= scores[row][column - 1]:
            row -= 1
        else:
            column -= 1
    aligned.reverse()
    stable: list[int] = []
    for previous_index, current_index in aligned:
        candidate = match_score(previous[previous_index], current[current_index])
        if candidate is not None and candidate[1]:
            stable.append(current_index)
    stable_set = set(stable)
    prefix: list[int] = []
    for current_index in range(len(current)):
        if current_index not in stable_set:
            break
        prefix.append(current_index)
    return tuple(prefix)


class DirectIPARecognizer:
    """CTC phone recognizer that never creates words or invokes G2P."""

    def __init__(self, *, normalize_input: bool = True) -> None:
        # Retain the old keyword for callers while making the trained
        # feature-extractor path mandatory in every mode.
        del normalize_input
        self._model: Any | None = None
        self._labels: dict[int, str] | None = None
        self._feature_extractor: Any | None = None

    def warmup(self) -> None:
        """Load model and vocabulary assets before audio capture begins."""
        self._ensure_model()
        self._ensure_feature_extractor()

    def recognize(
        self,
        samples: npt.ArrayLike,
        sample_rate: int,
        *,
        top_k: int = 4,
        valid_start_seconds: float = 0.0,
        valid_end_seconds: float | None = None,
    ) -> tuple[DetectedPhone, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        model, labels = self._ensure_model()

        canonical_samples = canonicalize_direct_audio(samples, sample_rate)
        window = prepare_direct_window(
            canonical_samples,
            DIRECT_SAMPLE_RATE,
            valid_start_seconds=valid_start_seconds,
            valid_end_seconds=valid_end_seconds,
        )
        extractor = self._ensure_feature_extractor()
        encoded = extractor(
            window.samples,
            sampling_rate=DIRECT_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = encoded.input_values
        attention_mask: Any | None = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            with torch.inference_mode():
                logits = model(input_values).logits[0]
        else:
            with torch.inference_mode():
                logits = model(input_values, attention_mask=attention_mask).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        ids = torch.argmax(probabilities, dim=-1)
        frame_seconds = len(window.samples) / DIRECT_SAMPLE_RATE / len(ids)
        spikes: list[tuple[str, int, float, tuple[tuple[str, float], ...]]] = []
        previous = -1
        for frame, token_tensor in enumerate(ids):
            token = int(token_tensor)
            if token != previous and token != 0:
                values, indices = torch.topk(
                    probabilities[frame], min(top_k + 1, probabilities.shape[-1])
                )
                alternatives = tuple(
                    (labels[int(index)], float(value))
                    for value, index in zip(values, indices, strict=True)
                    if int(index) != 0
                )[:top_k]
                spikes.append(
                    (labels[token], frame, float(probabilities[frame, token]), alternatives)
                )
            previous = token
        phones: list[DetectedPhone] = []
        for index, (ipa, frame, confidence, alternatives) in enumerate(spikes):
            left = 0 if index == 0 else round((spikes[index - 1][1] + frame) / 2)
            right = (
                len(ids)
                if index + 1 == len(spikes)
                else round((frame + spikes[index + 1][1]) / 2)
            )
            phones.append(
                DetectedPhone(
                    ipa,
                    left * frame_seconds,
                    right * frame_seconds,
                    confidence,
                    alternatives,
                )
            )
        clipped: list[DetectedPhone] = []
        for phone in phones:
            start = max(phone.start_seconds, window.valid_start_seconds)
            end = min(phone.end_seconds, window.valid_end_seconds)
            if end <= start:
                continue
            clipped.append(
                DetectedPhone(
                    phone.ipa,
                    start,
                    end,
                    phone.confidence,
                    phone.alternatives,
                )
            )
        return tuple(clipped)

    def _ensure_model(self) -> tuple[Any, dict[int, str]]:
        if self._model is None:
            self._model, vocab_path = _load_model_assets()
            vocabulary = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            self._labels = {int(index): symbol for symbol, index in vocabulary.items()}
        assert self._labels is not None
        return self._model, self._labels

    def _ensure_feature_extractor(self) -> Any:
        if self._feature_extractor is None:
            self._feature_extractor = _load_feature_extractor()
        return self._feature_extractor


def _load_model_assets() -> tuple[Any, str]:
    """Prefer an installed model cache and use the network only on first setup."""
    try:
        model = AutoModelForCTC.from_pretrained(
            _MODEL_ID,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        vocab_path = hf_hub_download(
            _MODEL_ID,
            "vocab.json",
            local_files_only=True,
        )
    except OSError:
        model = AutoModelForCTC.from_pretrained(
            _MODEL_ID,
            low_cpu_mem_usage=True,
        ).eval()
        vocab_path = hf_hub_download(_MODEL_ID, "vocab.json")
    return model, vocab_path


def _load_feature_extractor() -> Any:
    """Load the wav2vec2 waveform normalizer used during model training."""
    try:
        return AutoFeatureExtractor.from_pretrained(  # type: ignore[no-untyped-call]
            _MODEL_ID,
            local_files_only=True,
        )
    except OSError:
        return AutoFeatureExtractor.from_pretrained(_MODEL_ID)  # type: ignore[no-untyped-call]


class DirectAliasPlanner:
    """Resolve direct IPA detections with a detected bank-specific profile."""

    def __init__(self, bank: Voicebank, profile: VoicebankProfile | None = None) -> None:
        self._units = {unit.alias: unit for unit in bank.units}
        self.bank = bank
        self.profile = profile or detect_voicebank_profile(bank).profile

    def plan(self, phones: tuple[DetectedPhone, ...]) -> DirectPhoneticRecognition:
        if self.profile is None:
            return DirectPhoneticRecognition(
                phones,
                (),
                tuple(phone.ipa for phone in phones),
            )
        selected_phones = self._select_voicebank_compatible_path(phones)
        candidates = resolve_alias_candidates(
            tuple(phone.ipa for phone in selected_phones), self.bank, self.profile
        )
        mapped = [self.profile.symbol_for(phone.ipa) for phone in selected_phones]
        covered = {index for candidate in candidates for index in candidate.phone_indices}
        unmapped = tuple(
            phone.ipa
            for index, (phone, symbol) in enumerate(zip(phones, mapped, strict=True))
            if symbol is None or index not in covered
        )
        if not candidates:
            return DirectPhoneticRecognition(selected_phones, (), unmapped)
        anchors = [
            float(
                np.mean(
                    [
                        (
                            selected_phones[index].start_seconds
                            + selected_phones[index].end_seconds
                        )
                        / 2
                        for index in candidate.phone_indices
                    ]
                )
            )
            for candidate in candidates
        ]
        boundaries = [selected_phones[candidates[0].phone_indices[0]].start_seconds]
        boundaries.extend(
            float((left + right) / 2)
            for left, right in zip(anchors, anchors[1:], strict=False)
        )
        boundaries.append(
            selected_phones[candidates[-1].phone_indices[-1]].end_seconds
        )
        planned = tuple(
            DirectPlannedAlias(
                candidate.alias,
                self._units[candidate.alias],
                candidate.phone_indices,
                boundaries[index],
                boundaries[index + 1],
                float(
                    np.mean(
                        [
                            selected_phones[i].confidence
                            for i in candidate.phone_indices
                        ]
                    )
                ),
            )
            for index, candidate in enumerate(candidates)
        )
        return DirectPhoneticRecognition(selected_phones, planned, unmapped)

    def _select_voicebank_compatible_path(
        self, phones: tuple[DetectedPhone, ...]
    ) -> tuple[DetectedPhone, ...]:
        """Prefer a high-probability phone path that maps in this bank."""
        if self.profile is None or not phones:
            return phones
        primary_candidates = resolve_alias_candidates(
            tuple(phone.ipa for phone in phones), self.bank, self.profile
        )
        primary_covered = {
            index
            for candidate in primary_candidates
            for index in candidate.phone_indices
        }
        # Long, already-mappable utterances are the common case.  Avoid
        # repeatedly exploring their alternatives on every rolling update;
        # reserve the beam for short/ambiguous words where it matters most.
        if len(phones) > 16 and len(primary_covered) >= len(phones) * 0.8:
            return phones
        beam: list[tuple[tuple[DetectedPhone, ...], float]] = [((), 0.0)]
        for phone in phones:
            options = (phone.alternatives or ((phone.ipa, phone.confidence),))[:3]
            deduplicated: dict[str, float] = {}
            for ipa, confidence in options:
                deduplicated.setdefault(ipa, confidence)
            expanded: list[tuple[tuple[DetectedPhone, ...], float]] = []
            for path, score in beam:
                for ipa, confidence in deduplicated.items():
                    expanded.append(
                        (
                            path
                            + (
                                DetectedPhone(
                                    ipa,
                                    phone.start_seconds,
                                    phone.end_seconds,
                                    confidence,
                                    phone.alternatives,
                                ),
                            ),
                            score + float(np.log(max(confidence, 1e-6))),
                        )
                    )
            expanded.sort(
                key=lambda item: self._path_score(item[0], item[1]), reverse=True
            )
            beam = expanded[:12]
        best_path, _ = max(
            beam, key=lambda item: self._path_score(item[0], item[1])
        )
        return best_path

    def _path_score(
        self, phones: tuple[DetectedPhone, ...], acoustic_score: float
    ) -> float:
        if self.profile is None:
            return acoustic_score
        candidates = resolve_alias_candidates(
            tuple(phone.ipa for phone in phones), self.bank, self.profile
        )
        covered = {index for candidate in candidates for index in candidate.phone_indices}
        # Inventory coverage dominates small probability differences.
        return acoustic_score + 4.0 * len(covered) + 0.25 * len(candidates)
