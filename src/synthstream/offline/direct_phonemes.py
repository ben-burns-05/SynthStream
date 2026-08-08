"""Direct audio-to-IPA recognition and inventory-aware alias planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoModelForCTC

from synthstream.offline.voicebank_phonemizer import (
    VoicebankProfile,
    detect_voicebank_profile,
    resolve_alias_candidates,
)
from synthstream.voicebank import Voicebank, VoicebankUnit

FloatArray = npt.NDArray[np.float32]
_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


@dataclass(frozen=True, slots=True)
class DetectedPhone:
    ipa: str
    start_seconds: float
    end_seconds: float
    confidence: float


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


class DirectIPARecognizer:
    """CTC phone recognizer that never creates words or invokes G2P."""

    def __init__(self, *, normalize_input: bool = False) -> None:
        self._model: Any | None = None
        self._labels: dict[int, str] | None = None
        self._normalize_input = normalize_input
        self._feature_extractor: Any | None = None

    def warmup(self) -> None:
        """Load model and vocabulary assets before audio capture begins."""
        self._ensure_model()
        if self._normalize_input:
            self._ensure_feature_extractor()

    def recognize(self, samples: FloatArray, sample_rate: int) -> tuple[DetectedPhone, ...]:
        if sample_rate != 16_000:
            raise ValueError("direct IPA recognizer requires 16 kHz audio")
        model, labels = self._ensure_model()

        waveform = np.asarray(samples, dtype=np.float32).copy()
        attention_mask: Any | None = None
        if self._normalize_input:
            extractor = self._ensure_feature_extractor()
            encoded = extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = encoded.input_values
            attention_mask = getattr(encoded, "attention_mask", None)
        else:
            input_values = torch.from_numpy(waveform).unsqueeze(0)
        with torch.inference_mode():
            if attention_mask is None:
                logits = model(input_values).logits[0]
            else:
                logits = model(input_values, attention_mask=attention_mask).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        ids = torch.argmax(probabilities, dim=-1)
        frame_seconds = len(samples) / sample_rate / len(ids)
        spikes: list[tuple[str, int, float]] = []
        previous = -1
        for frame, token_tensor in enumerate(ids):
            token = int(token_tensor)
            if token != previous and token != 0:
                spikes.append((labels[token], frame, float(probabilities[frame, token])))
            previous = token
        phones: list[DetectedPhone] = []
        for index, (ipa, frame, confidence) in enumerate(spikes):
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
                )
            )
        return tuple(phones)

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
        candidates = resolve_alias_candidates(
            tuple(phone.ipa for phone in phones),
            self.bank,
            self.profile,
        )
        mapped = [self.profile.symbol_for(phone.ipa) for phone in phones]
        covered = {index for candidate in candidates for index in candidate.phone_indices}
        unmapped = tuple(
            phone.ipa
            for index, (phone, symbol) in enumerate(zip(phones, mapped, strict=True))
            if symbol is None or index not in covered
        )
        if not candidates:
            return DirectPhoneticRecognition(phones, (), unmapped)
        anchors = [
            float(
                np.mean(
                    [
                        (phones[index].start_seconds + phones[index].end_seconds) / 2
                        for index in candidate.phone_indices
                    ]
                )
            )
            for candidate in candidates
        ]
        boundaries = [phones[candidates[0].phone_indices[0]].start_seconds]
        boundaries.extend(
            float((left + right) / 2)
            for left, right in zip(anchors, anchors[1:], strict=False)
        )
        boundaries.append(phones[candidates[-1].phone_indices[-1]].end_seconds)
        planned = tuple(
            DirectPlannedAlias(
                candidate.alias,
                self._units[candidate.alias],
                candidate.phone_indices,
                boundaries[index],
                boundaries[index + 1],
                float(np.mean([phones[i].confidence for i in candidate.phone_indices])),
            )
            for index, candidate in enumerate(candidates)
        )
        return DirectPhoneticRecognition(phones, planned, unmapped)


class AikoDirectAliasPlanner(DirectAliasPlanner):
    """Compatibility wrapper for the original Aiko-specific public class."""

    def __init__(self, bank: Voicebank) -> None:
        capability = detect_voicebank_profile(bank)
        super().__init__(bank, capability.profile)
