"""Direct audio-to-IPA recognition and Aiko CVVC alias planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCTC

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
class DirectPhoneticRecognition:
    phones: tuple[DetectedPhone, ...]
    aliases: tuple[DirectPlannedAlias, ...]
    unmapped_phones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectPlannedAlias:
    """A real voicebank unit anchored to direct acoustic phone detections."""

    alias: str
    unit: VoicebankUnit
    phone_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    confidence: float


class DirectIPARecognizer:
    """CTC phone recognizer that never creates words or invokes G2P."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._labels: dict[int, str] | None = None

    def recognize(self, samples: FloatArray, sample_rate: int) -> tuple[DetectedPhone, ...]:
        if sample_rate != 16_000:
            raise ValueError("direct IPA recognizer requires 16 kHz audio")
        if self._model is None:
            self._model, vocab_path = _load_model_assets()
            vocabulary = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            self._labels = {int(index): symbol for symbol, index in vocabulary.items()}

        waveform = torch.from_numpy(
            np.asarray(samples, dtype=np.float32).copy()
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(waveform).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        ids = torch.argmax(probabilities, dim=-1)
        frame_seconds = len(samples) / sample_rate / len(ids)
        spikes: list[tuple[str, int, float]] = []
        previous = -1
        for frame, token_tensor in enumerate(ids):
            token = int(token_tensor)
            if token != previous and token != 0:
                assert self._labels is not None
                spikes.append((self._labels[token], frame, float(probabilities[frame, token])))
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


class AikoDirectAliasPlanner:
    """Translate detected IPA phones into existing Aiko CVVC recordings."""

    _SYMBOLS = {
        "aɪ": "I", "æ": "@", "ɑː": "9", "ɑ": "9", "ə": "u",
        "ɐ": "u", "ʌ": "u", "ɪ": "i", "i": "E", "iː": "E",
        "u": "o", "uː": "o", "oʊ": "O", "ɛ": "e", "ɜː": "3",
        "h": "h", "d": "d", "ɾ": "d", "ð": "dh", "t": "t",
        "k": "k", "j": "y", "ɹ": "r", "r": "r", "s": "s",
        "b": "b", "m": "m", "n": "n", "ŋ": "ng", "l": "l",
        "w": "w", "f": "f", "v": "v", "θ": "th", "ʃ": "sh",
        "ʒ": "zh", "p": "p", "ɡ": "g",
    }
    _VOWELS = {"I", "@", "9", "u", "i", "E", "o", "O", "e", "3"}

    def __init__(self, bank: Voicebank) -> None:
        self._units = {unit.alias: unit for unit in bank.units}

    def plan(self, phones: tuple[DetectedPhone, ...]) -> DirectPhoneticRecognition:
        mapped = [self._SYMBOLS.get(phone.ipa) for phone in phones]
        unmapped = tuple(
            phone.ipa
            for phone, symbol in zip(phones, mapped, strict=True)
            if symbol is None
        )
        candidates: list[tuple[str, tuple[int, ...]]] = []
        vowel_indices = [index for index, symbol in enumerate(mapped) if symbol in self._VOWELS]
        for number, vowel_index in enumerate(vowel_indices):
            vowel = mapped[vowel_index]
            assert vowel is not None
            previous_vowel = vowel_indices[number - 1] if number else -1
            onset_indices = tuple(
                index for index in range(previous_vowel + 1, vowel_index + 1) if mapped[index]
            )
            if onset_indices[:-1]:
                suffixes = [
                    (
                        "".join(mapped[index] or "" for index in onset_indices[start:]),
                        onset_indices[start:],
                    )
                    for start in range(len(onset_indices) - 1)
                ]
            else:
                suffixes = [("-" + vowel, onset_indices), (vowel, onset_indices)]
            selected = next(
                ((alias, indices) for alias, indices in suffixes if alias in self._units),
                None,
            )
            if selected is not None:
                candidates.append(selected)
            next_vowel = (
                vowel_indices[number + 1]
                if number + 1 < len(vowel_indices)
                else len(phones)
            )
            if vowel_index + 1 < next_vowel and mapped[vowel_index + 1]:
                alias = vowel + (mapped[vowel_index + 1] or "")
                if alias in self._units:
                    candidates.append((alias, (vowel_index, vowel_index + 1)))

        if not candidates:
            return DirectPhoneticRecognition(phones, (), unmapped)
        anchors = [
            np.mean(
                [
                    (phones[index].start_seconds + phones[index].end_seconds) / 2
                    for index in indices
                ]
            )
            for _, indices in candidates
        ]
        boundaries = [phones[candidates[0][1][0]].start_seconds]
        boundaries.extend(
            float((left + right) / 2)
            for left, right in zip(anchors, anchors[1:], strict=False)
        )
        boundaries.append(phones[candidates[-1][1][-1]].end_seconds)
        planned = tuple(
            DirectPlannedAlias(
                alias,
                self._units[alias],
                indices,
                boundaries[index],
                boundaries[index + 1],
                float(np.mean([phones[i].confidence for i in indices])),
            )
            for index, (alias, indices) in enumerate(candidates)
        )
        return DirectPhoneticRecognition(phones, planned, unmapped)
