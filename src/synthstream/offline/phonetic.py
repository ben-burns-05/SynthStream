"""English speech recognition and explicit UTAU alias planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import cmudict
import numpy as np
import numpy.typing as npt
import torch
import torchaudio  # type: ignore[import-untyped]

from synthstream.voicebank import Voicebank, VoicebankUnit

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RecognizedWord:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PlannedAlias:
    alias: str
    unit: VoicebankUnit
    phones: tuple[str, ...]
    word: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PhoneticRecognition:
    transcript: str
    words: tuple[RecognizedWord, ...]
    aliases: tuple[PlannedAlias, ...]
    unmapped_words: tuple[str, ...]


class EnglishCTCRecognizer:
    """Lazy wav2vec2 CTC frontend returning timestamped English words."""

    def __init__(self) -> None:
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self._model: torch.nn.Module | None = None

    def recognize(self, samples: FloatArray, sample_rate: int) -> tuple[RecognizedWord, ...]:
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32).copy()).unsqueeze(0)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.bundle.sample_rate
            )
        if self._model is None:
            self._model = self.bundle.get_model().eval()
        with torch.inference_mode():
            emissions, _ = self._model(waveform)
        probabilities = torch.softmax(emissions[0], dim=-1)
        token_ids = torch.argmax(probabilities, dim=-1)
        labels = self.bundle.get_labels()
        frame_seconds = waveform.shape[1] / self.bundle.sample_rate / len(token_ids)

        characters: list[tuple[str, int, int, float]] = []
        run_start = 0
        previous = int(token_ids[0])
        for frame in range(1, len(token_ids) + 1):
            token = int(token_ids[frame]) if frame < len(token_ids) else -1
            if token == previous:
                continue
            if previous != 0:
                score = float(torch.mean(probabilities[run_start:frame, previous]))
                characters.append((labels[previous], run_start, frame, score))
            previous = token
            run_start = frame

        words: list[RecognizedWord] = []
        current: list[tuple[str, int, int, float]] = []
        for item in characters + [("|", 0, 0, 0.0)]:
            if item[0] != "|":
                current.append(item)
                continue
            if current:
                words.append(
                    RecognizedWord(
                        "".join(part[0] for part in current).lower(),
                        current[0][1] * frame_seconds,
                        current[-1][2] * frame_seconds,
                        float(np.mean([part[3] for part in current])),
                    )
                )
                current = []
        if not words:
            return ()
        expanded = list(words)
        if expanded[0].start_seconds <= 0.15:
            expanded[0] = RecognizedWord(
                expanded[0].text, 0.0, expanded[0].end_seconds, expanded[0].confidence
            )
        for index in range(len(expanded) - 1):
            left, right = expanded[index], expanded[index + 1]
            gap = right.start_seconds - left.end_seconds
            if gap <= 0.15:
                boundary = left.end_seconds + gap / 2
                expanded[index] = RecognizedWord(
                    left.text, left.start_seconds, boundary, left.confidence
                )
                expanded[index + 1] = RecognizedWord(
                    right.text, boundary, right.end_seconds, right.confidence
                )
        return tuple(expanded)


class AikoEnglishAliasMap:
    """Map CMU phones into Aiko RockLoud's documented CVVC notation."""

    _VOWELS = {
        "AA": "9",
        "AE": "@",
        "AH": "u",
        "AO": "9",
        "AW": "8",
        "AY": "I",
        "EH": "e",
        "ER": "3",
        "EY": "A",
        "IH": "i",
        "IY": "E",
        "OW": "O",
        "OY": "Q",
        "UH": "6",
        "UW": "o",
    }
    _CONSONANTS = {
        "B": "b", "CH": "ch", "D": "d", "DH": "dh", "F": "f",
        "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l",
        "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r",
        "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v",
        "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
    }

    def __init__(self, bank: Voicebank) -> None:
        self.bank = bank
        self._units = {unit.alias: unit for unit in bank.units}

    @classmethod
    def supports(cls, bank: Voicebank) -> bool:
        aliases = {unit.alias for unit in bank.units}
        if not {"-I", "h@", "@d", "dh@", "@t"}.issubset(aliases):
            return False
        for readme in bank.root.rglob("readme.txt"):
            try:
                contents = readme.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if "kikyuune aiko" in contents and "phonemes guide" in contents:
                return True
        return False

    def plan(self, words: tuple[RecognizedWord, ...]) -> PhoneticRecognition:
        aliases: list[PlannedAlias] = []
        unmapped: list[str] = []
        for word in words:
            phones = _pronunciation(word.text)
            mapped = self._map_word(word, phones) if phones else ()
            if not mapped:
                unmapped.append(word.text)
                continue
            duration = max(word.end_seconds - word.start_seconds, 1e-3)
            for index, (alias, represented) in enumerate(mapped):
                start = word.start_seconds + duration * index / len(mapped)
                end = word.start_seconds + duration * (index + 1) / len(mapped)
                aliases.append(
                    PlannedAlias(
                        alias,
                        self._units[alias],
                        represented,
                        word.text,
                        start,
                        end,
                        word.confidence,
                    )
                )
        return PhoneticRecognition(
            " ".join(word.text for word in words), words, tuple(aliases), tuple(unmapped)
        )

    def _map_word(
        self, word: RecognizedWord, phones: tuple[str, ...]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        del word
        symbols = tuple(self._symbol(phone) for phone in phones)
        if any(symbol is None for symbol in symbols):
            return ()
        usable = tuple(symbol for symbol in symbols if symbol is not None)
        vowels = [index for index, phone in enumerate(phones) if _base_phone(phone) in self._VOWELS]
        if not vowels:
            return ()

        planned: list[tuple[str, tuple[str, ...]]] = []
        for vowel_number, vowel_index in enumerate(vowels):
            vowel = usable[vowel_index]
            previous_vowel = vowels[vowel_number - 1] if vowel_number else -1
            next_vowel = vowels[vowel_number + 1] if vowel_number + 1 < len(vowels) else len(phones)
            onset_symbols = usable[previous_vowel + 1 : vowel_index]
            coda_symbols = usable[vowel_index + 1 : next_vowel]

            if onset_symbols:
                onset_candidates = ["".join(onset_symbols) + vowel]
                if onset_symbols[-1] == "y" and len(onset_symbols) > 1:
                    onset_candidates.append("".join(onset_symbols[:-1]) + "yo")
            elif vowel_number:
                onset_candidates = [usable[previous_vowel] + vowel]
            else:
                onset_candidates = ["-" + vowel, vowel]
            onset_alias = self._first_existing(onset_candidates)
            if onset_alias:
                planned.append((onset_alias, phones[previous_vowel + 1 : vowel_index + 1]))
            if coda_symbols:
                contextual = (vowel, *coda_symbols)
                for pair_index, (left, right) in enumerate(
                    zip(contextual, contextual[1:], strict=False)
                ):
                    coda_alias = self._first_existing([left + right])
                    if coda_alias:
                        start = vowel_index + pair_index
                        planned.append((coda_alias, phones[start : start + 2]))
        return tuple(planned)

    def _symbol(self, phone: str) -> str | None:
        base = _base_phone(phone)
        return self._VOWELS.get(base) or self._CONSONANTS.get(base)

    def _first_existing(self, candidates: list[str]) -> str | None:
        return next((alias for alias in candidates if alias in self._units), None)


def recognize_aiko_english(
    samples: FloatArray,
    sample_rate: int,
    bank: Voicebank,
    recognizer: EnglishCTCRecognizer | None = None,
) -> PhoneticRecognition:
    frontend = recognizer or EnglishCTCRecognizer()
    return AikoEnglishAliasMap(bank).plan(frontend.recognize(samples, sample_rate))


@lru_cache(maxsize=1)
def _dictionary() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def _pronunciation(word: str) -> tuple[str, ...] | None:
    variants = _dictionary().get(re.sub(r"[^a-z']", "", word.lower()))
    return tuple(variants[0]) if variants else None


def _base_phone(phone: str) -> str:
    return re.sub(r"\d", "", phone.upper())
