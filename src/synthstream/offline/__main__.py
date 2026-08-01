"""Command-line entry point for offline timeline recognition."""

import argparse
import sys
from collections.abc import Sequence

from synthstream.offline.recognizer import recognize_wav


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recognize a human WAV with a voicebank")
    parser.add_argument("human_wav", help="human speech WAV to analyze")
    parser.add_argument("voicebank", help="UTAU voicebank directory")
    parser.add_argument("--output", "-o", required=True, help="timeline JSON output path")
    arguments = parser.parse_args(argv)

    timeline = recognize_wav(
        arguments.human_wav,
        arguments.voicebank,
        output_json=arguments.output,
    )
    print(
        f"Recognized {len(timeline.segments)} segments "
        f"from {timeline.input_duration_seconds:.3f}s of audio "
        f"with {timeline.recognition_mode}"
    )
    if timeline.transcript:
        print(f"Transcript: {timeline.transcript}")
    if timeline.unmapped_words:
        print(f"Unmapped words: {', '.join(timeline.unmapped_words)}")
    if timeline.detected_phonemes:
        print(_console_safe(f"Detected phonemes: {' '.join(timeline.detected_phonemes)}"))
    if timeline.unmapped_phonemes:
        print(f"Unmapped phonemes: {', '.join(timeline.unmapped_phonemes)}")
    return 0


def _console_safe(value: str) -> str:
    """Keep IPA diagnostics printable on legacy Windows console encodings."""
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


if __name__ == "__main__":
    raise SystemExit(main())
