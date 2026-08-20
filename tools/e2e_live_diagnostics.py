"""Run the live pipeline from a WAV and save timing/transport diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from synthstream.audio import (
    FakeDuplexAudioBackend,
    StreamDiagnosticEvent,
)
from synthstream.live import LiveDiagnosticEvent, LiveVoicebankEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicebank", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Feed the WAV as quickly as possible (useful for stress testing).",
    )
    args = parser.parse_args()

    source, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise SystemExit(f"input must be 16 kHz, got {sample_rate} Hz")
    mono = np.mean(source, axis=1).astype(np.float32, copy=False)
    backend = FakeDuplexAudioBackend()
    engine = LiveVoicebankEngine.from_voicebank(
        args.voicebank,
        backend,
        buffer_duration_seconds=5.0,
    )
    engine.prepare_direct_ipa()
    engine.start(background=True)
    captured: list[np.ndarray] = []
    block_size = engine.stream.block_size
    block_seconds = block_size / sample_rate
    feed_started = time.perf_counter()
    try:
        for start in range(0, len(mono), block_size):
            captured.append(backend.feed(mono[start : start + block_size]))
            if not args.no_realtime:
                time.sleep(block_seconds)
        # Let the worker finish the final recognition window before flushing it.
        if not args.no_realtime:
            time.sleep(max(0.5, block_seconds * 8))
        engine.flush()
        captured.append(backend.feed(np.zeros(block_size * 16, dtype=np.float32)))
    finally:
        engine.stop(flush=True)
    feed_seconds = time.perf_counter() - feed_started

    output = (
        np.concatenate(captured).astype(np.float32, copy=False)
        if captured
        else np.empty(0, dtype=np.float32)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, sample_rate)

    statistics = engine.statistics
    transport = engine.stream.statistics
    events = tuple(
        sorted(
            cast(
                tuple[LiveDiagnosticEvent | StreamDiagnosticEvent, ...],
                (*statistics.diagnostic_events, *transport.diagnostic_events),
            ),
            key=lambda event: event.monotonic_seconds,
        )
    )
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event.kind] = event_counts.get(event.kind, 0) + 1
    report = {
        "input_wav": str(args.input),
        "output_wav": str(args.output),
        "voicebank": str(args.voicebank),
        "input_seconds": len(mono) / sample_rate,
        "output_seconds": len(output) / sample_rate,
        "feed_wall_seconds": feed_seconds,
        "realtime_feed": not args.no_realtime,
        "event_counts": event_counts,
        "engine": {
            key: value
            for key, value in asdict(statistics).items()
            if key != "diagnostic_events"
        },
        "transport": {
            key: value
            for key, value in asdict(transport).items()
            if key != "diagnostic_events"
        },
        "diagnostic_events": [asdict(event) for event in events],
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "event_counts": event_counts}, indent=2))


if __name__ == "__main__":
    main()
