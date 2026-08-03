# Milestone 10 validation

Milestone 10 connects microphone input through analysis, incremental decoding, voicebank
rendering, and output buffering.

## Engine pipeline

`LiveVoicebankEngine` owns a `RealtimeAudioStream`, a feature extractor, a section matcher, a
Milestone 9 streaming decoder, and a `VoicebankRenderer`. The PortAudio callback remains
bounded and does not run model, matcher, or rendering work. A background worker calls
`process_available()` and queues only committed sections to the output ring buffer.

The same engine can be driven deterministically without a worker:

```python
engine.start(background=False)
backend.feed(microphone_samples)
engine.process_available()
engine.flush()
engine.stop()
```

`flush()` commits the final decoder path and renders any pending section. `LiveEngineStatistics`
reports processed input blocks, feature frames, committed segments, rendered output samples,
processing time, and worker errors.

## Scope and latency

The first live engine uses short analysis chunks and requires the stream and analysis sample
rates to match. Voicebank recordings are resampled to the duplex stream rate before output.
The bounded Milestone 9 streaming defaults provide the live latency guardrails; overlap-aware
rendering remains Milestone 11 work.

Fake-duplex tests verify that real voicebank-derived audio crosses the complete production
buffers, that the background worker starts and flushes cleanly, and that invalid rate pairings
are rejected before the stream starts.
