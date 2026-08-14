# Milestone 10 validation

Milestone 10 connects microphone input through direct IPA recognition, inventory-aware alias
planning, voicebank rendering, and output buffering.

## Engine pipeline

`LiveVoicebankEngine` owns a `RealtimeAudioStream`, the rolling wav2vec2 direct-IPA frontend,
the detected voicebank profile/alias planner, and a `VoicebankRenderer`. The PortAudio callback
remains bounded and does not run model, matcher, or rendering work. A background worker calls
`process_available()` and queues only committed sections to the output ring buffer.

Direct mode is the default and requires a supported English CVVC/VCCV/Presamp profile. It keeps
a 1.2-second recognition window, updates every 0.8 seconds, and commits aliases after a 0.4
second stability lag. The live transport defaults to four seconds of bounded input/output
buffering so model-inference bursts do not immediately drop microphone or rendered samples.
Each committed alias expands into its real OTO-derived onset, transition,
and sustain sections. `use_direct_ipa=False` explicitly selects the optimized Milestone 9
acoustic matcher for banks without a direct profile.

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

The first live engine uses short input chunks and requires the stream and analysis sample rates
to match. Voicebank recordings are resampled to the duplex stream rate before output. On the
development CPU, a 3.4-second Aiko fixture completes direct-IPA recognition and rendering in
about 2.3 seconds. Milestone 11 adds OTO-aware overlap and a bounded live
staging tail; see `docs/milestone11-validation.md` for the overlap guarantees.

Fake-duplex tests verify that real Aiko voicebank-derived audio crosses the complete direct-IPA
production path, that the acoustic fallback and background worker start cleanly, and that
invalid rate pairings are rejected before the stream starts.
