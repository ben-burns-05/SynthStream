# Milestone 8 validation

Milestone 8 connects the recognized unit/section timeline to the production voicebank
renderer and writes a complete WAV without requiring realtime audio hardware.

## Real-bank regression

- Input: `tests/fixtures/human/voices_sentence.wav`
- Voicebank: Kikyuune Aiko RockLoud CVVC EN
- Recognition: direct IPA CTC plus the `aiko-cvvc` profile
- Rendering: every voiced timeline section rendered from its real OTO WAV region
- Output rate: 44,100 Hz, the bank's source rate
- Output duration: 3.4 seconds
- Rendered voiced sections: 78
- Silence segments: preserved in the output buffer

The end-to-end test verifies that the output file exists, has the expected duration and rate,
contains nonzero voicebank audio, and is produced from the same timeline used for JSON export.

## CLI

```powershell
synthstream-recognize human.wav voicebank/my-bank `
  --output timeline.json `
  --output-wav voicebank-output.wav
```

For each non-silence timeline segment, synthesis resolves `unit_id` and `section_index`,
renders that exact OTO section to the timeline duration, resamples if necessary, and places it
at the corresponding output sample range. Silence segments are zero-filled. Unsupported banks
can still be rendered from a generic acoustic timeline, but their semantic alias selection
remains explicitly unvalidated.
