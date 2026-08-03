# Milestone 9 validation

Milestone 9 makes the segmental beam decoder incremental and adds fixed-lag commitment.

## Streaming behavior

`SegmentalBeamDecoder.stream(lookahead_frames=N)` returns a
`StreamingSegmentalBeamDecoder`. Each `push(FeatureBatch)` call appends new analysis frames,
extends only the newly reachable dynamic-programming endpoints, and retains the existing
beam and score cache. The update exposes:

- `provisional_path`: the current best path, which may change as future frames arrive;
- `committed_segments`: newly emitted sections that are safe to synthesize;
- `committed_path`: the complete prefix emitted so far;
- `decode_result`: current alternatives and beam diagnostics.

Before finalization, a segment is committed only when all surviving hypotheses share it and its
end lies at least `lookahead_frames` frames behind the newest input. `push(..., final=True)` or
`finish()` commits the winning path through the end of the stream.

The `SegmentalBeamDecoder.stream()` convenience method is tuned for live work by default:
candidate starts are limited to 2 per boundary, the active beam to 4 hypotheses, start history
to 80 frames (800 ms), and section duration to 100 frames (1 second). These bounds prevent
dynamic-programming work from growing without limit as a microphone session continues. Passing
`None` for an override restores the corresponding exhaustive offline setting.

The optimized matcher also reuses per-update feature matrices, amortizes callback-buffer
growth, precomputes silence prefix costs, vectorizes trajectory warping, and computes all
feature-component means in one reduction. On the development CPU, the 3.4-second Aiko fixture
takes about 1.3 seconds for bounded streaming decode and about 0.2 seconds for feature
extraction.

## Regression coverage

The decoder tests verify that:

1. Incremental processing produces the same final path as batch decoding.
2. Commitments respect the fixed lookahead lag.
3. An initially attractive path remains uncommitted while a later section can select a
   different path, then the corrected path is committed at finalization.
4. Invalid lookahead settings are rejected.
