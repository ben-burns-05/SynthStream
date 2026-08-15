# Human speech fixture

`voices_excerpt.wav` is a normalized 0.5-second excerpt beginning 0.6 seconds into `Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`.

The source recording is from the VOiCES corpus and is distributed under the Creative Commons Attribution 4.0 license. It is also distributed by the official PyTorch Audio tutorials at:

https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav

The excerpt is retained solely as a compact recorded-human-speech regression fixture.

`voices_sentence.wav` is the complete 3.4-second source recording. It says “I had
that curiosity beside me at this moment” and is retained for direct phoneme
recognition and real-voicebank integration tests under the same license.

The live short-word test cuts four labelled word windows from that recording,
with 40 ms of context on each side, then inserts silence between them:

| Word | Source interval (seconds) |
| --- | ---: |
| `I` | 0.624–0.704 |
| `ME` | 2.374–2.495 |
| `AT` | 2.535–2.595 |
| `THIS` | 2.635–2.796 |

These intervals come from the published forced-alignment example for this exact
VOiCES recording: <https://jalalal-tamami.github.io/Tutorial_WebMAUS_wav2vec2_whisper/forced_alignment_tutorial_JAT.html>.
