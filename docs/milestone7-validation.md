# Milestone 7 validation

Milestone 7 is accepted only when recorded human speech produces semantically compatible
aliases from a real installed voicebank. A structurally valid timeline of unrelated aliases is
not sufficient.

## Committed direct-phoneme regression

- Human source: VOiCES speaker `sp0307`, distributed by the official PyTorch Audio tutorial
- Committed input: `tests/fixtures/human/voices_sentence.wav` (3.4 seconds)
- Spoken sentence: “I had that curiosity beside me at this moment”
- Frontend: `facebook/wav2vec2-lv-60-espeak-cv-ft`
- Recognition method: waveform → IPA CTC phones; no words, dictionary, or G2P
- Direct output: `aɪ h æ d ð æ t k j uː ɹ ɪ ɑː s ɪ ɾ i b ɪ s aɪ d m i æ t ð ɪ s m oʊ m ə n t`
- Real bank: Kikyuune Aiko RockLoud CVVC EN, 1,552 units / 4,651 sections
- First selected aliases: `-I`, `h@`, `@d`, `dh@`, `@t`
- Unmapped phones: none

The test requires at least 20 distinct real aliases, verifies every unit against the loaded
bank, and checks continuous timing and trailing silence. Alias transitions are placed from
detected phone boundaries; each OTO section retains its own measured stretch ratio.

## Current scope

The verified IPA-to-alias map currently targets Kikyuune Aiko RockLoud CVVC EN. Other
banks retain the older MFCC/mel segmental fallback and report
`recognition_mode=acoustic-segmental`; that fallback is not semantically validated.
