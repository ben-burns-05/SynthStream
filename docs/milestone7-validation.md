# Milestone 7 validation

Milestone 7 is accepted only when recorded human speech produces semantically compatible
aliases from a real installed voicebank. A structurally valid timeline of unrelated aliases is
not sufficient.

## Direct-phoneme regression

- Human source: VOiCES speaker `sp0307`, distributed by the official PyTorch Audio tutorial
- Committed input: `tests/fixtures/human/voices_sentence.wav` (3.4 seconds)
- Spoken sentence: “I had that curiosity beside me at this moment”
- Frontend: `facebook/wav2vec2-lv-60-espeak-cv-ft`
- Recognition method: waveform → IPA CTC phones; no words, dictionary, or G2P
- Real Aiko bank: 1,552 units / 4,651 sections
- First selected aliases: `-I`, `h@`, `@d`, `dh@`, `@t`
- Unmapped phones: none

The test requires at least 20 distinct real aliases, verifies every unit against the loaded
bank, and checks continuous timing and trailing silence. Alias transitions are placed from
detected phone boundaries; each OTO section retains its own measured stretch ratio.

## Milestone 7.5 profile subset

The resolver now separates direct IPA recognition from voicebank-format aliasing. It detects
and validates these conventions against the actual OTO inventory:

- Aiko-style English CVVC (`aɪ h æ` → `-I`, `h@`)
- English VCCV (`aɪ h æ` → `-I`, `I h`, `h@` / `@ d`)
- English Presamp/CVVC using singer-local `presamp.ini` (`aɪ h æ` → `- aI`, `aI h{`)

The profile layer reads the declarative vowel, consonant, replacement and priority sections of
`presamp.ini`, queries the complete loaded alias inventory, and only accepts a profile when a
standard English transition probe reaches at least 90% alias coverage. Every candidate alias
is checked against the actual bank; unavailable forms are not fabricated.

The installed CZloid VCCV, Aiko CVVC and TETO English Presamp banks all pass this profile
test. The Japanese TETO bank is deliberately reported as unsupported for English rather than
being silently treated as an English bank.

## Current scope

Arbitrary UTAU aliases remain unsupported until a profile or user-supplied declarative rules
describe their language and alias convention. The generic MFCC/mel segmental fallback still
reports `recognition_mode=acoustic-segmental` and is not semantically validated.
