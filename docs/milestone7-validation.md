# Milestone 7 validation

Milestone 7 is accepted only when recorded human speech produces semantically compatible
aliases from a real installed voicebank. A structurally valid timeline of unrelated aliases is
not sufficient.

## Committed regression

- Human source: VOiCES speaker `sp0307`, distributed by the official PyTorch Audio tutorial
- Committed input: `tests/fixtures/human/voices_excerpt.wav`
- Spoken text: `I had that`
- Recognized text: `i had that`
- Real bank: Kikyuune Aiko RockLoud CVVC EN, 1,552 units / 4,651 sections
- Selected aliases: `-I`, `h@`, `@d`, `dh@`, `@t`
- Unmapped words: none

The test also requires all selected aliases and unit IDs to exist in the loaded bank, continuous
section timing, and correctly placed trailing silence.

## Independent-speaker diagnostic

The same production frontend was also run against PyTorch Audio's LibriSpeech `test-other`
sample `1688-142285-0007.wav`:

<https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav>

Reference text:

> I really was very much afraid of showing him how much shocked I was at some parts of what he said.

Production recognition:

> i really was very much afraid of showing him how much shocked i was at some part of what he said

The singular `part` is the model's only word difference. CMUdict mapped every recognized word;
the planner produced 120 real section/silence segments across 31 distinct Aiko aliases, with
leading and trailing silence preserved. The asset is not copied into the repository, so this is a
recorded reproducible diagnostic rather than an offline test dependency.

## Supported scope

Semantic mapping currently applies only when the loaded bank is verified as Kikyuune Aiko
RockLoud CVVC EN by both its alias inventory and its phoneme-guide readme. Other banks retain
the acoustic segmental fallback and explicitly report `recognition_mode=acoustic-segmental`;
that fallback is not considered semantically validated.
