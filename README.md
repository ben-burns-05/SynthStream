# SynthStream

SynthStream converts live human speech into audio assembled from an UTAU-style
voicebank. It is an experimental application: the result follows the speaker's
phonemes and approximate pitch, but it is not a conventional speech-to-text system.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or newer
- A microphone and an output device
- A supported English UTAU-style voicebank with `oto.ini` and its WAV recordings

The first use downloads approximately 1.3 GB of phoneme-model files. Later runs
reuse the local model cache.

## Install

From the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

## Start the live converter

On Windows, double-click `run_synthstream.bat`, or run:

```powershell
.venv\Scripts\python.exe -m synthstream
```

In the window:

1. Select the voicebank folder.
2. Select the microphone and output device.
3. Click **Start**.
4. Wait for the model-loading message to finish, then speak.
5. Click **Stop** when finished.

## Supported voicebanks

The default live mode currently supports these English alias conventions:

- Aiko-style English CVVC
- English VCCV
- English Presamp/CVVC banks containing `presamp.ini`

Tested examples include:

- `voicebank/Kikyuune Aiko RockLoud CVVC EN`
- `voicebank/TETO-English-150401`

Other UTAU banks may load and render, but unsupported alias conventions will not
have reliable direct phoneme recognition.

## Convert a prerecorded WAV

To create a timeline and a synthesized voicebank WAV:

```powershell
.venv\Scripts\python.exe -m synthstream.offline human.wav voicebank/TETO-English-150401 `
  --output timeline.json `
  --output-wav voicebank-output.wav
```

The input WAV can be mono or multichannel. The output is assembled from the
selected voicebank recordings, with timing and pitch transferred from the input.

## Diagnostics

The GUI provides two useful buttons:

- **Save diagnostics...** saves counters, device information, buffer levels, and errors.
- **Save recent mic WAV...** saves the exact microphone audio received by the worker.

Use these files when recognition is poor or there is no output. Check that the
selected devices are correct and that the input is not muted or clipping.

## Current limitations

- Recognition detects IPA phonemes directly; it does not produce words or use G2P.
- Short isolated words are less reliable than connected speech.
- Pitch is estimated once per alias and quantized to the nearest musical note.
- The voicebank must contain the aliases required by its detected convention.

Developer specifications, milestone notes, and test documentation are in `docs/`.
