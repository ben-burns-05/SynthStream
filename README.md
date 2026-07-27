# SynthStream

SynthStream is an early-stage desktop application for converting live human speech into audio assembled from an UTAU/VOCALOID-style voicebank.

## Development setup

Python 3.11 or newer is required. From a clean virtual environment:

```powershell
python -m pip install -e ".[dev]"
```

Run the application and quality checks with:

```powershell
synthstream
pytest
ruff check .
mypy
```

The immutable initial requirements are in `docs/initial_spec.md`. Implementation decisions and evolving requirements belong in `docs/spec.md`.

## Loading a voicebank

Milestone 1 provides a recursive UTAU `oto.ini` loader:

```python
from synthstream.voicebank import load_voicebank

bank = load_voicebank("voicebank/my-bank")
print(len(bank.units))
```

Parsed metadata is cached inside the selected bank's ignored `.synthstream-cache/` directory. The cache is automatically invalidated when an `oto.ini` file or referenced WAV changes.

## Rendering a voicebank unit

Milestone 2 can render a loaded unit at an independently selected duration and pitch, then route the result through a file or device sink:

```python
from synthstream.audio import WavFileSink
from synthstream.rendering import VoicebankRenderer
from synthstream.voicebank import load_voicebank

unit = load_voicebank("voicebank/my-bank").units[0]
result = VoicebankRenderer().render_unit(
    unit,
    duration_seconds=0.75,
    pitch_ratio=1.25,
)
result.send_to(WavFileSink("rendered.wav"))
```
