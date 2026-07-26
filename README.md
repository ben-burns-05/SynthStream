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

