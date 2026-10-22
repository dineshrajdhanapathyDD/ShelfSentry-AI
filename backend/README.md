# ShelfSentry AI - backend

Python backend for ShelfSentry AI: a bounded active-perception shelf
inspection system pairing substantive OpenCV 5 computer vision with a Strands
agent loop that reaches out through a restricted MCP tool server.

The backend is built **local/mock first**: the full pipeline (real OpenCV 5,
the agent loop, and MCP tools) runs offline with no AWS credentials. AWS Mode
is a verified follow-on.

## Package layout

```
shelfsentry/
  vision/    # OpenCV 5 validate / preprocess / analyze / score / annotate / video / version
  agent/     # bounded active-perception loop, decision-model abstraction, fallback policy
  mcp/       # restricted MCP vision-tool server
  data/      # typed Pydantic v2 contracts + StorageAdapter persistence
  api/        # mode-aware, versioned request handlers
  config/    # typed config + Local/AWS mode switch
  eval/      # labeled evaluation set + honest reporting runner
tests/       # mirrors the package layout
```

## Setup

From this `backend/` directory, with Python >= 3.11:

```bash
python -m pip install -e ".[test]"
# or, equivalently, from the pinned list:
python -m pip install -r requirements.txt
```

All dependency versions are pinned exactly (see `pyproject.toml` /
`requirements.txt`). The exact OpenCV version is documented there.

### Pinned versions (verified to install/import in this environment)

| Dependency | Pin | Notes |
| --- | --- | --- |
| opencv-python | `5.0.0.93` | `cv2.__version__ == 5.0.0` - the exact **OpenCV 5** version used |
| numpy | `2.2.6` | |
| pydantic | `2.12.5` | Pydantic v2 |
| strands-agents | `1.48.0` | Strands Agents SDK |
| mcp | `1.28.1` | Python MCP SDK |
| pytest | `9.1.1` | test runner |
| hypothesis | `6.170.2` | property-based testing |

> **OpenCV 5:** `opencv-python==5.0.0.93` is the only 5.x wheel currently
> published on PyPI and it installs on this environment's Python
> (3.13.15, win32), reporting `cv2.__version__ == 5.0.0`. We pin it exactly
> and never substitute an OpenCV 4 release. The runtime version is also
> verified at startup (`shelfsentry/vision/version.py`, delivered in a later
> task).

## Running tests

Single run (no watch mode); property tests generate >= 100 examples each via
the Hypothesis profile loaded in `tests/conftest.py`:

```bash
python -m pytest
```
