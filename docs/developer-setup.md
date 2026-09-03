# Developer setup

## Prerequisites

* Python 3.12+
* `ffmpeg` and `ffprobe` on `PATH` (the real media analysis path shells out
  to both; without them, ingest falls back to a conservative duration
  estimate and detectors log a warning and skip real analysis -- see
  `docs/limitations.md`)
* `tesseract` on `PATH` if you want license plate OCR (`enrichment.plate`,
  on by default) to actually read plate text rather than just detect
  plate-shaped regions -- see below. Without it, `enrich` logs one clear
  warning and skips OCR cleanly; nothing else is affected.
* Optionally Docker, if you'd rather not install these locally (the
  Dockerfile includes both ffmpeg and tesseract-ocr already)

Check ffmpeg and tesseract are available:

```bash
ffmpeg -version && ffprobe -version
tesseract --version
```

**ffmpeg**: Debian/Ubuntu `sudo apt-get install ffmpeg`; macOS `brew
install ffmpeg`; Windows: download a build from
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or
[ffmpeg.org](https://ffmpeg.org/download.html) and add its `bin/` folder to
`PATH`.

**tesseract**: Debian/Ubuntu `sudo apt-get install tesseract-ocr`; macOS
`brew install tesseract`; Windows: install from the
[UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
and either check "Add to PATH" during setup or add the install directory
(typically `C:\Program Files\Tesseract-OCR`) to `PATH` manually, then
restart your terminal. See `docs/local-ai.md`'s "License plate detection
and OCR" section for more.

## Install

```bash
git clone <repo-url>
cd gaggle
pip install -e .[dev]
pre-commit install   # optional but recommended
```

There are no database migrations to run -- `gaggle workspace
init --workspace ./workspace` creates the full directory layout and an empty
SQLite index in one step (see `docs/architecture.md`'s workspace layout).

## Try it against the bundled example media

```bash
gaggle workspace init --workspace ./workspace
gaggle ingest examples/sample_media --workspace ./workspace
gaggle analyze --workspace ./workspace
gaggle review queue --workspace ./workspace
```

`examples/sample_media` contains two short, real, ffmpeg-generated clips
(front camera: quiet -> motion + a horn-like audio spike -> quiet; rear
camera: quiet -> offset motion -> quiet) with no sidecar fixture files, so
this exercises the real OpenCV/scipy detection path end-to-end, not a
canned fixture. See `docs/cli-examples.md` for the full command reference,
including review, preservation, export, timeline queries, and the review
UI.

## Running checks

```bash
ruff check .              # lint
ruff format --check .     # formatting
mypy src                  # type checking (strict mode; scoped to src/, not tests/)
pytest                    # full test suite
pytest --cov=gaggle --cov-report=term-missing   # with coverage
```

Some tests are skipped automatically if `ffmpeg`/`ffprobe` aren't on `PATH`
(`pytestmark = pytest.mark.skipif(...)` at the top of the relevant test
modules) rather than failing -- if you see a block of skips, that's why.

## Docker

```bash
docker build -t gaggle .
docker run --rm -v "$(pwd)/workspace:/workspace" -v "$(pwd)/examples:/examples:ro" \
  gaggle ingest /examples/sample_media --workspace /workspace

docker compose up review-ui   # http://localhost:8000
docker compose --profile cli run --rm cli analyze --workspace /workspace
```

The image installs `ffmpeg` explicitly (see `Dockerfile`) so the real
analysis path works out of the box in the container, not just locally.

## Devcontainer

Open the repo in VS Code with the Dev Containers extension and "Reopen in
Container" -- it builds from the same `Dockerfile` and installs the
recommended `ruff`/`mypy`/`python` extensions automatically.

## Project layout quick reference

See `docs/architecture.md` for the full module map. The short version:
everything lives under `src/gaggle/<subsystem>/`, tests mirror
that structure under `tests/unit/` and `tests/integration/`, and
`AGENTS.md` at the repo root is the working reference for anyone (human or
agent) making changes -- read it before making structural changes.

## Common gotchas

* **`mypy` is scoped to `src/` only**, not `tests/`, in both CI and the
  documented local command above. Test files intentionally aren't held to
  the same strict-mode bar (pytest fixtures like `monkeypatch` and
  `tmp_path` don't always type-check cleanly under `strict = true` without
  extra plugin configuration).
* **The pre-commit `mypy` hook needs `additional_dependencies`** (already
  configured in `.pre-commit-config.yaml`) because the hook runs in its own
  isolated environment, not your project's virtualenv -- without those,
  every import of `pydantic`/`sqlalchemy`/etc. would fail to resolve and
  mypy would report spurious errors.
* **Workspace paths passed to the CLI/API are trusted.** There is currently
  no sandboxing of `--workspace`; run the CLI with the same care you'd give
  any tool that reads/writes arbitrary local paths.
