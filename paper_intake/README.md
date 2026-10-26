# Paper and GitHub intake (steps 1-4)

Python 3.10+ and Git required. This standalone CLI accepts a local PDF, a GitHub
repository URL, and an objective. It extracts pages and tables, optionally uses
OCR, optionally analyzes the paper with OpenAI, and inspects a committed Git
snapshot. It never imports or executes repository code, installs its dependencies,
or starts training. It does not yet map paper requirements to code or generate an
execution plan (steps 5 onward).

## Install separately from the project's model environment

From the UAP repository root in PowerShell:

```powershell
py -m venv .venv-intake
.\.venv-intake\Scripts\python.exe -m pip install -r paper_intake/requirements.txt
```

## Local inspection (no API key, no model calls)

```powershell
.\.venv-intake\Scripts\python.exe paper_intake/paper_intake.py `
  --paper "C:\path\to\paper.pdf" `
  --github https://github.com/adarsh-siva/UAP `
  --repo-path . `
  --objective "Identify the method and evaluation setup" `
  --render `
  --output intake-results/run-001
```

Choose a NEW output directory for each run. Omit `--repo-path .` to clone the
GitHub repository into the output directory. Git uses your existing local
credentials; no access token is embedded in the URL or report. For private
repositories, sign in to Git on your computer first. No browser login is started
by the CLI. Network/authentication failures are reported in manifest.json.

`--repo-path` reads HEAD's committed files, not uncommitted changes, and verifies
that origin matches `--github`. It does not fetch or modify your existing clone.
Use `--ref main` or an exact commit SHA to inspect an available local revision.
Remote cloning is shallow: older refs must already be available in a local clone.

## Paper understanding (optional API call)

Set OPENAI_API_KEY securely in your environment, then add these options to the
command above:

```text
--analyze --model YOUR_STRUCTURED_OUTPUT_MODEL
```

Choose a model available to your OpenAI API project that supports Responses
structured outputs. API billing is separate from a ChatGPT subscription. Only
paper text, extracted tables, and your objective are sent to OpenAI; repository
content and rendered images stay local. Calls set store=false. Do not put your
key in source code, a command argument, or a committed file.

Each chunk produces findings in seven categories: method, dataset, preprocessing,
architecture, hyperparameter, evaluation, and result. Each finding includes a
page number and quote. The program checks that quotes occur in that page's
extracted text and rejects unsupported quotes, invalid schemas, refusals, and
incomplete responses. Unreadable pages block analysis until OCR or a searchable PDF is supplied. This checks provenance, not whether the interpretation is
correct. Unknowns are labeled per chunk; another chunk may answer them.

Every page is included. Oversized individual pages fail explicitly instead of
being silently truncated; raise `--chunk-chars` if your model supports the size.
There is no automatic retry that could repeat API charges. On failure, extraction
and repository reports remain available, with an error manifest.

## Scanned PDFs and visual review

For scanned pages, install `pytesseract` in the intake environment and install the
Tesseract executable separately, making it available on PATH. Add `--ocr`.
OCR is used on pages with fewer than 40 embedded text characters. Mixed pages with
partial embedded text may require external OCR preprocessing. The default OCR
language is English. Missing OCR dependencies produce an error rather than fake
text. Without OCR, sparse pages are marked with warnings.

`--render` saves all page PNGs for human verification of tables, figures, and
mathematical notation. Table extraction is best-effort; equations/captions are
preserved in text/images, not reconstructed into a mathematical representation.
The model currently reads extracted text/tables, not images. Verify complex
layouts and formulas visually before relying on them.

## Outputs

- `report.md`: readable summary, entry-point hints, findings, and extraction warnings.
- `manifest.json`: objective, status, inspected commit, page count, error if any.
- `paper.json`: filename/hash, numbered page text, tables, extraction warnings.
- `pages/*.png`: page images when requested.
- `repository.json`: commit, full path inventory, selected text files, Python
  symbols/line numbers, main-guard hints, skipped files with reasons.
- `analysis.json`: evidence-backed paper findings when --analyze succeeds.
- `repository/`: no-checkout clone when no existing clone was supplied.

Repository inspection prioritizes README/dependency manifests, then root files.
It reads common source/config/document extensions, up to 100 KB per file and
2 MB total, reporting omissions. Symlinks, submodules, hidden files, binary
content, and common secret filenames are skipped. These filters are NOT a full
secret detector. Reports may contain private source text; review before sharing.
PDF limits: 100 MiB and 200 pages. The tool is a local CLI for user-supplied files,
not a hardened public upload server.

## Tests

```powershell
.\.venv-intake\Scripts\python.exe -m pip install reportlab
cd paper_intake
..\.venv-intake\Scripts\python.exe -m unittest -v test_paper_intake.py
```

Tests use a generated PDF and temporary Git repository. Model responses are
mocked; no API key, network, GPU, or training packages are required.

API format reference: https://developers.openai.com/api/docs/guides/structured-outputs
