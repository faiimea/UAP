"""Read tools for a LangChain agent: read_file and read_paper.

Requires: langchain-core, pypdf (pip install langchain-core pypdf), and curl for PDF URLs.
"""
import os
import re
import subprocess

from langchain_core.tools import tool

# Where read_paper keeps downloaded PDFs and their extracted text.
PAPER_CACHE_DIR = os.environ.get(
    "PAPER_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paper_cache"))


@tool
def read_file(path: str, start_line: int = 1, max_lines: int = 300) -> str:
    """Read lines from a text file (e.g. a cloned repo's training script, to find its
    command-line flags and how it prints metrics)."""
    try:
        with open(os.path.expanduser(path), errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"ERROR: {e}"
    chunk = lines[start_line - 1:start_line - 1 + max_lines]
    return "".join(f"{i}\t{l}" for i, l in enumerate(chunk, start_line)) + \
        (f"\n... ({len(lines)} lines total)" if len(lines) > start_line - 1 + max_lines else "")


@tool
def read_paper(source: str, search: str = "", start_line: int = 0, num_lines: int = 60) -> str:
    """Read a paper PDF (URL or local path) as numbered lines of plain text.
    - No `search` and no `start_line`: returns an outline - every table/figure caption and
      section heading with its line number. Start here.
    - `start_line` (no search): returns `num_lines` lines from there (e.g. a table found in
      the outline).
    - `search`: a case-insensitive regex; returns matching lines with context (only lines at
      or after `start_line`, if given).
    Tables are extracted as whitespace-separated rows with NO '|' separators, e.g.
    'Origin 93.86 100.00 94.60 93.89'. Search for a caption ('Table 1') or a row label,
    not for table markup. Column headers are usually one or two lines above the rows."""
    from pypdf import PdfReader
    cache_dir = PAPER_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", source)[-80:]
    txt_path = os.path.join(cache_dir, key + ".txt")
    if not os.path.exists(txt_path):
        pdf_path = source
        if re.match(r"https?://", source):
            pdf_path = os.path.join(cache_dir, key + ".pdf")
            p = subprocess.run(["curl", "-sSL", "-o", pdf_path, source], capture_output=True, text=True, timeout=300)
            if p.returncode != 0:
                return f"ERROR downloading: {p.stderr[-1000:]}"
        try:
            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf_path).pages)
        except Exception as e:
            return f"ERROR reading PDF: {e}"
        with open(txt_path, "w") as f:
            f.write(text)
    with open(txt_path) as f:
        lines = f.read().splitlines()
    caption_rx = re.compile(r"^\s*(Table|Figure|Fig\.)\s*\d+\s*[:.]|^\s*\d+(\.\d+)*\s+[A-Z][A-Za-z -]{2,60}$")
    outline = [f"{i}: {l.strip()[:150]}" for i, l in enumerate(lines) if caption_rx.search(l)]

    if start_line and not search:
        lo = max(0, start_line)
        # A span that starts just inside a table backs up to its caption, so the column
        # headers always come with the rows.
        for j in range(lo - 1, max(-1, lo - 9), -1):
            if re.match(r"^\s*(Table|Figure|Fig\.)\s*\d+", lines[j]):
                lo = j
                break
        span = lines[lo:lo + max(1, min(num_lines, 200))]
        return "\n".join(f"{i}: {l}" for i, l in enumerate(span, lo)) + f"\n({len(lines)} lines total)"
    if not search:
        return (f"{len(lines)} lines. Outline (captions and headings with line numbers); "
                f"read a span with start_line=<n>:\n" + "\n".join(outline))[:15000]
    try:
        rx = re.compile(search, re.IGNORECASE)
    except re.error as e:
        return f"ERROR: invalid regex: {e}"
    hits = [i for i, l in enumerate(lines) if i >= max(0, start_line) and rx.search(l)]
    if not hits:
        tables = [o for o in outline if re.search(r":\s*(Table|Figure|Fig\.)", o)]
        return (f"no lines match {search!r}. Remember tables have no '|' separators. "
                f"Captions in this paper (read one with start_line=<n>):\n" + "\n".join(tables))[:15000]
    # Context windows around hits, merged when they touch or nearly touch, so a table row
    # between two windows is never silently dropped.
    blocks = []
    for i in hits[:40]:
        lo, hi = max(0, i - 3), min(len(lines), i + 12)
        if blocks and lo <= blocks[-1][1] + 3:
            blocks[-1][1] = max(blocks[-1][1], hi)
        else:
            blocks.append([lo, hi])
    out = []
    for lo, hi in blocks:
        out.extend(f"{j}: {lines[j]}" for j in range(lo, hi))
        out.append("---")
    return "\n".join(out)[:15000] or "no matches"
