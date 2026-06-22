"""Render ``report/SLIDES.md`` into a landscape slide deck ``report/SLIDES.pdf``.

The brief requires a 20-minute presentation; this turns a lightweight Markdown
source -- slides separated by a line containing only ``---`` -- into a
print-ready, one-slide-per-page PDF, reusing the project's figures.

Usage::

    pip install markdown weasyprint
    python scripts/build_slides_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown  # noqa: E402
from weasyprint import HTML  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
MD_PATH = REPORT_DIR / "SLIDES.md"
PDF_PATH = REPORT_DIR / "SLIDES.pdf"

# Landscape 4:3 slides (extra vertical room so table+figure slides fit on ONE
# page), large type, one slide per printed page.
_CSS = """
@page { size: 25.4cm 19cm; margin: 0.9cm 1.2cm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; color: #1a1a1a; }
.slide { page-break-after: always; }
.slide:last-child { page-break-after: auto; }
h1 { font-size: 26pt; color: #1f3b6e; margin: 0 0 6px; }
h2 { font-size: 19pt; color: #1f3b6e; border-bottom: 2px solid #1f3b6e;
     padding-bottom: 3px; margin: 0 0 8px; }
h3 { font-size: 13pt; color: #444; margin: 3px 0; }
li { font-size: 12.5pt; line-height: 1.4; margin: 1px 0; }
p  { font-size: 12.5pt; line-height: 1.35; margin: 4px 0; }
table { border-collapse: collapse; width: 100%; font-size: 10.5pt; margin: 4px 0; }
th, td { border: 1px solid #999; padding: 3px 6px; text-align: left; }
th { background: #1f3b6e; color: #fff; }
tr:nth-child(even) td { background: #f3f6fb; }
img { max-width: 82%; max-height: 8.6cm; display: block; margin: 3px auto; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 11pt; }
strong { color: #1f3b6e; }
.center { text-align: center; }
"""


def build() -> Path:
    """Split SLIDES.md into per-slide pages and render the deck to PDF."""
    raw = MD_PATH.read_text(encoding="utf-8")
    # Slides are separated by a line that is exactly '---'.
    chunks = [c.strip() for c in raw.split("\n---\n") if c.strip()]
    slides_html = "".join(
        f"<div class='slide'>{markdown.markdown(c, extensions=['tables', 'sane_lists'])}</div>"
        for c in chunks
    )
    html_doc = f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>{slides_html}</body></html>"
    HTML(string=html_doc, base_url=str(REPORT_DIR)).write_pdf(str(PDF_PATH))
    return PDF_PATH


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}  ({out.stat().st_size // 1024} KB)")
