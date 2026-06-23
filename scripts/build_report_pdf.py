"""Render a Markdown document under ``report/`` to a polished PDF.

The brief asks for a PDF report; this converts a Markdown source into a
print-ready PDF with the figures embedded and the tables styled. It defaults to
``report/REPORT.md`` but accepts any Markdown file, so it also builds the
soutenance guide.

Pipeline: Markdown -> HTML (``markdown`` with table/code extensions) -> PDF
(``weasyprint``).  Relative image links such as ``../results/figures/x.png`` are
resolved against the ``report/`` directory via WeasyPrint's ``base_url``.

Usage::

    pip install markdown weasyprint
    python scripts/build_report_pdf.py                       # -> report/REPORT.pdf
    python scripts/build_report_pdf.py report/GUIDE_SOUTENANCE.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown  # noqa: E402
from weasyprint import HTML  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"

# Minimal, print-oriented stylesheet: A4 pages, readable type, bordered tables
# and figures that never overflow the page width.
_CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 15pt; margin-top: 18px; border-bottom: 1px solid #bbb; padding-bottom: 2px; }
h3 { font-size: 12pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.3pt; }
th, td { border: 1px solid #999; padding: 4px 7px; text-align: left; }
th { background: #f0f0f0; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; height: auto; display: block; margin: 8px auto; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8px; border-radius: 5px; overflow-x: auto;
      font-size: 8.6pt; line-height: 1.3; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #ccc; margin-left: 0; padding-left: 12px; color: #555; }
"""


def build(md_path: Path) -> Path:
    """Convert a Markdown file to a sibling PDF and return the output path."""
    md_path = Path(md_path)
    pdf_path = md_path.with_suffix(".pdf")
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_doc = f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>{html_body}</body></html>"
    # base_url = report/ so that '../results/...' image links resolve correctly.
    HTML(string=html_doc, base_url=str(REPORT_DIR)).write_pdf(str(pdf_path))
    return pdf_path


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT_DIR / "REPORT.md"
    out = build(src)
    print(f"Wrote {out}  ({out.stat().st_size // 1024} KB)")

