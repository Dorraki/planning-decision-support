"""Extract PDFs into per-page text and images with kreuzberg.

Every PDF under ``data/`` (outside ``data/extracted``) is written to a JSON
file at the mirrored path under ``data/extracted``:

    {"page_count": 2,
     "pages": [{"page_number": 1, "texts": ["..."], "images": ["<base64>"]}]}

Texts are kreuzberg's text elements in reading order, so headings stay on
their own lines; the policy index relies on that to find policy boundaries.
Images are kept so drawings and scanned pages can be read by a vision model.

    uv run extract
"""

import asyncio
import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from kreuzberg import (  # type: ignore[attr-defined]
    ExtractionConfig,
    ImageExtractionConfig,
    PageConfig,
    PdfConfig,
    ResultFormat,
    extract_file,
)
from loguru import logger

DATA_DIR = Path("data")
EXTRACTED_DIR = DATA_DIR / "extracted"
IMAGE_DPI = 300
MAX_CONCURRENT_PDFS = 16

CONFIG = ExtractionConfig(
    result_format=ResultFormat.ELEMENT_BASED,
    pages=PageConfig(extract_pages=True),
    images=ImageExtractionConfig(extract_images=True, target_dpi=IMAGE_DPI),
    pdf_options=PdfConfig(extract_metadata=True),
)


async def extract_pdf(path: Path) -> dict[str, Any]:
    """Return one PDF's text and images grouped by page.

    Args:
        path: PDF to extract.
    """
    result = await extract_file(path, config=CONFIG)
    pages: dict[int, dict[str, Any]] = {
        int(page["page_number"]): _empty_page(int(page["page_number"]))
        for page in result.pages or []
    }

    elements = sorted(result.elements or [], key=_element_index)
    for element in elements:
        text = str(element.get("text") or "").strip()
        page_number = (element.get("metadata") or {}).get("page_number")
        if element.get("element_type") in ("image", "page_break") or not text:
            continue
        if page_number is not None:
            pages.setdefault(int(page_number), _empty_page(int(page_number)))
            pages[int(page_number)]["texts"].append(text)

    for page in result.pages or []:
        for image in page.get("images", []):
            if isinstance(image.get("data"), bytes):
                number = int(image.get("page_number") or page["page_number"])
                pages.setdefault(number, _empty_page(number))
                encoded = base64.b64encode(image["data"]).decode("ascii")
                pages[number]["images"].append(encoded)

    return {"page_count": len(pages), "pages": [pages[n] for n in sorted(pages)]}


def _empty_page(number: int) -> dict[str, Any]:
    """Return a page with no content yet."""
    return {"page_number": number, "texts": [], "images": []}


def _element_index(element: Mapping[str, Any]) -> int:
    """Return an element's position in reading order."""
    return int((element.get("metadata") or {}).get("element_index") or 0)


async def extract_all(data_dir: Path, output_dir: Path) -> None:
    """Extract every PDF under ``data_dir`` that has no JSON output yet.

    Args:
        data_dir: Folder scanned recursively for PDFs.
        output_dir: Folder the mirrored JSON files are written under.
    """
    pdfs = [
        pdf for pdf in sorted(data_dir.rglob("*.pdf")) if output_dir not in pdf.parents
    ]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDFS)

    async def run(pdf: Path) -> None:
        target = output_dir / pdf.relative_to(data_dir).with_suffix(".json")
        if target.exists():
            return
        async with semaphore:
            document = await extract_pdf(pdf)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document), encoding="utf-8")
        logger.info(f"Wrote {target}")

    await asyncio.gather(*(run(pdf) for pdf in pdfs))


app = typer.Typer(
    name="extract",
    help="Extract PDFs under data/ into data/extracted/.",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def main(
    data_dir: Path = typer.Option(DATA_DIR, help="Folder to scan for PDFs."),
    output_dir: Path = typer.Option(EXTRACTED_DIR, help="Where to write JSON."),
) -> None:
    """Extract PDFs that have not been extracted yet.

    Args:
        data_dir: Folder to scan for PDFs.
        output_dir: Where to write JSON.
    """
    asyncio.run(extract_all(data_dir, output_dir))


if __name__ == "__main__":
    app()
