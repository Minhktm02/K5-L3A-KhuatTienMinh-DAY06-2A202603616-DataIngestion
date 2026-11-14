"""Extract 30 selected physical pages from the user-provided World Bank PDF."""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from starter.src.ingestion import load_pdf, normalize_text

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PDF_PATH = Path(
    os.getenv("WORLD_BANK_PDF", r"C:\Users\Admin\Downloads\9781464816987.pdf")
).expanduser()
DOCUMENT_ID = "world-bank-eca-spring-2021"
TITLE = "Europe and Central Asia Economic Update, Spring 2021: Data, Digitalization, and Governance"
SOURCE = "World Bank ECA Economic Update, Spring 2021 (ISBN 978-1-4648-1698-7)"
SELECTED_PAGES = list(range(19, 49))
SECTIONS = (
    (19, 23, "Global Context", 19),
    (24, 27, "Global Risks", 24),
    (28, 39, "Recent Developments", 28),
    (40, 48, "Regional Outlook", 40),
)
SNAPSHOT_PATH = CORPUS_DIR / "world_bank_eca_spring_2021_pages.jsonl"


def _section_for_page(page_number: int) -> tuple[str, int]:
    for first, last, label, heading_page in SECTIONS:
        if first <= page_number <= last:
            return label, heading_page
    raise ValueError(f"No verified report section assigned to physical page {page_number}")


def _write_new_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        logging.info("Verified existing identical output %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"World Bank source PDF was not found: {PDF_PATH}")
    raw_pdf = PDF_PATH.read_bytes()
    pdf_sha256 = hashlib.sha256(raw_pdf).hexdigest()

    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH), strict=False)
    physical_page_count = len(reader.pages)
    if physical_page_count != 172:
        raise ValueError(
            f"Expected the inspected 172-page source PDF, found {physical_page_count} pages"
        )
    extracted = load_pdf(PDF_PATH, DOCUMENT_ID, SOURCE, set(SELECTED_PAGES))
    extracted_by_page = {int(page.page): page for page in extracted}
    missing = sorted(set(SELECTED_PAGES) - set(extracted_by_page))
    if missing:
        raise ValueError(f"Selected source pages have no extractable text: {missing}")

    page_records = []
    snapshot_records = []
    for page_number in SELECTED_PAGES:
        page = extracted_by_page[page_number]
        section, heading_page = _section_for_page(page_number)
        text = normalize_text(page.text)
        if not text:
            raise ValueError(f"Selected source page {page_number} is empty after normalization")
        metadata = {
            "title": TITLE,
            "publication_year": 2021,
            "document_type": "economic_update",
            "section": section,
            "section_heading_page": heading_page,
        }
        snapshot_records.append(
            {
                "document_id": DOCUMENT_ID,
                "source": SOURCE,
                "page": page_number,
                "text": text,
                "char_count": len(text),
                "metadata": metadata,
            }
        )
        page_records.append(
            {
                "physical_page": page_number,
                "character_count": len(text),
                "processing_status": "text_extracted_and_normalized",
                "section": section,
                "section_heading_physical_page": heading_page,
            }
        )
        logging.info("Extracted physical page %d: %d normalized characters", page_number, len(text))

    if len(page_records) != 30 or len({record["physical_page"] for record in page_records}) != 30:
        raise ValueError("The selected corpus must contain exactly 30 distinct physical pages")

    snapshot_bytes = (
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in snapshot_records)
    ).encode("utf-8")
    accessed_at = datetime.now(timezone.utc).isoformat()
    if MANIFEST_PATH.is_file():
        existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing_manifest.get("pdf_sha256") != pdf_sha256:
            raise FileExistsError(
                f"Refusing to reuse manifest for a different source PDF: {MANIFEST_PATH}"
            )
        existing_accessed_at = existing_manifest.get("accessed_at")
        if not isinstance(existing_accessed_at, str) or not existing_accessed_at:
            raise ValueError(f"Existing manifest has no valid accessed_at timestamp: {MANIFEST_PATH}")
        accessed_at = existing_accessed_at
    manifest = {
        "schema_version": 2,
        "accessed_at": accessed_at,
        "source_id": DOCUMENT_ID,
        "title": TITLE,
        "source_path_or_url": str(PDF_PATH),
        "source_reference": "World Bank publication, ISBN 978-1-4648-1698-7",
        "license_or_permission": (
            "Creative Commons Attribution 3.0 IGO (CC BY 3.0 IGO), printed on physical "
            "page 172. Attribution: International Bank for Reconstruction and Development / "
            "The World Bank, 2021. Extracted pages were whitespace-normalized; the original "
            "PDF was not modified."
        ),
        "pdf_sha256": pdf_sha256,
        "physical_page_count": physical_page_count,
        "used_page_count": len(page_records),
        "used_pages": SELECTED_PAGES,
        "page_counting_method": (
            "Exactly 30 non-empty, text-extractable physical pages, 19-48 inclusive. "
            "Physical pages 1-18 are cover/front matter/contents or blank; the selected range "
            "is chapter content and excludes the references and license pages. Page IDs remain "
            "the original PDF physical page numbers."
        ),
        "selected_page_processing": page_records,
        "selected_page_extraction_errors": [],
        "page_snapshot": {
            "path": str(SNAPSHOT_PATH.relative_to(DATA_DIR)).replace("\\", "/"),
            "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "format": "JSONL; one normalized source page per record",
            "source_pdf_path_is_required_for_reextracting": True,
        },
        "section_metadata": {
            "field": "section",
            "type": "string",
            "value_source": "Printed section headings on physical pages 19, 24, 28, and 40",
            "values": [label for _, _, label, _ in SECTIONS],
        },
        "citation": (
            "International Bank for Reconstruction and Development / The World Bank. 2021. "
            "Europe and Central Asia Economic Update, Spring 2021: Data, Digitalization, "
            "and Governance. ISBN 978-1-4648-1698-7. CC BY 3.0 IGO."
        ),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    for path, content in ((SNAPSHOT_PATH, snapshot_bytes), (MANIFEST_PATH, manifest_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    _write_new_or_identical(SNAPSHOT_PATH, snapshot_bytes)
    _write_new_or_identical(MANIFEST_PATH, manifest_bytes)
    logging.info(
        "Prepared exactly %d pages from %s; PDF SHA-256 %s",
        len(page_records),
        PDF_PATH.name,
        pdf_sha256,
    )


if __name__ == "__main__":
    main()
