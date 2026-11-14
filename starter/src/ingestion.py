"""Read, normalize, and validate source data while retaining source locations."""
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

LOGGER = logging.getLogger(__name__)
MIN_CONTENT_UNIT_CHARS = 1500
MAX_CONTENT_UNIT_CHARS = 6000
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
SECTION_HEADING = re.compile(r"(?im)^\s*((?:CHAPTER|LETTER)\s+(?:[IVXLCDM]+|\d+)[^\r\n]*)\s*$")


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    selected_pages: set[int] | None = None,
) -> list[DocumentPage]:
    """Extract physical PDF pages, retaining original numbers; OCR is not implicit."""
    reader = PdfReader(str(path))
    if selected_pages and (min(selected_pages) < 1 or max(selected_pages) > len(reader.pages)):
        raise ValueError(f"Selected page numbers must be between 1 and {len(reader.pages)}")
    pages: list[DocumentPage] = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        if selected_pages is not None and index not in selected_pages:
            continue
        page_id = str(index)
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception:
            LOGGER.exception("Failed to extract %s page %s", source, page_id)
            continue
        if not text:
            LOGGER.warning("No extractable text on %s page %s; OCR may be needed", source, page_id)
            continue
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_id,
                text=text,
                metadata={"physical_page": index},
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    output: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            if output:
                blank_pending = True
            continue
        if blank_pending:
            output.append("")
            blank_pending = False
        output.append(line)
    return "\n".join(output).strip()


def _gutenberg_body(text: str, source: str) -> str:
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    if start is None or end is None or end.start() <= (start.end() if start else 0):
        LOGGER.warning("Project Gutenberg text markers not found or malformed in %s", source)
        return text
    return text[start.end():end.start()]


def _sections(document_id: str, body: str) -> list[tuple[str, str]]:
    headings = list(SECTION_HEADING.finditer(body))
    if not headings:
        return [("full_text", body.strip())]

    sections: list[tuple[str, str]] = []
    if body[:headings[0].start()].strip():
        sections.append(("opening", body[:headings[0].start()].strip()))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        label = normalize_text(heading.group(1))
        content = body[heading.start():end].strip()
        if content:
            sections.append((label, content))
    if not sections:
        raise ValueError(f"No non-empty sections found for {document_id}")
    return sections


def _paragraph_atoms(sections: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    atoms: list[tuple[str, str]] = []
    for section, text in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                atoms.append((section, paragraph))
                continue
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            piece = ""
            for sentence in sentences:
                if len(sentence) > max_chars:
                    if piece:
                        atoms.append((section, piece))
                        piece = ""
                    for start in range(0, len(sentence), max_chars):
                        atoms.append((section, sentence[start:start + max_chars]))
                    continue
                candidate = f"{piece} {sentence}".strip()
                if piece and len(candidate) > max_chars:
                    atoms.append((section, piece))
                    piece = sentence
                else:
                    piece = candidate
            if piece:
                atoms.append((section, piece))
    return atoms


def content_units(
    document_id: str,
    source: str,
    text: str,
    metadata: dict | None = None,
    min_chars: int = MIN_CONTENT_UNIT_CHARS,
    max_chars: int = MAX_CONTENT_UNIT_CHARS,
) -> list[DocumentPage]:
    """Create traceable, paragraph-aware source units of at least ``min_chars``."""
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("Require 1 <= min_chars <= max_chars")
    body = normalize_text(_gutenberg_body(text, source))
    atoms = _paragraph_atoms(_sections(document_id, body), max_chars)
    if not atoms:
        raise ValueError(f"No usable text found in {source}")

    grouped: list[tuple[str, str]] = []
    current_sections: list[str] = []
    current_parts: list[str] = []
    current_size = 0
    for section, part in atoms:
        separator_size = 2 if current_parts else 0
        if current_parts and current_size >= min_chars and current_size + separator_size + len(part) > max_chars:
            grouped.append(("; ".join(current_sections), "\n\n".join(current_parts)))
            current_sections, current_parts, current_size = [], [], 0
            separator_size = 0
        current_sections.append(section)
        current_parts.append(part)
        current_size += separator_size + len(part)
        if current_size >= max_chars:
            grouped.append(("; ".join(current_sections), "\n\n".join(current_parts)))
            current_sections, current_parts, current_size = [], [], 0

    if current_parts:
        remainder = "\n\n".join(current_parts)
        if len(remainder) < min_chars and grouped:
            previous_section, previous_text = grouped.pop()
            grouped.append((f"{previous_section}; {'; '.join(current_sections)}", f"{previous_text}\n\n{remainder}"))
        else:
            grouped.append(("; ".join(current_sections), remainder))

    source_metadata = metadata or {}
    pages = []
    for index, (section, unit_text) in enumerate(grouped, start=1):
        unit_id = f"{document_id}-unit-{index:04d}"
        normalized_unit = normalize_text(unit_text)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=unit_id,
                text=normalized_unit,
                metadata={
                    **source_metadata,
                    "section": section,
                    "source_unit_id": unit_id,
                    "unit_chars": len(normalized_unit),
                },
            )
        )
    return pages


def load_manifest(manifest_path: Path) -> tuple[list[DocumentPage], dict]:
    """Load a verified text snapshot or the earlier Project Gutenberg manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.resolve().parent
    snapshot = manifest.get("page_snapshot")
    if snapshot:
        snapshot_path = (root / snapshot["path"]).resolve()
        if not snapshot_path.is_relative_to(root.resolve()):
            raise ValueError(f"Page snapshot escapes data directory: {snapshot['path']}")
        raw_snapshot = snapshot_path.read_bytes()
        actual_hash = hashlib.sha256(raw_snapshot).hexdigest()
        if actual_hash != snapshot["sha256"]:
            raise ValueError("Page snapshot SHA-256 mismatch; corpus snapshot changed")
        records = [
            json.loads(line)
            for line in raw_snapshot.decode("utf-8").splitlines()
            if line.strip()
        ]
        pages = []
        seen_page_ids: set[str] = set()
        for record in records:
            page_id = str(record["page"])
            if page_id in seen_page_ids:
                raise ValueError(f"Duplicate source page in snapshot: {page_id}")
            seen_page_ids.add(page_id)
            text = normalize_text(record["text"])
            if not text:
                raise ValueError(f"Empty source page in snapshot: {page_id}")
            if len(text) != record["char_count"]:
                raise ValueError(f"Character count mismatch for source page {page_id}")
            pages.append(
                DocumentPage(
                    document_id=record["document_id"],
                    source=record["source"],
                    page=page_id,
                    text=text,
                    metadata=record["metadata"],
                )
            )
        pages.sort(key=lambda page: int(page.page))
        expected_pages = [str(number) for number in manifest["used_pages"]]
        if [page.page for page in pages] != expected_pages:
            raise ValueError("Page snapshot does not match the manifest's selected physical pages")
        if len(pages) != manifest["used_page_count"]:
            raise ValueError("Page snapshot count does not match the manifest")
        if len(pages) < 30:
            raise ValueError(f"Corpus has {len(pages)} content pages; at least 30 are required")
        return pages, manifest

    pages: list[DocumentPage] = []
    for source_record in manifest.get("sources", []):
        source_path = (root / source_record["source_path"]).resolve()
        if not source_path.is_relative_to(root.resolve()):
            raise ValueError(f"Corpus path escapes data directory: {source_record['source_path']}")
        raw = source_path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != source_record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {source_record['source_id']}: corpus changed")
        text = raw.decode("utf-8-sig")
        source_pages = content_units(
            source_record["source_id"],
            source_record["source_path_or_url"],
            text,
            {
                "category": source_record["category"],
                "author": source_record["author"],
                "publication_year": source_record["publication_year"],
                "language": source_record["language"],
            },
        )
        expected = source_record.get("content_units", [])
        actual = [
            {"page": page.page, "section": page.metadata["section"], "char_count": len(page.text)}
            for page in source_pages
        ]
        if actual != expected:
            raise ValueError(f"Manifest unit boundaries do not match {source_record['source_id']}")
        pages.extend(source_pages)
    if not pages:
        raise ValueError("Manifest contains no usable corpus pages")
    if len(pages) < 30:
        raise ValueError(f"Corpus has {len(pages)} content units; at least 30 are required")
    short_units = [page.page for page in pages if len(page.text) < MIN_CONTENT_UNIT_CHARS]
    if short_units:
        raise ValueError(f"Content units under {MIN_CONTENT_UNIT_CHARS} characters: {short_units}")
    LOGGER.info("Loaded %d verified content units from %d sources", len(pages), len(manifest["sources"]))
    return pages, manifest
