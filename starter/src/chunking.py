"""Fixed-size and paragraph-aware chunking that retain page-level provenance."""
import hashlib
import math
import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split normalized text into overlapping character windows."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Require size >= 1 and 0 <= overlap < size")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        if chunks and chunks[-1].metadata["source_offset_end"] >= len(page.text):
            break
        end = min(start + size, len(page.text))
        text = page.text[start:end]
        if text:
            chunks.append(_chunk(page, text, f"fixed-{start}-{end}", start, end))
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to a size limit; split only oversized paragraphs."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_size = 0
    cursor = 0

    def flush() -> None:
        nonlocal current, current_size, cursor
        if current:
            text = "\n\n".join(current)
            start = page.text.find(text, cursor)
            if start < 0:
                raise ValueError(f"Unable to map paragraph chunk to source page {page.page}")
            end = start + len(text)
            chunks.append(_chunk(page, text, f"paragraph-{len(chunks)}", start, end))
            cursor = end
            current, current_size = [], 0

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            flush()
            for piece_index, piece in enumerate(_split_long_paragraph(paragraph, max_size)):
                source_start = page.text.find(piece, cursor)
                if source_start < 0:
                    raise ValueError(f"Unable to map long paragraph to source page {page.page}")
                source_end = source_start + len(piece)
                chunks.append(
                    _chunk(
                        page,
                        piece,
                        f"long-paragraph-{len(chunks)}-{piece_index}",
                        source_start,
                        source_end,
                    )
                )
                cursor = source_end
            continue
        candidate_size = current_size + (2 if current else 0) + len(paragraph)
        if current and candidate_size > max_size:
            flush()
        current.append(paragraph)
        current_size += (2 if len(current) > 1 else 0) + len(paragraph)
    flush()
    return chunks


def _split_long_paragraph(text: str, max_size: int) -> list[str]:
    """Prefer sentence and word boundaries while avoiding tiny pieces."""
    piece_count = math.ceil(len(text) / max_size)
    target_size = math.ceil(len(text) / piece_count)
    min_piece_size = max(1, min(100, max_size // 2))
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?:[\"'’”)]*)\s+", text)
    ]
    pieces = []
    start = 0

    for piece_index in range(piece_count - 1):
        remaining_pieces = piece_count - piece_index - 1
        lower_end = max(start + min_piece_size, len(text) - max_size * remaining_pieces)
        upper_end = min(start + max_size, len(text) - min_piece_size * remaining_pieces)
        target_end = min(max(start + target_size, lower_end), upper_end)

        sentences = [end for end in sentence_ends if lower_end <= end <= upper_end]
        if sentences:
            cut = min(sentences, key=lambda boundary: abs(boundary - target_end))
        else:
            boundaries = []
            before = text.rfind(" ", lower_end - 1, target_end)
            after = text.find(" ", target_end, upper_end)
            if before >= lower_end - 1:
                boundaries.append(before + 1)
            if after >= 0:
                boundaries.append(after + 1)
            cut = min(boundaries, key=lambda boundary: abs(boundary - target_end)) if boundaries else target_end

        pieces.append(text[start:cut])
        start = cut

    pieces.append(text[start:])
    return pieces


def _chunk(page: DocumentPage, text: str, location: str, start: int, end: int) -> Chunk:
    chunk_id = hashlib.sha256(
        f"{page.document_id}\0{page.page}\0{location}\0{text}".encode("utf-8")
    ).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "source_offset_start": start,
            "source_offset_end": end,
        },
    )
