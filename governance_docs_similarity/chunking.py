"""Split documents into passages for embedding and alignment.

Paragraphs (blank-line separated, plus a break before section/article headers)
are packed into passages of up to `max_words`. A paragraph longer than that is
cut into overlapping word windows. The same function is used for corpus and
query documents so passages are comparable.
"""

from dataclasses import dataclass
import re

_HEADER = re.compile(
    r"^\s*(SEC(TION)?\.?\s+\d|Sec\.\s+\d|Section\s+\d|ARTICLE\b|Article\s+\d|CHAPTER\b|Chapter\s+\d"
    r"|PART\b|Part\s+\d|TITLE\b|§)",
    re.MULTILINE,
)
_BLANK_LINES = re.compile(r"\n\s*\n")
MIN_TAIL_WORDS = 20


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_idx: int
    text: str


def _paragraphs(text: str) -> list[str]:
    text = _HEADER.sub(lambda m: "\n\n" + m.group(0).lstrip(), text.replace("\r\n", "\n"))
    return [" ".join(p.split()) for p in _BLANK_LINES.split(text) if p.strip()]


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
    return windows


def chunk_text(text: str, max_words: int = 200, overlap: int = 30) -> list[str]:
    pieces: list[list[str]] = []
    for paragraph in _paragraphs(text):
        words = paragraph.split()
        if len(words) > max_words:
            pieces.extend(_windows(words, max_words, overlap))
        else:
            pieces.append(words)

    chunks: list[list[str]] = []
    current: list[str] = []
    for words in pieces:
        if current and len(current) + len(words) > max_words:
            chunks.append(current)
            current = []
        current = current + words
    if current:
        # Fold a tiny trailing fragment into the previous chunk rather than emitting it alone.
        if chunks and len(current) < MIN_TAIL_WORDS:
            chunks[-1] = chunks[-1] + current
        else:
            chunks.append(current)
    return [" ".join(c) for c in chunks]


def chunk_document(doc_id: str, text: str, max_words: int = 200, overlap: int = 30) -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, chunk_idx=i, text=t)
        for i, t in enumerate(chunk_text(text, max_words, overlap))
    ]
