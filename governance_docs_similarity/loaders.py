"""Load a new governance document from disk.

Only plain text is supported today. To add a format (PDF, HTML, URL), write a
`(path) -> str` function and register it in `_SUFFIX_TO_LOADER`.
"""

from collections.abc import Callable
from pathlib import Path

from governance_docs_similarity.document import Document


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_SUFFIX_TO_LOADER: dict[str, Callable[[Path], str]] = {
    ".txt": _read_text,
}


def load_document(path: str | Path, title: str | None = None) -> Document:
    path = Path(path)
    loader = _SUFFIX_TO_LOADER.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(_SUFFIX_TO_LOADER))
        raise ValueError(f"Unsupported file type {path.suffix!r} ({path}); supported: {supported}")
    text = loader(path)
    if not text.strip():
        raise ValueError(f"Document is empty: {path}")
    return Document(doc_id=path.stem, text=text, title=title or path.stem)
