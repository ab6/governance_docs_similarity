"""Load the AGORA corpus: full texts joined with document-level metadata."""

from pathlib import Path

import pandas as pd

from governance_docs_similarity.config import AGORA_DIR
from governance_docs_similarity.document import Document

METADATA_COLUMNS = {
    "Official name": "official_name",
    "Casual name": "casual_name",
    "Authority": "authority",
    "Collections": "collections",
    "Most recent activity": "most_recent_activity",
    "Most recent activity date": "date",
    "Short summary": "short_summary",
    "Tags": "tags",
}


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _dedupe_tags(tags: str) -> str:
    return ";".join(dict.fromkeys(t.strip() for t in tags.split(";") if t.strip()))


def load_metadata(agora_dir: Path = AGORA_DIR) -> dict[str, dict[str, str]]:
    """Return {AGORA ID: metadata dict} from documents.csv (empty if the file is missing)."""
    csv_path = agora_dir / "documents.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["AGORA ID", *METADATA_COLUMNS])
    metadata = {}
    for row in df.to_dict("records"):
        record = {key: _clean(row[col]) for col, key in METADATA_COLUMNS.items()}
        record["tags"] = _dedupe_tags(record["tags"])
        metadata[str(row["AGORA ID"])] = record
    return metadata


def load_agora_corpus(agora_dir: Path = AGORA_DIR) -> list[Document]:
    """Load every `fulltext/<AGORA ID>.txt` file as a Document with its metadata."""
    fulltext_dir = agora_dir / "fulltext"
    if not fulltext_dir.is_dir():
        raise FileNotFoundError(f"AGORA fulltext directory not found: {fulltext_dir}")
    metadata = load_metadata(agora_dir)
    documents = []
    for path in sorted(fulltext_dir.glob("*.txt"), key=lambda p: (len(p.stem), p.stem)):
        meta = metadata.get(path.stem, {})
        documents.append(
            Document(
                doc_id=path.stem,
                text=path.read_text(encoding="utf-8", errors="replace"),
                title=meta.get("official_name") or meta.get("casual_name") or "",
                metadata=meta,
            )
        )
    return documents
