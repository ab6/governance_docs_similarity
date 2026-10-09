"""Passage-level embedding index over the corpus, persisted to disk.

Chunks are stored contiguously per document so per-document reductions can use
`np.*.reduceat` over `doc_offsets`.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from governance_docs_similarity.chunking import chunk_document
from governance_docs_similarity.document import Document
from governance_docs_similarity.embeddings import Embedder

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.parquet"
DOCUMENTS_FILE = "documents.json"
MANIFEST_FILE = "manifest.json"


@dataclass
class CorpusIndex:
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalized
    chunks: pd.DataFrame  # columns: doc_id, chunk_idx, text
    documents: dict[str, Document]  # metadata only; text is not kept
    doc_ids: list[str]  # in index order
    doc_offsets: np.ndarray  # start row of each doc in `embeddings`
    manifest: dict

    @property
    def doc_sizes(self) -> np.ndarray:
        return np.diff(np.append(self.doc_offsets, len(self.embeddings)))

    def doc_rows(self, doc_id: str) -> slice:
        i = self.doc_ids.index(doc_id)
        return slice(int(self.doc_offsets[i]), int(self.doc_offsets[i] + self.doc_sizes[i]))

    def doc_chunk_texts(self, doc_id: str) -> list[str]:
        return self.chunks["text"].iloc[self.doc_rows(doc_id)].tolist()

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / EMBEDDINGS_FILE, self.embeddings)
        self.chunks.to_parquet(index_dir / CHUNKS_FILE, index=False)
        docs = {
            d.doc_id: {"title": d.title, "metadata": d.metadata} for d in self.documents.values()
        }
        (index_dir / DOCUMENTS_FILE).write_text(json.dumps(docs, ensure_ascii=False))
        (index_dir / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, index_dir: Path) -> CorpusIndex:
        if not (index_dir / MANIFEST_FILE).exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Build one first with `gds build-index`."
            )
        chunks = pd.read_parquet(index_dir / CHUNKS_FILE)
        chunks["doc_id"] = chunks["doc_id"].astype(str)
        raw_docs = json.loads((index_dir / DOCUMENTS_FILE).read_text())
        documents = {
            doc_id: Document(doc_id=doc_id, text="", title=d["title"], metadata=d["metadata"])
            for doc_id, d in raw_docs.items()
        }
        return cls._assemble(
            embeddings=np.load(index_dir / EMBEDDINGS_FILE),
            chunks=chunks,
            documents=documents,
            manifest=json.loads((index_dir / MANIFEST_FILE).read_text()),
        )

    @classmethod
    def _assemble(cls, embeddings, chunks, documents, manifest) -> CorpusIndex:
        starts = chunks.index[chunks["doc_id"] != chunks["doc_id"].shift()].to_numpy()
        doc_ids = chunks["doc_id"].iloc[starts].tolist()
        return cls(embeddings, chunks, documents, doc_ids, starts, manifest)

    def check_compatible(self, embedding_model: str, chunk_words: int, chunk_overlap: int):
        expected = {
            "embedding_model": embedding_model,
            "chunk_words": chunk_words,
            "chunk_overlap": chunk_overlap,
        }
        stale = {
            k: (self.manifest.get(k), v) for k, v in expected.items() if self.manifest.get(k) != v
        }
        if stale:
            details = ", ".join(f"{k}: index={a!r} settings={b!r}" for k, (a, b) in stale.items())
            raise ValueError(f"Index is stale ({details}). Rebuild with `gds build-index`.")


def build_index(
    documents: list[Document],
    embedder: Embedder,
    chunk_words: int,
    chunk_overlap: int,
    show_progress: bool = False,
) -> CorpusIndex:
    rows = [
        {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text}
        for doc in documents
        for c in chunk_document(doc.doc_id, doc.text, chunk_words, chunk_overlap)
    ]
    chunks = pd.DataFrame(rows, columns=["doc_id", "chunk_idx", "text"])
    if chunks.empty:
        raise ValueError("No text to index.")
    embeddings = embedder.encode(chunks["text"].tolist(), show_progress=show_progress)
    indexed = set(chunks["doc_id"])
    manifest = {
        "embedding_model": embedder.name,
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap,
        "n_documents": len(indexed),
        "n_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }
    documents_meta = {
        d.doc_id: Document(doc_id=d.doc_id, text="", title=d.title, metadata=d.metadata)
        for d in documents
        if d.doc_id in indexed
    }
    return CorpusIndex._assemble(embeddings, chunks, documents_meta, manifest)
