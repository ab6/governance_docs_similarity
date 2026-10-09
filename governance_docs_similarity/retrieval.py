"""Rank corpus documents by passage-level similarity to a query document.

For each corpus document we compute two coverage scores from the passage
similarity matrix:

- query_coverage: mean over query passages of their best match in the document
  (how much of the new document the corpus document explains)
- doc_coverage: mean over the document's passages of their best match in the
  query (how much of the corpus document survives in the new one)

The ranking score is their average, so a huge omnibus document that loosely
covers everything does not automatically beat a focused predecessor.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from governance_docs_similarity.chunking import chunk_document
from governance_docs_similarity.document import Document
from governance_docs_similarity.embeddings import Embedder
from governance_docs_similarity.index import CorpusIndex


@dataclass
class EncodedQuery:
    document: Document
    chunks: list[str]
    embeddings: np.ndarray
    similarities: np.ndarray  # (n_query_chunks, n_corpus_chunks)


@dataclass
class Match:
    document: Document
    score: float
    query_coverage: float
    doc_coverage: float


def encode_query(
    document: Document,
    index: CorpusIndex,
    embedder: Embedder,
    chunk_words: int,
    chunk_overlap: int,
) -> EncodedQuery:
    chunks = [
        c.text for c in chunk_document(document.doc_id, document.text, chunk_words, chunk_overlap)
    ]
    if not chunks:
        raise ValueError(f"Document {document.doc_id!r} has no text to compare.")
    embeddings = embedder.encode(chunks)
    return EncodedQuery(document, chunks, embeddings, embeddings @ index.embeddings.T)


def find_similar(
    query: EncodedQuery, index: CorpusIndex, k: int = 5, exclude_ids: Iterable[str] = ()
) -> list[Match]:
    sims = query.similarities
    offsets = index.doc_offsets
    # Best corpus passage per (query passage, document) -> (n_query, n_docs)
    best_per_doc = np.maximum.reduceat(sims, offsets, axis=1)
    query_coverage = best_per_doc.mean(axis=0)
    # Best query passage per corpus passage, averaged within each document.
    doc_coverage = np.add.reduceat(sims.max(axis=0), offsets) / index.doc_sizes
    scores = (query_coverage + doc_coverage) / 2

    excluded = set(map(str, exclude_ids))
    matches = []
    for i in np.argsort(-scores):
        doc_id = index.doc_ids[i]
        if doc_id in excluded:
            continue
        matches.append(
            Match(
                document=index.documents[doc_id],
                score=float(scores[i]),
                query_coverage=float(query_coverage[i]),
                doc_coverage=float(doc_coverage[i]),
            )
        )
        if len(matches) == k:
            break
    return matches


def novelty_share(
    query: EncodedQuery, index: CorpusIndex, threshold: float, exclude_ids=()
) -> float:
    """Fraction of query passages with no counterpart >= threshold anywhere in the corpus."""
    sims = query.similarities
    excluded = set(map(str, exclude_ids))
    if excluded:
        keep = ~index.chunks["doc_id"].isin(excluded).to_numpy()
        sims = sims[:, keep]
    if sims.shape[1] == 0:
        return 1.0
    return float((sims.max(axis=1) < threshold).mean())
