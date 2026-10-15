"""End-to-end comparison: query document -> similar documents -> aligned differences."""

from dataclasses import dataclass, field

from governance_docs_similarity.alignment import Alignment, align
from governance_docs_similarity.config import Settings
from governance_docs_similarity.describe import Describer, NullDescriber
from governance_docs_similarity.document import Document
from governance_docs_similarity.embeddings import Embedder
from governance_docs_similarity.index import CorpusIndex
from governance_docs_similarity.retrieval import (
    EncodedQuery,
    Match,
    encode_query,
    find_similar,
    novelty_share,
)


@dataclass
class MatchComparison:
    match: Match
    alignment: Alignment
    description: str | None = None


@dataclass
class ComparisonResult:
    query: EncodedQuery
    comparisons: list[MatchComparison]
    novelty_share: float
    settings: Settings
    excluded_ids: list[str] = field(default_factory=list)


def compare_document(
    document: Document,
    index: CorpusIndex,
    embedder: Embedder,
    settings: Settings,
    k: int | None = None,
    exclude_ids: list[str] | None = None,
    describer: Describer | None = None,
    describe_top: int | None = None,
    on_progress=lambda message: None,
) -> ComparisonResult:
    exclude_ids = exclude_ids or []
    describer = describer or NullDescriber()
    query = encode_query(document, index, embedder, settings.chunk_words, settings.chunk_overlap)
    matches = find_similar(query, index, k=k or settings.top_k, exclude_ids=exclude_ids)

    comparisons = []
    for rank, match in enumerate(matches):
        rows = index.doc_rows(match.document.doc_id)
        alignment = align(
            query.chunks,
            query.embeddings,
            index.chunks["text"].iloc[rows].tolist(),
            index.embeddings[rows],
            settings.unchanged_threshold,
            settings.modified_threshold,
        )
        description = None
        if describe_top is None or rank < describe_top:
            on_progress(f"Describing differences vs. {match.document.display_name}")
            description = describer.describe(document, match.document, alignment)
        comparisons.append(MatchComparison(match, alignment, description))

    return ComparisonResult(
        query=query,
        comparisons=comparisons,
        novelty_share=novelty_share(query, index, settings.modified_threshold, exclude_ids),
        settings=settings,
        excluded_ids=exclude_ids,
    )
