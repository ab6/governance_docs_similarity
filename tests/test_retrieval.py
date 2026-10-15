from governance_docs_similarity.document import Document
from governance_docs_similarity.index import CorpusIndex
from governance_docs_similarity.retrieval import encode_query, find_similar
from tests.conftest import DOCS


def _query(text, index, embedder, settings):
    return encode_query(
        Document("q", text), index, embedder, settings.chunk_words, settings.chunk_overlap
    )


def test_document_retrieves_itself_first(index, embedder, settings):
    for doc_id, (_, text) in DOCS.items():
        matches = find_similar(_query(text, index, embedder, settings), index, k=3)
        assert matches[0].document.doc_id == doc_id
        assert matches[0].score > 0.99


def test_exclude_ids(index, embedder, settings):
    query = _query(DOCS["1"][1], index, embedder, settings)
    matches = find_similar(query, index, k=3, exclude_ids=["1"])
    assert sorted(m.document.doc_id for m in matches) == ["2", "3"]


def test_index_round_trip(index, tmp_path, embedder, settings):
    index.save(tmp_path / "idx")
    loaded = CorpusIndex.load(tmp_path / "idx")
    assert loaded.doc_ids == index.doc_ids
    assert loaded.documents["3"].title == "Algorithmic Hiring Disclosure Act"
    loaded.check_compatible(embedder.name, settings.chunk_words, settings.chunk_overlap)
