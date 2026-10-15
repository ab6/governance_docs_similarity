import pytest

from governance_docs_similarity.corpus import load_agora_corpus
from governance_docs_similarity.loaders import load_document


def test_corpus_joins_metadata(agora_dir):
    docs = {d.doc_id: d for d in load_agora_corpus(agora_dir)}
    assert set(docs) == {"1", "2", "3"}
    assert docs["1"].title == "Facial Recognition Act"
    assert docs["1"].metadata["authority"] == "State governments"
    assert docs["1"].metadata["tags"] == "Tag A;Tag B"


def test_load_document_txt(tmp_path):
    path = tmp_path / "new_bill.txt"
    path.write_text("Some text")
    doc = load_document(path)
    assert (doc.doc_id, doc.text) == ("new_bill", "Some text")


def test_load_document_rejects_unsupported(tmp_path):
    path = tmp_path / "bill.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)
