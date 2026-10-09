from governance_docs_similarity.alignment import Label, align

CLAUSES = [
    "No law enforcement agency shall use facial recognition technology without a warrant.",
    "Agencies shall publish an annual report on the number of facial recognition searches.",
    "Vendors must submit their algorithms for independent accuracy and bias testing.",
]


def _align(query, match, embedder):
    return align(query, embedder.encode(query), match, embedder.encode(match), 0.92, 0.5)


def test_identical_documents_are_unchanged(embedder):
    alignment = _align(CLAUSES, CLAUSES, embedder)
    assert alignment.counts()[Label.UNCHANGED] == 3
    assert not alignment.by_label(Label.ABSENT)


def test_removed_clause_is_absent(embedder):
    alignment = _align(CLAUSES[:2], CLAUSES, embedder)
    absent = alignment.by_label(Label.ABSENT)
    assert [p.match_idx for p in absent] == [2]


def test_added_clause_is_new(embedder):
    query = CLAUSES + ["The state treasury shall fund robotics apprenticeship grants for rural youth."]
    alignment = _align(query, CLAUSES, embedder)
    assert [p.query_idx for p in alignment.by_label(Label.NEW)] == [3]


def test_edited_clause_is_modified(embedder):
    query = [CLAUSES[0].replace("without a warrant", "without a warrant or written consent")]
    alignment = _align(query, CLAUSES[:1], embedder)
    (passage,) = alignment.passages
    assert passage.label == Label.MODIFIED
    assert passage.ref == "Q0↔M0"
