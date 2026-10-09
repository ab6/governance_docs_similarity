from governance_docs_similarity.chunking import chunk_document, chunk_text


def test_short_document_is_single_chunk():
    assert chunk_text("Only fifteen words here in this very short policy document about AI systems.") == [
        "Only fifteen words here in this very short policy document about AI systems."
    ]


def test_paragraphs_are_packed_up_to_limit():
    text = "\n\n".join(" ".join([f"p{i}"] * 30) for i in range(4))
    chunks = chunk_text(text, max_words=70, overlap=5)
    assert [len(c.split()) for c in chunks] == [60, 60]


def test_section_headers_start_new_paragraphs():
    text = "Intro words " * 40 + "\nSEC. 2. RULES.\n" + "rule words " * 40
    chunks = chunk_text(text, max_words=100, overlap=0)
    assert len(chunks) == 2
    assert chunks[1].startswith("SEC. 2. RULES.")


def test_long_paragraph_uses_overlapping_windows():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), max_words=100, overlap=20)
    split = [c.split() for c in chunks]
    assert split[0][-20:] == split[1][:20]
    assert split[-1][-1] == "w249"


def test_tiny_tail_is_merged():
    text = " ".join(["a"] * 100) + "\n\n" + "tail words"
    chunks = chunk_text(text, max_words=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].endswith("tail words")


def test_empty_text_has_no_chunks():
    assert chunk_document("x", "   \n\n ") == []
