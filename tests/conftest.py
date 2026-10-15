"""Shared fixtures: a tiny AGORA-shaped corpus and a fast deterministic embedder."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from governance_docs_similarity.config import Settings
from governance_docs_similarity.corpus import load_agora_corpus
from governance_docs_similarity.index import build_index

DOCS = {
    "1": (
        "Facial Recognition Act",
        "SEC. 1. DEFINITIONS.\n\nFacial recognition technology means any system that identifies "
        "a person by comparing images of their face against a database of known faces.\n\n"
        "SEC. 2. PROHIBITION.\n\nNo law enforcement agency shall use facial recognition "
        "technology without first obtaining a warrant issued by a court of competent jurisdiction.",
    ),
    "2": (
        "Autonomous Vehicle Safety Act",
        "SEC. 1. SAFETY STANDARDS.\n\nThe Secretary of Transportation shall issue safety standards "
        "for autonomous vehicles operating on public roads, including crash reporting and "
        "cybersecurity requirements for vehicle software updates.",
    ),
    "3": (
        "Algorithmic Hiring Disclosure Act",
        "SEC. 1. DISCLOSURE.\n\nAn employer that uses an automated employment decision tool shall "
        "notify each candidate and disclose the qualifications and characteristics the tool "
        "assesses.\n\nSEC. 2. BIAS AUDIT.\n\nThe tool must undergo an independent bias audit "
        "every year and the results must be published on the employer website.",
    ),
}


class HashingEmbedder:
    """Bag-of-words hashing embedder: fast, offline, and identical text gives similarity 1."""

    name = "test-hashing"

    def __init__(self):
        self._vectorizer = HashingVectorizer(n_features=2**12, alternate_sign=False, norm="l2")

    def encode(self, texts, show_progress=False):
        return self._vectorizer.transform(texts).toarray().astype(np.float32)


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def settings():
    return Settings(
        embedding_model=HashingEmbedder.name,
        chunk_words=60,
        chunk_overlap=10,
        top_k=2,
        unchanged_threshold=0.92,
        modified_threshold=0.5,
    )


@pytest.fixture
def agora_dir(tmp_path):
    root = tmp_path / "agora"
    (root / "fulltext").mkdir(parents=True)
    rows = ["AGORA ID,Official name,Casual name,Authority,Collections,Most recent activity,"
            "Most recent activity date,Short summary,Tags"]
    for doc_id, (title, text) in DOCS.items():
        (root / "fulltext" / f"{doc_id}.txt").write_text(text)
        rows.append(f'{doc_id},{title},,State governments,U.S. state and local documents,'
                    f'Enacted,2024-0{doc_id}-01,Summary of {title}.,"Tag A;Tag A;Tag B"')
    (root / "documents.csv").write_text("\n".join(rows))
    return root


@pytest.fixture
def index(agora_dir, embedder, settings):
    return build_index(
        load_agora_corpus(agora_dir), embedder, settings.chunk_words, settings.chunk_overlap
    )
