"""Align passages of a query document with one matched corpus document."""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Label(StrEnum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    NEW = "new"  # in the query, no counterpart in the matched document
    ABSENT = "absent"  # in the matched document, no counterpart in the query


@dataclass
class AlignedPassage:
    label: Label
    score: float
    query_idx: int | None = None
    query_text: str | None = None
    match_idx: int | None = None
    match_text: str | None = None

    @property
    def ref(self) -> str:
        parts = []
        if self.query_idx is not None:
            parts.append(f"Q{self.query_idx}")
        if self.match_idx is not None:
            parts.append(f"M{self.match_idx}")
        return "↔".join(parts)


@dataclass
class Alignment:
    passages: list[AlignedPassage] = field(default_factory=list)

    def by_label(self, label: Label) -> list[AlignedPassage]:
        return [p for p in self.passages if p.label == label]

    def counts(self) -> dict[Label, int]:
        return {label: len(self.by_label(label)) for label in Label}


def align(
    query_chunks: list[str],
    query_embeddings: np.ndarray,
    match_chunks: list[str],
    match_embeddings: np.ndarray,
    unchanged_threshold: float,
    modified_threshold: float,
) -> Alignment:
    """Label each query passage by its best match, then add unmatched corpus passages.

    Passage refs: `Q<i>` is the i-th query passage, `M<j>` the j-th passage of the
    matched document.
    """
    sims = query_embeddings @ match_embeddings.T
    passages = []
    for i, row in enumerate(sims):
        j = int(row.argmax())
        score = float(row[j])
        if score >= unchanged_threshold:
            label = Label.UNCHANGED
        elif score >= modified_threshold:
            label = Label.MODIFIED
        else:
            passages.append(AlignedPassage(Label.NEW, score, i, query_chunks[i]))
            continue
        passages.append(AlignedPassage(label, score, i, query_chunks[i], j, match_chunks[j]))

    best_for_match = sims.max(axis=0)
    for j in np.flatnonzero(best_for_match < modified_threshold):
        passages.append(
            AlignedPassage(
                Label.ABSENT,
                float(best_for_match[j]),
                match_idx=int(j),
                match_text=match_chunks[j],
            )
        )
    return Alignment(passages)
