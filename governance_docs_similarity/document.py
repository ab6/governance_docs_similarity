"""Core document type shared by the corpus and new (query) documents."""

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.title or self.doc_id
