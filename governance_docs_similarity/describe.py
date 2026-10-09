"""Natural-language descriptions of how a new document differs from a match.

`Describer` is the swap point for LLM backends: `OllamaDescriber` runs a local
open model during development; a Claude API backend can be added alongside it
without touching the rest of the pipeline.
"""

import re
from typing import Protocol

from governance_docs_similarity.alignment import AlignedPassage, Alignment, Label
from governance_docs_similarity.document import Document

SYSTEM_PROMPT = """You are an expert analyst of AI governance law and policy.
You compare a NEW governance document with an EARLIER/EXISTING related document and explain
how the new one differs, so readers can see how regulation has evolved rather than treating
the new document as entirely novel.

Rules:
- Use only the passages provided. Do not invent provisions.
- Cite passage refs in square brackets for every claim, e.g. [Q3], [M7], [Q3↔M5].
  Q refs are passages of the NEW document; M refs are passages of the EXISTING document.
- Be specific: name the actors, obligations, deadlines, thresholds, and penalties that changed.
- If a section has nothing to report, write "No notable differences."
"""

SECTIONS = """Organize your answer under these Markdown headings (### level):
### Scope and applicability
### Obligations and requirements added or removed
### Enforcement, penalties, and oversight
### Definitions and terminology
### Other notable changes"""

PASSAGE_WORD_LIMIT = 180


def _truncate(text: str, limit: int = PASSAGE_WORD_LIMIT) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]) + " …"


def _doc_header(role: str, doc: Document) -> str:
    meta = doc.metadata
    lines = [f"{role}: {doc.display_name}"]
    for key, label in [
        ("authority", "Authority"),
        ("date", "Date"),
        ("collections", "Collections"),
        ("short_summary", "Summary"),
    ]:
        if meta.get(key):
            lines.append(f"  {label}: {meta[key]}")
    return "\n".join(lines)


def _format_item(p: AlignedPassage) -> str:
    if p.label == Label.MODIFIED:
        return (
            f"[{p.ref}] MODIFIED (similarity {p.score:.2f})\n"
            f"  NEW [Q{p.query_idx}]: {_truncate(p.query_text)}\n"
            f"  EXISTING [M{p.match_idx}]: {_truncate(p.match_text)}"
        )
    if p.label == Label.NEW:
        return f"[Q{p.query_idx}] ONLY IN NEW DOCUMENT\n  {_truncate(p.query_text)}"
    return f"[M{p.match_idx}] ONLY IN EXISTING DOCUMENT\n  {_truncate(p.match_text)}"


def difference_items(alignment: Alignment) -> list[AlignedPassage]:
    """The passages worth describing: everything except unchanged ones, modified first."""
    return (
        alignment.by_label(Label.MODIFIED)
        + alignment.by_label(Label.NEW)
        + alignment.by_label(Label.ABSENT)
    )


def build_prompt(query: Document, match: Document, alignment: Alignment, items) -> str:
    counts = alignment.counts()
    stats = ", ".join(f"{counts[label]} {label.value}" for label in Label)
    body = "\n\n".join(_format_item(p) for p in items)
    return (
        f"{_doc_header('NEW DOCUMENT', query)}\n\n{_doc_header('EXISTING DOCUMENT', match)}\n\n"
        f"Passage alignment summary: {stats}.\n\n"
        f"Differing passages:\n\n{body}\n\n"
        f"Describe how the NEW document differs from the EXISTING document.\n{SECTIONS}"
    )


def build_reduce_prompt(query: Document, match: Document, partials: list[str]) -> str:
    notes = "\n\n---\n\n".join(partials)
    return (
        f"{_doc_header('NEW DOCUMENT', query)}\n\n{_doc_header('EXISTING DOCUMENT', match)}\n\n"
        "Below are partial comparison notes, each written from a different subset of passages. "
        "Merge them into one coherent comparison. Remove duplicates, keep all passage citations, "
        f"and do not add claims that are not in the notes.\n\n{notes}\n\n{SECTIONS}"
    )


class Describer(Protocol):
    def describe(self, query: Document, match: Document, alignment: Alignment) -> str | None: ...


class NullDescriber:
    """No LLM: the report contains only the deterministic alignment."""

    def describe(self, query: Document, match: Document, alignment: Alignment) -> str | None:
        return None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaDescriber:
    """Describe differences with a local model served by Ollama (https://ollama.com)."""

    def __init__(
        self,
        model: str,
        host: str,
        batch_items: int = 12,
        max_items: int = 48,
        num_ctx: int = 16384,
        temperature: float = 0.2,
    ):
        import ollama

        self.model = model
        self.batch_items = batch_items
        self.max_items = max_items
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        self._client = ollama.Client(host=host)

    def _chat(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options=self.options,
            think=False,
        )
        return _THINK_BLOCK.sub("", response.message.content).strip()

    def check_available(self) -> None:
        """Raise a helpful error if the Ollama server or model is missing."""
        try:
            installed = {m.model for m in self._client.list().models}
        except Exception as err:
            raise RuntimeError(
                "Cannot reach Ollama. Install it (`brew install ollama`), start it "
                "(`ollama serve`), or rerun with --no-llm."
            ) from err
        if self.model not in installed and f"{self.model}:latest" not in installed:
            raise RuntimeError(
                f"Ollama model {self.model!r} not found. Run `ollama pull {self.model}`."
            )

    def describe(self, query: Document, match: Document, alignment: Alignment) -> str | None:
        items = difference_items(alignment)
        if not items:
            return "All passages are near-identical; no substantive differences detected."
        omitted = max(0, len(items) - self.max_items)
        items = items[: self.max_items]
        batches = [items[i : i + self.batch_items] for i in range(0, len(items), self.batch_items)]
        partials = [self._chat(build_prompt(query, match, alignment, b)) for b in batches]
        text = (
            partials[0]
            if len(partials) == 1
            else self._chat(build_reduce_prompt(query, match, partials))
        )
        if omitted:
            text += (
                f"\n\n_Note: {omitted} lower-priority differing passages were not sent to the "
                "model; see the alignment table._"
            )
        return text
