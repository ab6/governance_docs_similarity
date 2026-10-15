"""Render a ComparisonResult as Markdown."""

from governance_docs_similarity.alignment import AlignedPassage, Label
from governance_docs_similarity.pipeline import ComparisonResult, MatchComparison

SNIPPET_WORDS = 40


def _snippet(text: str | None) -> str:
    if not text:
        return ""
    words = text.split()
    snippet = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
    return snippet.replace("|", "\\|")


def _cell(value: str) -> str:
    return (value or "—").replace("|", "\\|").replace("\n", " ")


def _matches_table(result: ComparisonResult) -> list[str]:
    lines = [
        "| # | Document | AGORA ID | Score | Covers new doc | Covered by new doc | Authority | Date | Collections |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, c in enumerate(result.comparisons, 1):
        m, meta = c.match, c.match.document.metadata
        lines.append(
            f"| {rank} | {_cell(m.document.display_name)} | {m.document.doc_id} | {m.score:.3f} "
            f"| {m.query_coverage:.3f} | {m.doc_coverage:.3f} | {_cell(meta.get('authority', ''))} "
            f"| {_cell(meta.get('date', ''))} | {_cell(meta.get('collections', ''))} |"
        )
    return lines


def _overview(result: ComparisonResult) -> list[str]:
    s = result.settings
    n = len(result.query.chunks)
    lines = [
        f"- The new document was split into **{n} passages**.",
        (
            f"- **{result.novelty_share:.0%}** of its passages have no close counterpart "
            f"(similarity ≥ {s.modified_threshold}) anywhere in the corpus, "
            "so they are the most likely to be new policy."
        ),
    ]
    if result.comparisons:
        top = result.comparisons[0]
        counts = top.alignment.counts()
        carried = counts[Label.UNCHANGED] + counts[Label.MODIFIED]
        lines.append(
            f"- Closest predecessor: **{top.match.document.display_name}**. "
            f"{carried}/{n} new passages carry over from it "
            f"({counts[Label.UNCHANGED]} near-identical, {counts[Label.MODIFIED]} modified), and "
            f"{counts[Label.ABSENT]} of its passages do not appear in the new document."
        )
        dates = sorted(
            d for c in result.comparisons if (d := c.match.document.metadata.get("date"))
        )
        if dates:
            lines.append(f"- Top matches span activity dates **{dates[0]} → {dates[-1]}**.")
    if result.excluded_ids:
        lines.append(f"- Excluded from matching: {', '.join(result.excluded_ids)}.")
    return lines


def _alignment_row(p: AlignedPassage) -> str:
    return f"| {p.ref} | {p.label.value} | {p.score:.2f} | {_snippet(p.query_text)} | {_snippet(p.match_text)} |"


def _comparison_section(rank: int, c: MatchComparison) -> list[str]:
    doc = c.match.document
    counts = c.alignment.counts()
    lines = [f"## {rank}. {doc.display_name} (AGORA {doc.doc_id})", ""]
    if summary := doc.metadata.get("short_summary"):
        lines += [f"> {summary}", ""]
    lines += [
        "Passage alignment: " + ", ".join(f"{counts[label]} {label.value}" for label in Label),
        "",
    ]
    if c.description:
        lines += ["### How the new document differs", "", c.description, ""]
    differing = [p for p in c.alignment.passages if p.label != Label.UNCHANGED]
    if differing:
        lines += [
            "<details><summary>Differing passages</summary>",
            "",
            "| Ref | Label | Similarity | New document | Matched document |",
            "|---|---|---|---|---|",
            *(_alignment_row(p) for p in differing),
            "",
            "</details>",
            "",
        ]
    return lines


def render_report(result: ComparisonResult) -> str:
    query = result.query.document
    lines = [
        f"# Governance document comparison: {query.display_name}",
        "",
        "## Overview",
        "",
        *_overview(result),
        "",
        "## Most similar documents",
        "",
        *_matches_table(result),
        "",
        (
            "_Score is the mean of the two coverage columns: how well the matched document's "
            "passages explain the new document, and how much of the matched document survives "
            "in the new one._"
        ),
        "",
    ]
    for rank, c in enumerate(result.comparisons, 1):
        lines += _comparison_section(rank, c)
    lines += [
        "---",
        (
            f"_Embedding model: {result.settings.embedding_model}. Thresholds: unchanged ≥ "
            f"{result.settings.unchanged_threshold}, "
            f"modified ≥ {result.settings.modified_threshold}. "
            "Passage refs: Q = new document, M = matched document._"
        ),
    ]
    return "\n".join(lines) + "\n"


def render_matches(result_matches, query_name: str) -> str:
    lines = [f"Most similar documents to {query_name}:", ""]
    for rank, m in enumerate(result_matches, 1):
        meta = m.document.metadata
        lines.append(
            f"{rank:>2}. [{m.score:.3f}] {m.document.display_name} (AGORA {m.document.doc_id}"
            f"{', ' + meta['authority'] if meta.get('authority') else ''}"
            f"{', ' + meta['date'] if meta.get('date') else ''})"
        )
    return "\n".join(lines)
