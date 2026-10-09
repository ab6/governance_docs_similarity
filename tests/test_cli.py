from typer.testing import CliRunner

from governance_docs_similarity import cli, config
from governance_docs_similarity.describe import build_prompt, difference_items
from governance_docs_similarity.pipeline import compare_document
from governance_docs_similarity.report import render_report
from tests.conftest import DOCS, HashingEmbedder

AMENDED = DOCS["1"][1] + (
    "\n\nSEC. 3. PENALTIES.\n\nAn agency that violates this Act shall pay a civil penalty of "
    "ten thousand dollars per violation."
)


def test_compare_pipeline_and_report(index, embedder, settings):
    from governance_docs_similarity.document import Document

    query = Document("amended", AMENDED, title="Amended FRT Act")
    result = compare_document(query, index, embedder, settings)
    assert result.comparisons[0].match.document.doc_id == "1"
    report = render_report(result)
    assert "Amended FRT Act" in report
    assert "Facial Recognition Act" in report

    alignment = result.comparisons[0].alignment
    prompt = build_prompt(query, result.comparisons[0].match.document, alignment, difference_items(alignment))
    assert "EXISTING DOCUMENT: Facial Recognition Act" in prompt


def test_cli_compare_no_llm(index, tmp_path, monkeypatch, settings):
    index_dir = tmp_path / "idx"
    index.save(index_dir)
    doc = tmp_path / "amended.txt"
    doc.write_text(AMENDED)
    out = tmp_path / "report.md"
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "_embedder", lambda s: HashingEmbedder())

    result = CliRunner().invoke(
        cli.app,
        ["compare", str(doc), "--no-llm", "--index-dir", str(index_dir), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "Most similar documents" in out.read_text()


def test_cli_rejects_stale_index(index, tmp_path, settings, monkeypatch):
    index_dir = tmp_path / "idx"
    index.save(index_dir)
    doc = tmp_path / "amended.txt"
    doc.write_text(AMENDED)
    monkeypatch.setattr(config, "get_settings", lambda: settings.__class__(embedding_model="other"))
    result = CliRunner().invoke(cli.app, ["similar", str(doc), "--index-dir", str(index_dir)])
    assert result.exit_code != 0
    assert "stale" in str(result.exception)
