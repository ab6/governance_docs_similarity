"""`gds` command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from governance_docs_similarity import config

app = typer.Typer(
    help="Compare a new AI governance document with the AGORA corpus.", no_args_is_help=True
)


def _log(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BRIGHT_BLACK)


def _load_index(index_dir: Path, settings: config.Settings):
    from governance_docs_similarity.index import CorpusIndex

    index = CorpusIndex.load(index_dir)
    index.check_compatible(settings.embedding_model, settings.chunk_words, settings.chunk_overlap)
    return index


def _embedder(settings: config.Settings):
    from governance_docs_similarity.embeddings import SentenceTransformerEmbedder

    _log(f"Loading embedding model {settings.embedding_model}")
    return SentenceTransformerEmbedder(settings.embedding_model)


IndexDir = Annotated[Path, typer.Option(help="Where the corpus index is stored.")]
ExcludeIds = Annotated[
    list[str] | None,
    typer.Option("--exclude-id", help="AGORA ID(s) to leave out of matching (repeatable)."),
]


@app.command("build-index")
def build_index_cmd(
    agora_dir: Annotated[Path, typer.Option(help="AGORA dataset directory.")] = config.AGORA_DIR,
    index_dir: IndexDir = config.INDEX_DIR,
):
    """Chunk and embed the AGORA corpus and save the index."""
    from governance_docs_similarity.corpus import load_agora_corpus
    from governance_docs_similarity.index import build_index

    settings = config.get_settings()
    documents = load_agora_corpus(agora_dir)
    _log(f"Loaded {len(documents)} documents from {agora_dir}")
    index = build_index(
        documents,
        _embedder(settings),
        settings.chunk_words,
        settings.chunk_overlap,
        show_progress=True,
    )
    index.save(index_dir)
    typer.echo(
        f"Indexed {index.manifest['n_documents']} documents / {index.manifest['n_chunks']} "
        f"passages -> {index_dir}"
    )


@app.command()
def similar(
    document: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    k: Annotated[int, typer.Option(help="Number of matches.")] = 10,
    exclude_id: ExcludeIds = None,
    index_dir: IndexDir = config.INDEX_DIR,
):
    """List the corpus documents most similar to DOCUMENT (no difference analysis)."""
    from governance_docs_similarity.loaders import load_document
    from governance_docs_similarity.report import render_matches
    from governance_docs_similarity.retrieval import encode_query, find_similar

    settings = config.get_settings()
    index = _load_index(index_dir, settings)
    doc = load_document(document)
    query = encode_query(
        doc, index, _embedder(settings), settings.chunk_words, settings.chunk_overlap
    )
    matches = find_similar(query, index, k=k, exclude_ids=exclude_id or [])
    typer.echo(render_matches(matches, doc.display_name))


@app.command()
def compare(
    document: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    k: Annotated[int | None, typer.Option(help="Number of similar documents.")] = None,
    exclude_id: ExcludeIds = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Skip LLM descriptions; alignment only.")
    ] = False,
    llm_model: Annotated[str | None, typer.Option(help="Ollama model name.")] = None,
    describe_top: Annotated[
        int, typer.Option(help="Generate LLM descriptions for the top N matches only.")
    ] = 3,
    out: Annotated[Path | None, typer.Option(help="Report path (Markdown).")] = None,
    index_dir: IndexDir = config.INDEX_DIR,
):
    """Find documents similar to DOCUMENT and describe how it differs from them."""
    from governance_docs_similarity.describe import NullDescriber, OllamaDescriber
    from governance_docs_similarity.loaders import load_document
    from governance_docs_similarity.pipeline import compare_document
    from governance_docs_similarity.report import render_report

    settings = config.get_settings()
    if no_llm:
        describer = NullDescriber()
    else:
        describer = OllamaDescriber(
            model=llm_model or settings.llm_model,
            host=settings.llm_host,
            batch_items=settings.llm_batch_items,
        )
        try:
            describer.check_available()
        except RuntimeError as err:
            typer.secho(str(err), err=True, fg=typer.colors.RED)
            raise typer.Exit(1)

    index = _load_index(index_dir, settings)
    doc = load_document(document)
    result = compare_document(
        doc,
        index,
        _embedder(settings),
        settings,
        k=k,
        exclude_ids=exclude_id or [],
        describer=describer,
        describe_top=describe_top,
        on_progress=_log,
    )
    report = render_report(result)
    out = out or config.REPORTS_DIR / f"{document.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    typer.echo(report)
    _log(f"Report written to {out}")


if __name__ == "__main__":
    app()
