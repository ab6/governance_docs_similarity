# Governance Docs Similarity

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<img src="https://img.shields.io/badge/python-3.14-blue" />
<img src="https://img.shields.io/badge/license-MIT-green" />

**See how a new AI governance document relates to the ones that came before it.**

# Project Overview

AI regulation is rarely written from scratch. New bills reuse provisions from earlier bills.
National Defense Authorization Acts carry AI sections forward from year to year. Standalone bills
get folded into larger packages. Reading each new document as if it were entirely new hides how
the policy has actually changed.

This project provides a command-line tool, `gds`. You give it a new governance document and it:

1. **Finds the most similar documents** in a corpus of 1,251 AI governance documents (the
   [ETO AGORA](https://eto.tech/dataset-docs/agora/) dataset).
2. **Aligns their passages.** Each passage of the new document is labelled as *unchanged*,
   *modified* or *new*. Passages from the matched document that don't appear in the new one are
   labelled *absent*.
3. **Describes the differences.** A local open-source LLM (served by Ollama) writes the comparison
   under five headings: scope, obligations, enforcement, definitions, and other changes. Every
   claim cites the passages it is based on.
4. **Writes a Markdown report** with the ranked matches, the descriptions and the
   passage-by-passage alignment.

The LLM is behind a swappable `Describer` interface. A local model is used during development to
avoid API costs, and the plan is to move to the Claude API once the application matures.

# Installation and Setup

## Codes and Resources Used
- **Python:** 3.14.6
- **Package and environment manager:** [uv](https://docs.astral.sh/uv/)
- **Local LLM runtime:** [Ollama](https://ollama.com). Only needed for the generated difference
  descriptions; everything else runs without it.
- **Hardware:** developed on an Apple M3 Pro with 36 GB of RAM. Embeddings use MPS, CUDA or the
  CPU, whichever is available.

## Python Packages Used
- **Embeddings and modeling:** `sentence-transformers`, `torch`, `scikit-learn`
- **Data handling:** `pandas`, `pyarrow`, `numpy`
- **LLM client:** `ollama`
- **CLI and configuration:** `typer`, `python-dotenv`
- **Development:** `pytest`, `ruff`

## Setup
```bash
make create_environment   # uv venv with Python 3.14.6
make requirements         # uv sync

# Optional: local LLM for the difference descriptions
brew install ollama
ollama serve              # leave running in a separate terminal
ollama pull qwen3:8b
```

## Usage
```bash
# 1. Build the passage index (one time, about 1.5 minutes)
make index                                  # or: uv run gds build-index

# 2. List the most similar documents
uv run gds similar path/to/new_bill.txt --k 10

# 3. Full comparison report, written to reports/comparisons/<name>.md
uv run gds compare path/to/new_bill.txt
uv run gds compare path/to/new_bill.txt --no-llm       # alignment only, no Ollama needed
uv run gds compare data/raw/agora/fulltext/2682.txt --exclude-id 2682   # test with a corpus doc
```

Main options for `compare`:

| Option | What it does | Default |
|---|---|---|
| `--k` | Number of similar documents to report | `5` |
| `--describe-top` | Write LLM descriptions for only the top N matches | `3` |
| `--exclude-id` | AGORA ID to leave out of matching; can be repeated | none |
| `--llm-model` | Ollama model to use | `qwen3:8b` |
| `--no-llm` | Skip the LLM descriptions | off |
| `--out` | Path for the report | `reports/comparisons/<name>.md` |

You can override any default with a `GDS_*` environment variable or a `.env` file. The settings
are listed in [`config.py`](governance_docs_similarity/config.py). Examples:
`GDS_LLM_MODEL=llama3.1:8b`, `GDS_EMBEDDING_MODEL=...`, `GDS_MODIFIED_THRESHOLD=0.7`. If you change
the embedding model or the chunking settings, rebuild the index.

# Data

## Source Data
**AGORA (AI GOvernance and Regulatory Archive)**, version 1.33.0, from Georgetown CSET's
[Emerging Technology Observatory](https://eto.tech/dataset-docs/agora/). It is a collection of
AI-relevant laws, regulations, standards and other governance documents from the United States
and around the world.

| File | Contents |
|---|---|
| `fulltext/<AGORA ID>.txt` | Full text of 1,251 documents, from 15 words to about 53,000 (median about 1,000) |
| `documents.csv` | Metadata for each document: official name, authority, collections, dates, summaries and thematic tags |
| `segments.csv` | Annotated segments of each document |
| `collections.csv`, `authorities.csv` | Lookup tables for collections and issuing authorities |

The full datasheet, written following [Gebru et al., 2021](https://arxiv.org/abs/1803.09010), is
at [`data/datasheets/DATASHEET-AGORA.md`](data/datasheets/DATASHEET-AGORA.md). The corpus leans
toward U.S. law and policy.

## Data Acquisition
Download the AGORA dataset from ETO and unpack it to `data/raw/agora/`, so that you have
`data/raw/agora/documents.csv` and `data/raw/agora/fulltext/*.txt`. The raw data is not committed.
Use of the data is subject to ETO's terms of use; please cite ETO.

## Data Preprocessing
- **Metadata join:** each full-text file is matched to its row in `documents.csv` by AGORA ID.
  Duplicate tags are removed.
- **Chunking:** documents are split on paragraph breaks and on section, article, chapter, part
  and title headers. Paragraphs are packed into passages of up to 200 words. Longer paragraphs are
  split into windows that overlap by 30 words. New documents are chunked by the same code, so
  passages on both sides are comparable.
- **Embedding:** each passage is embedded with
  [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), which reads up to
  512 tokens and produces L2-normalized 384-dimensional vectors. The current corpus produces
  14,521 passages.
- **Index:** stored in `data/processed/agora_index/`. A manifest records the embedding model and
  chunk settings, and the CLI refuses to use an index that doesn't match the current settings.

# Code structure
```
├── LICENSE
├── Makefile                      <- make requirements | test | lint | index | compare
├── README.md
├── pyproject.toml                <- dependencies and the `gds` console script
├── uv.lock
├── data
│   ├── datasheets                <- AGORA datasheet and the scripts that generate it
│   ├── raw/agora                 <- AGORA dataset (not committed)
│   └── processed/agora_index     <- passage embedding index (generated)
├── reports
│   └── comparisons               <- generated comparison reports
├── tests                         <- pytest suite (offline: fixture corpus and hashing embedder)
└── governance_docs_similarity
    ├── cli.py                    <- `gds` commands: build-index, similar, compare
    ├── config.py                 <- paths and settings, overridable with GDS_* env vars
    ├── document.py               <- Document dataclass
    ├── corpus.py                 <- loads AGORA full text and metadata
    ├── loaders.py                <- input loading by file type (.txt today)
    ├── chunking.py               <- splits text into passages
    ├── embeddings.py             <- sentence-transformers wrapper
    ├── index.py                  <- builds, saves and loads the passage index
    ├── retrieval.py              <- ranks documents by passage coverage
    ├── alignment.py              <- labels passages unchanged/modified/new/absent
    ├── describe.py               <- Describer interface, Ollama backend, prompts
    ├── pipeline.py               <- end-to-end comparison
    ├── report.py                 <- Markdown report rendering
    ├── datasheet_generator.py    <- datasheet generator (Gebru et al.)
    └── model_card_generator.py   <- model card generator
```

# Results and evaluation

**Ranking.** For each corpus document, the tool computes two coverage scores:
- how well the document's passages explain the new document
- how much of the document survives in the new one

A document's score is the mean of the two. Counting both directions keeps large omnibus bills from
matching everything.

**Retrieval check (leave-one-out).** Each test document was removed from the corpus and its top 5
matches were checked:

| Test set | How a match counts as relevant | Precision@5 |
|---|---|---|
| All 37 FY2026 NDAA provisions | The match is an NDAA provision from an earlier fiscal year | 0.67 |
| 200 randomly sampled documents | The match shares a collection and an authority with the test document | 0.67 |

These relevance labels are rough: they come from the metadata, not from human review.

**Example.** For FY2026 NDAA §224, *National Security and Defense AI Institute*, the top matches
are the *GUARD Act of 2025* (score 0.94) and the earlier *GUARD Act* (0.94). Those are the
standalone bills whose text was folded into the NDAA. Next comes the FY2021 NDAA *National AI
Initiative Act* (0.81). Three of the four passages are carried over nearly word for word; the
remaining one differs only in the bill's preamble.

**Tests.** `make test` runs offline. It uses a small fixture corpus and a deterministic hashing
embedder.

# Future work
- **More input formats:** PDF, HTML and URL loaders, registered in `loaders.py`.
- **Claude API backend:** a `ClaudeDescriber` in `describe.py`, for better descriptions of long
  legal text.
- **Tuning:** set the unchanged/modified thresholds and the chunk size against labelled pairs, and
  exclude boilerplate such as enacting clauses from the alignment.
- **Evaluation:** turn the leave-one-out check into a reusable evaluation notebook, and add human
  judgments of description quality.
- **Interface:** a web interface once the CLI workflow is stable.

# Acknowledgments/References
- Emerging Technology Observatory, Georgetown CSET: [AGORA dataset](https://eto.tech/dataset-docs/agora/)
- Gebru et al., 2021: [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- Mitchell et al., 2019: [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [BGE embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5), BAAI
- Project layout from [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- README format from [pragyy/datascience-readme-template](https://github.com/pragyy/datascience-readme-template)

# License
[MIT License](LICENSE). Copyright (c) 2026, Amber McKenzie.
