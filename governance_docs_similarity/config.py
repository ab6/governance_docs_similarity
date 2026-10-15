"""Project paths and tunable defaults.

Every default can be overridden with an environment variable (or a `.env` file
at the project root) named `GDS_<SETTING>`, e.g. `GDS_LLM_MODEL=llama3.1:8b`.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
AGORA_DIR = DATA_DIR / "raw" / "agora"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "agora_index"
REPORTS_DIR = PROJECT_ROOT / "reports" / "comparisons"


def _env(name: str, default):
    value = os.getenv(f"GDS_{name}")
    if value is None:
        return default
    return type(default)(value)


@dataclass(frozen=True)
class Settings:
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    chunk_words: int = field(default_factory=lambda: _env("CHUNK_WORDS", 200))
    chunk_overlap: int = field(default_factory=lambda: _env("CHUNK_OVERLAP", 30))
    top_k: int = field(default_factory=lambda: _env("TOP_K", 5))
    # Passage alignment thresholds (cosine similarity).
    unchanged_threshold: float = field(default_factory=lambda: _env("UNCHANGED_THRESHOLD", 0.92))
    modified_threshold: float = field(default_factory=lambda: _env("MODIFIED_THRESHOLD", 0.75))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:8b"))
    llm_host: str = field(default_factory=lambda: _env("LLM_HOST", "http://localhost:11434"))
    # Max aligned items sent to the LLM in one prompt before map-reduce kicks in.
    llm_batch_items: int = field(default_factory=lambda: _env("LLM_BATCH_ITEMS", 12))


def get_settings() -> Settings:
    return Settings()
