"""Load a dataset from either a Hugging Face Hub id or a local path.

Both `datasets.load_dataset` and local file loading are handled behind one
entry point so callers (training, the CLI) don't need to know which kind of
source string they were given.
"""

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

_SUFFIX_TO_BUILDER = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
}


def load_local_dataset(path: str, split: str | None = None) -> Dataset | DatasetDict:
    """Load a dataset from a local CSV/JSON(L) file or a `save_to_disk` directory."""
    local_path = Path(path)
    if local_path.is_dir():
        return load_from_disk(str(local_path))

    builder = _SUFFIX_TO_BUILDER.get(local_path.suffix.lower())
    if builder is None:
        raise ValueError(
            f"Unsupported local dataset file type: {local_path.suffix!r} ({path})"
        )
    dataset = load_dataset(builder, data_files=str(local_path), split=split or "train")
    return dataset


def load_data(source: str, split: str | None = None) -> Dataset | DatasetDict:
    """Load a dataset from `source`.

    `source` is either a local filesystem path (file or `save_to_disk` directory)
    or a Hugging Face Hub dataset id (e.g. "cornell-movie-review-data/rotten_tomatoes").
    """
    if Path(source).exists():
        return load_local_dataset(source, split=split)
    return load_dataset(source, split=split)
