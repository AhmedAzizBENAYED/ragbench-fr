"""PIAF source file: download at a pinned revision and parse the parquet rows."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ragbench.data import DataError
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.log import get_logger

logger = get_logger(__name__)

COLUMNS = ("id", "title", "context", "question", "answers")


@dataclass(frozen=True)
class RawRecord:
    """One PIAF row, as published (no normalisation)."""

    id: str
    title: str
    context: str
    question: str
    answer_texts: tuple[str, ...]
    answer_starts: tuple[int, ...]


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(config: CorpusBuildConfig, raw_dir: Path) -> Path:
    """Download the source file at the pinned revision into `raw_dir` and check its SHA-256.

    Raises:
        DataError: the download failed or the file does not have the expected SHA-256.
    """
    try:
        result = hf_hub_download(
            repo_id=config.dataset,
            filename=config.filename,
            repo_type="dataset",
            revision=config.revision,
            local_dir=raw_dir,
        )
    # HfHubHTTPError (HTTP failures) and LocalEntryNotFoundError (offline) subclass OSError.
    except OSError as error:
        raise DataError(
            f"cannot download {config.filename} from {config.dataset}@{config.revision}: {error}"
        ) from error
    if not isinstance(result, str):
        raise DataError(f"unexpected download result for {config.filename}: {result!r}")

    path = Path(result)
    actual = file_sha256(path)
    if actual != config.raw_sha256:
        raise DataError(f"{path}: SHA-256 is {actual}, expected {config.raw_sha256}")
    logger.info("raw.downloaded", dataset=config.dataset, revision=config.revision, sha256=actual)
    return path


def _record(row: dict[str, Any]) -> RawRecord:
    answers = row.get("answers")
    if not isinstance(answers, dict):
        raise DataError(f"row {row.get('id')!r}: answers must be a mapping")
    texts, starts = answers.get("text"), answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
        raise DataError(f"row {row.get('id')!r}: answers.text and answers.answer_start mismatch")
    fields = [row.get(name) for name in ("id", "title", "context", "question")]
    if not all(isinstance(value, str) for value in fields):
        raise DataError(f"row {row.get('id')!r}: id, title, context and question must be strings")
    return RawRecord(
        id=row["id"],
        title=row["title"],
        context=row["context"],
        question=row["question"],
        answer_texts=tuple(str(text) for text in texts),
        answer_starts=tuple(int(start) for start in starts),
    )


def read_raw(path: Path) -> list[RawRecord]:
    """Parse the PIAF parquet file into records (the only place that uses pyarrow).

    Raises:
        DataError: the file cannot be parsed or a row does not have the PIAF schema.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        rows = pq.read_table(path, columns=list(COLUMNS)).to_pylist()
    except (pa.ArrowException, OSError) as error:
        raise DataError(f"{path}: cannot read parquet file: {error}") from error
    return [_record(row) for row in rows]
