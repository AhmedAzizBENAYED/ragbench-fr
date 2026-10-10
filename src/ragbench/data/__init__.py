"""Data: PIAF download, deduplicated corpus, frozen splits and manifest (spec §5)."""


class DataError(Exception):
    """The source data or a data artefact is missing, corrupted or inconsistent."""
