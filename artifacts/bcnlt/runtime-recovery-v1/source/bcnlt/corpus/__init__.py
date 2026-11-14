"""Provenance-first corpus ingestion for BCNLT training data."""

from .database import CorpusDatabase, SourceDocumentRecord
from .sources import (
    SourceSpec,
    ingest_local_source,
    ingest_opencodeinstruct_stream,
    ingest_stack_stream,
)

__all__ = [
    "CorpusDatabase",
    "SourceDocumentRecord",
    "SourceSpec",
    "ingest_local_source",
    "ingest_opencodeinstruct_stream",
    "ingest_stack_stream",
]
