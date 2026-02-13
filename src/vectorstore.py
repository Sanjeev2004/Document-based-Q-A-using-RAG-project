"""
Utilities for robust ChromaDB initialization.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime

from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIRECTORY


def _is_recoverable_chroma_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "pyo3_runtime.panicexception" in text
        or "range start index" in text
        or "could not connect to tenant default_tenant" in text
    )


def repair_chroma_directory() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_dir = CHROMA_PERSIST_DIRECTORY
    backup_dir = f"{source_dir}_corrupt_{ts}"

    if os.path.isdir(source_dir):
        shutil.move(source_dir, backup_dir)
    os.makedirs(source_dir, exist_ok=True)
    return backup_dir


def get_chroma_vectorstore(embeddings, allow_repair: bool = True) -> Chroma:
    """
    Build a Chroma vectorstore and auto-repair the persistence directory once
    when known Rust/tenant corruption errors occur.
    """
    try:
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
    except BaseException as exc:
        if not allow_repair or not _is_recoverable_chroma_error(exc):
            raise

        backup_dir = repair_chroma_directory()
        print(
            f"[Chroma Repair] Detected corrupted/incompatible DB. "
            f"Backed up old store to: {backup_dir}"
        )
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
