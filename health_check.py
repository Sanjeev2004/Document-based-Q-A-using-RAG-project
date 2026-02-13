"""
Pre-flight health checks for the RAG application.

Usage:
    python health_check.py
    python health_check.py --skip-llm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List

from huggingface_hub import InferenceClient

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
)
from src.vectorstore import repair_chroma_directory


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_env() -> CheckResult:
    if not HUGGINGFACE_API_KEY:
        return CheckResult(
            name="Environment",
            ok=False,
            detail="HUGGINGFACE_API_KEY is missing in .env.",
        )
    return CheckResult(
        name="Environment",
        ok=True,
        detail="Required environment variables look set.",
    )


def check_chroma() -> CheckResult:
    try:
        probe = (
            "from chromadb import PersistentClient; "
            f"c=PersistentClient(path={CHROMA_PERSIST_DIRECTORY!r}); "
            "print(len(c.list_collections()))"
        )
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            raw_detail = completed.stderr.strip() or completed.stdout.strip() or "Unknown error."
            detail = raw_detail.splitlines()[0]
            return CheckResult(
                name="ChromaDB",
                ok=False,
                detail=f"Failed to open ChromaDB: {detail}",
            )
        collections_count = completed.stdout.strip() or "0"
        return CheckResult(
            name="ChromaDB",
            ok=True,
            detail=f"Connected to {CHROMA_PERSIST_DIRECTORY}. Collections: {collections_count}.",
        )
    except BaseException as exc:
        return CheckResult(
            name="ChromaDB",
            ok=False,
            detail=f"Failed to open ChromaDB: {exc}",
        )


def check_huggingface_model_access() -> CheckResult:
    if not HUGGINGFACE_API_KEY:
        return CheckResult(
            name="Hugging Face Model Access",
            ok=False,
            detail="Skipped because HUGGINGFACE_API_KEY is missing.",
        )

    try:
        client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
        response = client.chat_completion(
            model=HUGGINGFACE_MODEL,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=8,
            temperature=0.0,
        )
        text = response.choices[0].message.content if response and response.choices else ""
        if not text:
            raise RuntimeError("Empty response from model.")
        return CheckResult(
            name="Hugging Face Model Access",
            ok=True,
            detail=f"Model reachable: {HUGGINGFACE_MODEL}",
        )
    except BaseException as exc:
        return CheckResult(
            name="Hugging Face Model Access",
            ok=False,
            detail=f"Failed to invoke model {HUGGINGFACE_MODEL}: {exc}",
        )


def run_checks(skip_llm: bool, repair_chroma: bool) -> List[CheckResult]:
    chroma_result = check_chroma()
    results = [check_env()]
    if repair_chroma and not chroma_result.ok:
        backup_dir = repair_chroma_directory()
        chroma_result = CheckResult(
            name="ChromaDB",
            ok=False,
            detail=(
                f"{chroma_result.detail} Auto-repair moved old DB to {backup_dir}. "
                "Re-run health check now."
            ),
        )
    results.append(chroma_result)
    if skip_llm:
        results.append(
            CheckResult(
                name="Hugging Face Model Access",
                ok=True,
                detail="Skipped by --skip-llm flag.",
            )
        )
    else:
        results.append(check_huggingface_model_access())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-flight checks for the RAG app.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip remote LLM access check.",
    )
    parser.add_argument(
        "--repair-chroma",
        action="store_true",
        help="If Chroma check fails, move current DB aside and recreate an empty one.",
    )
    args = parser.parse_args()

    results = run_checks(skip_llm=args.skip_llm, repair_chroma=args.repair_chroma)
    failed = [r for r in results if not r.ok]

    print("RAG Health Check")
    print("=" * 40)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    if failed:
        print("=" * 40)
        print(f"Overall: FAIL ({len(failed)} check(s) failed)")
        return 1

    print("=" * 40)
    print("Overall: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
