"""Shared RAG service for the group chatbot and evaluation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation


def build_followup_query(question: str, history: list[dict] | None = None) -> str:
    """Add a tiny amount of conversation memory for short follow-up questions."""
    if not history:
        return question

    short_followup = len(question.split()) <= 8
    has_reference_word = any(word in question.lower() for word in ["này", "đó", "trên", "vậy", "họ"])
    if not (short_followup or has_reference_word):
        return question

    previous_user_turns = [turn["content"] for turn in history if turn.get("role") == "user"]
    if not previous_user_turns:
        return question

    return f"Ngữ cảnh câu hỏi trước: {previous_user_turns[-1]}\nCâu hỏi hiện tại: {question}"


def answer_question(question: str, history: list[dict] | None = None, top_k: int = 5) -> dict:
    """Run retrieval + generation and normalize the shape used by UI/eval."""
    rewritten_query = build_followup_query(question, history)
    result = generate_with_citation(rewritten_query, top_k=top_k)
    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieval_source": result.get("retrieval_source", "none"),
        "context": result.get("context", ""),
    }


def source_label(source: dict, index: int) -> str:
    metadata = source.get("metadata", {})
    name = metadata.get("source") or metadata.get("title") or f"Source {index}"
    score = source.get("score", 0.0)
    retrieval_source = source.get("source", "unknown")
    return f"{index}. {name} | {retrieval_source} | score={score:.3f}"
