"""Shared LLM factory for all agents."""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return an OpenAI chat client."""
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=api_key,
        temperature=0.3,
    )
