"""End-to-end test client for the Legal Multi-Agent System.

Sends a legal question to the Customer Agent and prints the response.
"""

import asyncio
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


def _extract_text_from_parts(parts: Any) -> str:
    """Extract text from A2A Part-like objects."""
    text = ""
    for part in parts or []:
        payload = getattr(part, "root", part)
        value = getattr(payload, "text", None)
        if value:
            text += value
    return text


def _extract_text(event: Any) -> str:
    """Extract text from Message, Task artifacts, or Task status message."""
    result = event
    if isinstance(event, tuple):
        result = event[0]

    text = _extract_text_from_parts(getattr(result, "parts", None))
    if text:
        return text

    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        for artifact in artifacts:
            text += _extract_text_from_parts(getattr(artifact, "parts", None))
        if text:
            return text

    status = getattr(result, "status", None)
    status_message = getattr(status, "message", None)
    return _extract_text_from_parts(getattr(status_message, "parts", None))


async def main() -> None:
    print(f"Connecting to Customer Agent at {CUSTOMER_AGENT_URL}")
    print(f"Question: {QUESTION}")
    print("-" * 60)

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        # Resolve agent card
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: Could not reach Customer Agent at {card_url}")
            print(f"  {e}")
            print("Make sure all services are running (./start_all.sh)")
            sys.exit(1)

        from a2a.client import ClientConfig, ClientFactory, create_text_message_object
        from a2a.types import AgentCard

        agent_card = AgentCard.model_validate(card_resp.json())
        print(f"Connected to agent: {agent_card.name} v{agent_card.version}")
        print("-" * 60)

        client = ClientFactory(
            ClientConfig(streaming=False, httpx_client=http_client)
        ).create(agent_card)
        message = create_text_message_object(content=QUESTION)

        print("Sending request (this may take 30-60s while agents chain)...\n")
        result_text = ""
        async for event in client.send_message(message):
            result_text = _extract_text(event) or result_text

        if not result_text:
            print("No text response received from the agent.")
            return

        print("RESPONSE:")
        print("=" * 60)
        print(result_text)
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user before the agent returned a final response.")
        sys.exit(130)
