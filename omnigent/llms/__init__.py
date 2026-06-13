"""
Multi-provider LLM client with OpenAI Responses API interface.

Usage::

    from omnigent.llms import Client

    client = Client()
    resp = client.responses.create(
        input=[{"role": "user", "content": "Hello"}],
        instructions="You are a helpful assistant.",
        model="anthropic/claude-sonnet-4-20250514",
    )
"""

from omnigent.llms.client import Client
from omnigent.llms.context_window import get_model_context_window

__all__ = ["Client", "get_model_context_window"]
