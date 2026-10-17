"""LLM and embedding helpers shared by all agents.

Gemini models are called through Google's OpenAI-compatible endpoint with
GEMINI_API_KEY; OpenAI models and embeddings use OPENAI_API_KEY.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from openai import OpenAI

import config

DEFAULT_SYSTEM_PROMPT = "You are an expert in analog circuit design."

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```")


def _client_for(model: str) -> OpenAI:
    if model.startswith("gpt"):
        return OpenAI(api_key=config.require_env("OPENAI_API_KEY"))
    return OpenAI(api_key=config.require_env("GEMINI_API_KEY"), base_url=config.GEMINI_BASE_URL)


def ask_messages(messages: List[Dict[str, str]], model: str = config.DESIGN_MODEL) -> str:
    """Send a chat history to the model and return the reply text."""
    response = _client_for(model).chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def ask(prompt: str, model: str = config.DESIGN_MODEL, system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Send one user prompt with a system prompt and return the reply text."""
    return ask_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=model,
    )


def embed(text: str) -> List[float]:
    """Return the embedding of text (OpenAI text-embedding-3-large)."""
    client = OpenAI(api_key=config.require_env("OPENAI_API_KEY"))
    response = client.embeddings.create(input=text, model=config.EMBEDDING_MODEL)
    return response.data[0].embedding


def extract_code(text: str) -> List[str]:
    """Return the contents of all fenced code blocks in text."""
    return [m.strip() for m in _CODE_BLOCK_RE.findall(text)]


def parse_assignment(code: str, name: str) -> Any:
    """Return the literal assigned to `name` in an LLM code snippet, without executing it."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"No assignment to {name!r} found in LLM output.")
