"""Reflection Module (Introspective Optimization Layer)."""

from __future__ import annotations

import config
import llm


def get_reflection(history: str) -> str:
    """Diagnose why a design iteration failed, at the topology level, for the next iteration."""
    prompt = history + "\n You were unsuccessful in completing the design task in this trial. The above is the entire chat history of the design process in this agent. In a few sentences, diagnose a possible reason for failure that aims to mitigate the same failure. Use complete sentences. Note that the next trial will start from topology selection with this reflection. Do not mention detail sizing as the next trial will have a new sizing process under another topology. Please focus on the topology level and include the current topology because the next trial will not know the topology used in this trial. "
    return llm.ask(prompt, model=config.DESIGN_MODEL)
