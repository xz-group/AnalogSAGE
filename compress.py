"""Compression Module (Stage Context Fusion Layer).

designTask.py calls compress_feedback after a failed sizing; the summary is passed to the
Topology Refinement Agent and to the Reflection Module.
"""

from __future__ import annotations

import config
import llm


def compress_feedback(history: str, special_prompt: str = "") -> str:
    """Compress the design history of failed iterations into feedback for earlier stages."""
    prompt = history + "\n You were unsuccessful in completing the design task in the iterations. The above is the whole chat history of the design process. Please compress the design and the performance result among iterations as feedback to the previous steps to make it generate a better design. Use complete sentences. Summarize the key points and avoid redundancy. Be concise and to the point. "
    prompt += special_prompt
    return llm.ask(prompt, model=config.DESIGN_MODEL)
