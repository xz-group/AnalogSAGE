"""Topology Selection Agent: knowledge retrieval and the natural-language topology proposal."""

from __future__ import annotations

import logging
from typing import TextIO, Tuple

import knowledge
import llm
from circuits import Circuit

logger = logging.getLogger(__name__)


def get_knowledge_docs(
    circuit: Circuit, spec: str, model: str, log: TextIO, memory_prompt: str = "", reflection_prompt: str = ""
) -> str:
    """Ask the LLM for a knowledge query and return the Knowledge Module summary ('' on failure)."""
    zero_query = memory_prompt + reflection_prompt + f"""
I want to design {circuit.design_noun} using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
I have an knowledge agent that can help me find relevant knowledge documents about analog circuit design from a knowledge base.
Please help me to write a query to ask the knowledge agent to get relevant knowledge documents to help me design the {circuit.short_noun}.
First analyze the specifications and then output the query text in a code block.
Example
```text
{circuit.knowledge_example}
```
"""
    try:
        answer = llm.ask(zero_query, model=model)
        log.write("Q:\n" + zero_query + "\nA:\n" + answer + "\n")
        knowledge_query = llm.extract_code(answer)[0]
        docs = knowledge.getKnowledge(knowledge_query, mytop_k=5)
        log.write("Knowledge docs:\n" + str(docs) + "\n")
        return docs
    except Exception as e:  # retrieval is optional; continue without knowledge
        logger.warning("Knowledge retrieval skipped: %s", e)
        log.write(f"Knowledge retrieval skipped: {e}\n")
        return ""


def propose_topology(
    circuit: Circuit, spec: str, knowledge_docs: str, model: str, log: TextIO, memory_prompt: str = ""
) -> Tuple[str, str]:
    """Return (prompt, answer): a natural-language description of candidate topologies."""
    first_query = memory_prompt + "\n" + f"""
Please describe possible topology designs of {circuit.proposal_noun} that meets the following specifications:
{spec}
I have the following relevant knowledge documents to help me design the {circuit.short_noun}:
{knowledge_docs}
"""
    answer = llm.ask(first_query, model=model)
    log.write("Q:\n" + first_query + "\nA:\n" + answer + "\n")
    return first_query, answer
