"""Knowledge Module: query decomposition, Design Knowledge RAG retrieval and summarization."""

from __future__ import annotations

import logging

import config
import llm
import queryPinecone

logger = logging.getLogger(__name__)


def getKnowledge(query_text, mytop_k: int = 5, stage: str = "topology design") -> str:
    """Split the query into sub-queries, retrieve papers for each, and summarize them for `stage`."""
    history = [
        {
            "role": "system",
            "content": "You are an expert in analog circuit design.",
        },
    ]

    prompt = f"""
You are a knowledge retrieval agent for analog circuit design. Given a query, you will first convert the knowledge query into several RAG query and then, you will summary the retrieval documents.
Here is the query: {query_text}.
The current design stage is {stage}.
The knowledge base contains many published papers about analog circuit design but do not have the specific domain knowledge for a certain PDK.
First analysis the query and revise it into several sub-queries to retrieve relevant knowledge.
You should return the queries in the following python format:
```python
queries = ["query1", "query2", ...]
```
Do not change the variable name "queries". At most 5 elements in queries.
"""
    query_result = llm.ask_messages(history + [{"role": "user", "content": prompt}], model=config.DESIGN_MODEL)
    history = history + [{"role": "assistant", "content": query_result}]
    queries = llm.parse_assignment(llm.extract_code(query_result)[0], "queries")
    logger.info("Decomposed knowledge queries: %s", queries)

    results = queryPinecone.queryPinecone(queries, mytop_k)
    new_prompt = "Now I have retrieved the following relevant knowledge documents:\n"
    for res in results:
        new_prompt += res["text"] + "\n"
    new_prompt += (
        f"Please summarize the relevant knowledge to help me answer the original query: {query_text}. "
        f"Provide detailed and technical information. The current design stage is {stage}. "
        "Do not give unrelated information about the query."
    )
    return llm.ask_messages(history + [{"role": "user", "content": new_prompt}], model=config.DESIGN_MODEL)
