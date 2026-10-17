"""Design Knowledge RAG: embedding search over chunks of analog-design papers in a Pinecone index.

The index and namespace come from PINECONE_INDEX and PINECONE_NAMESPACE (see config.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from pinecone import Pinecone

import config
import llm


def queryPinecone(query_text_list: List[str], mytop_k: int = 10) -> List[Dict[str, Any]]:
    """Return the metadata (including 'text') of the top-k chunks for every query."""
    index = Pinecone(api_key=config.require_env("PINECONE_API_KEY")).Index(name=config.pinecone_index())
    namespace = config.pinecone_namespace()
    ans: List[Dict[str, Any]] = []
    for query_text in query_text_list:
        matches = index.query(
            vector=llm.embed(query_text),
            top_k=mytop_k,
            include_values=False,
            include_metadata=True,
            namespace=namespace,
        )["matches"]
        ans.extend(match["metadata"] for match in matches)
    return ans
