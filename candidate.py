"""Candidates Module: hybrid (vector + BM25) retrieval over the topology library, then LLM ranking."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

import config
import llm
from circuits import Circuit

logger = logging.getLogger(__name__)


class SimpleRAG:
    """Topology descriptions ({id}_v1.txt) with their embeddings (data{id}_v1.pkl)."""

    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        self.embedding_matrix = None
        self.bm25 = None

    def load_data(self, pkl_folder: Path, txt_folder: Path) -> None:
        self.docs = []
        for txt_file in sorted(os.listdir(txt_folder)):
            if not txt_file.endswith(".txt"):
                continue
            base_name = txt_file.replace(".txt", "")
            pkl_path = os.path.join(pkl_folder, f"data{base_name}.pkl")
            if not os.path.exists(pkl_path):
                logger.warning("Missing embedding for %s", txt_file)
                continue
            with open(os.path.join(txt_folder, txt_file), "r", encoding="utf-8") as f:
                text = f.read()
            with open(pkl_path, "rb") as f:
                embedding = np.array(pickle.load(f)).reshape(-1)
            self.docs.append({"filename": txt_file, "text": text, "embedding": embedding})
        if not self.docs:
            raise ValueError("No valid documents loaded.")
        self.embedding_matrix = np.vstack([doc["embedding"] for doc in self.docs])
        self.bm25 = BM25Okapi([doc["text"].split() for doc in self.docs])

    def cosine_similarity(self, q_embedding: np.ndarray) -> np.ndarray:
        q = q_embedding / (np.linalg.norm(q_embedding) + 1e-10)
        m = self.embedding_matrix
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
        return np.dot(m, q)

    def vector_query(self, q_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        sims = self.cosine_similarity(np.array(q_embedding).reshape(-1))
        top_idx = np.argsort(-sims)[:top_k]
        return [{"idx": int(i), "filename": self.docs[i]["filename"], "text": self.docs[i]["text"]} for i in top_idx]

    def bm25_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(query_text.split())
        top_idx = np.argsort(-scores)[:top_k]
        return [{"idx": int(i), "filename": self.docs[i]["filename"], "text": self.docs[i]["text"]} for i in top_idx]

    def query_dual(self, query_text: str, q_embedding: List[float], bm25_k: int = 5, vector_k: int = 5) -> List[Dict[str, Any]]:
        """Union of the vector top-k, the BM25 top-k and the top-5 of an equal-weight hybrid score."""
        alpha = 0.5
        hybrid_k = 5
        vector_results = self.vector_query(q_embedding, vector_k)
        bm25_results = self.bm25_query(query_text, bm25_k)
        sims = self.cosine_similarity(np.array(q_embedding).reshape(-1))
        bm25_scores = self.bm25.get_scores(query_text.split())
        vec_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-10)
        bm_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        hybrid_scores = alpha * vec_norm + (1 - alpha) * bm_norm
        hybrid_results = [
            {"idx": int(i), "filename": self.docs[i]["filename"], "text": self.docs[i]["text"]}
            for i in np.argsort(-hybrid_scores)[:hybrid_k]
        ]
        combined: Dict[int, Dict[str, Any]] = {}
        for item in vector_results + bm25_results + hybrid_results:
            combined.setdefault(item["idx"], {"filename": item["filename"], "text": item["text"]})
        logger.info("Retrieved topologies: %s", [item["filename"] for item in combined.values()])
        return list(combined.values())


def getNetlist(circuit: Circuit, topology_id: int) -> str:
    return f"Netlist id: {topology_id}\n" + circuit.topology_description(topology_id) + "\n\n"


def getCandidates(circuit: Circuit, query_text: str, spec: str, knowledge_docs: str) -> Tuple[str, List[int]]:
    """Return (descriptions of the retrieved topologies, topology ids ranked by the LLM)."""
    rag = SimpleRAG()
    rag.load_data(circuit.rag_dir, circuit.rag_dir)
    second_query = f"""
Topology description:
{query_text}
Please write a prompt to search for this topology design in a RAG system and only output the topology description prompt text directly without any additional explanation.
Sample output:
{circuit.candidate_example}
"""
    answer = llm.ask(second_query, model=config.DESIGN_MODEL)
    results = rag.query_dual(answer, llm.embed(answer), bm25_k=5, vector_k=5)
    mynewcandidate = ""
    for res in results:
        mynewcandidate += getNetlist(circuit, int(res["filename"].split("_")[0]))
    third_query = f"""
I have the spec requirements of {circuit.spec_noun} as follows:
{spec}
And I have a topology design description that could meet the spec requirements as follows:
{query_text}
I also have some candidate netlists that may meet the spec requirements as follows:
{mynewcandidate}
There are some retrievaled knowledge.
{knowledge_docs}
Please help me to select the best candidate netlist that meets the spec requirements.
First compare each candidate with the topology design description and spec requirement and eliminate those that do not match the topology design and finally output the topology candidate netlist id directly sorted based on the possibility that they meet the spec requirements.
Example:
```python
mycandidate = [35,17] # 35 is the best candidate, 17 is the second best candidate, etc. Give the best 5 candidates.
```
Please do not modify the variable name "mycandidate" and you MUST put it in a python code block like the example.
"""
    answer = llm.ask(third_query, model=config.DESIGN_MODEL)
    try:
        ranked = [int(i) for i in llm.parse_assignment(llm.extract_code(answer)[0], "mycandidate")]
    except (IndexError, ValueError, SyntaxError, TypeError) as e:
        logger.warning("Could not parse candidate list: %s", e)
        ranked = []
    return mynewcandidate, ranked
