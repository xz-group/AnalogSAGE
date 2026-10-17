"""Run-time configuration for AnalogSAGE.

Secrets and service names are read only from environment variables:
  GEMINI_API_KEY      Gemini models (OpenAI-compatible endpoint)
  OPENAI_API_KEY      OpenAI models and text embeddings
  PINECONE_API_KEY    literature knowledge base (Design Knowledge RAG)
  PINECONE_INDEX      name of the Pinecone index of the knowledge base
  PINECONE_NAMESPACE  namespace inside that index (optional; default namespace if unset)
  SKY130_PDK_ROOT     SKY130 PDK directory that contains libs.tech/ngspice/
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS_ROOT = ROOT / "tasks"
TOPOLOGY_ROOT = ROOT / "topologies"
TESTBENCH_ROOT = ROOT / "simulation"
RESULTS_ROOT = ROOT / "results"

# LLM settings used in the experiments.
DESIGN_MODEL = "gemini-2.5-flash"
REFINE_QUERY_MODEL = "gemini-2.5-flash-lite"
REFINE_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-3-large"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Design loop.
MAX_ITERATION = 3
TOP_K_CANDIDATES = 5
MAX_SIMULATIONS = 150
LLM_BUDGET = 40
REFINE_CHECK_ATTEMPTS = 3  # PySpice answers per refinement (the first + repairs after a failed check)
NGSPICE_TIMEOUT_S = 180


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if it is unset."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def pinecone_index() -> str:
    """Name of the Pinecone index that holds the Design Knowledge RAG."""
    return require_env("PINECONE_INDEX")


def pinecone_namespace() -> str:
    """Namespace inside the Pinecone index ('' selects the default namespace)."""
    return os.environ.get("PINECONE_NAMESPACE", "").strip()


def sky130_pdk_root() -> Path:
    """Return the SKY130 PDK root that contains libs.tech/ngspice/."""
    root = Path(require_env("SKY130_PDK_ROOT")).expanduser()
    if not (root / "libs.tech" / "ngspice").is_dir():
        raise RuntimeError(f"SKY130_PDK_ROOT={root} does not contain libs.tech/ngspice/.")
    return root
