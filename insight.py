"""Insight Module and Evolution Memory.

After a design task succeeds, the Insight Module summarizes the design rationale. Later
tasks of the same series then receive, for every earlier task that has an entry, its
specification, the successful topology and this insight (in a fixed shuffled order).
Seed entries from tasks/<circuit>/seed_experience.json are added to the pool when present
and take precedence over entries produced in the current run.
"""

from __future__ import annotations

import json
import random
from typing import Dict, Tuple

import config
import llm
from circuits import Circuit

Experience = Dict[str, Tuple[int, str]]  # spec id -> (topology id, insight text)


def get_insight(history: str, circuit: Circuit) -> str:
    """Summarize a successful design process as guidance for later tasks."""
    prompt = history + f"\n You have successfully completed the design task. The above is the entire chat history of the design process. In a few sentences, give an experience observation you find in this design process which could be used to guide another {circuit.insight_noun} design under a different specification requirement. Use complete sentences."
    return llm.ask(prompt, model=config.DESIGN_MODEL)


def series_base(spec_id: int) -> int:
    """Tasks of one series share the leading digits: 1-15, 10001-10016, 20001-20005, ..., 600001."""
    if spec_id >= 10000:
        return spec_id - spec_id % 10000
    return 0


def load_seed_experience(circuit: Circuit) -> Experience:
    path = circuit.tasks_dir / "seed_experience.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: (int(v["topology_id"]), v["experience"]) for k, v in data.items() if not k.startswith("_")}


def build_memory_prompt(circuit: Circuit, spec_id: int, experience: Experience) -> str:
    """Evolution Memory prompt for spec_id: entries of earlier tasks in the same series only."""
    base = series_base(spec_id)
    index = spec_id - base
    if index == 1:
        return ""
    pool = dict(experience)
    pool.update(load_seed_experience(circuit))
    valid_ids = [i for i in range(1, index) if str(i + base) in pool]
    random.Random(42).shuffle(valid_ids)
    prompt = circuit.memory_header
    for i in valid_ids:
        topology_id, insight_text = pool[str(i + base)]
        requirement = (circuit.tasks_dir / f"{i + base}.prompt").read_text(encoding="utf-8")
        topology = circuit.topology_description(topology_id)
        prompt += f"""
    Query {i} Requirement:
    {requirement}
    Query {i} Topology description:
    {topology}
    Query {i} Experience description:
    {insight_text}
    """
    return prompt
