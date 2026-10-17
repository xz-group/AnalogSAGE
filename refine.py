"""Topology Refinement Agent: small structural edits to a topology after a failed sizing.

The agent writes the refined topology as a PySpice subcircuit class. pyspice_netlist converts
it into an ngspice netlist and Circuit.check_netlist checks it; if the conversion or a check
fails, the errors are returned to the LLM for a corrected class (REFINE_CHECK_ATTEMPTS in total).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
import knowledge
import llm
import pyspice_netlist
from circuits import Circuit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Refinement:
    """A refined topology: the PySpice class written by the LLM and the netlist PySpice renders."""

    pyspice_code: str
    netlist: str


def _convert_and_check(circuit: Circuit, answer: str, topology: str) -> Tuple[Optional[Refinement], List[str]]:
    code = next((block for block in llm.extract_code(answer) if "class " in block), None)
    if code is None:
        return None, ["no python code block with a PySpice SubCircuitFactory class was found"]
    try:
        netlist = pyspice_netlist.pyspice_to_netlist(code, circuit.refined_name, circuit.ports)
    except pyspice_netlist.NetlistError as e:
        return None, e.errors
    errors = circuit.check_netlist(netlist, topology)
    return (None if errors else Refinement(code, netlist)), errors


def _repair_prompt(errors: List[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        "The PySpice code did not pass the netlist check:\n"
        f"{listed}\n"
        "Fix these problems and output the complete corrected class in one python code block, "
        "following the same rules."
    )


def getMyRefine(
    circuit: Circuit,
    spec_id: int,
    topology: str,
    topology_id: int,
    base_performance: str = "",
    feedback: str = "",
) -> Optional[Refinement]:
    """Return the refined topology (PySpice source and checked netlist), or None.

    `feedback` is the Compression Module summary of the failed sizing attempt.
    """
    log_dir = config.RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"refine_{circuit.name}_{spec_id}_topo{topology_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    )
    spec = (circuit.tasks_dir / f"{spec_id}.prompt").read_text(encoding="utf-8")
    iter_text = ""
    if base_performance:
        iter_text += f"""
In the previous iteration, I have the original topology which reaches the following specifications under the best sizing:
'performance': {base_performance}
"""
    if feedback:
        iter_text += f"""
Summary of the failed design attempt:
{feedback}
"""

    with log_path.open("w", encoding="utf-8") as log:
        prompt = iter_text + f"""
Please design {circuit.design_noun} using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to judge whether it is needed to do a topology refinement.

There is a knowledge agent that can help to find relevant knowledge documents about analog circuit design from a knowledge base.
You can write a query to ask the knowledge agent to get relevant knowledge documents to help you design the {circuit.short_noun}.
To ask the agent, please first analyze the specifications and then output the query text in a code block.
You do not need to do detail refinement now, just need to find relevant knowledge documents.

Example
```text
{circuit.refine_query_example}
```
"""
        first_query = llm.ask(prompt, model=config.REFINE_QUERY_MODEL)
        log.write("First query:\n" + prompt + "\nFirst query result:\n" + first_query + "\n")
        query_text_list = llm.extract_code(first_query)
        try:
            knowledge_results = knowledge.getKnowledge(query_text_list, mytop_k=5)
        except Exception as e:  # retrieval is optional; refine without it
            logger.warning("Knowledge retrieval failed during refinement: %s", e)
            log.write(f"Knowledge retrieval failed: {e}\n")
            knowledge_results = ""

        prompt = iter_text + f"""
Please design {circuit.design_noun} using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to judge whether it is needed to do a topology refinement.

The following relevant knowledge documents are retrieved to help to design the {circuit.short_noun}:
{knowledge_results}
{circuit.refine_instruction}{circuit.refine_output_rules}"""
        log.write("Second query:\n" + prompt + "\n")
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": llm.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(1, config.REFINE_CHECK_ATTEMPTS + 1):
            answer = llm.ask_messages(messages, model=config.REFINE_MODEL)
            log.write(f"Second query result (attempt {attempt}):\n{answer}\n")
            refinement, errors = _convert_and_check(circuit, answer, topology)
            if refinement is not None:
                log.write(
                    "PySpice code:\n" + refinement.pyspice_code + "\n"
                    "Converted netlist (passed all checks):\n" + refinement.netlist
                )
                return refinement
            log.write("Netlist check failed:\n" + "\n".join(f"- {e}" for e in errors) + "\n")
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": _repair_prompt(errors)},
            ]
        log.write(f"No refined topology passed the checks in {config.REFINE_CHECK_ATTEMPTS} attempts.\n")
    return None
