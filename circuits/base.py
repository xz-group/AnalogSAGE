"""Base class for circuit adapters.

An adapter holds everything that differs between circuit types: prompt wording,
parameter-range rules, how a topology is composed into the testbench, and how the
testbench results are turned into the metrics named in tasks.json.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import config
from pyspice_netlist import structural_errors, subckt_models
from simulation import parse_check, render_testbench, write_varspice

Constraints = Dict[str, Any]

PARAM_QUOTED_RE = re.compile(r"'([^']+)'")
PARAM_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

_PYSPICE_RULES = '''
Output the refined topology as PySpice code: one SubCircuitFactory class in one python code block, as in this example.
```python
from PySpice.Spice.Netlist import SubCircuitFactory


class RefinedCircuit(SubCircuitFactory):
    NAME = "{name}"
    NODES = {nodes}

    def __init__(self):
        super().__init__()
{example}
```
Rules for the PySpice code:
- NAME and NODES MUST be exactly as in the example; NODES is the port order of the testbench.
- __init__ may only contain element calls with literal arguments: self.X(name, subcircuit, node, ..., **params) for SKY130 devices, and self.R / self.C / self.V / self.I(name, node1, node2, value). Do not use loops, variables or other code.
- Write every sizing value as a string with the variable name or a simple expression of variables, e.g. l="P1_L", w="P1_W*1", m="P1_M". Keep the **original variables** for unchanged devices and introduce new variables for any new device. Use plain numbers only for fixed values.
- Node and element names may only use letters, digits and underscores.
The class is converted with PySpice into a SPICE subcircuit and checked before simulation: ports, device terminals, dangling nodes, and an ngspice parse with the SKY130 models.
'''


class Circuit(ABC):
    """Circuit-type specific parts of the AnalogSAGE flow."""

    name: str = ""                  # folder key under tasks/, topologies/ and simulation/
    design_noun: str = ""           # "an LDO (low-dropout regulator)"
    short_noun: str = ""            # "LDO"
    proposal_noun: str = ""         # noun phrase in the topology-proposal prompt
    spec_noun: str = ""             # noun phrase in the candidate-selection prompt
    knowledge_example: str = ""     # example knowledge query (Topology Selection Agent)
    candidate_example: str = ""     # example topology description (Candidates Module)
    refine_query_example: str = ""  # example knowledge query (Topology Refinement Agent)
    refine_instruction: str = ""    # what the refinement agent may change, and circuit-specific rules
    pyspice_example: str = ""       # element calls of the PySpice example in the refinement prompt
    insight_noun: str = ""          # used in "guide another <insight_noun> design"
    memory_header: str = ""         # first line of the Evolution Memory prompt
    sizing_intro: str = ""          # first lines of the Parameter Optimization Agent prompt
    range_rule: str = ""            # parameter-range rule shown to the sizing LLM
    range_rule_language: str = "python"
    relation_example: str = ""      # third example relation in the sizing prompt
    retrieval_subject: str = ""     # "analog op-amp sizing"
    retrieval_fallback: str = ""    # retrieval query used when the LLM returns nothing
    title: str = ""                 # first line of the composed netlist
    subckt_header: str = ""         # .subckt line expected by the testbench
    netlist_file: str = ""          # netlist file included by the testbench

    # ---- paths -------------------------------------------------------------------------
    @property
    def tasks_dir(self) -> Path:
        return config.TASKS_ROOT / self.name

    @property
    def netlist_dir(self) -> Path:
        return config.TOPOLOGY_ROOT / self.name / "netlists"

    @property
    def rag_dir(self) -> Path:
        return config.TOPOLOGY_ROOT / self.name / "rag"

    @property
    def testbench_dir(self) -> Path:
        return config.TESTBENCH_ROOT / self.name

    @property
    def workdir(self) -> Path:
        return config.RESULTS_ROOT / "work" / self.name

    @property
    def check_workdir(self) -> Path:
        return config.RESULTS_ROOT / "work" / f"{self.name}_check"

    @property
    def subckt_name(self) -> str:
        return self.subckt_header.split()[1]

    @property
    def ports(self) -> Tuple[str, ...]:
        return tuple(self.subckt_header.split()[2:])

    @property
    def refined_name(self) -> str:
        """Subcircuit name the Topology Refinement Agent must use."""
        return f"{self.subckt_name}_refined"

    @property
    def refine_output_rules(self) -> str:
        """Output format of the Topology Refinement Agent: a PySpice class (see pyspice_netlist)."""
        return _PYSPICE_RULES.format(name=self.refined_name, nodes=self.ports, example=self.pyspice_example)

    # ---- tasks and topologies ------------------------------------------------------------
    def load_spec_text(self, spec_id: int) -> str:
        path = self.tasks_dir / f"{spec_id}.prompt"
        if not path.exists():
            raise FileNotFoundError(f"Spec prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def load_constraints(self, spec_id: int) -> Constraints:
        tasks = json.loads((self.tasks_dir / "tasks.json").read_text(encoding="utf-8"))
        constraints = tasks.get("filters", {}).get(str(spec_id))
        if constraints is None:
            raise KeyError(f"Spec id {spec_id} not found in {self.tasks_dir / 'tasks.json'}.")
        return constraints

    def topology_path(self, topology_id: int) -> Path:
        return self.netlist_dir / f"{topology_id}.txt"

    def topology_description(self, topology_id: int) -> str:
        return (self.rag_dir / f"{topology_id}_v1.txt").read_text(encoding="utf-8")

    # ---- sizing parameters ---------------------------------------------------------------
    def extract_param_names(self, netlist_text: str) -> List[str]:
        """Sizing variables are the quoted tokens of non-comment netlist lines."""
        names: List[str] = []
        seen = set()
        for raw in netlist_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            for quoted in PARAM_QUOTED_RE.findall(line):
                for token in PARAM_TOKEN_RE.findall(quoted):
                    if token not in seen:
                        seen.add(token)
                        names.append(token)
        if not names:
            raise ValueError("No sizing parameters found in netlist.")
        return names

    @abstractmethod
    def is_integer_param(self, name: str) -> bool:
        """True for device multiplicities, which are rounded to integers."""

    @abstractmethod
    def build_bounds(self, param_names: List[str], netlist_text: str) -> np.ndarray:
        """Hard search range [low, high] for every parameter, shape (n, 2)."""

    # ---- netlist composition and simulation ----------------------------------------------
    def compose(self, topology_text: str, workdir: Path) -> str:
        """Put the device lines of a topology under the testbench's .subckt header."""
        lines = topology_text.splitlines()
        start = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith(".subckt")), None)
        end = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip().lower().startswith(".ends")), None)
        if start is None or end is None or end <= start:
            raise ValueError("Topology netlist must contain one .subckt ... .ends block.")
        merged = [self.title, self.subckt_header, *lines[start + 1 : end], f".ends {self.subckt_name}"]
        merged_text = "\n".join(merged) + "\n"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / self.netlist_file).write_text(merged_text, encoding="utf-8")
        self.write_aux(workdir, merged_text)
        return merged_text

    def write_aux(self, workdir: Path, merged_text: str) -> None:
        """Write topology-dependent testbench files (none by default)."""

    def check_netlist(self, netlist_text: str, original_topology: str) -> List[str]:
        """Checks a refined netlist must pass before it is sized; returns the errors ([] = passed).

        1. structure: ports connected, device terminal counts, only SKY130 devices or
           subcircuits the input topology already used, no dangling nodes;
        2. sizing variables: at least one quoted variable;
        3. ngspice: the composed netlist loads with the SKY130 models, every sizing variable
           set to the middle of its hard range (no analysis is run).
        """
        errors = structural_errors(netlist_text, self.ports, subckt_models(original_topology))
        if errors:
            return errors
        workdir = self.check_workdir
        merged = self.compose(netlist_text, workdir)
        try:
            names = self.extract_param_names(merged)
        except ValueError as e:
            return [str(e)]
        bounds = self.build_bounds(names, merged)
        midpoints = {}
        for name, (lo, hi) in zip(names, bounds):
            value = (float(lo) + float(hi)) / 2.0
            midpoints[name] = float(round(value)) if self.is_integer_param(name) else value
        write_varspice(workdir, midpoints)
        deck = render_testbench(
            config.TESTBENCH_ROOT / "netlist_check.spice",
            workdir,
            {
                "@NETLIST@": self.netlist_file,
                "@AUX@": ".include ./aux.spice" if (workdir / "aux.spice").exists() else "",
                "@INSTANCE@": f"xchk {' '.join(self.ports)} {self.subckt_name}",
            },
        )
        return parse_check(workdir, deck.name)

    @abstractmethod
    def simulate(self, params: Dict[str, float], constraints: Constraints, workdir: Path) -> Dict[str, Any]:
        """Run the testbench for one sizing and return the metrics named in tasks.json."""
