"""Low-dropout regulator adapter.

Testbench (simulation/ldo/tb.spice), sky130 tt, VDD = 1.8 V, Vref = 1.2 V, unity feedback
(target Vout = 1.2 V), CL = 100 pF:
  dcgain, GBW, phase_margin  loop gain broken with a 1T L/C at the feedback node, Iload = 10 mA
  PSRR      -vdb(vout) at 0.1 Hz with AC = 1 on VDD, Iload = 10 mA
  LNR       PP(vout)/AVG(vout)/0.36 over VDD 1.62..1.98 V at 10 mA        (1/V)
  LR        PP(vout)/AVG(vout)/9.99 mA over Iload 10 uA..10 mA           (1/A)
  vos       |vout(10 mA) - 1.2 V|                                           (V)
  Power     I(VDD) * 1.8 V at Iload = 10 uA, including the load current   (uW)
  undershoot / overshoot  1.2 V - min(vout) / max(vout) - 1.2 V for a 10 uA <-> 10 mA step
  dropout_mv              (1.8 V - vout(10 mA)) * 1000 (AutoSizer-compatible definition)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from simulation import parse_values, render_testbench, run_ngspice, write_varspice

from .base import Circuit, Constraints

_RANGE_RULE = """def is_integer_param(name: str) -> bool:
    upper = name.upper()
    if upper.endswith("_L") or "_L_" in upper or upper.endswith("_W") or "_W_" in upper:
        return False
    return (
        upper.endswith("_M")
        or "_M_" in upper
        or upper.endswith("_NM")
        or upper.endswith("_PM")
        or upper.startswith("M")
        or upper == "NMIR"
    )

def infer_param_range(name: str) -> Tuple[float, float]:
    upper = name.upper()
    if upper.endswith("_L") or "_L_" in upper or upper.startswith("L"):
        return (0.35, 2.0)
    if upper.endswith("_W") or "_W_" in upper or upper.startswith("W"):
        return (0.5, 7.0)
    if is_integer_param(name):
        return (1.0, 200.0)
    if "CAP" in upper or upper.startswith("C"):
        return (1e-13, 1e-10)
    if upper.startswith("R"):
        return (100.0, 1e6)
    if upper.startswith("I"):
        return (5e-7, 1e-4)
    if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
        return (0.0, 1.8)
    if upper.startswith("V"):
        return (0.0, 1.8)
    return (0.35, 10.0)"""

_REFINE_INSTRUCTION = """Please *refine and modify the topology if needed (You could do small changes on the topology like adding compensation, feed-forward capacitors, buffer stages or other small structures without changing the main structure. Do not do large changes like changing from one stage to two stages)* and then output the refined topology as PySpice code in the format below.

Rules for the refined topology:
- Use sky130 device models (sky130_fd_pr__nfet_01v8, sky130_fd_pr__pfet_01v8, ...) like the input netlist.
- The testbench will drive: gnda=0, vdda=1.8V, vref=1.2V, vfb tied to vout (unity feedback), load 10uA-10mA at vout with 100pF.
- Do not instantiate any other subcircuit; the netlist must be self-contained.
"""

_PYSPICE_EXAMPLE = """        self.X("pass", "sky130_fd_pr__pfet_01v8", "vout", "gate", "vdda", "vdda", l="PASS_L", w="PASS_W*1", m="PASS_M")
        self.C("0", "gate", "vout", "CAPACITOR_0")
        self.I("b1", "vdda", "ib1", "IB1")
        # one call per device of the refined topology"""


class LDO(Circuit):
    name = "ldo"
    design_noun = "an LDO (low-dropout regulator)"
    short_noun = "LDO"
    proposal_noun = "an LDO (low-dropout regulator, under skywater PDK, SKY130 process node)"
    spec_noun = proposal_noun
    knowledge_example = "I want to design an LDO with high PSRR and fast load transient response. How can I achieve this?"
    candidate_example = (
        "The circuit is a low-dropout regulator with a PMOS pass transistor. The error amplifier is a "
        "five-transistor OTA with an NMOS differential input pair and PMOS current-mirror load, and the "
        "pass device gate is driven directly by the error amplifier output with a compensation capacitor "
        "at the gate node."
    )
    refine_query_example = "How to improve the PSRR and load transient response of an LDO with a PMOS pass transistor?"
    refine_instruction = _REFINE_INSTRUCTION
    pyspice_example = _PYSPICE_EXAMPLE
    insight_noun = "LDO"
    memory_header = "I have designed these LDO topologies with spec requirements."
    sizing_intro = (
        "You are optimizing analog LDO (low-dropout regulator) sizing with mixed LLM + numerical optimization.\n"
        "The testbench: VDD=1.8V, Vref=1.2V, target Vout=1.2V (unity feedback), load 10uA to 10mA, "
        "CL=100pF, SKY130 process."
    )
    range_rule = _RANGE_RULE
    range_rule_language = "python"
    relation_example = '"VB2 > VB1"'
    retrieval_subject = "analog LDO (low-dropout regulator) sizing"
    retrieval_fallback = "LDO sizing strategy to satisfy multi-objective specs in SKY130."
    title = ".title LDO"
    subckt_header = ".subckt ldo gnda vdda vref vfb vout"
    netlist_file = "myldo.txt"

    def is_integer_param(self, name: str) -> bool:
        upper = name.upper()
        if upper.endswith("_L") or "_L_" in upper or upper.endswith("_W") or "_W_" in upper:
            return False
        return (
            upper.endswith("_M")
            or "_M_" in upper
            or upper.endswith("_NM")
            or upper.endswith("_PM")
            or upper.startswith("M")
            or upper == "NMIR"
        )

    def infer_param_range(self, name: str) -> Tuple[float, float]:
        upper = name.upper()
        if upper.endswith("_L") or "_L_" in upper or upper.startswith("L"):
            return (0.35, 2.0)
        if upper.endswith("_W") or "_W_" in upper or upper.startswith("W"):
            return (0.5, 7.0)
        if self.is_integer_param(name):
            return (1.0, 200.0)
        if "CAP" in upper or upper.startswith("C"):
            return (1e-13, 1e-10)
        if upper.startswith("R"):
            return (100.0, 1e6)
        if upper.startswith("I"):
            return (5e-7, 1e-4)
        if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
            return (0.0, 1.8)
        if upper.startswith("V"):
            return (0.0, 1.8)
        return (0.35, 10.0)

    def build_bounds(self, param_names: List[str], netlist_text: str) -> np.ndarray:
        return np.asarray([self.infer_param_range(n) for n in param_names], dtype=np.float64)

    def simulate(self, params: Dict[str, float], constraints: Constraints, workdir: Path) -> Dict[str, Any]:
        write_varspice(workdir, params)
        tb = render_testbench(self.testbench_dir / "tb.spice", workdir)
        raw = parse_values(
            run_ngspice(workdir, tb.name, "result.txt"),
            ["dcgain", "gain_bandwidth_product", "phase_margin", "psrr", "lnr", "lr",
             "vos", "power", "undershoot", "overshoot", "dropout_mv"],
        )
        return {
            **raw,
            "GBW": raw["gain_bandwidth_product"],
            "phase_margin (deg)": raw["phase_margin"],
            "PSRR": raw["psrr"],
            "LNR": abs(raw["lnr"]) if raw["lnr"] is not None else None,
            "LR": abs(raw["lr"]) if raw["lr"] is not None else None,
            "Power": raw["power"],
        }
