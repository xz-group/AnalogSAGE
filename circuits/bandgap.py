"""Bandgap voltage-reference adapter.

Testbench (simulation/bandgap/tb.spice), AnalogGym voltage-reference methodology on sky130 tt,
VDDA = 1.8 V, no load on vref:
  vref     v(vref) at 25 C                                              (V)
  tc       |PP(vref)/AVG(vref)|/170 * 1e6 over -45..125 C               (ppm/C)
  ls       |PP(vref)/AVG(vref)|/0.36 over VDDA 1.62..1.98 V              (1/V)
  psrr     -vdb(vref) at 10 Hz with AC = 1 on VDDA                       (dB)
  psrr_hf  -vdb(vref) at 1 MHz                                           (dB)
  power    |I(VDDA)| * 1.8 V at 25 C                                     (uW)
aux.spice is written per topology: a VIREF source holds a real current-reference output at
1 V, and behavioural 'amplifier' blocks get a macromodel.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation import parse_values, render_testbench, run_ngspice, write_varspice

from .base import PARAM_QUOTED_RE, PARAM_TOKEN_RE, Circuit, Constraints

CATEGORY_RANGES: Dict[str, Tuple[float, float]] = {
    "Lfet": (0.35, 25.0),      # MOS channel length, um
    "Wfet": (0.42, 50.0),      # MOS width, um
    "Mint": (1.0, 20.0),       # device multiplier (integer; also BJT area multiple)
    "Cum": (1.0, 30.0),        # MiM-cap w/l, um
    "Rlen": (1.0, 500.0),      # sky130 poly-resistor length, um
    "Rohm": (1e3, 5e6),        # plain resistance, ohm
    "Cfarad": (1e-13, 5e-11),  # plain capacitance, F
    "Iamp": (1e-6, 1e-4),      # current source, A
    "Vvolt": (0.0, 1.8),       # voltage source, V
}

AMP3_SUBCKT = """.subckt amplifier inn inp out
Gm 0 out inp inn 1e-4
Ro out 0 1e6
Cc out 0 3p
.ends amplifier
"""

AMP2_SUBCKT = """.subckt amplifier a b
Rstub a b 1e12
.ends amplifier
"""

_RANGE_RULE = """Parameters ending in _M (device/BJT multipliers) are INTEGERS.
Ranges are classified from the netlist context (units follow the device the
variable is used on):
- MOS l='..._L' : channel length in um, hard range [0.35, 25]
- MOS w='..._W' : width in um, hard range [0.42, 50]
- any m='..._M' : integer multiplier, hard range [1, 20]
- sky130 poly-resistor l='RESISTOR_x' : resistor LENGTH in um, hard range [1, 500]
  (res_high_po_0p69 ~530 ohm/um, res_xhigh_po_0p69 ~2947 ohm/um at 27C)
- MiM-cap w/l='CAPACITOR_x' : capacitor side in um, hard range [1, 30]
- plain R element value 'RESISTOR_x' : resistance in ohm, hard range [1e3, 5e6]
- plain C element value 'CAPACITOR_x' : capacitance in F, hard range [1e-13, 5e-11]
- plain I element value : current in A, hard range [1e-6, 1e-4]"""

_REFINE_INSTRUCTION = """Please *refine and modify the topology if needed (You could do small changes on the topology like adding compensation capacitors, cascode devices in the mirrors, startup elements, trimming/output resistor branches or other small structures without changing the main structure. Do not do large changes like replacing the whole core or amplifier)* and then output the refined topology as PySpice code in the format below.

Rules for the refined topology:
- Use sky130 device models (sky130_fd_pr__nfet_01v8, sky130_fd_pr__pfet_01v8, sky130_fd_pr__pnp_05v5_W3p40L3p40, sky130_fd_pr__res_high_po_0p69, sky130_fd_pr__res_xhigh_po_0p69, sky130_fd_pr__cap_mim_m3_1, ...) like the input netlist.
- The testbench will drive: gnda=0, vdda=1.8V (swept 1.62-1.98V for line sensitivity, AC=1 for PSRR), temperature swept -45..125C. If the input topology has a real current-reference output, the testbench holds the iref pin at 1V; keep that iref branch. If the input topology instead ties iref to vref (a line like "Rlink vref iref 0"), keep that element as self.R("link", "vref", "iref", 0).
- Do not instantiate any other subcircuit; the netlist must be self-contained. The only exception: if the input topology already uses the behavioral 'amplifier' block, you may keep it (the testbench supplies it).
"""

_PYSPICE_EXAMPLE = """        self.X("q1", "sky130_fd_pr__pnp_05v5_W3p40L3p40", "gnda", "gnda", "va", m="Q1_M")
        self.X("r1", "sky130_fd_pr__res_high_po_0p69", "nb", "r1m", "gnda", l="RESISTOR_1")
        self.C("0", "vref", "gnda", "CAPACITOR_0")
        # one call per device of the refined topology"""


class Bandgap(Circuit):
    name = "bandgap"
    design_noun = "a bandgap voltage reference"
    short_noun = "bandgap voltage reference"
    proposal_noun = "a bandgap voltage reference (under skywater PDK, SKY130 process node)"
    spec_noun = proposal_noun
    knowledge_example = (
        "I want to design a bandgap voltage reference with low temperature coefficient and high PSRR. "
        "How can I achieve this?"
    )
    candidate_example = (
        "The circuit is a Banba-style current-mode bandgap reference. A PMOS current mirror forces equal "
        "currents into a unit PNP and a parallel N-times PNP plus resistor branch through an error amplifier, "
        "and the mirrored current into an output resistor produces the temperature-compensated reference voltage."
    )
    refine_query_example = (
        "How to improve the temperature coefficient and PSRR of a current-mode bandgap reference with a PMOS mirror?"
    )
    refine_instruction = _REFINE_INSTRUCTION
    pyspice_example = _PYSPICE_EXAMPLE
    insight_noun = "bandgap voltage-reference"
    memory_header = "I have designed these bandgap voltage-reference topologies with spec requirements."
    sizing_intro = (
        "You are optimizing analog bandgap voltage-reference sizing with mixed LLM + numerical optimization.\n"
        "The testbench (AnalogGym voltage-reference methodology on SKY130): VDDA=1.8V, GNDA=0V;\n"
        "temperature swept -45..125C -> vref (V, at 25C), tc (ppm/C, pp/avg/170*1e6), power (uW, at 25C);\n"
        "supply swept 1.62..1.98V -> ls (line sensitivity, /V, pp/avg/0.36);\n"
        "AC 1 on VDDA -> psrr (dB, -vdb(vref) at 10Hz) and psrr_hf (dB, at 1MHz).\n"
        "If the topology has a real current-reference output, the iref pin is held at 1V by the testbench;\n"
        'topologies with "Rlink vref iref 0" are pure voltage references.'
    )
    range_rule = _RANGE_RULE
    range_rule_language = "text"
    relation_example = '"Q2_M == 8 * Q1_M"'
    retrieval_subject = "analog bandgap voltage-reference sizing"
    retrieval_fallback = "Bandgap voltage reference sizing strategy to satisfy multi-objective specs in SKY130."
    title = ".title bandgap"
    subckt_header = ".subckt bandgap gnda vdda vref iref"
    netlist_file = "mybandgap.txt"

    # ---- sizing parameters ---------------------------------------------------------------
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

    @staticmethod
    def classify_params(netlist_text: str) -> Dict[str, str]:
        """Map each sizing variable to a CATEGORY_RANGES key from the device it is used on."""
        cat: Dict[str, str] = {}
        for raw in netlist_text.splitlines():
            s = raw.strip()
            if not s or s[0] in "*.":
                continue
            low = s.lower()
            is_fet = "fet" in low
            is_mim = "cap_mim" in low or "cap_var" in low
            is_res_prim = "__res_" in low
            for m in re.finditer(r"\b([lwmLWM])\s*=\s*'([^']+)'", s):
                key = m.group(1).lower()
                for token in PARAM_TOKEN_RE.findall(m.group(2)):
                    if key == "m":
                        c = "Mint"
                    elif is_fet:
                        c = "Lfet" if key == "l" else "Wfet"
                    elif is_mim:
                        c = "Cum"
                    elif is_res_prim:
                        c = "Rlen"
                    else:
                        c = ""
                    if c:
                        cat.setdefault(token, c)
            if low[0] in ("r", "c", "i", "v"):
                q = PARAM_QUOTED_RE.findall(s)
                if q:
                    c = {"r": "Rohm", "c": "Cfarad", "i": "Iamp", "v": "Vvolt"}[low[0]]
                    for token in PARAM_TOKEN_RE.findall(q[0]):
                        cat.setdefault(token, c)
        return cat

    def infer_param_range(self, name: str, category_map: Optional[Dict[str, str]] = None) -> Tuple[float, float]:
        if category_map and name in category_map:
            return CATEGORY_RANGES[category_map[name]]
        upper = name.upper()
        if upper.endswith("_L") or "_L_" in upper or upper.startswith("L"):
            return CATEGORY_RANGES["Lfet"]
        if upper.endswith("_W") or "_W_" in upper or upper.startswith("W"):
            return CATEGORY_RANGES["Wfet"]
        if self.is_integer_param(name):
            return CATEGORY_RANGES["Mint"]
        if "CAP" in upper or upper.startswith("C"):
            return CATEGORY_RANGES["Cfarad"]
        if upper.startswith("R"):
            return CATEGORY_RANGES["Rohm"]
        if upper.startswith("I"):
            return CATEGORY_RANGES["Iamp"]
        if upper.startswith("V"):
            return CATEGORY_RANGES["Vvolt"]
        return (0.35, 25.0)

    def build_bounds(self, param_names: List[str], netlist_text: str) -> np.ndarray:
        category_map = self.classify_params(netlist_text) if netlist_text else {}
        return np.asarray([self.infer_param_range(n, category_map) for n in param_names], dtype=np.float64)

    # ---- netlist composition and simulation ----------------------------------------------
    @staticmethod
    def analyze_topology(netlist_text: str) -> Tuple[bool, int]:
        """Return (has_rlink, amplifier_arity). has_rlink: iref is a 0-ohm alias of vref."""
        has_rlink = bool(
            re.search(r"(?im)^\s*r\w*\s+vref\s+iref\s", netlist_text)
            or re.search(r"(?im)^\s*r\w*\s+iref\s+vref\s", netlist_text)
        )
        amp_arity = 0
        for ln in netlist_text.splitlines():
            s = ln.strip()
            if not s or s.startswith("*"):
                continue
            t = s.split()
            if t and t[-1].lower() == "amplifier":
                amp_arity = len(t) - 2
        return has_rlink, amp_arity

    def write_aux(self, workdir: Path, merged_text: str) -> None:
        has_rlink, amp_arity = self.analyze_topology(merged_text)
        lines = ["* aux: topology-dependent testbench pieces (auto-written by sizing_mix)"]
        if amp_arity == 3:
            lines.append(AMP3_SUBCKT.rstrip())
        elif amp_arity == 2:
            lines.append(AMP2_SUBCKT.rstrip())
        if has_rlink:
            lines.append("* iref is aliased to vref via Rlink inside the topology; no VIREF source")
        else:
            lines.append("VIREF iref 0 DC 1")
        (workdir / "aux.spice").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def simulate(self, params: Dict[str, float], constraints: Constraints, workdir: Path) -> Dict[str, Any]:
        write_varspice(workdir, params)
        tb = render_testbench(self.testbench_dir / "tb.spice", workdir)
        raw = parse_values(
            run_ngspice(workdir, tb.name, "result.txt"),
            ["vrefnom", "tcppm", "powuw", "lsperv", "psrrdc", "psrrhf"],
        )
        return {
            **raw,
            "vref": raw["vrefnom"],
            "tc": raw["tcppm"],
            "ls": raw["lsperv"],
            "psrr": raw["psrrdc"],
            "psrr_hf": raw["psrrhf"],
            "power": raw["powuw"],
        }
