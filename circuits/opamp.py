"""Operational-amplifier adapter.

Testbenches (simulation/opamp/):
  tb_acdc.spice  AC/DC: open-loop gain, GBW, phase margin, CMRR, PSRR (+/-), power, TC, offset.
  tb_tran.spice  transient: slew rate and settling time. Run only when a task constrains
                 sr, setting_time or FOML (the AnalogGym tasks).

Metric conventions follow the op-amp dataset used for the experiments:
  Power  = -I(VDD) * 1.8 V in uW at 25 C (unity-gain follower, VCM = 0.72 V)
  cmrrdc, PSRP, PSRN = |vdb| at 0.1 Hz;  GBW = 0 dB crossing of the open-loop gain
  phase_margin (deg) = phase of the open-loop gain at that crossing (inverting drive)
  TC  = PP(vout)/AVG(vout)/165 over -40..125 C (1/C);  vos = vout(25 C) - 0.72 V
  sr  = mean of the rising and falling slew rates, V/s
  setting_time = max of the rising- and falling-edge settling times (1% band), s
  FOML = sr[V/s] * CLoad[pF] / (1000 * Power[uW]) = SR[V/us] * CLoad[pF] / Power[mW]
  FOMS = GBW[Hz] * CLoad[pF] / (1000 * Power[uW]) = GBW[MHz] * CLoad[pF] / Power[mW]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation import parse_values, render_testbench, run_ngspice, write_varspice

from .base import Circuit, Constraints

DEFAULT_CLOAD_PF = 1.0  # load of tb_acdc.spice used for tasks without a CLoad
SETTLING_BAND = 0.01
TRANSIENT_METRICS = {"sr", "setting_time", "FOML"}

_RANGE_RULE = """def infer_param_range(name: str) -> Tuple[float, float]:
    upper = name.upper()
    if upper.endswith("_L") or "_L_" in upper:
        return (0.15, 1.0)
    if upper.endswith("_W") or "_W_" in upper:
        return (0.5, 10.0)
    if upper.endswith("_M") or "_M_" in upper:
        return (1.0, 100.0)
    if "CAPACITOR" in upper or upper.startswith("CCOMP"):
        return (1e-12, 1e-10)
    if "RESISTOR" in upper:
        return (1.0, 1e6)
    if "CURRENT" in upper or upper.startswith("IB"):
        return (5e-7, 2e-5)
    if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
        return (0.0, 1.8)
    if upper.startswith("V"):
        return (0.0, 1.8)
    return (0.15, 10.0)"""

_REFINE_INSTRUCTION = """Please *refine and modify the topology if needed (You could do small changes on the topology like adding Gain Boosting or compensation or other small structures without changing the main structure. Do not do large changes like changing from one stage to two stages)* and then output the refined topology as PySpice code in the format below.

Rules for the refined topology:
- Use sky130 device models (sky130_fd_pr__nfet_01v8, sky130_fd_pr__pfet_01v8, ...) like the input netlist.
- The testbench will drive: gnda=0, vdda=1.8V, input common mode 0.72V; it measures the open-loop response from vinn to vout, CMRR and PSRR in a unity-gain follower, and power, with a load capacitor at vout.
- Do not instantiate any other subcircuit; the netlist must be self-contained.
"""

_PYSPICE_EXAMPLE = """        self.X("m1", "sky130_fd_pr__pfet_01v8", "net1", "vinn", "net3", "vdda", l="P1_L", w="P1_W*1", m="P1_M")
        self.C("0", "net1", "vout", "CAPACITOR_0")
        self.V("b1", "vb1", "gnda", "VB1")
        # one call per device of the refined topology"""


class OpAmp(Circuit):
    name = "opamp"
    design_noun = "an operational amplifier"
    short_noun = "op-amp"
    proposal_noun = "a op-amp (under skywater PDK, SKY130 process node)"
    spec_noun = "an op-amp (under skywater PDK, SKY130 process node)"
    knowledge_example = "I want to design an op-amp with high gain and high bandwidth. How can I achieve this?"
    candidate_example = (
        "The circuit is a two-stage operational amplifier. The first stage consists of an NMOS "
        "differential pair with PMOS diode-connected loads, while the second stage is a common-source "
        "amplifier with a self-cascoded PMOS driver and an NMOS current-source load to achieve high gain."
    )
    refine_query_example = "How to optimize the topology of a two-stage operational amplifier with Miller compensation?"
    refine_instruction = _REFINE_INSTRUCTION
    pyspice_example = _PYSPICE_EXAMPLE
    insight_noun = "op-amp"
    memory_header = "I have designed these topologies with spec requirements."
    sizing_intro = "You are optimizing analog op-amp sizing with mixed LLM + numerical optimization."
    range_rule = _RANGE_RULE
    range_rule_language = "python"
    relation_example = '"VB2 > VB1"'
    retrieval_subject = "analog op-amp sizing"
    retrieval_fallback = "Op-amp sizing strategy to satisfy multi-objective specs in SKY130."
    title = ".title OpAmp"
    subckt_header = ".subckt op_amp gnda vdda vinn vinp vout"
    netlist_file = "myopamp.txt"

    # ---- sizing parameters ---------------------------------------------------------------
    def is_integer_param(self, name: str) -> bool:
        upper = name.upper()
        return upper.endswith("_M") or "_M_" in upper

    def infer_param_range(self, name: str) -> Tuple[float, float]:
        upper = name.upper()
        if upper.endswith("_L") or "_L_" in upper:
            return (0.15, 1.0)
        if upper.endswith("_W") or "_W_" in upper:
            return (0.5, 10.0)
        if upper.endswith("_M") or "_M_" in upper:
            return (1.0, 100.0)
        if "CAPACITOR" in upper or upper.startswith("CCOMP"):
            return (1e-12, 1e-10)
        if "RESISTOR" in upper:
            return (1.0, 1e6)
        if "CURRENT" in upper or upper.startswith("IB"):
            return (5e-7, 2e-5)
        if upper.startswith("VB") or upper.startswith("VBIAS") or upper.startswith("VREF"):
            return (0.0, 1.8)
        if upper.startswith("V"):
            return (0.0, 1.8)
        return (0.15, 10.0)

    def build_bounds(self, param_names: List[str], netlist_text: str) -> np.ndarray:
        return np.asarray([self.infer_param_range(n) for n in param_names], dtype=np.float64)

    # ---- simulation ----------------------------------------------------------------------
    def simulate(self, params: Dict[str, float], constraints: Constraints, workdir: Path) -> Dict[str, Any]:
        cload_pf = float(constraints.get("_context", {}).get("Cload_pF", DEFAULT_CLOAD_PF))
        tokens = {"@CLOAD@": f"{cload_pf}p"}
        write_varspice(workdir, params)

        acdc = render_testbench(self.testbench_dir / "tb_acdc.spice", workdir, tokens)
        raw = parse_values(
            run_ngspice(workdir, acdc.name, "result.txt"),
            ["dcgain_", "gain_bandwidth_product_", "phase_margin", "ivdd25",
             "cmrrdc", "dcpsrp", "dcpsrn", "tc", "vos25"],
        )
        power = -raw["ivdd25"] * 1.8 * 1e6 if raw["ivdd25"] is not None else None
        result: Dict[str, Any] = {
            **raw,
            "Power": power,
            "dcgain": raw["dcgain_"],
            "GBW": raw["gain_bandwidth_product_"],
            "phase_margin (deg)": raw["phase_margin"],
            "cmrrdc": _abs(raw["cmrrdc"]),
            "PSRP": _abs(raw["dcpsrp"]),
            "PSRN": _abs(raw["dcpsrn"]),
            "TC": raw["tc"],
            "vos": raw["vos25"],
            "abs(TC)": _abs(raw["tc"]),
            "abs(vos)": _abs(raw["vos25"]),
        }

        if TRANSIENT_METRICS & set(constraints):
            tran = render_testbench(self.testbench_dir / "tb_tran.spice", workdir, tokens)
            meas = parse_values(run_ngspice(workdir, tran.name, "result_tran.txt"), ["sr_rise", "sr_fall"])
            rates = [meas["sr_rise"], meas["sr_fall"]]
            result["sr_rise"], result["sr_fall"] = rates
            result["sr"] = float(np.mean(rates)) if all(r is not None for r in rates) else None
            result["setting_time"] = settling_time(workdir / "tran.dat")

        result["FOML"] = _figure_of_merit(result.get("sr"), cload_pf, power)
        result["FOMS"] = _figure_of_merit(result["GBW"], cload_pf, power)
        return result


def _abs(value: Optional[float]) -> Optional[float]:
    return abs(value) if value is not None else None


def _figure_of_merit(numerator: Optional[float], cload_pf: float, power_uw: Optional[float]) -> Optional[float]:
    if numerator is None or power_uw is None or power_uw <= 0:
        return None
    return numerator * cload_pf / (1000.0 * power_uw)


def read_tran_dat(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read an ngspice ASCII raw file with vectors time, v(vout3), v(visr)."""
    if not path.exists():
        return None
    rows: List[List[float]] = []
    current: List[float] = []
    in_values = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Values:"):
            in_values = True
            continue
        if not in_values or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:  # "<index> <time>" starts a new point
            if current:
                rows.append(current)
            current = [float(parts[1])]
        else:
            current.append(float(parts[0]))
    if current:
        rows.append(current)
    data = np.array([r for r in rows if len(r) == 3])
    if data.size == 0:
        return None
    return data[:, 0], data[:, 1], data[:, 2]


def settling_time(path: Path, band: float = SETTLING_BAND) -> Optional[float]:
    """AnalogGym settling time: per edge, the first time after which the output stays within
    `band` (relative) of the final input level; the result is the larger of the two edges.
    Returns None if either edge never settles."""
    data = read_tran_dat(path)
    if data is None:
        return None
    t, vout, vin = data
    dv = np.diff(vin)
    if not np.any(dv > 0) or not np.any(dv < 0):
        return None
    t0, t1 = t[np.where(dv > 0)[0][0]], t[np.where(dv < 0)[0][0]]
    v0, v1 = np.median(vin[t < t0]), np.median(vin[(t > t0) & (t < t1)])
    times = []
    for start, end, ref in ((t0, t1, v1), (t1, t.max(), v0)):
        idx = (t >= start) & (t <= end)
        seg_t, delta = t[idx], (vout[idx] - ref) / ref
        settled = next((i for i in range(len(delta)) if np.all(np.abs(delta[i:]) < band)), None)
        if settled is None:
            return None
        times.append(float(seg_t[settled] - start))
    return max(times)
