"""ngspice helpers shared by the circuit adapters."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import config

logger = logging.getLogger(__name__)

PDK_TOKEN = "@SKY130_PDK_ROOT@"
_NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

PARSED_MARKER = "NETLIST_PARSED"  # echoed by netlist_check.spice once the circuit is loaded
# Warnings alone are not errors: the SKY130 model files raise "Model issue ... unrecognized
# parameter - ignored", and a 0-ohm link resistor raises "Value of resistor ... too small".
_PARSE_ERROR_RE = re.compile(
    r"(?i)error|unknown|undefined|too (?:many|few)|could not|can't find|cannot|not a valid|interrupted"
)
# Batch mode always ends a deck without analyses with this notice; it is not a netlist error.
_BATCH_NOTICE = ("incomplete or empty netlist", 'or no ".plot"', "no simulations run")
MAX_PARSE_ERRORS = 8


def render_testbench(template: Path, workdir: Path, substitutions: Optional[Mapping[str, str]] = None) -> Path:
    """Copy a testbench template into workdir, filling in the PDK path and other tokens."""
    text = template.read_text(encoding="utf-8")
    tokens = {PDK_TOKEN: str(config.sky130_pdk_root()), **(substitutions or {})}
    for token, value in tokens.items():
        text = text.replace(token, value)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / template.name
    out.write_text(text, encoding="utf-8")
    return out


def write_varspice(workdir: Path, params: Mapping[str, float]) -> None:
    """Write the sizing parameters as a .PARAM block (three values per line)."""
    kvs = [f"{name}={value}" for name, value in params.items()]
    lines = [".PARAM"] + ["+ " + " ".join(kvs[i : i + 3]) for i in range(0, len(kvs), 3)]
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "var.spice").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ngspice(workdir: Path, testbench: str, output: str) -> Optional[str]:
    """Run ngspice in batch mode; return the output text, or None if the run failed."""
    out_path = workdir / output
    out_path.unlink(missing_ok=True)  # never read a stale result after a crash
    try:
        subprocess.run(
            ["ngspice", "-b", testbench, "-o", output],
            cwd=workdir,
            check=True,
            timeout=config.NGSPICE_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("ngspice failed for %s in %s: %s", testbench, workdir, e)
    if not out_path.exists():
        return None
    return out_path.read_text(encoding="utf-8", errors="ignore")


def parse_check(workdir: Path, deck: str, output: str = "check.txt") -> List[str]:
    """Load a deck without analyses in ngspice and return its netlist errors ([] if it parses).

    ngspice exits with status 1 for any deck without analyses in batch mode, so the result is
    read from the log: error and warning lines, and the PARSED_MARKER echoed after loading.
    """
    out_path = workdir / output
    out_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["ngspice", "-b", deck, "-o", output],
            cwd=workdir,
            timeout=config.NGSPICE_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return [f"ngspice could not run the netlist check: {e}"]
    if not out_path.exists():
        return ["ngspice wrote no output for the netlist check"]
    text = out_path.read_text(encoding="utf-8", errors="ignore")
    errors: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if _PARSE_ERROR_RE.search(s) and not any(notice in s for notice in _BATCH_NOTICE) and s[:240] not in errors:
            errors.append(s[:240])  # PDK model lines echoed in errors can be thousands of characters
    if not errors and PARSED_MARKER not in text:
        errors.append("ngspice stopped before the netlist was loaded")
    return errors[:MAX_PARSE_ERRORS]


def parse_values(text: Optional[str], names: Iterable[str]) -> Dict[str, Optional[float]]:
    """Read `name = value` lines printed by .meas / print (ngspice prints names in lower case)."""
    values: Dict[str, Optional[float]] = {}
    for name in names:
        match = re.search(rf"\b{re.escape(name)}\s*=\s*{_NUMBER}", text or "")
        values[name] = float(match.group(1)) if match else None
    return values
