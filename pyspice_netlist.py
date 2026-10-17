"""PySpice conversion and structural checks for refined topologies.

The Topology Refinement Agent writes the refined topology as a PySpice SubCircuitFactory
class. The class is never executed: its source is parsed with `ast`, only element calls
with literal arguments are accepted (self.X / R / C / V / I), and the calls are replayed
on a PySpice SubCircuit, which renders the ngspice netlist. structural_errors then checks
the rendered netlist (ports, device pin counts, dangling nodes); Circuit.check_netlist adds
an ngspice parse of the netlist with the SKY130 models.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)  # PySpice 1.5 has invalid escapes in docstrings
    from PySpice.Spice.Netlist import SubCircuit

ELEMENTS = ("X", "R", "C", "V", "I")
FACTORY_BASES = ("SubCircuitFactory", "SubCircuit")
GROUND = "0"

IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
MODEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PARAM_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SPICE_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:[a-zA-Z]+)?$")
EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_+\-*/(). ]+$")

# Terminal counts of the subcircuits in the topology library (the testbench supplies
# 'amplifier' for topologies that already use it).
PIN_RULES: Tuple[Tuple[re.Pattern, Set[int]], ...] = (
    (re.compile(r"^sky130_fd_pr__[np]fet_"), {4}),
    (re.compile(r"^sky130_fd_pr__npn_"), {4}),
    (re.compile(r"^sky130_fd_pr__pnp_"), {3}),
    (re.compile(r"^sky130_fd_pr__res_"), {3}),
    (re.compile(r"^sky130_fd_pr__cap_mim_"), {2}),
    (re.compile(r"^amplifier$"), {2, 3}),
)
PDK_PREFIX = "sky130_fd_pr__"


class NetlistError(ValueError):
    """The PySpice code could not be converted; `errors` lists every problem found."""

    def __init__(self, errors: List[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


# ---- conversion ----------------------------------------------------------------------------
def pyspice_to_netlist(code: str, name: str, nodes: Sequence[str]) -> str:
    """Convert the PySpice class in `code` into a .subckt netlist named `name` with ports `nodes`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise NetlistError([f"Python syntax error at line {e.lineno}: {e.msg}"]) from None

    errors: List[str] = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.Import, ast.ImportFrom)) and not _is_docstring(node):
            errors.append(f"line {node.lineno}: only imports and one class are allowed at module level")
    if len(classes) != 1:
        raise NetlistError(errors + [f"expected exactly one PySpice class, found {len(classes)}"])
    cls = classes[0]

    if not any(_base_name(b) in FACTORY_BASES for b in cls.bases):
        errors.append(f"class {cls.name} must derive from SubCircuitFactory")
    attrs = _class_constants(cls)
    class_name = attrs.get("NAME", attrs.get("__name__"))
    class_nodes = attrs.get("NODES", attrs.get("__nodes__"))
    if not isinstance(class_name, str) or class_name.lower() != name.lower():
        errors.append(f'NAME must be "{name}", got {class_name!r}')
    if not isinstance(class_nodes, (tuple, list)) or [str(n).lower() for n in class_nodes] != [
        n.lower() for n in nodes
    ]:
        errors.append(f"NODES must be {tuple(nodes)} in this order, got {class_nodes!r}")

    init = next((s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"), None)
    if init is None:
        raise NetlistError(errors + [f"class {cls.name} has no __init__ with the element calls"])

    subckt = SubCircuit(name, *nodes)
    n_elements = 0
    for stmt in init.body:
        if _is_docstring(stmt) or isinstance(stmt, ast.Pass) or _is_super_init(stmt):
            continue
        call = _element_call(stmt)
        if call is None:
            errors.append(
                f"line {stmt.lineno}: __init__ may only contain self.X/R/C/V/I(...) calls with literal arguments"
            )
            continue
        kind, args, kwargs, problems = call
        if problems:
            errors.extend(f"line {stmt.lineno}: {p}" for p in problems)
            continue
        problems = _check_element(kind, args, kwargs)
        if problems:
            errors.extend(f"line {stmt.lineno}: {p}" for p in problems)
            continue
        try:
            getattr(subckt, kind)(*args, **kwargs)
            n_elements += 1
        except (NameError, ValueError, TypeError, KeyError) as e:  # e.g. duplicate element name
            errors.append(f"line {stmt.lineno}: PySpice rejected self.{kind}(...): {e}")
    if n_elements == 0 and not errors:
        errors.append("the class defines no elements")
    if errors:
        raise NetlistError(errors)
    return str(subckt).strip() + "\n"


def _is_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _class_constants(cls: ast.ClassDef) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                values[stmt.targets[0].id] = ast.literal_eval(stmt.value)
            except ValueError:
                continue
    return values


def _is_super_init(stmt: ast.stmt) -> bool:
    """super().__init__(...), SubCircuitFactory.__init__(self) or SubCircuit.__init__(self, ...)."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    func = stmt.value.func
    if not (isinstance(func, ast.Attribute) and func.attr == "__init__"):
        return False
    owner = func.value
    if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super":
        return True
    return isinstance(owner, ast.Name) and owner.id in FACTORY_BASES


def _element_call(stmt: ast.stmt) -> Optional[Tuple[str, List[Any], Dict[str, Any], List[str]]]:
    """Return (kind, args, kwargs, problems) for a self.<kind>(...) statement, else None."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    func = call.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self"):
        return None
    if func.attr not in ELEMENTS:
        return func.attr, [], {}, [f"self.{func.attr}(...) is not allowed; use one of {', '.join(ELEMENTS)}"]
    problems: List[str] = []
    args: List[Any] = []
    for arg in call.args:
        try:
            args.append(ast.literal_eval(arg))
        except ValueError:
            problems.append(f"argument {ast.unparse(arg)!r} of self.{func.attr} is not a literal")
    kwargs: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            problems.append(f"**kwargs is not allowed in self.{func.attr}")
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            problems.append(f"value of {kw.arg}= in self.{func.attr} is not a literal")
    return func.attr, args, kwargs, problems


def _check_element(kind: str, args: List[Any], kwargs: Dict[str, Any]) -> List[str]:
    """Validate the literal arguments in place: names, nodes and SPICE values (quoted if symbolic)."""
    problems: List[str] = []
    min_args = 3 if kind == "X" else 4
    if len(args) < min_args or (kind != "X" and len(args) != 4):
        expected = "(name, subcircuit, node, ...)" if kind == "X" else "(name, node1, node2, value)"
        return [f"self.{kind} expects {expected}, got {len(args)} positional arguments"]
    args[0] = str(args[0])
    if not IDENT_RE.match(args[0]):
        problems.append(f"element name {args[0]!r} may only use letters, digits and underscores")
    node_slice = slice(2, None) if kind == "X" else slice(1, 3)
    for node in args[node_slice]:
        if not isinstance(node, (str, int)) or isinstance(node, bool) or not IDENT_RE.match(str(node)):
            problems.append(f"node {node!r} may only use letters, digits and underscores")
    if kind == "X" and (not isinstance(args[1], str) or not MODEL_RE.match(args[1])):
        problems.append(f"subcircuit name {args[1]!r} is not a valid identifier")
    for key in list(kwargs):
        if not PARAM_KEY_RE.match(key) or key == "raw_spice":
            problems.append(f"parameter name {key!r} is not allowed")
            continue
        value, problem = _spice_value(kwargs[key])
        if problem:
            problems.append(f"{key}={kwargs[key]!r}: {problem}")
        kwargs[key] = value
    if kind != "X":
        # R / C may name a SKY130 device model instead of a value (R1 a b sky130_fd_pr__res_generic_po w=2 l='R1')
        model = args[3].strip() if isinstance(args[3], str) else ""
        if kind in ("R", "C") and model.lower().startswith(PDK_PREFIX) and MODEL_RE.match(model):
            args[3] = model
        else:
            value, problem = _spice_value(args[3])
            if problem:
                problems.append(f"value {args[3]!r}: {problem}")
            args[3] = value
        if kwargs:  # PySpice accepts only a few named parameters for R/C/V/I; pass the checked ones verbatim
            raw = " ".join(f"{k}={v}" for k, v in kwargs.items())
            kwargs.clear()
            kwargs["raw_spice"] = raw
    for i in range(2 if kind == "X" else 1, len(args) if kind == "X" else 3):
        args[i] = str(args[i])
    return problems


def _spice_value(value: Any) -> Tuple[str, Optional[str]]:
    """Numbers stay as they are; a sizing variable or expression is quoted for ngspice."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return "", "must be a number or a string with a sizing variable or expression"
    if isinstance(value, (int, float)):
        return repr(value), None
    text = value.strip().strip("'\"{}").strip()
    if SPICE_NUMBER_RE.match(text):
        return text, None
    if not text or not EXPRESSION_RE.match(text) or not re.search(r"[A-Za-z_]", text):
        return "", "use a sizing variable or a simple arithmetic expression of sizing variables"
    return f"'{text}'", None


# ---- structural checks ----------------------------------------------------------------------
def structural_errors(netlist: str, ports: Sequence[str], allowed_subckts: Set[str]) -> List[str]:
    """Port connections, device pin counts, allowed subcircuits and dangling nodes of a .subckt."""
    errors: List[str] = []
    terminals: Dict[str, int] = {}
    port_set = {p.lower() for p in ports}
    allowed = {s.lower() for s in allowed_subckts}
    for line in _body_lines(netlist):
        tokens = line.split()
        kind = tokens[0][0].upper()
        if kind == "X":
            positional = [t for t in tokens[1:] if "=" not in t]
            model, nodes = positional[-1], positional[:-1]
            low = model.lower()
            if not low.startswith(PDK_PREFIX) and low not in allowed:
                errors.append(f"{tokens[0]}: subcircuit {model} is neither a SKY130 device nor used by the input topology")
            rule = next((counts for pattern, counts in PIN_RULES if pattern.match(low)), None)
            if rule is not None and len(nodes) not in rule:
                errors.append(f"{tokens[0]}: {model} needs {sorted(rule)} terminals, got {len(nodes)}")
        else:
            nodes = tokens[1:3]
        for node in nodes:
            terminals[node.lower()] = terminals.get(node.lower(), 0) + 1
    if not terminals:
        return ["the subcircuit has no elements"]
    for port in ports:
        if terminals.get(port.lower(), 0) == 0:
            errors.append(f"port {port} is not connected to any element")
    dangling = sorted(n for n, count in terminals.items() if count == 1 and n not in port_set and n != GROUND)
    if dangling:
        errors.append(f"dangling nodes (only one terminal connected): {', '.join(dangling)}")
    return errors


def subckt_models(netlist: str) -> Set[str]:
    """Subcircuit names instantiated by the X elements of a netlist."""
    models: Set[str] = set()
    for line in _body_lines(netlist):
        tokens = line.split()
        if tokens[0][0].upper() == "X":
            models.add([t for t in tokens[1:] if "=" not in t][-1])
    return models


def _body_lines(netlist: str) -> List[str]:
    lines: List[str] = []
    inside = False
    for raw in netlist.splitlines():
        s = raw.strip()
        low = s.lower()
        if low.startswith(".subckt"):
            inside = True
        elif low.startswith(".ends"):
            inside = False
        elif inside and s and not s.startswith("*") and len(s.split()) >= 3:
            lines.append(s)
    return lines
