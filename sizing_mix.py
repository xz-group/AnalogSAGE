"""Parameter Optimization Agent and Numerical Module.

The LLM turns the specification, topology and simulation history into an optimizer
choice (BO or GA), narrowed parameter ranges, parameter relations and proposed samples;
the numerical optimizer runs the search and every candidate is simulated with ngspice.
Circuit-specific parts (prompt wording, parameter ranges, testbench) come from a
circuits.Circuit adapter.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import config
import llm
from circuits import Circuit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an expert in analog circuit design and numerical optimization."

# Objective J(x) of the Numerical Module (paper, Methodology):
#   J(x) = -LAMBDA * sum_i max(0, rho_i * (s_i - y_i)) / (|s_i| + EPS)
# rho_i = +1 for lower bounds (">=", ">") and -1 for upper bounds ("<=", "<"), y_i is the
# simulated metric and s_i its target. J = 0 only when every specification is met, and a
# satisfied constraint adds nothing, so extra margin is not rewarded. EPS is in the unit of
# s_i and only prevents division by zero. A metric that is missing or non-finite adds
# -MISSING_PENALTY; a simulation that returns no result scores FAILED_SCORE. Keys of
# tasks.json that start with "_" (e.g. "_context") carry test conditions, not constraints.
# A design passes only if every constraint holds exactly (no tolerance).
LAMBDA = 4.0
EPS = 1e-12
MISSING_PENALTY = 10.0
FAILED_SCORE = -1e9
LOWER_OPS = (">=", ">")
UPPER_OPS = ("<=", "<")

IntegerTest = Callable[[str], bool]


@dataclass
class LLMPlan:
    optimizer: str
    samples: List[Dict[str, Any]]
    ranges: Dict[str, Tuple[float, float]]
    relations: List[str]
    raw_response: str


def ask_llm(prompt: str, mymodel: str) -> str:
    return llm.ask(prompt, model=mymodel, system=SYSTEM_PROMPT)


# ---- parameter vectors ---------------------------------------------------------------------
def clip_vector(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def sample_uniform(n: int, rng: np.random.Generator, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, bounds.shape[0]))


def vector_to_params(
    x: np.ndarray, param_names: List[str], bounds: np.ndarray, is_integer: IntegerTest
) -> Dict[str, float]:
    x_clip = clip_vector(np.asarray(x, dtype=np.float64), bounds)
    out: Dict[str, float] = {}
    for i, name in enumerate(param_names):
        value = float(x_clip[i])
        if is_integer(name):
            value = float(int(round(value)))
            value = float(np.clip(value, bounds[i, 0], bounds[i, 1]))
        out[name] = value
    return out


def dict_to_vector(
    sample: Dict[str, Any],
    param_names: List[str],
    bounds: np.ndarray,
    rng: np.random.Generator,
    is_integer: IntegerTest,
    fallback: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    values: List[float] = []
    fallback = fallback or {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        raw = sample.get(name, fallback.get(name, None))
        if raw is None:
            value = float(rng.uniform(lo, hi))
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float(rng.uniform(lo, hi))
        value = float(np.clip(value, lo, hi))
        if is_integer(name):
            value = float(int(round(value)))
            value = float(np.clip(value, lo, hi))
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def params_to_vector(
    params: Dict[str, float], param_names: List[str], bounds: np.ndarray, is_integer: IntegerTest
) -> np.ndarray:
    out = []
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        value = float(params.get(name, lo))
        value = float(np.clip(value, lo, hi))
        if is_integer(name):
            value = float(int(round(value)))
            value = float(np.clip(value, lo, hi))
        out.append(value)
    return np.asarray(out, dtype=np.float64)


# ---- objective -------------------------------------------------------------------------------
def _spec_items(constraints: Dict[str, Any]) -> List[Tuple[str, Dict[str, float]]]:
    return [(k, v) for k, v in constraints.items() if not k.startswith("_")]


def _metric_value(result: Dict[str, Any], key: str) -> Optional[float]:
    raw = result.get(key, None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _satisfied(value: float, op: str, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    raise ValueError(f"Unsupported constraint operator {op!r}")


def result_meets_constraints(result: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    for key, cond in _spec_items(constraints):
        value = _metric_value(result, key)
        if value is None:
            return False
        for op, target in cond.items():
            if not _satisfied(value, op, float(target)):
                return False
    return True


def score_result(result: Dict[str, Any], constraints: Dict[str, Any]) -> float:
    """J(x): zero when all constraints hold, otherwise the scaled sum of normalized violations."""
    if not isinstance(result, dict):
        return FAILED_SCORE
    score = 0.0
    for key, cond in _spec_items(constraints):
        value = _metric_value(result, key)
        if value is None:
            score -= MISSING_PENALTY
            continue
        for op, target in cond.items():
            target = float(target)
            if op in LOWER_OPS:
                violation = max(0.0, target - value)
            elif op in UPPER_OPS:
                violation = max(0.0, value - target)
            else:
                raise ValueError(f"Unsupported constraint operator {op!r}")
            score -= LAMBDA * violation / (abs(target) + EPS)
    return float(score)


def get_violations(result: Dict[str, Any], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for key, cond in _spec_items(constraints):
        raw_value = result.get(key, None) if isinstance(result, dict) else None
        value = _metric_value(result, key) if isinstance(result, dict) else None
        for op, target in cond.items():
            target_f = float(target)
            if value is None:
                reason = "missing_result" if not isinstance(result, dict) else "missing_or_nonfinite"
                violations.append({"metric": key, "op": op, "target": target_f, "value": raw_value, "reason": reason})
            elif not _satisfied(value, op, target_f):
                violations.append(
                    {"metric": key, "op": op, "target": target_f, "value": value, "reason": "constraint_not_met"}
                )
    return violations


# ---- numerical optimizers (Bayesian optimization and genetic algorithm) -----------------------
def normalize_X(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = np.maximum(hi - lo, 1e-12)
    return (X - lo) / span


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 0.2, variance: float = 1.0) -> np.ndarray:
    x1_sq = np.sum(X1**2, axis=1, keepdims=True)
    x2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    dist2 = np.maximum(x1_sq + x2_sq - 2.0 * (X1 @ X2.T), 0.0)
    return variance * np.exp(-0.5 * dist2 / (length_scale**2))


def gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    noise: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(X_train, X_train) + noise * np.eye(X_train.shape[0])
    Ks = rbf_kernel(X_train, X_test)
    Kss_diag = np.diag(rbf_kernel(X_test, X_test))

    L = np.linalg.cholesky(K + 1e-12 * np.eye(K.shape[0]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    var = np.maximum(Kss_diag - np.sum(v * v, axis=0), 1e-12)
    return mu, np.sqrt(var)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = (mu - y_best) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def propose_bo(
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
    n_candidates: int = 1024,
) -> np.ndarray:
    if len(history_X) < 6:
        return sample_uniform(1, rng, bounds)[0]
    X = np.asarray(history_X, dtype=np.float64)
    y = np.asarray(history_y, dtype=np.float64)

    candidates = sample_uniform(n_candidates, rng, bounds)
    Xn = normalize_X(X, bounds)
    Cn = normalize_X(candidates, bounds)
    mu, sigma = gp_posterior(Xn, y, Cn)
    ei = expected_improvement(mu, sigma, float(np.max(y)))
    return candidates[int(np.argmax(ei))]


def _tournament_select(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = min(3, len(population))
    idx = rng.choice(len(population), size=k, replace=False)
    win = idx[int(np.argmax(fitness[idx]))]
    return population[win].copy()


def propose_ga(
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.08,
) -> np.ndarray:
    n = len(history_X)
    if n < 4:
        return sample_uniform(1, rng, bounds)[0]

    X = np.asarray(history_X, dtype=np.float64)
    y = np.asarray(history_y, dtype=np.float64)
    top_k = min(max(8, n // 2), n)
    keep_idx = np.argsort(y)[-top_k:]
    population = X[keep_idx]
    fitness = y[keep_idx]

    p1 = _tournament_select(population, fitness, rng)
    p2 = _tournament_select(population, fitness, rng)
    alpha = rng.uniform(0.0, 1.0, size=p1.shape[0])
    child = alpha * p1 + (1.0 - alpha) * p2

    span = bounds[:, 1] - bounds[:, 0]
    mask = rng.random(child.shape[0]) < mutation_rate
    noise = rng.normal(0.0, mutation_scale * span, size=child.shape[0])
    child = child + mask * noise
    return clip_vector(child, bounds)


def propose_with_optimizer(
    optimizer: str,
    history_X: List[np.ndarray],
    history_y: List[float],
    rng: np.random.Generator,
    bounds: np.ndarray,
) -> np.ndarray:
    opt = optimizer.upper()
    if opt == "GA":
        return propose_ga(history_X, history_y, rng, bounds)
    return propose_bo(history_X, history_y, rng, bounds)


# ---- LLM plan parsing and parameter relations ---------------------------------------------------
def _code_blocks(text: str) -> List[str]:
    return re.findall(r"```(?:json|python|text|txt)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_like(raw: str) -> Optional[Any]:
    if not raw:
        return None
    candidates = [raw.strip()]
    obj_text = _first_balanced_object(raw)
    if obj_text:
        candidates.append(obj_text.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return None


RELATION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*(.+?)\s*$")
RHS_NUM_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
RHS_VAR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
RHS_K_MUL_VAR_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
RHS_VAR_MUL_K_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def _eval_rhs(rhs: str, params: Dict[str, float]) -> Optional[float]:
    m = RHS_NUM_RE.match(rhs)
    if m:
        return float(m.group(1))
    m = RHS_VAR_RE.match(rhs)
    if m:
        key = m.group(1)
        if key in params:
            return float(params[key])
        return None
    m = RHS_K_MUL_VAR_RE.match(rhs)
    if m:
        k = float(m.group(1))
        key = m.group(2)
        if key in params:
            return k * float(params[key])
        return None
    m = RHS_VAR_MUL_K_RE.match(rhs)
    if m:
        key = m.group(1)
        k = float(m.group(2))
        if key in params:
            return float(params[key]) * k
        return None
    return None


def apply_relations_to_params(
    params: Dict[str, float],
    relations: List[str],
    param_names: List[str],
    bounds: np.ndarray,
    is_integer: IntegerTest,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Enforce LLM-proposed relations (==, >=, >, <=, <) as hard constraints before simulation."""
    if not relations:
        return dict(params), []

    adjusted = dict(params)
    name_to_idx = {n: i for i, n in enumerate(param_names)}
    applied: List[Dict[str, Any]] = []
    eps = 1e-9

    # Multiple passes to propagate chained dependencies.
    for _ in range(3):
        for expr in relations:
            m = RELATION_RE.match(expr)
            if not m:
                continue
            lhs, op, rhs = m.group(1), m.group(2), m.group(3)
            if lhs not in adjusted or lhs not in name_to_idx:
                continue
            rhs_val = _eval_rhs(rhs, adjusted)
            if rhs_val is None or not np.isfinite(rhs_val):
                continue

            old = float(adjusted[lhs])
            new = old
            if op == "==":
                new = rhs_val
            elif op == ">=":
                new = max(old, rhs_val)
            elif op == ">":
                new = max(old, rhs_val + eps)
            elif op == "<=":
                new = min(old, rhs_val)
            elif op == "<":
                new = min(old, rhs_val - eps)

            idx = name_to_idx[lhs]
            lo, hi = float(bounds[idx, 0]), float(bounds[idx, 1])
            new = float(np.clip(new, lo, hi))
            if is_integer(lhs):
                new = float(int(round(new)))
                new = float(np.clip(new, lo, hi))

            if abs(new - old) > 1e-15:
                adjusted[lhs] = new
                applied.append({"relation": expr, "lhs": lhs, "old": old, "new": new})

    return adjusted, applied


def parse_llm_plan(response: str, default_optimizer: str = "BO") -> LLMPlan:
    parsed_obj: Optional[Dict[str, Any]] = None
    for block in _code_blocks(response) + [response]:
        obj = _parse_json_like(block)
        if isinstance(obj, dict):
            parsed_obj = obj
            break

    if parsed_obj is None:
        return LLMPlan(default_optimizer.upper(), [], {}, [], response)

    raw_optimizer = str(parsed_obj.get("optimizer", default_optimizer)).upper().strip()
    optimizer = "GA" if raw_optimizer == "GA" else "BO"

    samples_raw = parsed_obj.get("samples", [])
    samples: List[Dict[str, Any]] = []
    if isinstance(samples_raw, dict):
        samples_raw = [samples_raw]
    if isinstance(samples_raw, list):
        for item in samples_raw:
            if isinstance(item, dict):
                samples.append(item)

    ranges: Dict[str, Tuple[float, float]] = {}
    ranges_raw = parsed_obj.get("ranges", {})
    if isinstance(ranges_raw, dict):
        for k, v in ranges_raw.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    lo = float(v[0])
                    hi = float(v[1])
                    ranges[str(k)] = (lo, hi)
                except (TypeError, ValueError):
                    continue

    relations: List[str] = []
    relations_raw = parsed_obj.get("relations", parsed_obj.get("constraints", []))
    if isinstance(relations_raw, str):
        relations_raw = [relations_raw]
    if isinstance(relations_raw, list):
        for item in relations_raw:
            if isinstance(item, str) and item.strip():
                relations.append(item.strip())

    return LLMPlan(optimizer=optimizer, samples=samples, ranges=ranges, relations=relations, raw_response=response)


def _build_llm_prompt(
    circuit: Circuit,
    stage: str,
    n_samples: int,
    spec_text: str,
    constraints: Dict[str, Any],
    topology: str,
    param_names: List[str],
    bounds: np.ndarray,
    hard_bounds: np.ndarray,
    rag_context: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
) -> str:
    range_lines = []
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        hlo, hhi = hard_bounds[i]
        range_lines.append(f"- {name}: current=[{lo}, {hi}], hard=[{hlo}, {hhi}]")
    history_text = json.dumps(history_tail, ensure_ascii=False, indent=2) if history_tail else "[]"
    best_text = json.dumps(best_result, ensure_ascii=False, indent=2) if best_result else "{}"
    constraints_text = json.dumps(constraints, ensure_ascii=False, indent=2)
    rag_text = rag_context.strip() if rag_context else ""
    return f"""
{circuit.sizing_intro}
This is stage: {stage}.

Spec text:
{spec_text}

Hard JSON constraints for pass/fail:
{constraints_text}

Current topology netlist:
{topology}

Retrieved knowledge context (RAG, may be empty):
{rag_text}

Current best known result:
{best_text}

Recent simulation history:
{history_text}

Parameter names and ranges:
{chr(10).join(range_lines)}

Range inference rule used by the system (follow this exactly):
```{circuit.range_rule_language}
{circuit.range_rule}
```

Task:
1) Choose optimizer for the next numerical proposal. Must be "BO" or "GA".
2) Propose exactly {n_samples} sizing samples for initialization/refresh.
3) Each sample must include every parameter exactly once with valid numeric values in range.
4) Propose a NEW range for every parameter as the next numerical-tool search range.
   Range must satisfy: hard_min <= new_min < new_max <= hard_max.
   You must provide ranges for all parameters in every response.
5) You may optionally provide sizing relations/constraints for variables, such as:
   - "P2_W == 2 * P1_W"
   - "N3_M >= N1_M"
   - {circuit.relation_example}

Return EXACTLY ONE JSON object in one code block and nothing else:
```json
{{
  "optimizer": "BO",
  "ranges": {{
    "PARAM_A": [0.0, 1.0],
    "PARAM_B": [0.0, 1.0]
  }},
  "relations": [
    "PARAM_B >= PARAM_A",
    "PARAM_C == 2 * PARAM_A"
  ],
  "samples": [
    {{"PARAM_A": 0.0, "PARAM_B": 0.0}}
  ]
}}
```
Note: You can choose either "BO" or "GA" for optimizer.
""".strip()


def request_llm_plan(mymodel: str, default_optimizer: str, **prompt_args: Any) -> LLMPlan:
    prompt = _build_llm_prompt(**prompt_args)
    try:
        response = ask_llm(prompt, mymodel)
    except Exception as e:  # network/API errors must not stop the numerical search
        logger.warning("LLM plan request failed: %s", e)
        return LLMPlan(default_optimizer.upper(), [], {}, [], f"LLM_ERROR: {e}")
    return parse_llm_plan(response, default_optimizer=default_optimizer)


def sanitize_bounds_update(
    proposed_ranges: Dict[str, Tuple[float, float]],
    param_names: List[str],
    current_bounds: np.ndarray,
    hard_bounds: np.ndarray,
) -> np.ndarray:
    new_bounds = current_bounds.copy()
    for i, name in enumerate(param_names):
        if name not in proposed_ranges:
            continue
        lo_raw, hi_raw = proposed_ranges[name]
        hlo, hhi = hard_bounds[i]
        lo = float(np.clip(lo_raw, hlo, hhi))
        hi = float(np.clip(hi_raw, hlo, hhi))
        if lo > hi:
            lo, hi = hi, lo
        # guarantee non-zero span
        if abs(hi - lo) < 1e-15:
            eps = max((hhi - hlo) * 1e-6, 1e-12)
            lo = max(hlo, lo - eps)
            hi = min(hhi, hi + eps)
            if lo >= hi:
                lo, hi = hlo, hhi
        new_bounds[i, 0] = lo
        new_bounds[i, 1] = hi
    return new_bounds


def _sanitize_samples(
    samples: List[Dict[str, Any]],
    target_count: int,
    param_names: List[str],
    bounds: np.ndarray,
    rng: np.random.Generator,
    is_integer: IntegerTest,
    fallback_params: Optional[Dict[str, float]] = None,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for sample in samples:
        out.append(dict_to_vector(sample, param_names, bounds, rng, is_integer, fallback=fallback_params))
        if len(out) >= target_count:
            return out
    while len(out) < target_count:
        out.append(sample_uniform(1, rng, bounds)[0])
    return out


def _history_tail(records: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    return records[-k:] if len(records) > k else records


def _best_designs(records: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(records, key=lambda r: float(r.get("score", -1e18)), reverse=True)
    best: List[Dict[str, Any]] = []
    for rec in ranked[: max(0, top_n)]:
        best.append(
            {
                "sim": rec.get("sim"),
                "score": rec.get("score"),
                "source": rec.get("source"),
                "optimizer": rec.get("optimizer"),
                "params": rec.get("params"),
                "result": rec.get("result"),
                "violations": rec.get("violations"),
            }
        )
    return best


# ---- literature retrieval for sizing ------------------------------------------------------------
def build_retrieval_query(
    circuit: Circuit,
    mymodel: str,
    spec_text: str,
    constraints: Dict[str, Any],
    topology: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
) -> str:
    prompt = f"""
You are preparing one concise retrieval query for {circuit.retrieval_subject}.
Focus on parameter sizing strategy, not topology generation.

Spec text:
{spec_text}

Hard constraints:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Current topology:
{topology}

Best result so far:
{json.dumps(best_result, ensure_ascii=False, indent=2) if best_result else "{}"}

Recent history:
{json.dumps(history_tail, ensure_ascii=False, indent=2) if history_tail else "[]"}

Return exactly one query in one text code block:
```text
your retrieval query here
```
""".strip()
    response = ask_llm(prompt, mymodel)
    blocks = _code_blocks(response)
    if blocks:
        q = blocks[0].strip()
        if q:
            return q
    q = response.strip()
    return q if q else circuit.retrieval_fallback


def get_retrieval_context(
    circuit: Circuit,
    mymodel: str,
    spec_text: str,
    constraints: Dict[str, Any],
    topology: str,
    history_tail: List[Dict[str, Any]],
    best_result: Optional[Dict[str, Any]],
    rag_top_k: int = 5,
) -> Tuple[str, str]:
    """Return (knowledge summary, debug note). Retrieval failures are logged and skipped."""
    try:
        import knowledge
    except ImportError as e:
        return "", f"RAG skipped: import knowledge failed ({e})"
    try:
        query = build_retrieval_query(circuit, mymodel, spec_text, constraints, topology, history_tail, best_result)
    except Exception as e:
        return "", f"RAG skipped: query generation failed ({e})"
    try:
        context = knowledge.getKnowledge(query, mytop_k=rag_top_k, stage="parameter sizing")
        return context or "", f"RAG query: {query}"
    except Exception as e:
        return "", f"RAG retrieval failed for query [{query}] ({e})"


# ---- main loop -----------------------------------------------------------------------------------
def sizing_mix(
    circuit: Circuit,
    topology_id: int,
    spec_id: int,
    mymodel: str = config.DESIGN_MODEL,
    max_simulations: int = 1000,
    llm_budget: int = 400,
    feedback_interval: int = 20,
    initial_llm_samples: int = 5,
    refresh_llm_samples: int = 5,
    rag_top_k: int = 5,
    seed: int = 42,
    topology_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Size one topology for one specification.

    The LLM sets the optimizer, ranges, relations and samples at the start and after every
    `feedback_interval` simulations until `llm_budget` simulations have been used; the search
    then continues numerically inside the last LLM-defined ranges. The run stops at the first
    design that meets every constraint or after `max_simulations` simulations.
    """
    rng = np.random.default_rng(seed)
    is_integer = circuit.is_integer_param
    workdir = circuit.workdir

    topology_path = topology_path or circuit.topology_path(topology_id)
    merged_topology = circuit.compose(topology_path.read_text(encoding="utf-8"), workdir)
    spec_text = circuit.load_spec_text(spec_id)
    constraints = circuit.load_constraints(spec_id)

    param_names = circuit.extract_param_names(merged_topology)
    hard_bounds = circuit.build_bounds(param_names, merged_topology)
    bounds = hard_bounds.copy()
    log_dir = config.RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"sizing_mix_{circuit.name}_{spec_id}_topo{topology_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    )

    prompt_args = dict(
        circuit=circuit,
        spec_text=spec_text,
        constraints=constraints,
        topology=merged_topology,
        param_names=param_names,
        hard_bounds=hard_bounds,
    )

    history_X: List[np.ndarray] = []
    history_y: List[float] = []
    sim_records: List[Dict[str, Any]] = []
    pending_queue: List[Tuple[str, np.ndarray]] = []
    active_optimizer = "BO"
    active_relations: List[str] = []
    llm_enabled = llm_budget > 0

    best_score = -1e18
    best_params: Optional[Dict[str, float]] = None
    best_perf: Optional[Dict[str, Any]] = None
    solved_params: Optional[Dict[str, float]] = None
    solved_perf: Optional[Dict[str, Any]] = None
    met_specs = False

    init_rag_debug = "LLM disabled"
    init_llm_response = "LLM disabled"
    if llm_enabled:
        init_rag_context, init_rag_debug = get_retrieval_context(
            circuit, mymodel, spec_text, constraints, merged_topology, [], None, rag_top_k
        )
        init_plan = request_llm_plan(
            mymodel,
            active_optimizer,
            stage="initial",
            n_samples=initial_llm_samples,
            bounds=bounds,
            rag_context=init_rag_context,
            history_tail=[],
            best_result=None,
            **prompt_args,
        )
        init_llm_response = init_plan.raw_response
        active_optimizer = init_plan.optimizer
        active_relations = init_plan.relations
        bounds = sanitize_bounds_update(init_plan.ranges, param_names, bounds, hard_bounds)
        init_vectors = _sanitize_samples(
            init_plan.samples, initial_llm_samples, param_names, bounds, rng, is_integer, fallback_params=None
        )
        pending_queue.extend([("llm_init", x) for x in init_vectors])

    def bounds_json() -> str:
        return json.dumps({n: [float(bounds[i, 0]), float(bounds[i, 1])] for i, n in enumerate(param_names)})

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"circuit={circuit.name}, topology_id={topology_id}, spec_id={spec_id}, model={mymodel}\n")
        log.write(f"topology_path={topology_path}\n")
        log.write(
            f"max_simulations={max_simulations}, llm_budget={llm_budget}, "
            f"feedback_interval={feedback_interval}, seed={seed}\n"
        )
        log.write(f"active_optimizer(init)={active_optimizer}\n")
        log.write(f"active_relations(init)={json.dumps(active_relations, ensure_ascii=False)}\n")
        log.write(f"active_bounds(init)={bounds_json()}\n")
        log.write(f"init_rag_debug={init_rag_debug}\n")
        log.write(f"initial_llm_response:\n{init_llm_response}\n\n")

        sim_count = 0
        while sim_count < max_simulations:
            if llm_enabled and sim_count >= llm_budget:
                # LLM-guided phase over: keep the last LLM ranges, relations and optimizer.
                llm_enabled = False
                pending_queue = []
                log.write(f"[switch@{sim_count}] pure numerical mode within the last LLM ranges and relations\n")
                log.write(f"[switch@{sim_count}] active_bounds -> {bounds_json()}\n")

            if pending_queue:
                source, x = pending_queue.pop(0)
            else:
                x = propose_with_optimizer(active_optimizer, history_X, history_y, rng, bounds)
                source = f"{active_optimizer.lower()}_proposal"

            params = vector_to_params(x, param_names, bounds, is_integer)
            params, relation_applied = apply_relations_to_params(params, active_relations, param_names, bounds, is_integer)
            result = circuit.simulate(params, constraints, workdir)
            score = score_result(result, constraints)
            violations = get_violations(result, constraints)
            x_effective = params_to_vector(params, param_names, bounds, is_integer)

            history_X.append(x_effective.copy())
            history_y.append(score)
            sim_count += 1

            record = {
                "sim": sim_count,
                "source": source,
                "optimizer": active_optimizer,
                "score": score,
                "params": params,
                "result": result,
                "violations": violations,
                "relations": active_relations,
                "relation_applied": relation_applied,
            }
            sim_records.append(record)

            log.write(f"[sim {sim_count}] source={source}, optimizer={active_optimizer}, score={score:.6f}\n")
            log.write(f"params={json.dumps(params, ensure_ascii=False)}\n")
            if relation_applied:
                log.write(f"RELATION_APPLIED={json.dumps(relation_applied, ensure_ascii=False)}\n")
            log.write(f"result={json.dumps(result, ensure_ascii=False)}\n")
            log.write(f"VIOLATIONS={json.dumps(violations, ensure_ascii=False)}\n")

            if score > best_score:
                best_score = score
                best_params = params
                best_perf = result

            if result_meets_constraints(result, constraints):
                met_specs = True
                solved_params = params
                solved_perf = result
                log.write(f"PASS at sim={sim_count}\n")
                break

            if llm_enabled and sim_count % feedback_interval == 0 and sim_count < min(max_simulations, llm_budget):
                best_result = {"best_score": best_score, "best_params": best_params, "best_result": best_perf}
                refresh_rag_context, refresh_rag_debug = get_retrieval_context(
                    circuit, mymodel, spec_text, constraints, merged_topology,
                    _history_tail(sim_records, 8), best_result, rag_top_k,
                )
                refresh_plan = request_llm_plan(
                    mymodel,
                    active_optimizer,
                    stage=f"refresh_after_{sim_count}_sims",
                    n_samples=refresh_llm_samples,
                    bounds=bounds,
                    rag_context=refresh_rag_context,
                    history_tail=_history_tail(sim_records, 8),
                    best_result=best_result,
                    **prompt_args,
                )
                active_optimizer = refresh_plan.optimizer
                active_relations = refresh_plan.relations
                bounds = sanitize_bounds_update(refresh_plan.ranges, param_names, bounds, hard_bounds)
                log.write(f"[feedback@{sim_count}] active_optimizer -> {active_optimizer}\n")
                log.write(f"[feedback@{sim_count}] active_relations -> {json.dumps(active_relations, ensure_ascii=False)}\n")
                log.write(f"[feedback@{sim_count}] active_bounds -> {bounds_json()}\n")
                log.write(f"[feedback@{sim_count}] rag_debug -> {refresh_rag_debug}\n")
                log.write(f"[feedback@{sim_count}] llm_response:\n{refresh_plan.raw_response}\n")

                # Add one numerical proposal immediately, then add LLM proposals.
                numerical_next = propose_with_optimizer(active_optimizer, history_X, history_y, rng, bounds)
                pending_queue.append((f"{active_optimizer.lower()}_next", numerical_next))
                llm_vectors = _sanitize_samples(
                    refresh_plan.samples, refresh_llm_samples, param_names, bounds, rng, is_integer,
                    fallback_params=best_params,
                )
                pending_queue.extend([("llm_refresh", x_new) for x_new in llm_vectors])

        log.write("\n=== SUMMARY ===\n")
        log.write(f"met_specs={met_specs}\n")
        log.write(f"sim_count={sim_count}\n")
        log.write(f"best_score={best_score}\n")
        log.write(f"best_params={json.dumps(best_params, ensure_ascii=False)}\n")
        log.write(f"best_perf={json.dumps(best_perf, ensure_ascii=False)}\n")
        log.write(f"best_designs={json.dumps(_best_designs(sim_records, top_n=5), ensure_ascii=False)}\n")

    return {
        "topology_id": topology_id,
        "spec_id": spec_id,
        "met_specs": met_specs,
        "simulations": len(sim_records),
        "active_optimizer": active_optimizer,
        "result_params": solved_params if met_specs else best_params,
        "result_performance": solved_perf if met_specs else best_perf,
        "best_score": best_score,
        "best_designs": _best_designs(sim_records, top_n=5),
        "log_file": str(log_path),
        "history": sim_records,
    }
