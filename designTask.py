"""AnalogSAGE design flow: from a natural-language specification to a sized, simulated design.

One test of one design task:
  while the specification is not met and iteration < MAX_ITERATION:
    0. memory      Evolution Memory of earlier tasks in the same series (insight.build_memory_prompt)
                   plus the reflections of failed iterations of this test
    1. knowledge   the Topology Selection Agent queries the Knowledge Module
    2. proposal    natural-language description of candidate topologies
    3. selection   the Candidates Module ranks library topologies; the first one not yet tried
                   in this test is used
    4. sizing      Parameter Optimization Agent + Numerical Module with ngspice (sizing_mix)
    5. compression on failure the Compression Module summarizes this iteration's history and
                   simulated performance
    6. refinement  the Topology Refinement Agent edits the topology using that summary; the
                   refined topology is re-sized
    7. reflection  the Reflection Module explains the failure, from the summary, for the next
                   iteration
Success is decided only by simulation: every constraint of the task in tasks.json must hold.
Reflections are reset for every test; insights are only shown to tasks with a larger id in
the same series and accumulate in-process in the order the tasks are run.

Example:
  python designTask.py --circuit ldo --tasks 20001 20002 20003 20004 20005 --tests 1-15
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import candidate
import compress
import config
import insight
import refine
import reflection
import sizing_mix
import tolology
from circuits import CIRCUITS, Circuit, get_circuit

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_sizing(
    circuit: Circuit,
    topology_id: int,
    spec_id: int,
    model: str,
    max_simulations: int,
    llm_budget: int,
    topology_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return sizing_mix.sizing_mix(
        circuit,
        topology_id,
        spec_id,
        mymodel=model,
        max_simulations=max_simulations,
        llm_budget=llm_budget,
        topology_path=topology_path,
    )


def _log_sizing(log: TextIO, label: str, result: Dict[str, Any]) -> None:
    log.write(
        f"{label}: met_specs={result['met_specs']}, sims={result['simulations']}, "
        f"best_score={result['best_score']}\nbest_perf={json.dumps(result['result_performance'])}\n"
        f"sizing log: {result['log_file']}\n"
    )


def designTask(
    circuit: Circuit,
    spec_id: int,
    experience: insight.Experience,
    k: int = config.TOP_K_CANDIDATES,
    model: str = config.DESIGN_MODEL,
    testK: int = 1,
    max_simulations: int = config.MAX_SIMULATIONS,
    llm_budget: int = config.LLM_BUDGET,
) -> Optional[Dict[str, Any]]:
    """Run one independent test of one design task. Returns the sizing result of the
    successful attempt (met_specs=True) or of the best attempt."""
    results_dir = config.RESULTS_ROOT / circuit.name
    refined_dir = results_dir / "netlist_refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"designTask_{spec_id}_k{k}_test{testK}_{_timestamp()}.log"
    spec = (circuit.tasks_dir / f"{spec_id}.prompt").read_text(encoding="utf-8")
    logger.info("designTask circuit=%s spec=%s test=%s log=%s", circuit.name, spec_id, testK, log_path)

    my_reflection: List[str] = []
    best_overall: Optional[Dict[str, Any]] = None
    tried_topologies: List[int] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"designTask circuit={circuit.name} spec={spec_id} k={k} test={testK} model={model}\n")
        iteration = 0
        while iteration < config.MAX_ITERATION:
            iteration += 1
            log.write(f"===== iteration {iteration} =====\n")
            reflection_prompt = ""
            if my_reflection:
                reflection_prompt = "\n Here are the reflections summarized from the previous trial:" + str(my_reflection) + "\n"
            memory_prompt = insight.build_memory_prompt(circuit, spec_id, experience)
            if memory_prompt:
                log.write("Memory prompt:\n" + memory_prompt + "\n")

            # 1-2. knowledge retrieval and topology proposal
            knowledge_docs = tolology.get_knowledge_docs(circuit, spec, model, log, memory_prompt, reflection_prompt)
            first_query, answer = tolology.propose_topology(circuit, spec, knowledge_docs, model, log, memory_prompt)
            history = first_query + "\n" + answer

            # 3. topology selection: size the first ranked candidate not tried in this test
            _, ranked = candidate.getCandidates(circuit, answer, spec, knowledge_docs)
            candidates = ranked[:k]
            log.write(f"Candidates: {candidates}\n")
            selected = next((c for c in candidates if c not in tried_topologies), None)
            if selected is None:
                log.write(f"All candidates already tried: {candidates}; skip to next iteration.\n")
                continue
            tried_topologies.append(selected)
            log.write(f"Selected candidate: {selected} (tried so far: {tried_topologies})\n")

            # 4. sizing
            result = _run_sizing(circuit, selected, spec_id, model, max_simulations, llm_budget)
            _log_sizing(log, f"topology {selected}", result)
            if best_overall is None or result["best_score"] > best_overall["best_score"]:
                best_overall = result
            if result["met_specs"]:
                _record_success(circuit, spec_id, selected, result, history, experience, log, False, testK)
                return result

            # 5. compression: summarize this iteration (history + simulated performance)
            failure_history = (
                history
                + "\nNow I get this design (topology id "
                + str(selected)
                + "):\nThe sized performance is\n"
                + json.dumps(result["result_performance"])
            )
            try:
                feedback = compress.compress_feedback(failure_history)
                log.write("Compressed feedback:\n" + feedback + "\n")
            except Exception as e:  # without a summary, refine and reflect on the full history
                logger.warning("Compression failed: %s", e)
                log.write(f"Compression failed: {e}\n")
                feedback = ""

            # 6. topology refinement with the compressed feedback, then re-size
            try:
                refined = refine.getMyRefine(
                    circuit,
                    spec_id,
                    circuit.topology_path(selected).read_text(encoding="utf-8"),
                    selected,
                    base_performance=json.dumps(result["result_performance"]),
                    feedback=feedback,
                )
            except Exception as e:  # refinement is best effort; fall through to reflection
                logger.warning("Refine failed: %s", e)
                log.write(f"Refine failed: {e}\n")
                refined = None
            if refined:
                refined_path = refined_dir / f"refined_{spec_id}_topo{selected}_test{testK}_{_timestamp()}.txt"
                refined_path.write_text(refined.netlist, encoding="utf-8")
                refined_path.with_suffix(".py").write_text(refined.pyspice_code + "\n", encoding="utf-8")
                log.write(f"Refined netlist saved: {refined_path} (PySpice source: {refined_path.with_suffix('.py')})\n")
                try:
                    result_r = _run_sizing(
                        circuit, selected, spec_id, model, max_simulations, llm_budget, topology_path=refined_path
                    )
                except Exception as e:  # an LLM netlist may not simulate; keep the unrefined result
                    logger.warning("Refined sizing failed: %s", e)
                    log.write(f"Refined sizing failed: {e}\n")
                    result_r = None
                if result_r is not None:
                    _log_sizing(log, f"refined topology {selected}", result_r)
                    if result_r["best_score"] > best_overall["best_score"]:
                        best_overall = result_r
                    if result_r["met_specs"]:
                        _record_success(circuit, spec_id, selected, result_r, history, experience, log, True, testK)
                        return result_r

            # 7. reflection for the next iteration of this test, from the compressed feedback
            try:
                reflection_input = (
                    feedback + f"\nThe topology id of this design is {selected}." if feedback else failure_history
                )
                my_reflection.append(reflection.get_reflection(reflection_input))
                log.write("Reflection:\n" + my_reflection[-1] + "\n")
            except Exception as e:  # keep iterating without this reflection
                log.write(f"Reflection failed: {e}\n")
            log.write(f"Iteration {iteration} failed.\n")

        log.write("FAILED to meet specs within MAX_ITERATION.\n")
        if best_overall is not None:
            log.write(
                f"best attempt: topology={best_overall['topology_id']}, score={best_overall['best_score']}\n"
                f"best_perf={json.dumps(best_overall['result_performance'])}\n"
            )
    return best_overall


def _record_success(
    circuit: Circuit,
    spec_id: int,
    topo_id: int,
    result: Dict[str, Any],
    history: str,
    experience: insight.Experience,
    log: TextIO,
    refined: bool,
    testK: int,
) -> None:
    tag = " (refined)" if refined else ""
    logger.info("SUCCESS: %s spec %s met with topology %s%s", circuit.name, spec_id, topo_id, tag)
    log.write(f"SUCCESS with topology {topo_id}{tag}\n")
    log.write(f"params={json.dumps(result['result_params'])}\n")
    log.write(f"performance={json.dumps(result['result_performance'])}\n")
    try:
        new_insight = insight.get_insight(
            history
            + "\nFinal topology id: "
            + str(topo_id)
            + "\nFinal performance: "
            + json.dumps(result["result_performance"]),
            circuit,
        )
        experience[str(spec_id)] = (topo_id, new_insight)
        log.write("New experience:\n" + new_insight + "\n")
    except Exception as e:  # the design still counts as a success
        log.write(f"Experience generation failed: {e}\n")
    summary = {
        "circuit": circuit.name,
        "spec_id": spec_id,
        "topology_id": topo_id,
        "refined": refined,
        "testK": testK,
        "params": result["result_params"],
        "performance": result["result_performance"],
        "simulations": result["simulations"],
        "sizing_log": result["log_file"],
    }
    summary_path = config.RESULTS_ROOT / circuit.name / (
        f"success_{spec_id}_topo{topo_id}{'_refined' if refined else ''}_test{testK}_{_timestamp()}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_tests(text: str) -> List[int]:
    """'1-15' -> [1..15]; '1,3,5' -> [1, 3, 5]."""
    tests: List[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            tests.extend(range(lo, hi + 1))
        elif part.strip():
            tests.append(int(part))
    return tests


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run AnalogSAGE design tasks.")
    parser.add_argument("--circuit", required=True, choices=sorted(CIRCUITS))
    parser.add_argument("--tasks", type=int, nargs="+", required=True, help="spec ids, run in this order")
    parser.add_argument("--tests", default="1-15", help="test indices, e.g. 1-15 or 1,2,3")
    parser.add_argument("--k", type=int, default=config.TOP_K_CANDIDATES, help="candidates kept per selection")
    parser.add_argument("--model", default=config.DESIGN_MODEL)
    parser.add_argument("--max-simulations", type=int, default=config.MAX_SIMULATIONS)
    parser.add_argument("--llm-budget", type=int, default=config.LLM_BUDGET)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    circuit = get_circuit(args.circuit)
    experience: insight.Experience = {}
    batch_path = config.RESULTS_ROOT / circuit.name / f"batch_{_timestamp()}.txt"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with batch_path.open("w", encoding="utf-8") as batch:
        batch.write("Batch results\n")
        for spec_id in args.tasks:
            for testK in parse_tests(args.tests):
                batch.write(f"Model {args.model} Query {spec_id} Test {testK}\n")
                batch.write("Current Experience:\n" + str(experience) + "\n")
                batch.flush()
                result = designTask(
                    circuit, spec_id, experience, k=args.k, model=args.model, testK=testK,
                    max_simulations=args.max_simulations, llm_budget=args.llm_budget,
                )
                met = bool(result and result.get("met_specs"))
                detail = f" (topology {result['topology_id']}, sims {result['simulations']})" if result else ""
                batch.write(f"Test {testK} final success: {met}{detail}\n")
                batch.flush()
    logger.info("Batch results: %s", batch_path)


if __name__ == "__main__":
    main()
