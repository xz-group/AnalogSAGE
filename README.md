# AnalogSAGE

AnalogSAGE turns a natural-language specification into a sized analog circuit. It combines a
topology selection agent, a topology refinement agent, and a parameter optimization agent with
memory layers. Every design is checked by ngspice simulation on the SKY130 PDK. The flow
supports three circuit types: operational amplifiers, low-dropout regulators (LDOs), and
bandgap voltage references.

## Modules

| File | Component |
| --- | --- |
| `designTask.py` | Design flow and command-line entry point |
| `tolology.py` | Topology Selection Agent (knowledge query and topology proposal) |
| `candidate.py` | Candidates Module (vector + BM25 retrieval over the topology library, LLM ranking) |
| `refine.py` | Topology Refinement Agent (writes the refined topology as a PySpice class) |
| `pyspice_netlist.py` | PySpice conversion of refined topologies and their structural checks |
| `sizing_mix.py` | Parameter Optimization Agent and Numerical Module (BO/GA + ngspice), objective J(x) |
| `knowledge.py`, `queryPinecone.py` | Knowledge Module and Design Knowledge RAG (Pinecone) |
| `reflection.py` | Reflection Module (Introspective Optimization Layer) |
| `insight.py` | Insight Module and Evolution Memory |
| `compress.py` | Compression Module (summarizes a failed sizing attempt for refinement and reflection) |
| `circuits/` | Circuit adapters: prompts, parameter ranges, netlist composition, metric extraction |
| `simulation/` | Testbench templates per circuit type, the netlist-check deck, and the ngspice runner |
| `tasks/<circuit>/` | Task prompts (`<id>.prompt`), constraints (`tasks.json`), seed memory |
| `topologies/<circuit>/` | Topology library: netlists and RAG descriptions with embeddings |

## Setup

1. Install Python 3.10 or newer and run `pip install -r requirements.txt`. PySpice is used only
   to build netlists, so its ngspice shared-library interface is not needed.
2. Install ngspice (tested with ngspice-47).
3. Install the SKY130 PDK.
4. Set the environment variables:

```bash
export GEMINI_API_KEY=...      # Gemini models (OpenAI-compatible endpoint)
export OPENAI_API_KEY=...      # text-embedding-3-large (topology retrieval and RAG)
export PINECONE_API_KEY=...    # Design Knowledge RAG
export PINECONE_INDEX=...      # name of the Pinecone index of the knowledge base
export PINECONE_NAMESPACE=...  # namespace inside the index (optional)
export SKY130_PDK_ROOT=/path/to/sky130_pdk   # directory that contains libs.tech/ngspice/
```

The knowledge base is a Pinecone index of paper chunks embedded with `text-embedding-3-large`;
the metadata of each record must contain the chunk in a `text` field. If the index is not
reachable, knowledge retrieval is skipped and the flow continues without it.

## Run

```bash
python designTask.py --circuit opamp   --tasks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --tests 1-15
python designTask.py --circuit ldo     --tasks 20001 20002 20003 20004 20005 --tests 1-15
python designTask.py --circuit bandgap --tasks 30001 30002 30003 30004 30005 --tests 1-15
```

Tasks run in the order given. Insights from successful tasks are kept in memory for the rest of
the process and are shown only to later tasks of the same series. Reflections are reset for every
test. Outputs are written to `results/`:

- `results/<circuit>/`: `designTask_*.log` (one per test), `success_*.json`, `batch_*.txt`, and
  `netlist_refined/` with each refined netlist (`.txt`) and its PySpice source (`.py`).
- `results/logs/`: `sizing_mix_*.log` and `refine_*.log`.
- `results/work/<circuit>/`: the ngspice working directory; `results/work/<circuit>_check/` holds the
  netlist check of refined topologies.

## Tasks

| Series | Circuit | Task ids | Used in the paper as |
| --- | --- | --- | --- |
| proposed amplifier tasks | opamp | 1-15 | A1-A15 |
| AutoSizer amplifier tasks | opamp | 40001-40009 | S1-S9 |
| AutoSizer LDO task | ldo | 500001 | S10 |
| AutoSizer bandgap task | bandgap | 600001 | S11 |
| AnalogGym amplifier tasks | opamp | 10001-10016 | G1-G16 |
| proposed LDO tasks | ldo | 20001-20005 | L1-L5 |
| proposed bandgap tasks | bandgap | 30001-30005 | B1-B5 |

Each `tasks.json` maps a task id to its constraints, for example `{"Power": {"<=": 1000}, ...}`.
Keys that start with `_` give test conditions rather than constraints. For example,
`_context.Cload_pF` sets the load capacitance of the AnalogGym tasks.

## Objective

Every candidate is scored with

J(x) = -λ Σ_i max(0, ρ_i (s_i − y_i)) / (|s_i| + ε),  with λ = 4 and ε = 1e-12

where ρ_i = +1 for lower bounds (`>=`, `>`) and −1 for upper bounds (`<=`, `<`). ε has the unit
of s_i and only prevents division by zero. J = 0 only when every constraint holds, and margin
beyond a satisfied target is not rewarded. A missing or non-finite metric adds −10, and a
simulation that returns no result scores −1e9. A design passes only if every constraint holds
exactly, with no tolerance.

## Topology refinement

After a failed sizing, the Topology Refinement Agent writes the refined topology as a PySpice
`SubCircuitFactory` class, with the subcircuit name `<name>_refined` and the testbench ports.
The class is never executed. `pyspice_netlist.py` reads it with `ast`, accepts only
`self.X/R/C/V/I(...)` calls with literal arguments, and replays these calls on a PySpice
`SubCircuit`, which renders the ngspice netlist. Before the refined topology is sized,
`Circuit.check_netlist` checks:

- the structure: every port is connected, device terminal counts are right, only SKY130
  devices or subcircuits already used by the input topology appear, and no node is dangling;
- that the netlist has at least one quoted sizing variable;
- that ngspice loads the netlist with the SKY130 models when every variable is at the middle
  of its hard range (`simulation/netlist_check.spice`, no analysis is run).

If the conversion or a check fails, the errors are sent back to the LLM for a corrected class.
`REFINE_CHECK_ATTEMPTS` (3) bounds the number of answers per refinement; when none passes, the
iteration continues without a refined topology.

## Testbenches

All testbenches use the SKY130 tt corner and VDD = 1.8 V. The module docstrings in `circuits/`
give the full definitions.

- **Op-amp (`tb_acdc.spice`).**
  - Gain, GBW, and phase margin are taken from the open-loop response, with the loop broken by
    a 1 TH/1 TF network and the AC signal applied to the inverting input. GBW is the 0 dB
    crossing.
  - CMRR, PSRR+ (`PSRP`), and PSRR− (`PSRN`) are measured at 0.1 Hz in a unity-gain follower.
    PSRN is the rejection of the negative supply.
  - Power, TC, and offset come from a DC follower at VCM = 0.72 V.
  - The load capacitance is 1 pF unless the task sets `_context.Cload_pF`.
- **Op-amp (`tb_tran.spice`).** This runs only for tasks that constrain `sr`, `setting_time`,
  or `FOML`. A 0.3→0.5 V step drives a follower.
  - The slew rate is the mean of the rising and falling rates, each measured to the 0.4 V crossing.
  - The settling time is the larger of the two edges, using a 1 % band.
  - It is adapted from AnalogGym `TB_Amplifier_Tran.cir`. The falling-edge crossing is searched
    only after the falling input step.
- **LDO (`tb.spice`).** Vref = 1.2 V with unity feedback, a 10 µA–10 mA load, and CL = 100 pF.
  - Loop gain, GBW, phase margin, and PSRR are measured at 10 mA.
  - LNR and LR are relative: the peak-to-peak change of Vout divided by its mean, per volt or
    per ampere.
  - Power is measured at a 10 µA load and includes the load current.
  - Undershoot and overshoot are measured from 1.2 V for a 10 µA ↔ 10 mA step.
- **Bandgap (`tb.spice`).**
  - vref is taken at 25 °C.
  - TC uses the box method over −45…125 °C.
  - LS covers a VDD sweep of 1.62–1.98 V.
  - PSRR is measured at 10 Hz and 1 MHz; power at 25 °C.
  - `aux.spice` holds a current-reference output at 1 V and models behavioural amplifier blocks.

## Default settings

| Setting | Value |
| --- | --- |
| LLM | `gemini-2.5-flash`; the refinement knowledge query uses `gemini-2.5-flash-lite` |
| Embeddings | `text-embedding-3-large` |
| Design iterations | at most 3 |
| Candidates per selection | 5 |
| Sizing budget | 150 simulations per topology |
| Refinement answers | at most 3 PySpice classes per refinement (first answer + 2 repairs) |
| LLM-guided phase | first 40 simulations; the plan is refreshed every 20 simulations with 5 LLM samples |
| Seed | 42 for the optimizer and the memory order |

See `config.py`.
