# crusher-design

Open engineering foundation for jaw crusher design. Parametric templates,
multi-physics solvers (FEA, MBD, DEM), whole-machine multi-objective
optimisation, global sensitivity analysis, wear-aging lifecycle
simulation, and a failure-mode library — all model-agnostic so any
LLM (Opus, Mythos, GPT, Gemini) can drive the design loop through MCP.

Built around best-of-breed open projects rather than rolled-our-own
solvers. Our value is the integration layer, the crusher-domain
templates, and the trust-tiered knowledge base — not solver
re-implementation.

```
                  ┌────────────────────────────────────┐
                  │   LLM architect (via MCP servers)  │
                  └─────────────────┬──────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │   integration layer (this repo)             │
              │   assembly · loop · knowledge · memory     │
              └─────────────────────┬──────────────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
     CalculiX     gmsh        Project Chrono     LIGGGHTS     pymoo
     (FEA)     (mesh)         (MBD, optional)    (DEM)       (NSGA-III)
                                    │
                                    ▼
                          CadQuery (parametric CAD)
                                    │
                                    ▼
                       SALib (Sobol global sensitivity)
```

---

## What's inside

| Layer | Component | Purpose |
|---|---|---|
| **Catalog** | `catalog/{models,parts}.yaml` | Crusher SKUs and part classes |
| **Templates** | `templates/<part>/{geometry.py, instances/, load_cases.yaml}` | Parametric CAD + per-(part, model) validated configs |
| **Knowledge** | `knowledge/{materials, ores, manufacturing, standards}.yaml`<br>`knowledge/sources/` (multi-tier trust resolver) | Engineering data + provenance tagging |
| **MBD** | `mcp-servers/mbd/` | Closed-form 4-bar kinematics + dynamics + ISO 281 bearing life; optional Project Chrono transient runner |
| **FEA** | `mcp-servers/fea/` | gmsh mesh → CalculiX solve → FRD parse pipeline + NAFEMS benchmarks |
| **DEM** | `mcp-servers/dem/` | LIGGGHTS wrapper for granular flow + wear-contact mapping (scaffolded) |
| **DFM** | `mcp-servers/dfm/` | Casting/machining/welding rule engine |
| **Assembly** | `assembly/`, `loop/assembly_loop.py` | Whole-machine graph, force-path, system aggregation |
| **Optimisation** | `loop/pareto.py`, `loop/pareto_pymoo.py` | Pareto rank + NSGA-II + NSGA-III diversity selection |
| **Sensitivity** | `loop/sensitivity.py` | SALib-backed Sobol first-order + total-order indices |
| **Surrogate** | `loop/system_surrogate.py` | Permutation-invariant set network for system KPIs |
| **Wear** | `loop/wear_evolution.py` | Archard wear law + lifecycle integration |
| **Failure modes** | `knowledge/failure_modes.yaml` | 18 encoded crusher failure modes with citations |
| **Drivers** | `bin/{run_design,run_assembly,run_sensitivity,ingest_drawing,ingest_mill_cert,data_status}.py` | CLI entry points |

---

## Quick start

Requirements: Python 3.11+.

```bash
pip install -e .
python -m pytest tests/                          # 139 tests, < 30 s
python bin/data_status.py                        # show real-vs-placeholder coverage
python bin/run_sensitivity.py toggle_plate PE_400x600 --samples 256
python bin/run_assembly.py PE_400x600 --algorithm nsga3 --top-k 5
```

Heavier dependencies are **optional** — modules degrade gracefully:

| For | Install | Skipped tests run when present |
|---|---|---|
| Local CAD geometry (`templates/*.geometry.py build()`) | `pip install cadquery-ocp` | — |
| Real FEA (gmsh + CalculiX) | `pip install gmsh` + `brew install calculix-ccx` + `pip install calculix-frd-py` | NAFEMS benchmarks |
| Higher-fidelity MBD | `conda install -c projectchrono pychrono` | flexible-body cases |
| GPU surrogate training | `pip install torch` | surrogate train/predict |

---

## Test philosophy

Every test in this repo is gated by **analytical truth** or **physical
conservation laws**, not by agreement with prior code:

- Archard wear: `V = K·N·s/H` reproduced to `rel=1e-12`
- Sobol indices: Ishigami function against published analytical values
- FEA: cantilever tip deflection against Bernoulli–Euler `PL³/(3EI)`
- MBD: force balance ΣF = 0 and moment balance ΣM = 0
- ISO 281 bearing life: constant-load identity
- Pareto: dominance definition (a ≥ b on all, > b on at least one)

If a future change breaks the math, an analytical test fails immediately —
not after a regression in production.

---

## Data realness

Every numeric value in the repo today is one of:

- **Placeholder** (instance YAMLs labelled with `notes: PLACEHOLDER`)
- **Handbook** (MakeItFrom material properties, tagged `tier: 4` by the
  trust resolver)
- **Class-typical** (PE-series crusher specs, generic 4-bar mechanism
  geometry)
- **Empty** (`knowledge/sources/techmakkers_internal.yaml` mill-cert log
  is currently `[]`)

The architecture is real. The data is not. As soon as a real toggle-plate
drawing is ingested via `bin/ingest_drawing.py`, the multi-tier trust
resolver in `mcp-servers/knowledge/` automatically prefers it over the
handbook — no code change required.

Run `python bin/data_status.py` to see the current realness percentage.

---

## Open-source projects this builds on

| Project | License | Used for |
|---|---|---|
| [CalculiX](http://www.calculix.de/) | GPL | FEA solver |
| [gmsh](https://gmsh.info/) | GPL | mesh generation |
| [CadQuery](https://cadquery.readthedocs.io/) | Apache 2.0 | parametric CAD |
| [Project Chrono](https://projectchrono.org/) | BSD-3 | optional higher-fidelity MBD |
| [LIGGGHTS](https://www.cfdem.com/liggghts) | GPL | DEM granular simulation |
| [pymoo](https://pymoo.org/) | Apache 2.0 | NSGA-II / NSGA-III |
| [SALib](https://salib.readthedocs.io/) | Apache 2.0 | global sensitivity (Sobol) |
| [PyTorch](https://pytorch.org/) (optional) | BSD-3 | surrogate model |
| [Anthropic MCP](https://modelcontextprotocol.io/) | MIT | LLM tool transport |

Pinning to maintained, peer-reviewed implementations rather than
rolling our own is intentional. We hold the integration layer to the
same standard via analytical-truth tests.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Project status

Active. Architecture-first phase: build the system, then ingest real
TechMakkers data. Run `python bin/data_status.py` at any time to see
how much of the system is grounded in measured data vs. placeholders.
