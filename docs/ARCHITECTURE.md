# Architecture

## Data model: part × model

```
catalog/
  models.yaml      crusher model registry (PE_250x400, PE_400x600, ...)
  parts.yaml       part class registry (toggle_plate, swing_jaw, eccentric_shaft, ...)

templates/<part>/
  geometry.py            ONE parametric script per part class (the IP)
  metadata.yaml          part class info
  load_cases.yaml        default load case templates
  instances/
    <model>.yaml         validated params for this part on this crusher model
    <model>.history.jsonl     append-only design history per (part, model)

knowledge/         materials, manufacturing rules, standards (model-agnostic)
memory/            engineering rationale, decisions, lessons (compounds)
mcp-servers/       geometry, fea, knowledge — tool layer for any LLM
loop/              orchestrator + fitness
bin/               CLI helpers (scaffold_part, scaffold_instance)
```

## How a design run works

```
(part, model) baseline params  ──┐
                                 ├─► loop/design_loop.sweep()
LLM-proposed param mutations  ───┘         │
                                           ▼
                              for each variant:
                                geometry MCP  → STEP
                                fea MCP       → mesh + solve + metrics
                                knowledge MCP → material props + safety factor
                                fitness.py    → composite score
                                           │
                                           ▼
                            append to <model>.history.jsonl
                            return ranked top-K
```

## CLI

```bash
# Add a new part class
python bin/scaffold_part.py eccentric_shaft \
    --criticality safety_critical --material forged_steel

# Add a new (part, model) instance — optionally copy from existing
python bin/scaffold_instance.py toggle_plate PE_500x750 --from PE_400x600

# Dry-run sweep on a (part, model)
python -m loop.design_loop
```

## Why model-agnostic (Mythos-ready)

Same templates, same MCP tools, same fitness, same memory. Swap the LLM
that drives the loop — Opus 4.7 today, Mythos tomorrow — and you get:
- 2M context: all part-model instances loaded simultaneously → cross-model knowledge transfer (toggle plate insight from PE_400x600 informs PE_600x900 design automatically)
- ULTRAPLAN: week-long evolutionary sweeps across the full catalog
- KAIROS: continuous monitoring of `<model>.history.jsonl` for drift/regressions

No rewrite. Architecture amplifies with model capability.

## Compounding (the moat)

| Asset | Where it lives | Grows when |
|-------|----------------|------------|
| Part class library | `templates/<part>/geometry.py` | New part class scaffolded |
| Model instances | `templates/<part>/instances/<model>.yaml` | New crusher model added or existing one re-validated |
| Validation history | `<model>.history.jsonl` | Every design run appended |
| Engineering memory | `memory/decisions/`, `memory/lessons/` | Every design decision or field lesson logged |
| Field telemetry | (future — separate ingest pipeline) | Every deployed unit reports |

By Month 12, catalog × instances × history files = a training-grade
dataset no software-only competitor can replicate.
