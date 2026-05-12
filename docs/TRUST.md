# Trust model — how the architecture chooses what's correct

When two sources disagree on a property (e.g. Mn13 yield strength), the
architecture must pick one. This document explains the rules.

## Six tiers (lower = higher trust)

| Tier | Source class | Example |
|------|--------------|---------|
| 1 | techmakkers internal measured | mill cert, fab measurement, field telemetry |
| 2 | techmakkers internal validated | FEA + engineer sign-off, decision memory |
| 3 | external authoritative | NIST SRD, Materials Project, ISO/BIS, peer-reviewed |
| 4 | external handbook | MakeItFrom, manufacturer datasheets, arXiv |
| 5 | external industry | 911Metallurgist Wi tables, trade journals |
| 6 | external unverified | GrabCAD geometry, Wikipedia, forums |

Source declarations live in `knowledge/sources.yaml`.
Per-source data files live in `knowledge/sources/<source_id>.yaml`.

## Resolver rules

1. Collect every candidate value for the (material, property) pair across all sources.
2. Apply task-specific elevation from `sources.yaml -> task_overrides`. Example: for `wear_life` calculations, `techmakkers_field_telemetry` is elevated above its base tier.
3. Sort by (effective tier, recency).
4. Return chosen value + ALL alternatives + the reason.
5. If top two candidates disagree by >15%, attach `conflict_warning` — the LLM must surface this to the engineer.

## Task overrides currently configured

| Task | Elevated sources | Why |
|------|------------------|-----|
| `wear_life` | field telemetry, fab measurement | YOUR materials in YOUR conditions beat handbook |
| `fatigue` | FEA validated, peer-reviewed | Geometry+load specific; generic S-N curves are weak |
| `novel_alloy` | Materials Project, NIST, papers | No internal data exists yet |
| `bond_work_index` | papers, Bond 1961, 911Metallurgist | Prefer modern measurement, fall back to canonical |

## Why this works

- The LLM never silently picks a wrong number — it sees all candidates.
- The hierarchy is auditable: every choice has a `source_id` and `reason`.
- Adding new data raises trust: each measurement appended to `techmakkers_internal.yaml` automatically wins over external handbook values for that property/material.
- The system **improves over time** as you measure more — without retraining anything. The data does the work, not the model.

## How to update

- New mill cert: append to `knowledge/sources/techmakkers_internal.yaml` under `mill_certs:`. Done.
- New field observation: append under `field_observations:`. Done.
- New external source: declare in `sources.yaml`, add per-source YAML file under `knowledge/sources/`. Done.
- Material now in production but field data disagrees with handbook: the resolver automatically prefers your data. No code change needed.

## What this does NOT replace

- An engineer of record signing off on a design. The system surfaces information; the human decides what ships.
- Standards compliance certification. The system organizes; certified labs measure.
- Engineering judgement on edge cases. Conflict warnings flag for review; do not auto-resolve safety-critical mismatches.
