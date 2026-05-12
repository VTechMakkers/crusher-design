# Engineering Memory — TechMakkers Crusher Design

Index of design rationale, decisions, and lessons. Each entry is a separate
file under `decisions/` or `lessons/`. Append-only. Compounds across model
generations — any future LLM (Opus, Mythos, beyond) reads this same memory.

## How to add
- `decisions/<short-slug>.md` — a design decision and why
- `lessons/<short-slug>.md` — something learned from field, failure, or QC

Each file starts with frontmatter:

```
---
date: YYYY-MM-DD
part: toggle_plate | swing_jaw | eccentric_shaft | ...
type: decision | lesson
status: active | superseded
---
```

## Active entries

(empty — first entry will appear here)
