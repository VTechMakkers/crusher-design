# Hardware-optimized architecture

Target rig: AMD Ryzen 5 7600X · MSI B650 · 32 GB DDR5 · 1 TB NVMe Gen4 · RTX 3090 (24 GB) · RTX 3080 (10 GB).

## Workload allocation

| Component | Lives on | Why |
|---|---|---|
| Qwen3-Coder-30B (AWQ-4bit) | **RTX 3090** | Main worker — code, FEA/DEM decks, param mutations, YAML edits. ~17 GB VRAM, room for KV cache. |
| Qwen3-7B (INT8) | **RTX 3080** | Fast routine — classify, summarize, embeddings, field extraction. ~8 GB VRAM. |
| LIGGGHTS DEM | **RTX 3090** (when LLM idle) OR **RTX 3080** | GPU-accelerated DEM via LAMMPS GPU package. ~100K–500K particles. |
| CalculiX FEA | **CPU** (Ryzen 5 7600X, 12 threads) | Linear-static + nonlinear via OpenMP. Add PETSc-GPU for ~2× speedup on linear solve. |
| gmsh meshing | **CPU** | Mesh generation; fast on Ryzen. |
| Surrogate model training/inference | **RTX 3090** (when not serving) | PyTorch MLP; train in seconds, predict in microseconds. |
| Opus 4.7 (cloud) | API | Orchestration, hard reasoning, cross-template insight, final review. |

## Token routing (the 70/30 split)

`mcp-servers/local_llm/router.py` decides per task:

| Task category | Backend |
|---|---|
| param_mutation, cadquery_edit, fea_deck_generate, dem_deck_generate, mesh_script, yaml_edit, result_parse | local_coder (3090) |
| classify, embedding, summarize_short, extract_field | local_fast (3080) |
| design_plan, cross_template_insight, conflict_resolve, final_review, novel_problem | cloud (Opus 4.7) |

Result: ~70% of token volume goes local. Estimated monthly Opus cost drops from ~$300–600 to ~$100–200 for equivalent design work.

## DEM pipeline (the competitive edge)

```
CadQuery (CPU) → STL → mcp-servers/dem/server.py
                                  │
                                  ▼
                       LIGGGHTS deck templated
                       (templates/<part>/dem_template.lammps)
                                  │
                                  ▼
                       liggghts run --gpu (RTX 3090)
                                  │
                                  ▼
                       extract_metrics → {TPH, wear_map, P80, energy}
                                  │
                                  ▼
                       loop/dem_fitness.py → DEM composite score
                                  │
                                  ▼
                       combined_score(structural, dem) → total fitness
```

For the swing jaw plate, DEM informs the **tooth pitch / depth / angle** — exactly the parameters that determine TPH, wear pattern, and product gradation. Apollo / Propel / Stalwart do NOT do this. Sandvik / Metso do on bigger rigs.

## Memory budget (32 GB → upgrade path)

- CalculiX 1M-element model: ~6 GB peak
- LIGGGHTS 200K particles + meshes: ~3 GB
- gmsh: ~2 GB working
- vLLM models in VRAM (not system RAM)
- OS + utilities: ~4 GB
- **Headroom on 32 GB**: tight for parallel FEA + DEM. Upgrade to 64 GB at first opportunity (board supports 192 GB total).

## Thermal / reliability notes

- Dual-GPU pulls 600–700 W under load. Confirm PSU + airflow.
- Sustained DEM + LLM loads can throttle. Watch nvidia-smi `Pwr/Cap`.
- LIGGGHTS prefers single high-bandwidth GPU over split workload — keep 3090 dedicated to DEM during sweeps; pause LLM serving if needed.

## Bootstrap

```bash
git clone <crusher-design> /opt/crusher-design
cd /opt/crusher-design
sudo bash infra/setup_local.sh
sudo systemctl start vllm-coder vllm-fast
python3 -m loop.design_loop                # smoke test
```

## Cost comparison

| Setup | Monthly cost (est) | TPH prediction accuracy |
|---|---|---|
| All-Opus on a laptop, no DEM | $300–600 | handbook ±30% |
| This rig + 70/30 routing + DEM | ~$110–210 ($100–200 Opus + $10 electricity) | DEM ±5–15% |

The rig pays for itself in 6–12 months on token savings alone — separate from the competitive edge DEM provides.
