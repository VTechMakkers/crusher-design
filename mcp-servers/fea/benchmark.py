"""
NAFEMS-style FEA pipeline validation.

Each benchmark has:
  - an analytical (closed-form) reference solution
  - a small primitive geometry built directly in gmsh (no STEP needed)
  - a CalculiX deck assembled by `calculix_deck.build_deck`
  - tolerance bands stating how close FEA must match analytical

Pass criteria deliberately allow for known FEA error sources: stress
concentration at clamped supports (use long thin geometries to minimize),
shear deflection at low L/h (use L/h ≥ 8), coarse-mesh under-prediction
of peak stress (use second-order elements).

Two benchmarks today:
  1. Uniaxial tension bar — σ and δ both have clean closed-form,
     no boundary-singularity stress concentration.
  2. Cantilever bending — Bernoulli-Euler tip deflection. The fixed-end
     stress is a singularity in FEA, so we check tip deflection only.

These are intentionally simple. A wrong solver setup will fail them by
margins much larger than the listed tolerance.
"""
from __future__ import annotations
import math
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calculix_deck import Boundary, Material, PointLoad, build_deck
from .gmsh_mesher import mesh_box
from .frd_parser import parse as parse_frd


STEEL_NOMINAL = Material(
    name="STEEL_NOMINAL",
    youngs_modulus_MPa=210_000.0,
    poisson_ratio=0.30,
    density_kg_m3=7850.0,
)


@dataclass
class BenchmarkResult:
    name: str
    analytical: dict[str, float]
    fea: dict[str, float]
    relative_error: dict[str, float]
    passed: bool
    tolerance: dict[str, float]
    notes: str = ""


def _run_ccx(deck_text: str, work_dir: Path, deck_stem: str = "job") -> Path:
    """Write deck, run CalculiX, return path to FRD."""
    if not shutil.which("ccx"):
        raise ImportError(
            "CalculiX `ccx` binary not on PATH; install via "
            "`brew install calculix-ccx` or `apt install calculix-ccx`"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / f"{deck_stem}.inp"
    deck_path.write_text(deck_text)
    subprocess.run(["ccx", deck_stem], cwd=work_dir,
                   check=True, capture_output=True)
    return work_dir / f"{deck_stem}.frd"


def tension_bar(*, work_dir: Path,
                 length_mm: float = 100.0,
                 cross_section_mm: float = 20.0,
                 axial_load_N: float = 80_000.0,
                 material: Material = STEEL_NOMINAL,
                 mesh_size_mm: float = 4.0,
                 sigma_tol: float = 0.02,
                 delta_tol: float = 0.03) -> BenchmarkResult:
    """Uniaxial tension on a square bar.

    Analytical:
      σ = P / A
      δ = P L / (A E)

    No stress concentration; clamped face is a uniform displacement
    constraint, not a singularity. FEA should match to ~1% with a
    reasonable mesh.
    """
    A = cross_section_mm * cross_section_mm
    sigma_analytical = axial_load_N / A
    delta_analytical = axial_load_N * length_mm / (A * material.youngs_modulus_MPa)

    mesh_inp = work_dir / "tension.inp"
    mesh = mesh_box(length_mm=length_mm,
                     width_mm=cross_section_mm,
                     height_mm=cross_section_mm,
                     fixed_face="x_min", load_face="x_max",
                     out_inp=mesh_inp, mesh_size_mm=mesh_size_mm)
    # Distribute the total load equally across nodes on the load face
    load_nodes = mesh.physical_groups.get("x_max", [])
    if not load_nodes:
        raise RuntimeError("x_max physical group has no nodes — meshing failed")
    per_node = axial_load_N / len(load_nodes)
    deck = build_deck(
        mesh_inp_filename=mesh_inp.name,
        material=material,
        boundaries=[Boundary("x_min", 1), Boundary("x_min", 2), Boundary("x_min", 3)],
        point_loads=[PointLoad("x_max", 1, per_node)],
    )
    frd = _run_ccx(deck, work_dir, deck_stem="tension")
    fea = parse_frd(frd)
    if fea.get("_not_implemented"):
        raise RuntimeError(f"FRD parse failed: {fea}")

    sigma_fea = fea["max_von_mises_MPa"]
    delta_fea = fea["max_displacement_mm"]
    sigma_err = abs(sigma_fea - sigma_analytical) / sigma_analytical
    delta_err = abs(delta_fea - delta_analytical) / delta_analytical
    return BenchmarkResult(
        name="tension_bar",
        analytical={"sigma_MPa": sigma_analytical,
                     "delta_mm": delta_analytical},
        fea={"sigma_MPa": sigma_fea, "delta_mm": delta_fea},
        relative_error={"sigma": sigma_err, "delta": delta_err},
        tolerance={"sigma": sigma_tol, "delta": delta_tol},
        passed=(sigma_err < sigma_tol and delta_err < delta_tol),
    )


def cantilever_bending(*, work_dir: Path,
                        length_mm: float = 200.0,
                        cross_section_mm: float = 20.0,
                        tip_load_N: float = 1_000.0,
                        material: Material = STEEL_NOMINAL,
                        mesh_size_mm: float = 4.0,
                        delta_tol: float = 0.05) -> BenchmarkResult:
    """Cantilever beam with tip load applied transverse to the long axis.

    Analytical (Bernoulli-Euler):
      I = w h^3 / 12
      δ_tip = P L^3 / (3 E I)

    Length/height ratio must be ≥ 8 for Bernoulli-Euler to be accurate to
    a few percent; the default L=200, h=20 gives L/h = 10. Stress at the
    fixed end is a singularity in any continuum FEA so we don't check it
    in this benchmark — use the tension test for stress accuracy.
    """
    h = cross_section_mm
    w = cross_section_mm
    I = w * (h ** 3) / 12.0
    if length_mm / h < 8.0:
        raise ValueError(
            f"L/h = {length_mm/h:.1f} < 8; Bernoulli-Euler is not accurate "
            "enough for a strict comparison. Increase length_mm."
        )
    delta_analytical = (tip_load_N * length_mm ** 3) \
                        / (3.0 * material.youngs_modulus_MPa * I)

    mesh_inp = work_dir / "cantilever.inp"
    mesh = mesh_box(length_mm=length_mm,
                     width_mm=cross_section_mm,
                     height_mm=cross_section_mm,
                     fixed_face="x_min", load_face="x_max",
                     out_inp=mesh_inp, mesh_size_mm=mesh_size_mm)
    load_nodes = mesh.physical_groups.get("x_max", [])
    if not load_nodes:
        raise RuntimeError("x_max physical group has no nodes — meshing failed")
    per_node = tip_load_N / len(load_nodes)
    deck = build_deck(
        mesh_inp_filename=mesh_inp.name,
        material=material,
        # Fully clamp the x=0 end (all 3 translational DOFs)
        boundaries=[Boundary("x_min", 1), Boundary("x_min", 2), Boundary("x_min", 3)],
        # Tip load in -z (lateral)
        point_loads=[PointLoad("x_max", 3, -per_node)],
    )
    frd = _run_ccx(deck, work_dir, deck_stem="cantilever")
    fea = parse_frd(frd)
    if fea.get("_not_implemented"):
        raise RuntimeError(f"FRD parse failed: {fea}")
    delta_fea = fea["max_displacement_mm"]
    delta_err = abs(delta_fea - delta_analytical) / delta_analytical
    return BenchmarkResult(
        name="cantilever_bending",
        analytical={"delta_tip_mm": delta_analytical},
        fea={"delta_tip_mm": delta_fea,
              "sigma_max_MPa": fea["max_von_mises_MPa"]},
        relative_error={"delta_tip": delta_err},
        tolerance={"delta_tip": delta_tol},
        passed=(delta_err < delta_tol),
        notes="fixed-end stress is a continuum-FEA singularity; "
              "only tip deflection is gated. Use tension_bar for stress accuracy.",
    )
