"""
CalculiX input deck builder.

Produces production-grade `.inp` text from structured inputs. Kept as pure
text manipulation so the build is deterministic, testable without any
solver installed, and easy to inspect when something goes wrong.

A CalculiX deck for a static structural problem has, in order:

  *INCLUDE the gmsh-generated mesh INP (defines nodes + elements + node
            sets named after the gmsh Physical Groups)
  *MATERIAL  + *ELASTIC + *DENSITY            (material properties)
  *SOLID SECTION                              (assign material to element set)
  *STEP                                       (begin a step)
    *STATIC
    *BOUNDARY                                 (clamp specified node sets)
    *CLOAD                                    (apply loads to specified sets)
    *NODE FILE, *EL FILE                      (request output)
  *END STEP

This file knows nothing about gmsh; the mesh INP is assumed to define the
node sets referenced in BCs and loads.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Linear-elastic isotropic material in SI-derived units used by
    CalculiX (mm, tonne, s, N, MPa)."""
    name: str
    youngs_modulus_MPa: float
    poisson_ratio: float
    density_kg_m3: float

    @property
    def density_tonne_per_mm3(self) -> float:
        # CalculiX consistent units: when length=mm and force=N, density
        # must be in tonne/mm^3. 1 kg/m^3 = 1e-12 tonne/mm^3.
        return self.density_kg_m3 * 1.0e-12

    def validate(self) -> None:
        assert self.youngs_modulus_MPa > 0
        assert 0.0 <= self.poisson_ratio < 0.5
        assert self.density_kg_m3 > 0


@dataclass(frozen=True)
class Boundary:
    """Clamp a degree of freedom on a node set.
       dof: 1=x, 2=y, 3=z (translational only for linear static)."""
    nset_name: str
    dof: int
    value: float = 0.0

    def validate(self) -> None:
        assert self.dof in (1, 2, 3), f"unsupported dof {self.dof}"


@dataclass(frozen=True)
class PointLoad:
    """Apply a force component (N) to every node in a node set."""
    nset_name: str
    dof: int
    force_N: float

    def validate(self) -> None:
        assert self.dof in (1, 2, 3)


def build_deck(*,
                mesh_inp_filename: str,
                material: Material,
                element_set_name: str = "Eall",
                boundaries: list[Boundary],
                point_loads: list[PointLoad],
                step_name: str = "static") -> str:
    """Return the full CalculiX deck text.

    `mesh_inp_filename` must be a path relative to the working directory
    where ccx will be invoked. CalculiX's *INCLUDE doesn't follow absolute
    paths reliably on all platforms; relative is the safe convention.
    """
    material.validate()
    for b in boundaries:
        b.validate()
    for p in point_loads:
        p.validate()

    lines: list[str] = [
        f"*INCLUDE, INPUT={mesh_inp_filename}",
        "",
        f"*MATERIAL, NAME={material.name}",
        "*ELASTIC",
        f"{material.youngs_modulus_MPa},{material.poisson_ratio}",
        "*DENSITY",
        f"{material.density_tonne_per_mm3:.6e}",
        "",
        f"*SOLID SECTION, ELSET={element_set_name}, MATERIAL={material.name}",
        "",
        f"*STEP, NAME={step_name}",
        "*STATIC",
    ]

    if boundaries:
        lines.append("*BOUNDARY")
        for b in boundaries:
            lines.append(f"{b.nset_name},{b.dof},{b.dof},{b.value}")

    if point_loads:
        lines.append("*CLOAD")
        for p in point_loads:
            lines.append(f"{p.nset_name},{p.dof},{p.force_N}")

    lines.extend([
        "*NODE FILE",
        "U",
        "*EL FILE",
        "S",
        "*NODE PRINT, NSET=Nall",
        "U",
        "*EL PRINT, ELSET=Eall",
        "S",
        "*END STEP",
        "",
    ])
    return "\n".join(lines)
