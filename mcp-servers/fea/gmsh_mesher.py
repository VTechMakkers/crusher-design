"""
gmsh meshing wrapper.

Two entry points:
  - mesh_step():        STEP file -> CalculiX INP (used by production parts)
  - mesh_box():         build + mesh a primitive box directly in gmsh
                        (used by NAFEMS benchmarks; no CAD library needed)

Both produce an INP that defines named node sets corresponding to gmsh
Physical Groups, which CalculiX can reference in *BOUNDARY and *CLOAD
cards by name.

Requires gmsh (CLI or Python API). When neither is available the
functions raise ImportError with a clear install hint rather than
producing silent stub output.
"""
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _require_gmsh_cli() -> str:
    path = shutil.which("gmsh")
    if not path:
        raise ImportError(
            "gmsh not found on PATH; install via `brew install gmsh` "
            "or `apt install gmsh`"
        )
    return path


def _try_gmsh_python():
    try:
        import gmsh  # type: ignore[import-not-found]
        return gmsh
    except ImportError:
        return None


@dataclass
class MeshResult:
    inp_path: Path
    n_nodes: int
    n_elements: int
    physical_groups: dict[str, list[int]]


def mesh_step(step_path: Path, *,
               out_inp: Path | None = None,
               mesh_size_mm: float = 8.0,
               second_order: bool = True) -> MeshResult:
    """Mesh a STEP file with 3D tets, write CalculiX INP.

    Uses the gmsh CLI directly; reads any Physical Groups defined in the
    STEP via OpenCASCADE attributes. For simple shapes those won't exist,
    so production part templates should declare them in a companion
    geometry.py helper (a future task).
    """
    _require_gmsh_cli()
    step = Path(step_path).resolve()
    out = (out_inp or step.with_suffix(".inp")).resolve()
    order = 2 if second_order else 1
    geo = out.with_suffix(".geo")
    geo.write_text(
        f'Merge "{step}";\n'
        f'Mesh.ElementOrder = {order};\n'
        f'Mesh.CharacteristicLengthMin = {mesh_size_mm * 0.5};\n'
        f'Mesh.CharacteristicLengthMax = {mesh_size_mm};\n'
        f'Mesh 3;\n'
        f'Save "{out}";\n'
    )
    subprocess.run(
        ["gmsh", str(geo), "-3", "-format", "inp", "-o", str(out)],
        check=True, capture_output=True,
    )
    return _read_inp_summary(out)


def mesh_box(*, length_mm: float, width_mm: float, height_mm: float,
              fixed_face: str = "x_min",
              load_face: str = "x_max",
              out_inp: Path,
              mesh_size_mm: float = 4.0,
              second_order: bool = True) -> MeshResult:
    """Build + mesh a rectangular box directly with gmsh, with Physical
    Surfaces tagged for the named BC and load faces.

    Faces map to box coordinates:
      x_min: face at x=0          x_max: face at x=length_mm
      y_min: y=0                  y_max: y=width_mm
      z_min: z=0                  z_max: z=height_mm

    Used by NAFEMS-style benchmarks where the geometry is a clean
    primitive and there is no STEP file involved.
    """
    gmsh_py = _try_gmsh_python()
    if gmsh_py is None:
        raise ImportError(
            "gmsh Python API not available; `pip install gmsh` on the "
            "machine that will run the benchmarks (CLI alone cannot tag "
            "named Physical Surfaces from a script easily)"
        )
    gmsh_py.initialize()
    try:
        gmsh_py.option.setNumber("General.Terminal", 0)
        box = gmsh_py.model.occ.addBox(0, 0, 0, length_mm, width_mm, height_mm)
        gmsh_py.model.occ.synchronize()

        # Identify the 6 faces by centroid location
        faces = gmsh_py.model.getBoundary([(3, box)], oriented=False)
        face_tags: dict[str, int] = {}
        eps = min(length_mm, width_mm, height_mm) * 0.01
        for (_, tag) in faces:
            com = gmsh_py.model.occ.getCenterOfMass(2, tag)
            if abs(com[0] - 0) < eps:
                face_tags["x_min"] = tag
            elif abs(com[0] - length_mm) < eps:
                face_tags["x_max"] = tag
            elif abs(com[1] - 0) < eps:
                face_tags["y_min"] = tag
            elif abs(com[1] - width_mm) < eps:
                face_tags["y_max"] = tag
            elif abs(com[2] - 0) < eps:
                face_tags["z_min"] = tag
            elif abs(com[2] - height_mm) < eps:
                face_tags["z_max"] = tag

        # Tag the requested faces as Physical Surfaces with stable names
        if fixed_face not in face_tags:
            raise ValueError(f"fixed_face {fixed_face!r} not on box")
        if load_face not in face_tags:
            raise ValueError(f"load_face {load_face!r} not on box")
        gmsh_py.model.addPhysicalGroup(2, [face_tags[fixed_face]], name=fixed_face)
        gmsh_py.model.addPhysicalGroup(2, [face_tags[load_face]], name=load_face)
        # Tag the volume so element set Eall maps correctly
        gmsh_py.model.addPhysicalGroup(3, [box], name="solid")

        gmsh_py.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_mm * 0.5)
        gmsh_py.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
        gmsh_py.option.setNumber("Mesh.ElementOrder", 2 if second_order else 1)
        gmsh_py.model.mesh.generate(3)
        out = Path(out_inp).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        gmsh_py.write(str(out))
    finally:
        gmsh_py.finalize()
    return _read_inp_summary(out)


def _read_inp_summary(inp_path: Path) -> MeshResult:
    """Crude parse of the INP to count nodes/elements and discover any
    NSET names (gmsh emits one NSET per Physical Surface)."""
    text = inp_path.read_text(errors="ignore")
    n_nodes = 0
    n_elements = 0
    nsets: dict[str, list[int]] = {}
    in_nodes = False
    in_elements = False
    in_nset: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("*NODE"):
            in_nodes = True; in_elements = False; in_nset = None; continue
        if s.startswith("*ELEMENT"):
            in_nodes = False; in_elements = True; in_nset = None; continue
        if s.startswith("*NSET"):
            # *NSET,NSET=name
            name = ""
            for tok in s.split(","):
                tok = tok.strip()
                if tok.upper().startswith("NSET="):
                    name = tok.split("=", 1)[1]
            in_nodes = False; in_elements = False; in_nset = name
            nsets.setdefault(name, [])
            continue
        if s.startswith("*"):
            in_nodes = False; in_elements = False; in_nset = None; continue
        if not s:
            continue
        if in_nodes:
            n_nodes += 1
        elif in_elements:
            n_elements += 1
        elif in_nset is not None:
            for tok in s.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    nsets[in_nset].append(int(tok))
    return MeshResult(inp_path=inp_path, n_nodes=n_nodes,
                       n_elements=n_elements, physical_groups=nsets)
