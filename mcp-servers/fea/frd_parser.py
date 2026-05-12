"""
CalculiX .frd result parser.

CalculiX writes nodal + elemental results in an ASCII .frd file. The
format has a fixed-column header followed by repeating result blocks
identified by 4-byte type codes. Hand-rolling a correct parser is fragile;
we delegate to `calculix-frd-py` if installed, otherwise return a clear
`_not_implemented` marker so callers know to install it.

This module exists to keep the FEA server thin: the server orchestrates,
this module handles the format.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


def parse(frd_path: str | Path) -> dict[str, Any]:
    """Parse a CalculiX .frd file. Returns one of:

      {"max_von_mises_MPa": float, "max_displacement_mm": float, "ok": True}
      {"_not_implemented": True, "reason": "...", "frd_path": "..."}
    """
    p = Path(frd_path).resolve()
    if not p.exists():
        return {"_not_implemented": True,
                "reason": f"frd file not found: {p}"}
    try:
        from calculix_frd_py import read_frd  # type: ignore[import-not-found]
    except ImportError:
        return {"_not_implemented": True,
                "reason": "install calculix-frd-py to parse FRD results",
                "frd_path": str(p)}
    result = read_frd(str(p))
    stress_block = result.get_field("STRESS") if hasattr(result, "get_field") else None
    disp_block = result.get_field("DISP") if hasattr(result, "get_field") else None
    if stress_block is None:
        return {"_not_implemented": True,
                "reason": "FRD has no STRESS block — verify solver completed"}
    # Compute per-node von Mises from stress tensor components
    max_vm = _max_von_mises(stress_block)
    max_disp = _max_displacement_magnitude(disp_block) if disp_block else 0.0
    return {"max_von_mises_MPa": float(max_vm),
            "max_displacement_mm": float(max_disp),
            "ok": True}


def _max_von_mises(stress_block) -> float:
    """Compute max nodal von Mises from a (Nx6) stress tensor field.

    σ_vm = sqrt( ((σx-σy)^2 + (σy-σz)^2 + (σz-σx)^2 + 6*(τxy^2+τyz^2+τzx^2)) / 2 )

    Different versions of calculix-frd-py expose the data differently. We
    look for the conventional fields; if absent, raise so the caller knows
    to inspect the FRD manually rather than getting a wrong number.
    """
    import math
    rows = getattr(stress_block, "data", None) or stress_block
    best = 0.0
    for row in rows:
        # Expected component order: SXX, SYY, SZZ, SXY, SYZ, SZX
        try:
            sxx, syy, szz, sxy, syz, szx = row[:6]
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"unexpected FRD stress row shape {row!r} — "
                f"calculix-frd-py API may have changed"
            ) from e
        vm = math.sqrt(
            ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
             + 6.0 * (sxy * sxy + syz * syz + szx * szx)) / 2.0
        )
        if vm > best:
            best = vm
    return best


def _max_displacement_magnitude(disp_block) -> float:
    """Max |u| over all nodes; expects (Nx3) DISP rows."""
    import math
    rows = getattr(disp_block, "data", None) or disp_block
    best = 0.0
    for row in rows:
        try:
            ux, uy, uz = row[:3]
        except (TypeError, ValueError):
            continue
        m = math.sqrt(ux * ux + uy * uy + uz * uz)
        if m > best:
            best = m
    return best
