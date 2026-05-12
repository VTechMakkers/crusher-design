"""
Global sensitivity analysis on the per-part design loop.

Computes Sobol indices via SALib's Saltelli sampling. Two values per
parameter are reported:

  first_order (S_i):   fraction of the KPI's variance attributable to
                       parameter i alone, ignoring interactions. Σ S_i ≤ 1.
  total_order (S_T_i): fraction of variance from i including ALL its
                       interactions. Σ S_T_i ≥ 1, with equality iff the
                       model is additive (no parameter interactions).

`interaction_strength_i = S_T_i - S_i` measures how much parameter i's
effect on the KPI depends on the values of other parameters.

What this gives the engineer that local one-at-a-time sensitivity does
not:
  - Global (over the full design space) not local (around one baseline)
  - Variance-decomposition, not slope. Tells you "param X explains 47%
    of the KPI's spread" — a quantitative ranking, not a heuristic.
  - Interaction detection. If two parameters interact (e.g. tooth_pitch
    × tooth_depth), local methods miss it; Sobol shows it as
    S_T - S > 0.

Implementation: SALib (Apache 2.0) is the de facto Python standard for
Saltelli + Sobol. Used widely in aerospace and climate-science global
sensitivity work. We wrap it with crusher-domain plumbing: bound
inference, KPI extraction, invalid-sample handling, ranked output.

Performance: Saltelli's method uses N(2k+2) function evaluations for
first-order + total-order indices on k parameters with N base samples.
For k=8 and N=1024, that's 18 432 evals. With the dry-run physics in
`design_loop.evaluate` each eval is sub-millisecond; full analysis runs
in ~15 s. With real FEA + DEM each eval would take minutes — that's
when the system_surrogate (already built) becomes essential.
"""
from __future__ import annotations
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop import design_loop


# ---------------------------------------------------------------------------
# Result dataclass

@dataclass(frozen=True)
class SensitivityResult:
    """Ranked variance decomposition of a single KPI across parameters."""
    target_kpi: str
    parameter_names: tuple[str, ...]
    first_order: dict[str, float]
    first_order_ci: dict[str, float]
    total_order: dict[str, float]
    total_order_ci: dict[str, float]
    n_samples_base: int
    n_evaluations: int
    n_invalid_samples: int
    bounds: dict[str, tuple[float, float]]

    def ranked_first_order(self) -> list[tuple[str, float]]:
        return sorted(self.first_order.items(), key=lambda x: -x[1])

    def ranked_total_order(self) -> list[tuple[str, float]]:
        return sorted(self.total_order.items(), key=lambda x: -x[1])

    def interaction_strength(self) -> dict[str, float]:
        """S_T_i - S_i.  Large = parameter i has strong interaction effects."""
        return {n: max(0.0, self.total_order[n] - self.first_order[n])
                for n in self.parameter_names}

    def summary_lines(self) -> list[str]:
        """Human-readable per-parameter ranking for printing or PR notes."""
        lines = [f"Sobol sensitivity of {self.target_kpi}",
                 f"  base samples: {self.n_samples_base}, "
                 f"total evaluations: {self.n_evaluations}, "
                 f"invalid: {self.n_invalid_samples}",
                 "", "  param          S1 (first)  S1±95%CI    ST (total)  interaction"]
        for n in sorted(self.parameter_names, key=lambda x: -self.total_order[x]):
            s1 = self.first_order[n]
            s1_ci = self.first_order_ci[n]
            st = self.total_order[n]
            inter = self.interaction_strength()[n]
            lines.append(f"  {n:14s}  {s1:+.3f}      ±{s1_ci:.3f}       "
                          f"{st:+.3f}       {inter:.3f}")
        return lines


# ---------------------------------------------------------------------------
# Bounds inference

def perturbation_bounds_from_baseline(baseline_params: dict[str, Any],
                                       fraction: float = 0.20
                                       ) -> dict[str, tuple[float, float]]:
    """Build symmetric ± fraction bounds around the baseline.

    Useful when the engineer wants to explore the design space *near* a
    validated configuration without re-deriving validate() bounds. The
    20% default is a sensible "neighbourhood" — large enough to expose
    real sensitivity, small enough that DFM + validate() failures stay
    rare."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in baseline_params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        v = float(value)
        if v == 0:
            continue
        spread = abs(v) * fraction
        bounds[name] = (v - spread, v + spread)
    return bounds


# ---------------------------------------------------------------------------
# Evaluation wrapper

def _extract_kpi(record: dict[str, Any], target_kpi: str) -> float:
    """Pull a single scalar KPI value from a design_loop.evaluate record.
    Raises KeyError with the available keys if `target_kpi` is unknown —
    intentionally not caught by the per-sample evaluator so a mistyped KPI
    surfaces as a hard error on the very first sample rather than silently
    invalidating the whole analysis."""
    metrics = record.get("metrics", {})
    fitness = record.get("fitness", {})
    if target_kpi in metrics:
        return float(metrics[target_kpi])
    if target_kpi in fitness:
        return float(fitness[target_kpi])
    raise KeyError(
        f"target_kpi {target_kpi!r} not in record. "
        f"available metrics: {sorted(metrics)}; "
        f"available fitness keys: {sorted(fitness)}"
    )


def _evaluate_for_kpi(*, part: str, model: str,
                      params: dict[str, Any],
                      target_kpi: str,
                      baseline_material: str) -> float | None:
    """Run design_loop.evaluate on a single sample, extract one scalar KPI.
    Returns None if the sample fails validation (params out of part's
    geometric bounds). KPI-extraction errors are NOT caught — they
    propagate so a mistyped target_kpi fails loudly."""
    try:
        record = design_loop.evaluate(
            part=part, model=model, params=params,
            material=baseline_material, runner=None,
        )
    except (AssertionError, ValueError):
        return None
    return _extract_kpi(record, target_kpi)


# ---------------------------------------------------------------------------
# Main analysis driver

def analyze_part(*,
                  part: str,
                  model: str,
                  target_kpi: str = "composite",
                  parameter_bounds: dict[str, tuple[float, float]] | None = None,
                  n_samples_base: int = 1024,
                  seed: int = 0,
                  invalid_sentinel: float | None = None,
                  ) -> SensitivityResult:
    """Compute Sobol first-order and total-order indices for one (part, model).

    Parameters
    ----------
    part, model         : (part class, crusher model) — must have an instance YAML
    target_kpi          : which KPI to decompose. One of
                            'composite' (fitness composite, default)
                            'mass_score', 'safety_factor_score', 'wear_score'
                            'manufacturability_score', 'safety_factor_value'
                            'mass_kg', 'max_von_mises_MPa', 'max_displacement_mm'
    parameter_bounds    : {param_name: (lo, hi)} for each parameter to sweep.
                          If None, defaults to ±20 % around the instance baseline.
    n_samples_base      : N in Saltelli's N(2k+2) total evaluation count
    seed                : RNG seed for reproducible Saltelli sequences
    invalid_sentinel    : value to substitute when a sample fails
                          DFM/validation. None → samples are filtered out
                          (and the count reported in n_invalid_samples).
    """
    try:
        from SALib.sample import sobol as salib_sobol_sample
        from SALib.analyze import sobol as salib_sobol_analyze
    except ImportError as e:
        raise ImportError(
            "SALib not installed; `pip install SALib` (Apache 2.0)"
        ) from e

    import numpy as np

    instance = design_loop.load_instance(part, model)
    baseline = dict(instance["params"])
    material = instance["material"]

    if parameter_bounds is None:
        parameter_bounds = perturbation_bounds_from_baseline(baseline)
    if not parameter_bounds:
        raise ValueError(
            "no numeric parameters found in baseline; "
            "supply parameter_bounds explicitly"
        )

    # Validate target_kpi against the baseline record once, upfront. A
    # typo here (e.g. "composit") would otherwise look like every sample
    # failing validation and produce silent meaningless output.
    baseline_record = design_loop.evaluate(
        part=part, model=model, params=baseline,
        material=material, runner=None,
    )
    _extract_kpi(baseline_record, target_kpi)   # raises with available keys

    names = sorted(parameter_bounds)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [list(parameter_bounds[n]) for n in names],
    }

    # SALib 1.5: use `sobol.sample` (the older `saltelli.sample` is deprecated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        X = salib_sobol_sample.sample(problem, n_samples_base,
                                        calc_second_order=False,
                                        seed=seed)
    n_evaluations = X.shape[0]

    Y = np.zeros(n_evaluations, dtype=float)
    n_invalid = 0
    for i, row in enumerate(X):
        params = dict(baseline)
        for j, name in enumerate(names):
            params[name] = float(row[j])
        y = _evaluate_for_kpi(part=part, model=model, params=params,
                               target_kpi=target_kpi,
                               baseline_material=material)
        if y is None:
            n_invalid += 1
            if invalid_sentinel is None:
                # Carry forward the last valid Y so SALib doesn't see NaN.
                # This inflates correlation between adjacent samples and
                # biases Sobol estimators; acceptable at low invalid rates
                # (≲5%), unreliable above that. The boundary warning below
                # flags when the user is in trouble.
                Y[i] = Y[i - 1] if i > 0 else 0.0
            else:
                Y[i] = float(invalid_sentinel)
        else:
            Y[i] = y

    invalid_fraction = n_invalid / max(n_evaluations, 1)
    if invalid_fraction > 0.10:
        warnings.warn(
            f"{invalid_fraction:.1%} of samples failed validation "
            f"({n_invalid} of {n_evaluations}). Sobol indices may be "
            f"unreliable — widen parameter_bounds or set invalid_sentinel.",
            stacklevel=2,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        Si = salib_sobol_analyze.analyze(
            problem, Y, calc_second_order=False,
            seed=seed, print_to_console=False,
        )

    return SensitivityResult(
        target_kpi=target_kpi,
        parameter_names=tuple(names),
        first_order={n: float(Si["S1"][i]) for i, n in enumerate(names)},
        first_order_ci={n: float(Si["S1_conf"][i]) for i, n in enumerate(names)},
        total_order={n: float(Si["ST"][i]) for i, n in enumerate(names)},
        total_order_ci={n: float(Si["ST_conf"][i]) for i, n in enumerate(names)},
        n_samples_base=n_samples_base,
        n_evaluations=int(n_evaluations),
        n_invalid_samples=n_invalid,
        bounds={n: tuple(parameter_bounds[n]) for n in names},
    )


# ---------------------------------------------------------------------------
# Analytical-function driver — for tests that verify our wrapper against a
# function with closed-form Sobol indices.

def analyze_function(*,
                      problem: dict[str, Any],
                      func: Callable[[Any], float],
                      n_samples_base: int = 1024,
                      seed: int = 0) -> SensitivityResult:
    """Run Sobol on an arbitrary scalar function for analytical validation.

    `problem` follows SALib's schema: {num_vars, names, bounds}.
    `func` takes a 1-D numpy array (one sample row) and returns a scalar.
    """
    from SALib.sample import sobol as salib_sobol_sample
    from SALib.analyze import sobol as salib_sobol_analyze
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        X = salib_sobol_sample.sample(problem, n_samples_base,
                                        calc_second_order=False, seed=seed)
    Y = np.array([func(row) for row in X], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        Si = salib_sobol_analyze.analyze(
            problem, Y, calc_second_order=False, seed=seed,
            print_to_console=False,
        )
    names = list(problem["names"])
    return SensitivityResult(
        target_kpi="<analytical>",
        parameter_names=tuple(names),
        first_order={n: float(Si["S1"][i]) for i, n in enumerate(names)},
        first_order_ci={n: float(Si["S1_conf"][i]) for i, n in enumerate(names)},
        total_order={n: float(Si["ST"][i]) for i, n in enumerate(names)},
        total_order_ci={n: float(Si["ST_conf"][i]) for i, n in enumerate(names)},
        n_samples_base=n_samples_base,
        n_evaluations=int(X.shape[0]),
        n_invalid_samples=0,
        bounds={n: tuple(problem["bounds"][i]) for i, n in enumerate(names)},
    )
