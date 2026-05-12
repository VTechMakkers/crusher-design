"""
Sensitivity analysis tests.

Layered:
  1. Wrapper correctness on functions with closed-form Sobol indices
     (Ishigami, additive linear, single-parameter pure dependence).
     These give us absolute confidence that our SALib wrapper is set up
     correctly — if it reproduces a paper-correct answer, the integration
     is right.
  2. Property tests (variance decomposition identities, dataclass
     invariants).
  3. Smoke tests on the actual crusher design loop.

Tolerances on stochastic tests come from Monte Carlo error theory:
Saltelli's variance estimator converges at ~1/sqrt(N). For N=2048,
expected absolute error on a well-conditioned Sobol index is < 0.04.
We add headroom and use 0.06.
"""
from __future__ import annotations
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# SALib is a hard dep for sensitivity; mark skip when absent so the rest
# of the suite remains green on environments without it.
salib = pytest.importorskip("SALib", reason="SALib not installed")

from loop.sensitivity import (analyze_function, analyze_part,
                                perturbation_bounds_from_baseline)


# Tolerance grounded in Monte Carlo error of Sobol at N=2048.
SOBOL_ABS_TOL = 0.06

# ---------------------------------------------------------------------------
# Closed-form analytical baselines


def test_ishigami_first_and_total_indices_match_published_values():
    """The Ishigami function (Saltelli & Bratchikov 1995) is the standard
    Sobol test problem. Closed-form indices (a=7, b=0.1):
        S1 ≈ {X1: 0.314, X2: 0.442, X3: 0.000}
        ST ≈ {X1: 0.558, X2: 0.442, X3: 0.244}
    Reproducing these end-to-end validates our entire pipeline."""
    a, b = 7.0, 0.1
    problem = {
        "num_vars": 3,
        "names": ["X1", "X2", "X3"],
        "bounds": [[-math.pi, math.pi]] * 3,
    }
    def ishigami(x):
        return (math.sin(x[0])
                + a * math.sin(x[1]) ** 2
                + b * x[2] ** 4 * math.sin(x[0]))

    result = analyze_function(problem=problem, func=ishigami,
                                n_samples_base=2048, seed=0)
    assert result.first_order["X1"] == pytest.approx(0.314, abs=SOBOL_ABS_TOL)
    assert result.first_order["X2"] == pytest.approx(0.442, abs=SOBOL_ABS_TOL)
    assert abs(result.first_order["X3"]) < SOBOL_ABS_TOL
    assert result.total_order["X1"] == pytest.approx(0.558, abs=SOBOL_ABS_TOL)
    assert result.total_order["X2"] == pytest.approx(0.442, abs=SOBOL_ABS_TOL)
    assert result.total_order["X3"] == pytest.approx(0.244, abs=SOBOL_ABS_TOL)


def test_additive_linear_function_has_no_interactions():
    """Y = 3*X1 + 5*X2 + X3, all U(0,1). Analytical first-order = total-order
    (no interactions); shares are (9, 25, 1) / (9+25+1) = (0.257, 0.714, 0.029)."""
    problem = {
        "num_vars": 3,
        "names": ["X1", "X2", "X3"],
        "bounds": [[0.0, 1.0]] * 3,
    }
    def linear(x):
        return 3 * x[0] + 5 * x[1] + x[2]

    r = analyze_function(problem=problem, func=linear,
                           n_samples_base=2048, seed=0)
    expected = {"X1": 9 / 35, "X2": 25 / 35, "X3": 1 / 35}
    for name, exp in expected.items():
        assert r.first_order[name] == pytest.approx(exp, abs=SOBOL_ABS_TOL)
        # Additive function → first_order ≈ total_order
        assert r.total_order[name] == pytest.approx(exp, abs=SOBOL_ABS_TOL)
        # Interaction strength near 0
        assert r.interaction_strength()[name] < 0.05


def test_single_active_parameter_gets_all_variance():
    """Y depends only on X1: Sobol S1[X1] ≈ 1, S1[X2] ≈ 0, S1[X3] ≈ 0."""
    problem = {
        "num_vars": 3,
        "names": ["X1", "X2", "X3"],
        "bounds": [[0.0, 1.0]] * 3,
    }
    def only_x1(x):
        return 10.0 * x[0]

    r = analyze_function(problem=problem, func=only_x1,
                           n_samples_base=2048, seed=0)
    assert r.first_order["X1"] == pytest.approx(1.0, abs=SOBOL_ABS_TOL)
    assert abs(r.first_order["X2"]) < SOBOL_ABS_TOL
    assert abs(r.first_order["X3"]) < SOBOL_ABS_TOL


# ---------------------------------------------------------------------------
# Property + reproducibility tests


def test_same_seed_gives_same_indices():
    problem = {
        "num_vars": 2, "names": ["a", "b"],
        "bounds": [[0.0, 1.0]] * 2,
    }
    func = lambda x: x[0] * x[1]
    r1 = analyze_function(problem=problem, func=func,
                            n_samples_base=1024, seed=42)
    r2 = analyze_function(problem=problem, func=func,
                            n_samples_base=1024, seed=42)
    assert r1.first_order == r2.first_order
    assert r1.total_order == r2.total_order


def test_different_seeds_give_different_indices_but_close():
    problem = {
        "num_vars": 2, "names": ["a", "b"],
        "bounds": [[0.0, 1.0]] * 2,
    }
    func = lambda x: x[0] + x[1]
    r1 = analyze_function(problem=problem, func=func, n_samples_base=1024, seed=1)
    r2 = analyze_function(problem=problem, func=func, n_samples_base=1024, seed=2)
    # Different RNG streams → different point estimates
    assert r1.first_order != r2.first_order
    # But within Monte Carlo tolerance of each other
    for k in ("a", "b"):
        assert abs(r1.first_order[k] - r2.first_order[k]) < 0.1


def test_n_evaluations_matches_saltelli_formula():
    """First-order-only Saltelli uses N(k+2) samples, not N(2k+2)."""
    problem = {"num_vars": 3, "names": ["a", "b", "c"],
                "bounds": [[0.0, 1.0]] * 3}
    func = lambda x: x[0] + x[1] + x[2]
    r = analyze_function(problem=problem, func=func, n_samples_base=512)
    expected_evals = 512 * (3 + 2)
    assert r.n_evaluations == expected_evals


def test_ranked_outputs_sort_descending():
    problem = {"num_vars": 3, "names": ["a", "b", "c"],
                "bounds": [[0.0, 1.0]] * 3}
    func = lambda x: 10 * x[0] + 3 * x[1] + x[2]
    r = analyze_function(problem=problem, func=func, n_samples_base=1024, seed=0)
    ranked = r.ranked_total_order()
    assert ranked[0][0] == "a"     # largest coefficient → largest total share
    assert ranked[-1][0] == "c"
    # Strictly descending
    for prev, nxt in zip(ranked, ranked[1:]):
        assert prev[1] >= nxt[1]


def test_summary_lines_contain_expected_columns():
    problem = {"num_vars": 2, "names": ["a", "b"],
                "bounds": [[0.0, 1.0]] * 2}
    func = lambda x: x[0] + x[1]
    r = analyze_function(problem=problem, func=func, n_samples_base=512, seed=0)
    text = "\n".join(r.summary_lines())
    assert "S1 (first)" in text
    assert "ST (total)" in text
    assert "interaction" in text
    assert "a" in text and "b" in text


# ---------------------------------------------------------------------------
# Bounds inference


def test_perturbation_bounds_handle_zero_baseline():
    """A parameter at 0.0 has no relative perturbation — bounds should not
    include it (no symmetric ± fraction of zero)."""
    bounds = perturbation_bounds_from_baseline({"a": 100.0, "b": 0.0,
                                                  "c": 5.0}, fraction=0.2)
    assert "a" in bounds
    assert "c" in bounds
    assert "b" not in bounds   # zero baseline excluded
    assert bounds["a"] == (80.0, 120.0)


def test_perturbation_bounds_skip_non_numeric():
    bounds = perturbation_bounds_from_baseline({"label": "foo", "size_mm": 42})
    assert "label" not in bounds
    assert "size_mm" in bounds


def test_perturbation_bounds_reject_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        perturbation_bounds_from_baseline({"x": 1.0}, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        perturbation_bounds_from_baseline({"x": 1.0}, fraction=1.5)


# ---------------------------------------------------------------------------
# Real crusher design loop smoke test


def test_analyze_toggle_plate_smoke():
    """End-to-end on the real design loop. Verifies plumbing, not absolute
    values — placeholder physics produces a particular sensitivity pattern
    that will change when real FEA + DEM land. We only check:
      - the analysis completes
      - results are well-formed (S in [0, 1+tol], indices sum reasonable)
      - no large invalid-sample count (means bounds are sensible)
    """
    result = analyze_part(
        part="toggle_plate", model="PE_400x600",
        target_kpi="composite",
        n_samples_base=128,    # small for fast test
        seed=0,
    )
    assert result.target_kpi == "composite"
    assert len(result.parameter_names) >= 4
    # Sobol indices should sit in [0, ~1+CI] for a well-defined problem;
    # negative S1 within CI of zero is normal Monte Carlo noise.
    for n in result.parameter_names:
        s1 = result.first_order[n]
        st = result.total_order[n]
        assert -0.1 <= s1 <= 1.2, f"S1[{n}]={s1} outside plausible range"
        assert -0.1 <= st <= 1.2, f"ST[{n}]={st} outside plausible range"
    # Most samples should be valid (bounds default to ±20% which is usually safe)
    assert result.n_invalid_samples < result.n_evaluations // 4


def test_mistyped_target_kpi_raises_immediately():
    """Regression for the broad-except silent-failure bug: passing a typo
    like 'composit' must raise KeyError with the available keys listed,
    NOT silently mark every sample invalid and complete the analysis."""
    with pytest.raises(KeyError, match="available"):
        analyze_part(
            part="toggle_plate", model="PE_400x600",
            target_kpi="composit",   # missing trailing 'e'
            n_samples_base=64, seed=0,
        )


def test_high_invalid_rate_warns(monkeypatch):
    """When >10% of samples fail validation, the user must see a warning.
    The dry-run path doesn't naturally exercise validate() failures, so we
    monkeypatch the per-sample evaluator to force a 1-in-3 failure rate."""
    import loop.sensitivity as sens

    n_calls = [0]
    def fake_eval(*, part, model, params, target_kpi, baseline_material):
        n_calls[0] += 1
        if n_calls[0] % 3 == 0:
            return None        # simulated validation failure
        return 1.0
    monkeypatch.setattr(sens, "_evaluate_for_kpi", fake_eval)

    with pytest.warns(UserWarning, match="failed validation"):
        sens.analyze_part(
            part="toggle_plate", model="PE_400x600",
            target_kpi="composite",
            n_samples_base=64, seed=0,
        )


def test_cli_runs_end_to_end():
    proc = subprocess.run(
        [sys.executable, "bin/run_sensitivity.py",
         "toggle_plate", "PE_400x600", "--samples", "64", "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    import json
    out = json.loads(proc.stdout)
    assert out["target_kpi"] == "composite"
    assert "first_order" in out
    assert "total_order" in out
    assert "interaction_strength" in out
