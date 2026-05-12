"""
FMEA (Failure Mode and Effects Analysis) rule engine.

Evaluates the 18 failure modes declared in `knowledge/failure_modes.yaml`
against a design's computed metrics, returning a ranked report:

  - which failure modes pass (margin > 0) and by how much
  - which fail (margin < 0) and by how much
  - which can't be evaluated because a required metric is missing
  - sorted by severity (critical → major → minor) and by smallest margin

Rule kinds supported (declared in `failure_modes.yaml.rule.kind`):
  ratio_threshold      numerator / denominator vs threshold + direction
  stress_vs_strength   stress · safety_factor vs strength
  wear_lifetime        wear depth at service hour vs allowable depth
  fatigue_life         stress amplitude vs S-N endurance (material or FAT class)
  contact_pressure     contact stress vs material allowable (Hertzian or flat)

Engine convention:
  margin > 0  → safe; magnitude is the fractional safety margin
  margin = 0  → at threshold (knife-edge)
  margin < 0  → failed; magnitude is the fractional overshoot
  margin = None → couldn't evaluate (missing metric); engine reports a
                  `skipped_reason` rather than fabricating a result.

The engine doesn't compute stresses or wear — it consumes metrics
produced upstream by FEA, MBD, DEM, wear simulation, etc. This keeps
FMEA purely about evaluation; what metrics are produced is the design
loop's concern.

Citations come from the failure_modes.yaml entries (Timoshenko & Gere,
Shigley, IIW Hobbacher, ISO 281, VDI 2230, ASME PVP, ...). Every
evaluation carries its citation forward into the report so a customer
or regulator can trace the basis of every claim.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Input + output dataclasses

@dataclass
class DesignMetrics:
    """The bundle of computed metric values that FMEA rules consume.

    `metrics`            scalar metrics by name (e.g. peak_stress_MPa = 320.0)
    `materials`          part_class -> material name (used for fatigue lookups)
    `material_props`     material name -> dict of properties (UTS, yield, ...)
                          ; if omitted, the engine loads from
                          knowledge/materials.yaml on demand
    `wear_trajectory`    LifecycleTrajectory from loop.wear_evolution, if any
                          ; required for wear_lifetime rules
    """
    metrics: dict[str, float]
    materials: dict[str, str] = field(default_factory=dict)
    material_props: dict[str, dict[str, Any]] = field(default_factory=dict)
    wear_trajectory: Any = None


@dataclass(frozen=True)
class FailureModeEvaluation:
    failure_mode: str
    part_class: str
    severity: str
    rule_kind: str
    passes: bool | None              # None when skipped_reason is set
    margin: float | None
    evaluation_detail: dict[str, Any]
    citation: str
    consequence: str
    prevention_rule: str
    skipped_reason: str | None = None

    @property
    def evaluated(self) -> bool:
        return self.skipped_reason is None


@dataclass(frozen=True)
class FMEAReport:
    part_class: str
    evaluations: list[FailureModeEvaluation]

    # ------- query API --------

    def evaluable(self) -> list[FailureModeEvaluation]:
        return [e for e in self.evaluations if e.evaluated]

    def unable_to_evaluate(self) -> list[FailureModeEvaluation]:
        return [e for e in self.evaluations if not e.evaluated]

    def passing(self) -> list[FailureModeEvaluation]:
        return [e for e in self.evaluable() if e.passes]

    def failing(self) -> list[FailureModeEvaluation]:
        return [e for e in self.evaluable() if not e.passes]

    def critical_failures(self) -> list[FailureModeEvaluation]:
        return [e for e in self.failing() if e.severity == "critical"]

    def passes_all(self) -> bool:
        """True iff every EVALUABLE failure mode passes. (Skipped modes
        are not counted — they are pending more metrics.)"""
        return not self.failing()

    def ranked_by_margin(self) -> list[FailureModeEvaluation]:
        """Smallest margin first (closest to failure). Failed modes
        (negative margin) come before passing modes."""
        return sorted(self.evaluable(), key=lambda e: e.margin)

    def ranked_by_risk(self) -> list[FailureModeEvaluation]:
        """Severity-weighted ranking. Critical failures first; within each
        severity, smallest margin first."""
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        return sorted(
            self.evaluable(),
            key=lambda e: (severity_order.get(e.severity, 99), e.margin),
        )

    # ------- output --------

    def summary_lines(self) -> list[str]:
        out = [f"FMEA report — {self.part_class}",
                f"  {len(self.evaluable())} evaluated, "
                f"{len(self.unable_to_evaluate())} skipped, "
                f"{len(self.failing())} failing"]
        out.append("")
        out.append("  failure_mode                          severity  margin     status")
        for e in self.ranked_by_risk():
            marker = "FAIL" if not e.passes else "ok"
            margin_str = f"{e.margin:+.3f}" if e.margin is not None else "  ---"
            out.append(f"  {e.failure_mode:38s}{e.severity:10s}{margin_str:11s}{marker}")
        if self.unable_to_evaluate():
            out.append("")
            out.append("  Skipped (missing metrics):")
            for e in self.unable_to_evaluate():
                out.append(f"    - {e.failure_mode}: {e.skipped_reason}")
        return out


# ---------------------------------------------------------------------------
# Internal helpers


class _Skip(Exception):
    """Raised by an evaluator when a required metric is missing.
    The engine catches it and converts to a FailureModeEvaluation with
    skipped_reason set — explicit, never silent."""


def _metric(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        raise _Skip(f"missing metric {name!r}")
    return float(metrics[name])


# ---------------------------------------------------------------------------
# S-N fatigue helpers — used by `fatigue_life` rule kind


def fat_class_endurance(fat_class: str, target_cycles: float) -> float:
    """Endurance stress (MPa) for a weld detail at `target_cycles`,
    per IIW Hobbacher (2008) two-slope S-N curve.

    The FAT class is the endurance stress at 2·10⁶ cycles. The curve has:
      - slope m=3 from 10⁴ to 5·10⁶ cycles
      - slope m=5 from 5·10⁶ to 10⁸ cycles
      - constant amplitude fatigue limit (CAFL) above 10⁸ cycles

    Example: FAT 71 at 2·10⁶ cycles returns exactly 71 MPa.
    """
    fat_value_str = fat_class.replace("FAT", "").strip()
    fat = float(fat_value_str)
    if fat <= 0 or target_cycles <= 0:
        raise ValueError("fat_class and target_cycles must be positive")
    n_ref = 2.0e6
    n_transition = 5.0e6
    n_cafl = 1.0e8

    if target_cycles <= n_transition:
        return fat * (n_ref / target_cycles) ** (1.0 / 3.0)

    σ_transition = fat * (n_ref / n_transition) ** (1.0 / 3.0)
    if target_cycles <= n_cafl:
        return σ_transition * (n_transition / target_cycles) ** (1.0 / 5.0)

    σ_cafl = σ_transition * (n_transition / n_cafl) ** (1.0 / 5.0)
    return σ_cafl


def material_endurance(material_props: dict[str, Any],
                        target_cycles: float) -> float:
    """Endurance stress (MPa) for a wrought-steel material at target_cycles,
    using the Shigley simplification:

        σ_e' = 0.5 · σ_UTS    for UTS ≤ 1400 MPa
        σ_e' = 700 MPa        for UTS > 1400 MPa  (plateau)

    σ_e' is the rotating-beam fatigue limit at ~10⁶ cycles. Below 10⁶,
    log-linear (Basquin) interpolation from σ_a(10³) = 0.9·σ_UTS:

        σ_a(N) = 0.9·UTS · (σ_e'/(0.9·UTS))^((log N − 3) / 3)   for 10³≤N≤10⁶

    For N > 10⁶ we hold at σ_e' (high-cycle plateau, valid for steels in
    the absence of corrosion / temperature derating).

    References:
      Shigley's Mechanical Engineering Design, 11th ed. (Budynas 2020),
      Eq. 6-8 + §6-9.
    """
    uts = material_props.get("ultimate_strength_MPa")
    if uts is None:
        raise _Skip("material has no ultimate_strength_MPa")
    if target_cycles <= 0:
        raise ValueError("target_cycles must be positive")
    uts = float(uts)
    σ_e_endurance = 0.5 * uts if uts <= 1400.0 else 700.0
    σ_a_10_3 = 0.9 * uts

    if target_cycles <= 1.0e3:
        return σ_a_10_3
    if target_cycles >= 1.0e6:
        return σ_e_endurance

    # Log-linear interpolation between (10³, 0.9·UTS) and (10⁶, σ_e_endurance)
    log_n = math.log10(target_cycles)
    return σ_a_10_3 * (σ_e_endurance / σ_a_10_3) ** ((log_n - 3.0) / 3.0)


# ---------------------------------------------------------------------------
# Per-rule-kind evaluators
#
# Each evaluator returns (margin, detail_dict). Raise _Skip to signal an
# evaluation is impossible with the metrics provided.


def _eval_ratio_threshold(rule: dict[str, Any],
                            design_metrics: DesignMetrics
                            ) -> tuple[float, dict[str, Any]]:
    num_name = rule["numerator"]
    den_name = rule["denominator"]
    threshold = float(rule["threshold"])
    direction = rule["direction"]
    if direction not in ("above_is_failure", "below_is_failure"):
        raise ValueError(f"unknown direction {direction!r} in ratio_threshold rule")

    # `denominator` may be either a metric name OR a literal numeric constant
    # (used in failure_modes.yaml for cases like "bolt_preload_loss_pct / 100").
    num = _metric(design_metrics.metrics, num_name)
    try:
        den = float(den_name)
    except (TypeError, ValueError):
        den = _metric(design_metrics.metrics, den_name)
    if den == 0:
        raise _Skip(f"denominator {den_name!r} is zero — ratio undefined")

    ratio = num / den
    if direction == "above_is_failure":
        # safe when ratio < threshold; margin = (threshold - ratio) / threshold
        margin = (threshold - ratio) / threshold if threshold != 0 else -ratio
    else:  # below_is_failure
        margin = (ratio - threshold) / threshold if threshold != 0 else ratio

    return margin, {
        "numerator": num_name, "denominator": den_name,
        "numerator_value": num, "denominator_value": den,
        "ratio": ratio, "threshold": threshold, "direction": direction,
    }


def _eval_stress_vs_strength(rule: dict[str, Any],
                               design_metrics: DesignMetrics
                               ) -> tuple[float, dict[str, Any]]:
    stress = _metric(design_metrics.metrics, rule["stress_metric"])
    strength = _metric(design_metrics.metrics, rule["strength_metric"])
    sf = float(rule["safety_factor"])
    direction = rule["direction"]
    if strength <= 0:
        raise _Skip(f"strength metric {rule['strength_metric']!r} ≤ 0")

    if direction == "above_is_failure":
        # safe when stress · SF ≤ strength
        margin = (strength - stress * sf) / strength
    elif direction == "below_is_failure":
        margin = (stress * sf - strength) / strength
    else:
        raise ValueError(f"unknown direction {direction!r}")

    return margin, {
        "stress_MPa": stress, "strength_MPa": strength,
        "safety_factor": sf, "direction": direction,
    }


def _eval_wear_lifetime(rule: dict[str, Any],
                         design_metrics: DesignMetrics
                         ) -> tuple[float, dict[str, Any]]:
    if design_metrics.wear_trajectory is None:
        raise _Skip("no wear_trajectory provided")
    region = rule["region"]
    target_hours = float(rule["at_service_hours"])
    max_depth_m = float(rule["max_depth_mm"]) / 1000.0

    try:
        state = design_metrics.wear_trajectory.state_at_hours(target_hours)
    except (AttributeError, ValueError) as exc:
        raise _Skip(f"wear trajectory has no state at {target_hours} hr: {exc}")
    if region not in state.depth_per_region_m:
        raise _Skip(f"region {region!r} absent from wear trajectory")
    actual_depth_m = state.depth_per_region_m[region]

    margin = (max_depth_m - actual_depth_m) / max_depth_m
    return margin, {
        "region": region, "service_hours": target_hours,
        "actual_depth_mm": actual_depth_m * 1000.0,
        "max_depth_mm": max_depth_m * 1000.0,
    }


def _eval_fatigue_life(rule: dict[str, Any],
                        design_metrics: DesignMetrics
                        ) -> tuple[float, dict[str, Any]]:
    σ_amp = _metric(design_metrics.metrics, rule["stress_amplitude_MPa"])
    target_n = float(rule["target_cycles"])

    if "detail_category" in rule:
        endurance = fat_class_endurance(rule["detail_category"], target_n)
        endurance_source = f"IIW {rule['detail_category']} at {target_n:.1e} cycles"
    else:
        # Material-based endurance — look up the part's material's props
        part_class = rule.get("_part_class", "")
        mat_name = design_metrics.materials.get(part_class)
        if mat_name is None:
            raise _Skip(
                f"fatigue_life needs material for part {part_class!r}; "
                f"add design_metrics.materials[{part_class!r}] = '<material>'"
            )
        props = (design_metrics.material_props.get(mat_name)
                  or _load_material_props(mat_name))
        if props is None:
            raise _Skip(
                f"material {mat_name!r} not in design_metrics.material_props "
                f"or knowledge/materials.yaml"
            )
        endurance = material_endurance(props, target_n)
        endurance_source = (f"Shigley σ_e'(N={target_n:.1e}) from "
                             f"{mat_name} UTS={props.get('ultimate_strength_MPa')} MPa")

    if endurance <= 0:
        raise _Skip("endurance ≤ 0 — degenerate")
    margin = (endurance - σ_amp) / endurance
    return margin, {
        "stress_amplitude_MPa": σ_amp,
        "endurance_MPa": endurance,
        "target_cycles": target_n,
        "endurance_source": endurance_source,
    }


def _eval_contact_pressure(rule: dict[str, Any],
                             design_metrics: DesignMetrics
                             ) -> tuple[float, dict[str, Any]]:
    pressure = _metric(design_metrics.metrics, rule["pressure_metric"])
    allowable = _metric(design_metrics.metrics, rule["allowable_metric"])
    sf = float(rule.get("safety_factor", 1.0))
    if allowable <= 0:
        raise _Skip("allowable contact stress ≤ 0")
    margin = (allowable - pressure * sf) / allowable
    return margin, {
        "pressure_MPa": pressure, "allowable_MPa": allowable,
        "safety_factor": sf,
    }


_RULE_EVALUATORS: dict[str, Callable[[dict, DesignMetrics],
                                       tuple[float, dict]]] = {
    "ratio_threshold": _eval_ratio_threshold,
    "stress_vs_strength": _eval_stress_vs_strength,
    "wear_lifetime": _eval_wear_lifetime,
    "fatigue_life": _eval_fatigue_life,
    "contact_pressure": _eval_contact_pressure,
}


# ---------------------------------------------------------------------------
# YAML + materials loading


def _load_failure_modes(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load((root / "knowledge/failure_modes.yaml").read_text())
    return data["failure_modes"]


def _load_material_props(material: str,
                           root: Path = ROOT) -> dict[str, Any] | None:
    mats = yaml.safe_load((root / "knowledge/materials.yaml").read_text())
    return mats.get(material)


# ---------------------------------------------------------------------------
# Top-level engine


def evaluate_failure_modes(*,
                             part_class: str,
                             design_metrics: DesignMetrics,
                             failure_modes: dict[str, dict[str, Any]] | None = None,
                             metric_aliases: dict[str, str] | None = None,
                             root: Path = ROOT) -> FMEAReport:
    """Run the FMEA pass for one part class.

    `metric_aliases` lets callers map a rule's metric name to a metric they
    actually have. Example: failure_modes.yaml asks for
    'peak_compressive_stress_MPa' but the design loop emits
    'max_von_mises_MPa'. Pass:

        metric_aliases={"peak_compressive_stress_MPa": "max_von_mises_MPa"}

    and the engine looks up 'max_von_mises_MPa' when a rule references the
    aliased name. Aliases are an explicit caller decision — the engine
    never auto-maps. If a rule needs a metric and there's no alias and no
    metric of that name, the evaluation is recorded as skipped.
    """
    fms = failure_modes if failure_modes is not None else _load_failure_modes(root)
    applicable = {
        name: spec for name, spec in fms.items()
        if part_class in spec.get("applies_to", [])
    }

    # Apply aliases by injecting them into a shallow metrics view
    if metric_aliases:
        aliased = dict(design_metrics.metrics)
        for rule_name, source_name in metric_aliases.items():
            if rule_name not in aliased and source_name in aliased:
                aliased[rule_name] = aliased[source_name]
        design_metrics = DesignMetrics(
            metrics=aliased,
            materials=design_metrics.materials,
            material_props=design_metrics.material_props,
            wear_trajectory=design_metrics.wear_trajectory,
        )

    evaluations: list[FailureModeEvaluation] = []
    for name, spec in applicable.items():
        rule = dict(spec["rule"])
        rule["_part_class"] = part_class       # forwarded to fatigue lookup
        kind = rule["kind"]
        evaluator = _RULE_EVALUATORS.get(kind)
        if evaluator is None:
            evaluations.append(FailureModeEvaluation(
                failure_mode=name, part_class=part_class,
                severity=spec["severity"], rule_kind=kind,
                passes=None, margin=None, evaluation_detail={},
                citation=spec.get("citation", ""),
                consequence=spec.get("consequence", ""),
                prevention_rule=spec.get("prevention_rule", ""),
                skipped_reason=f"unknown rule kind {kind!r}",
            ))
            continue

        try:
            margin, detail = evaluator(rule, design_metrics)
        except _Skip as exc:
            evaluations.append(FailureModeEvaluation(
                failure_mode=name, part_class=part_class,
                severity=spec["severity"], rule_kind=kind,
                passes=None, margin=None, evaluation_detail={},
                citation=spec.get("citation", ""),
                consequence=spec.get("consequence", ""),
                prevention_rule=spec.get("prevention_rule", ""),
                skipped_reason=str(exc),
            ))
            continue

        evaluations.append(FailureModeEvaluation(
            failure_mode=name, part_class=part_class,
            severity=spec["severity"], rule_kind=kind,
            passes=(margin >= 0.0), margin=margin,
            evaluation_detail=detail,
            citation=spec.get("citation", ""),
            consequence=spec.get("consequence", ""),
            prevention_rule=spec.get("prevention_rule", ""),
        ))

    return FMEAReport(part_class=part_class, evaluations=evaluations)
