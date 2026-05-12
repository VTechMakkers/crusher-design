"""
Wear-aging lifecycle simulation via the Archard wear model.

Archard's wear law (Archard 1953, *J. Appl. Phys.* 24):

    V = K · N · s / H

where V is removed volume, K is the dimensionless wear coefficient,
N is normal load, s is sliding distance, H is the hardness of the
softer material (in pressure units).

For a region of area A subject to normal force N_region and sliding
distance s_region, the wear depth follows:

    h_region = K · (N_region / A) · s_region / H
             = K · P · s / H                          (P = N/A)

In rate form (used for lifecycle integration):

    dh/dt = K · P · v / H                             (v = sliding velocity)

The DEM solver produces a time-history of (N, s) per surface region per
crank revolution; multiplying by hours of service and the Archard
constants yields the wear-depth field at any service time t.

What this module enables:
  - simulate_lifecycle(initial_geometry, duty_cycle, hours)
      → wear-depth field at successive service hours
  - effective_geometry_at(t)
      → "the part 5,000 hours into service"
  - mass_loss_at(t)
      → total mass removed (input for aftermarket parts revenue model)
  - all subject to a conservation-of-mass identity that the tests enforce
    as a hard invariant

Pairs with:
  - mcp-servers/dem/   → produces the contact + sliding maps Archard consumes
  - loop/rbdo.py       → wears the geometry then propagates the worn
                          stress state through Monte-Carlo reliability
                          (next phase)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Material wear data

@dataclass(frozen=True)
class WearPair:
    """Archard wear constants for a (wear-surface material, abrasive) pair.

    K is dimensionless; H is the hardness of the wear-surface material
    in pressure units (Pa). The ratio K/H carries SI dimension of inverse
    pressure — convenient because P (contact pressure) is in Pa and
    s (sliding) in m, so K·P·s/H is dimensionless × Pa × m / Pa = m,
    giving wear depth in meters as expected.
    """
    surface_material: str
    abrasive: str
    K_dimensionless: float
    hardness_Pa: float
    citation: str = ""

    def validate(self) -> None:
        if not 1.0e-7 < self.K_dimensionless < 1.0e-1:
            raise ValueError(
                f"K={self.K_dimensionless} outside physical range "
                f"[1e-7, 1e-1] for abrasive wear. Verify the citation."
            )
        if self.hardness_Pa <= 0:
            raise ValueError("hardness_Pa must be positive")

    @property
    def wear_rate_constant(self) -> float:
        """K/H in 1/Pa.  Multiplied by (P · s) gives wear depth in meters."""
        return self.K_dimensionless / self.hardness_Pa


# ---------------------------------------------------------------------------
# Contact exposure (the DEM-side input)

@dataclass(frozen=True)
class RegionExposure:
    """The (N, s) tuple for one surface region over one duty period.

    The duty period is typically one crank revolution worth of crushing.
    The exposure is then scaled by hours of service to produce wear.
    """
    region_name: str
    area_m2: float
    normal_force_N: float           # time-averaged over the duty period
    sliding_distance_m: float       # cumulative sliding over the period
    duty_period_s: float            # length of period the exposure was averaged over

    def validate(self) -> None:
        for label, v in (("area_m2", self.area_m2),
                          ("duty_period_s", self.duty_period_s)):
            if v <= 0:
                raise ValueError(f"{label} must be positive (got {v})")
        if self.normal_force_N < 0:
            raise ValueError("normal_force_N must be non-negative")
        if self.sliding_distance_m < 0:
            raise ValueError("sliding_distance_m must be non-negative")


# ---------------------------------------------------------------------------
# Archard core (dimensional-analysis-verified)

def archard_wear_depth(*, force_N: float, sliding_m: float, area_m2: float,
                        K_dimensionless: float, hardness_Pa: float) -> float:
    """Wear depth (m) from Archard's law for a region of given area.

    h = K · P · s / H = K · (N / A) · s / H
    """
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    if hardness_Pa <= 0:
        raise ValueError("hardness_Pa must be positive")
    if force_N < 0 or sliding_m < 0 or K_dimensionless < 0:
        raise ValueError("force, sliding, and K must be non-negative")
    pressure_Pa = force_N / area_m2
    return K_dimensionless * pressure_Pa * sliding_m / hardness_Pa


def archard_wear_volume(*, force_N: float, sliding_m: float,
                          K_dimensionless: float,
                          hardness_Pa: float) -> float:
    """Wear volume (m^3) from Archard's law.

    V = K · N · s / H

    Note this is the area-independent form. For depth on a known area,
    use `archard_wear_depth`.
    """
    if hardness_Pa <= 0:
        raise ValueError("hardness_Pa must be positive")
    if force_N < 0 or sliding_m < 0 or K_dimensionless < 0:
        raise ValueError("force, sliding, and K must be non-negative")
    return K_dimensionless * force_N * sliding_m / hardness_Pa


# ---------------------------------------------------------------------------
# Lifecycle integration

@dataclass
class WearState:
    """Cumulative wear depth per region at a point in service time."""
    service_hours: float
    depth_per_region_m: dict[str, float]

    def total_volume_m3(self, region_areas_m2: dict[str, float]) -> float:
        return sum(self.depth_per_region_m[r] * region_areas_m2[r]
                    for r in self.depth_per_region_m)

    def total_mass_kg(self, region_areas_m2: dict[str, float],
                       density_kg_m3: float) -> float:
        return density_kg_m3 * self.total_volume_m3(region_areas_m2)

    def peak_depth_mm(self) -> tuple[str, float]:
        """Return (region_name, depth_mm) of the most-worn region."""
        if not self.depth_per_region_m:
            return ("", 0.0)
        worst = max(self.depth_per_region_m,
                     key=lambda r: self.depth_per_region_m[r])
        return (worst, 1000.0 * self.depth_per_region_m[worst])


@dataclass(frozen=True)
class LifecycleTrajectory:
    """Wear states sampled at a series of service times."""
    states: list[WearState]
    wear_pair: WearPair
    duty_active_fraction: float

    def state_at_hours(self, hours: float) -> WearState:
        """Linear-interpolate wear depth at an intermediate service time.

        Archard wear accumulates linearly with sliding distance, which in
        a steady duty cycle is linear in time → linear interpolation in
        time is exact, not approximate (within the duty model)."""
        if not self.states:
            raise ValueError("trajectory has no states")
        for i in range(len(self.states) - 1):
            t0, t1 = self.states[i].service_hours, self.states[i + 1].service_hours
            if t0 <= hours <= t1:
                if t1 == t0:
                    return self.states[i]
                alpha = (hours - t0) / (t1 - t0)
                regions = set(self.states[i].depth_per_region_m) \
                          | set(self.states[i + 1].depth_per_region_m)
                interp = {
                    r: (1 - alpha) * self.states[i].depth_per_region_m.get(r, 0.0)
                       + alpha * self.states[i + 1].depth_per_region_m.get(r, 0.0)
                    for r in regions
                }
                return WearState(service_hours=hours, depth_per_region_m=interp)
        # outside the bracketed range → return the closer endpoint
        if hours < self.states[0].service_hours:
            return self.states[0]
        return self.states[-1]


def simulate_lifecycle(*,
                        exposures: list[RegionExposure],
                        wear_pair: WearPair,
                        total_service_hours: float,
                        duty_active_fraction: float = 1.0,
                        sample_times_hours: list[float] | None = None,
                        ) -> LifecycleTrajectory:
    """Integrate Archard wear over the full service life.

    Parameters
    ----------
    exposures               : per-region DEM exposure for one duty period
    wear_pair               : Archard constants for the surface/abrasive pair
    total_service_hours     : end of the lifecycle
    duty_active_fraction    : fraction of clock-time the crusher is
                              actually crushing (typical 0.6-0.85 — captures
                              feed gaps, maintenance, weekends)
    sample_times_hours      : checkpoints to record. Default is
                              [0, 0.1, 0.25, 0.5, 0.75, 1.0] × total

    Wear is linear in sliding distance, which under a steady duty model is
    linear in time. We therefore do not need a finite-step integrator —
    the closed-form `h(t) = (K/H) · P · v · t_active` is exact. Sampling
    at intermediate checkpoints is for downstream FEA at those times.
    """
    wear_pair.validate()
    if total_service_hours <= 0:
        raise ValueError("total_service_hours must be positive")
    if not 0.0 < duty_active_fraction <= 1.0:
        raise ValueError("duty_active_fraction must be in (0, 1]")
    for ex in exposures:
        ex.validate()
    if not exposures:
        raise ValueError("must provide at least one region exposure")

    if sample_times_hours is None:
        sample_times_hours = [
            total_service_hours * f for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        ]
    if any(t < 0 or t > total_service_hours for t in sample_times_hours):
        raise ValueError(
            "sample_times_hours must lie within [0, total_service_hours]"
        )

    # Wear depth per region after one duty period
    period_depth: dict[str, float] = {
        ex.region_name: archard_wear_depth(
            force_N=ex.normal_force_N, sliding_m=ex.sliding_distance_m,
            area_m2=ex.area_m2,
            K_dimensionless=wear_pair.K_dimensionless,
            hardness_Pa=wear_pair.hardness_Pa,
        ) for ex in exposures
    }
    period_seconds = max(ex.duty_period_s for ex in exposures)
    if period_seconds <= 0:
        raise ValueError("duty_period_s must be positive for all regions")

    states: list[WearState] = []
    for t_hours in sorted(sample_times_hours):
        active_seconds = t_hours * 3600.0 * duty_active_fraction
        n_periods = active_seconds / period_seconds
        depth_field = {r: d * n_periods for r, d in period_depth.items()}
        states.append(WearState(service_hours=t_hours,
                                  depth_per_region_m=depth_field))

    return LifecycleTrajectory(
        states=states, wear_pair=wear_pair,
        duty_active_fraction=duty_active_fraction,
    )


