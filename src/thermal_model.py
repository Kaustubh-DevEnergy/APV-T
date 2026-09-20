"""
Thermal model: ISO 9806:2017 steady-state model
for bifacial Insulating Glass Flat-Plate Collector (IGFPC).

Efficiency equation:
    eta = eta_0 - a1*(Tm - Ta)/G - a2*(Tm - Ta)^2/G

Coefficients from Summ et al. (2025), Applied Thermal Engineering.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, np.ndarray, pd.Series]


class IGFPCThermalModel:
    """Steady-state thermal efficiency and power model for bifacial IGFPC."""

    def __init__(
        self,
        eta_0: float = 0.856,
        a1: float = 4.069,
        a2: float = 0.009,
        collector_area: float = 12.422,
        rear_optical_factor: float = 0.85,
        iam_b0: float = 0.1,
    ):
        self.eta_0 = float(eta_0)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.collector_area = float(collector_area)
        self.rear_optical_factor = float(rear_optical_factor)
        self.iam_b0 = float(iam_b0)

    # ------------------------------------------------------------------ #
    def effective_irradiance(
        self, G_front: ArrayLike, G_rear: ArrayLike
    ) -> np.ndarray:
        """Combine front + rear (optically derated) into G_eff."""
        g_front = np.clip(np.nan_to_num(np.asarray(G_front, dtype=float)), 0.0, None)
        g_rear = np.clip(np.nan_to_num(np.asarray(G_rear, dtype=float)), 0.0, None)
        return g_front + self.rear_optical_factor * g_rear

    # ------------------------------------------------------------------ #
    def iam_correction(self, aoi_degrees: ArrayLike) -> np.ndarray:
        """ASHRAE incidence-angle modifier. Optional refinement."""
        aoi = np.asarray(aoi_degrees, dtype=float)
        theta_rad = np.radians(aoi)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_theta = np.cos(theta_rad)
            K = 1.0 - self.iam_b0 * (1.0 / cos_theta - 1.0)
        K = np.where(np.abs(aoi) > 80.0, 0.0, K)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(K, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def steady_state_efficiency(
        self,
        G_effective: ArrayLike,
        T_mean_fluid: ArrayLike,
        T_ambient: ArrayLike,
    ) -> np.ndarray:
        """ISO 9806:2017 steady-state thermal efficiency."""
        G = np.asarray(G_effective, dtype=float)
        Tm = np.asarray(T_mean_fluid, dtype=float)
        Ta = np.asarray(T_ambient, dtype=float)

        G_b, Tm_b, Ta_b = np.broadcast_arrays(G, Tm, Ta)
        dT = Tm_b - Ta_b

        valid = G_b > 0.0
        G_safe = np.where(valid, G_b, 1.0)

        eta = self.eta_0 - self.a1 * dT / G_safe - self.a2 * (dT ** 2) / G_safe
        eta = np.where(valid, eta, 0.0)
        return np.clip(eta, 0.0, self.eta_0)

    # ------------------------------------------------------------------ #
    def thermal_power(
        self,
        G_front: ArrayLike,
        G_rear: ArrayLike,
        T_mean_fluid: ArrayLike,
        T_ambient: ArrayLike,
    ) -> np.ndarray:
        """Useful thermal power output [W]."""
        G_eff = self.effective_irradiance(G_front, G_rear)
        eta = self.steady_state_efficiency(G_eff, T_mean_fluid, T_ambient)
        Q = eta * G_eff * self.collector_area
        return np.clip(Q, 0.0, None)

    # ------------------------------------------------------------------ #
    def simulate(
        self,
        irradiance_df: pd.DataFrame,
        T_mean_fluid: float = 65.0,
    ) -> pd.DataFrame:
        """Run thermal simulation over irradiance time series."""
        required = {"G_front", "G_rear"}
        missing = required - set(irradiance_df.columns)
        if missing:
            raise ValueError(f"irradiance_df missing columns: {missing}")

        G_front = irradiance_df["G_front"].to_numpy(dtype=float)
        G_rear = irradiance_df["G_rear"].to_numpy(dtype=float)

        if "temp_air" in irradiance_df.columns:
            T_ambient = irradiance_df["temp_air"].to_numpy(dtype=float)
        else:
            warnings.warn("No temp_air column; assuming 20 degC.", stacklevel=2)
            T_ambient = np.full(len(irradiance_df), 20.0)

        T_m = np.full(len(irradiance_df), float(T_mean_fluid))

        # Bifacial (front + rear)
        G_eff_bifacial = self.effective_irradiance(G_front, G_rear)
        eta_bifacial = self.steady_state_efficiency(G_eff_bifacial, T_m, T_ambient)
        Q_bifacial_W = np.clip(eta_bifacial * G_eff_bifacial * self.collector_area, 0.0, None)

        # Monofacial reference (front only)
        G_eff_mono = self.effective_irradiance(G_front, np.zeros_like(G_rear))
        eta_monofacial = self.steady_state_efficiency(G_eff_mono, T_m, T_ambient)
        Q_monofacial_W = np.clip(eta_monofacial * G_eff_mono * self.collector_area, 0.0, None)

        return pd.DataFrame(
            {
                "G_effective": G_eff_bifacial,
                "efficiency": eta_bifacial,
                "Q_bifacial_W": Q_bifacial_W,
                "Q_bifacial_kWh": Q_bifacial_W / 1000.0,
                "Q_monofacial_W": Q_monofacial_W,
                "Q_monofacial_kWh": Q_monofacial_W / 1000.0,
                "eta_bifacial": eta_bifacial,
                "eta_monofacial": eta_monofacial,
            },
            index=irradiance_df.index,
        )

    # ------------------------------------------------------------------ #
    def calculate_kpis(self, thermal_results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute KPIs from thermal simulation results."""
        df = thermal_results_df

        yield_bifacial = df["Q_bifacial_kWh"].sum() / self.collector_area
        yield_monofacial = df["Q_monofacial_kWh"].sum() / self.collector_area

        if yield_monofacial > 0:
            gain_pct = (yield_bifacial - yield_monofacial) / yield_monofacial * 100.0
        else:
            gain_pct = float("nan")

        operating_mask = df["Q_bifacial_W"] > 0
        mean_eff = (
            float(df.loc[operating_mask, "efficiency"].mean())
            if operating_mask.any() else 0.0
        )

        peak_power = float(df["Q_bifacial_W"].max())
        operating_hours = int(operating_mask.sum())
        total_yield_kWh = df["Q_bifacial_kWh"].sum()

        if peak_power > 0:
            capacity_factor = float(total_yield_kWh / (peak_power / 1000.0 * 8760.0))
        else:
            capacity_factor = 0.0

        return {
            "annual_yield_bifacial_kWh_per_m2": float(yield_bifacial),
            "annual_yield_monofacial_kWh_per_m2": float(yield_monofacial),
            "bifacial_thermal_gain_pct": float(gain_pct),
            "mean_efficiency_bifacial": mean_eff,
            "peak_power_W": peak_power,
            "operating_hours": operating_hours,
            "capacity_factor": capacity_factor,
        }

    # ------------------------------------------------------------------ #
    def efficiency_curve_data(
        self,
        G_values: Sequence[float] = (200, 400, 600, 800, 1000),
        T_ambient: float = 20.0,
        T_mean_min: float = 20.0,
        T_mean_max: float = 120.0,
        n_points: int = 101,
    ) -> pd.DataFrame:
        """Generate efficiency-vs-reduced-temperature curve data for plotting."""
        T_mean_sweep = np.linspace(T_mean_min, T_mean_max, n_points)
        frames = []
        for G in G_values:
            G_arr = np.full(n_points, float(G))
            Ta_arr = np.full(n_points, float(T_ambient))
            eta = self.steady_state_efficiency(G_arr, T_mean_sweep, Ta_arr)
            reduced_T = (T_mean_sweep - T_ambient) / G if G > 0 else np.full(n_points, np.nan)
            frames.append(pd.DataFrame({
                "G": G_arr, "T_mean": T_mean_sweep,
                "reduced_temperature": reduced_T, "efficiency": eta,
            }))
        return pd.concat(frames, ignore_index=True)


# ============================================================================= #
# SENSITIVITY ANALYSIS
# ============================================================================= #

class BifacialThermalAnalysis:
    """Higher-level sensitivity utilities built on IGFPCThermalModel."""

    def __init__(self, base_model: Optional[IGFPCThermalModel] = None):
        self.base_model = base_model if base_model is not None else IGFPCThermalModel()

    def temperature_sensitivity(
        self,
        irradiance_df: pd.DataFrame,
        T_values: Sequence[float] = (40, 50, 60, 65, 70, 80),
    ) -> pd.DataFrame:
        """Sweep mean fluid temperature."""
        rows = []
        for T in T_values:
            results = self.base_model.simulate(irradiance_df, T_mean_fluid=T)
            kpis = self.base_model.calculate_kpis(results)
            rows.append({
                "T_mean": float(T),
                "yield_bifacial": kpis["annual_yield_bifacial_kWh_per_m2"],
                "yield_monofacial": kpis["annual_yield_monofacial_kWh_per_m2"],
                "gain_pct": kpis["bifacial_thermal_gain_pct"],
            })
        return pd.DataFrame(rows)

    def rear_factor_sensitivity(
        self,
        irradiance_df: pd.DataFrame,
        factors: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
        T_mean_fluid: float = 65.0,
    ) -> pd.DataFrame:
        """Sweep rear optical factor."""
        rows = []
        for factor in factors:
            model = IGFPCThermalModel(
                eta_0=self.base_model.eta_0,
                a1=self.base_model.a1,
                a2=self.base_model.a2,
                collector_area=self.base_model.collector_area,
                rear_optical_factor=float(factor),
                iam_b0=self.base_model.iam_b0,
            )
            results = model.simulate(irradiance_df, T_mean_fluid=T_mean_fluid)
            kpis = model.calculate_kpis(results)
            rows.append({
                "rear_optical_factor": float(factor),
                "yield_bifacial": kpis["annual_yield_bifacial_kWh_per_m2"],
                "yield_monofacial": kpis["annual_yield_monofacial_kWh_per_m2"],
                "gain_pct": kpis["bifacial_thermal_gain_pct"],
            })
        return pd.DataFrame(rows)


# ============================================================================= #
# DEMO / SELF-TEST
# ============================================================================= #
if __name__ == "__main__":
    hours = np.arange(24)
    times = pd.date_range("2026-06-21 00:00", periods=24, freq="1h", tz="Europe/Berlin")

    def _bell_curve(h, peak_hour, width, peak_val):
        curve = peak_val * np.exp(-0.5 * ((h - peak_hour) / width) ** 2)
        return np.clip(curve, 0.0, None)

    G_front = _bell_curve(hours, peak_hour=9.0, width=2.5, peak_val=600.0)
    G_rear = _bell_curve(hours, peak_hour=15.0, width=2.5, peak_val=400.0)
    daylight_mask = (hours >= 5) & (hours <= 21)
    G_front = np.where(daylight_mask, G_front, 0.0)
    G_rear = np.where(daylight_mask, G_rear, 0.0)

    weather = pd.DataFrame(
        {"G_front": G_front, "G_rear": G_rear, "temp_air": 20.0}, index=times
    )

    model = IGFPCThermalModel()
    print("=== IGFPC bifacial thermal model demo ===")
    print(f"eta_0={model.eta_0}, a1={model.a1}, a2={model.a2}, "
          f"area={model.collector_area:.3f} m^2, rear_factor={model.rear_optical_factor}\n")

    results = model.simulate(weather, T_mean_fluid=65.0)
    print(results[["G_effective", "efficiency", "Q_bifacial_W", "Q_monofacial_W",
                    "eta_bifacial", "eta_monofacial"]].round(2).to_string(index=False))

    kpis = model.calculate_kpis(results)
    print("\nKPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    bifacial_total = results["Q_bifacial_kWh"].sum()
    monofacial_total = results["Q_monofacial_kWh"].sum()
    print(f"\nBifacial total:   {bifacial_total:.3f} kWh")
    print(f"Monofacial total: {monofacial_total:.3f} kWh")
    assert bifacial_total > monofacial_total, "Bifacial should exceed monofacial!"
    print("Confirmed: bifacial > monofacial.")

    # Sensitivity analyses
    analysis = BifacialThermalAnalysis(base_model=model)
    print("\n--- Temperature sensitivity ---")
    print(analysis.temperature_sensitivity(weather, T_values=[40, 50, 60, 65, 70, 80]).round(4).to_string(index=False))

    print("\n--- Rear optical factor sensitivity ---")
    print(analysis.rear_factor_sensitivity(weather).round(4).to_string(index=False))
