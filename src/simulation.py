"""
Master simulation orchestrator for APV-T systems.

Combines bifacial irradiance model + ISO 9806 thermal model
+ AquaCrop crop shading + Land Equivalent Ratio (LER).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from irradiance_model import BifacialVerticalIrradiance
from thermal_model import IGFPCThermalModel
from crop_model import AquaCropSimulation
from config import SiteConfig, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig


# ============================================================================= #
# 1. RESULT CONTAINER
# ============================================================================= #

@dataclass
class SimulationResult:
    """Container for a complete APV-T simulation run."""

    config: Dict[str, Any]
    hourly_results: pd.DataFrame
    irradiance_kpis: Dict[str, Any]
    thermal_kpis: Dict[str, Any]
    reference_thermal_yield_kWh_per_m2: float = 0.0

    # --- Derived KPIs --- #
    @property
    def annual_ghi(self) -> float:
        if "ghi" in self.hourly_results.columns:
            return float(self.hourly_results["ghi"].sum() / 1000.0)
        return float("nan")

    @property
    def annual_front_irradiance(self) -> float:
        return float(self.hourly_results["G_front"].sum() / 1000.0)

    @property
    def annual_rear_irradiance(self) -> float:
        return float(self.hourly_results["G_rear"].sum() / 1000.0)

    @property
    def annual_total_irradiance(self) -> float:
        return float(self.hourly_results["G_total"].sum() / 1000.0)

    @property
    def bifacial_irradiance_gain(self) -> float:
        front = self.annual_front_irradiance
        if front <= 0:
            return float("nan")
        return float(self.annual_rear_irradiance / front * 100.0)

    @property
    def annual_thermal_yield_bifacial(self) -> float:
        return float(self.thermal_kpis.get("annual_yield_bifacial_kWh_per_m2", float("nan")))

    @property
    def annual_thermal_yield_monofacial(self) -> float:
        return float(self.thermal_kpis.get("annual_yield_monofacial_kWh_per_m2", float("nan")))

    @property
    def bifacial_thermal_gain(self) -> float:
        return float(self.thermal_kpis.get("bifacial_thermal_gain_pct", float("nan")))

    @property
    def estimated_crop_yield(self) -> float:
        return float(self.config.get("estimated_crop_yield_t_per_ha", float("nan")))

    @property
    def LER(self) -> float:
        return float(self.config.get("LER", float("nan")))

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Flatten all headline KPIs into a single dict."""
        flat: Dict[str, Any] = {
            "annual_ghi_kWh_per_m2": self.annual_ghi,
            "annual_front_irradiance_kWh_per_m2": self.annual_front_irradiance,
            "annual_rear_irradiance_kWh_per_m2": self.annual_rear_irradiance,
            "annual_total_irradiance_kWh_per_m2": self.annual_total_irradiance,
            "bifacial_irradiance_gain_pct": self.bifacial_irradiance_gain,
            "annual_thermal_yield_bifacial_kWh_per_m2": self.annual_thermal_yield_bifacial,
            "annual_thermal_yield_monofacial_kWh_per_m2": self.annual_thermal_yield_monofacial,
            "bifacial_thermal_gain_pct": self.bifacial_thermal_gain,
            "reference_thermal_yield_kWh_per_m2": self.reference_thermal_yield_kWh_per_m2,
            "estimated_crop_yield_t_per_ha": self.estimated_crop_yield,
            "LER": self.LER,
        }
        flat.update({f"irr_{k}": v for k, v in self.irradiance_kpis.items()})
        flat.update({f"thermal_{k}": v for k, v in self.thermal_kpis.items()})
        return flat

    # ------------------------------------------------------------------ #
    def summary(self) -> None:
        """Print formatted KPI summary to stdout."""
        line = "=" * 62
        print(line)
        print("APV-T SIMULATION SUMMARY")
        print(line)

        print("\n--- Irradiance ---")
        print(f"  {'Annual GHI:':<40}{self.annual_ghi:>10.1f} kWh/m^2")
        print(f"  {'Annual front POA:':<40}{self.annual_front_irradiance:>10.1f} kWh/m^2")
        print(f"  {'Annual rear POA:':<40}{self.annual_rear_irradiance:>10.1f} kWh/m^2")
        print(f"  {'Annual total POA:':<40}{self.annual_total_irradiance:>10.1f} kWh/m^2")
        print(f"  {'Bifacial irradiance gain:':<40}{self.bifacial_irradiance_gain:>10.1f} %")

        print("\n--- Geometric view factors ---")
        for key, val in self.irradiance_kpis.items():
            if isinstance(val, float):
                print(f"  {key + ':':<40}{val:>10.4f}")
            else:
                print(f"  {key + ':':<40}{val!s:>10}")

        print("\n--- Thermal ---")
        print(f"  {'Bifacial thermal yield:':<40}{self.annual_thermal_yield_bifacial:>10.1f} kWh/m^2")
        print(f"  {'Monofacial thermal yield:':<40}{self.annual_thermal_yield_monofacial:>10.1f} kWh/m^2")
        print(f"  {'Bifacial thermal gain:':<40}{self.bifacial_thermal_gain:>10.1f} %")
        print(f"  {'Reference tilted yield:':<40}{self.reference_thermal_yield_kWh_per_m2:>10.1f} kWh/m^2")

        print("\n--- Agrivoltaic crop impact ---")
        print(f"  {'Estimated crop yield:':<40}{self.estimated_crop_yield:>10.2f} t/ha")
        print(f"  {'Reference yield:':<40}{self.config.get('reference_yield', float('nan')):>10.2f} t/ha")

        print("\n--- Land Equivalent Ratio ---")
        print(f"  {'LER:':<40}{self.LER:>10.3f}")
        print(line)

    # ------------------------------------------------------------------ #
    def monthly_summary(self) -> pd.DataFrame:
        """Aggregate hourly results into monthly totals/averages."""
        df = self.hourly_results.copy()
        month = df.index.month

        agg: Dict[str, pd.Series] = {}
        if "ghi" in df.columns:
            agg["GHI_kWh_m2"] = df["ghi"].groupby(month).sum() / 1000.0
        agg["G_front_kWh_m2"] = df["G_front"].groupby(month).sum() / 1000.0
        agg["G_rear_kWh_m2"] = df["G_rear"].groupby(month).sum() / 1000.0
        agg["G_total_kWh_m2"] = df["G_total"].groupby(month).sum() / 1000.0
        agg["Q_bifacial_kWh"] = df["Q_bifacial_kWh"].groupby(month).sum()
        agg["Q_monofacial_kWh"] = df["Q_monofacial_kWh"].groupby(month).sum()
        agg["mean_efficiency"] = df["efficiency"].groupby(month).mean()
        if "bifacial_gain" in df.columns:
            agg["mean_bifacial_gain"] = df["bifacial_gain"].groupby(month).mean()

        monthly = pd.DataFrame(agg)
        monthly.index.name = "month"
        return monthly

    # ------------------------------------------------------------------ #
    def save(self, directory: Union[str, Path]) -> None:
        """Persist results: hourly CSV + KPI JSON + config JSON."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.hourly_results.to_csv(out_dir / "hourly_results.csv")

        kpis = self.to_dict()
        kpis_clean = {
            k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
            for k, v in kpis.items()
        }
        with open(out_dir / "kpis.json", "w") as f:
            json.dump(kpis_clean, f, indent=2, default=str)

        with open(out_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)


# ============================================================================= #
# 2. MAIN SIMULATION ORCHESTRATOR
# ============================================================================= #

class APVTSimulation:
    """
    End-to-end APV-T simulation pipeline.

    Steps:
      1. Bifacial irradiance (view-factor model)
      2. Thermal simulation (ISO 9806 IGFPC model)
      3. Reference tilted-surface yield (LER denominator)
      4. Crop shading via AquaCrop / empirical fallback
      5. Land Equivalent Ratio
    """

    def __init__(
        self,
        site_config: SiteConfig,
        apv_config: APVConfig,
        collector_config: IGFPCConfig,
        thermal_config: ThermalOperatingConfig,
        crop_config: CropConfig,
        monthly_albedo: Optional[Dict[int, float]] = None,
    ):
        self.site_config = site_config
        self.apv_config = apv_config
        self.collector_config = collector_config
        self.thermal_config = thermal_config
        self.crop_config = crop_config
        self.monthly_albedo = monthly_albedo

        self.irradiance_model = BifacialVerticalIrradiance(
            site_config, apv_config, collector_config, monthly_albedo
        )
        self.thermal_model = IGFPCThermalModel(
            eta_0=collector_config.eta_0,
            a1=collector_config.a_1,
            a2=collector_config.a_2,
            collector_area=collector_config.area,
            rear_optical_factor=apv_config.rear_optical_factor,
        )

    # ------------------------------------------------------------------ #
    def run(self, weather_df: pd.DataFrame) -> SimulationResult:
        """Run the complete APV-T simulation pipeline."""

        # --- Step 1: bifacial irradiance --- #
        irr_results = self.irradiance_model.simulate(weather_df)

        # --- Step 2: merge ambient temperature + raw weather --- #
        cols = {c.lower(): c for c in weather_df.columns}
        merged = irr_results.copy()
        if "temp_air" in cols:
            merged["temp_air"] = weather_df[cols["temp_air"]].values
        else:
            merged["temp_air"] = 20.0
        for raw_col in ("ghi", "dni", "dhi"):
            if raw_col in cols:
                merged[raw_col] = weather_df[cols[raw_col]].values

        # --- Step 3: thermal simulation --- #
        thermal_results = self.thermal_model.simulate(
            merged, T_mean_fluid=self.thermal_config.T_mean
        )

        # --- Step 4: merge irradiance + thermal --- #
        hourly_results = merged.join(thermal_results, how="left")

        # --- Step 5: reference tilted yield (LER denominator) --- #
        reference_yield_kWh_m2 = self.run_reference_thermal(weather_df, tilt=35.0)

        # --- Step 6: crop shading via AquaCrop --- #
        GCR = self.collector_config.module_height / self.collector_config.row_spacing
        par_reduction = float(np.clip(GCR * 0.35, 0.05, 0.70))

        crop_sim = AquaCropSimulation(crop_name="Barley", planting_date="05/01")
        crop_res = crop_sim.simulate(
            weather_df=weather_df,
            shade_fraction=par_reduction,
            reference_yield_override=self.crop_config.reference_yield,
        )
        crop_yield = crop_res.apv_yield_t_ha
        crop_ratio = crop_res.yield_ratio

        # --- Step 7: Land Equivalent Ratio --- #
        thermal_kpis = self.thermal_model.calculate_kpis(hourly_results)
        thermal_yield_bifacial = thermal_kpis["annual_yield_bifacial_kWh_per_m2"]

        if reference_yield_kWh_m2 > 0:
            energy_ratio = thermal_yield_bifacial / reference_yield_kWh_m2
        else:
            energy_ratio = float("nan")
        LER = crop_ratio + energy_ratio

        # --- Step 8: assemble result --- #
        irradiance_kpis = {
            "vf_sky_front": self.irradiance_model.vf_sky_front,
            "vf_sky_rear": self.irradiance_model.vf_sky_rear,
            "vf_ground_front": self.irradiance_model.vf_ground_front,
            "vf_ground_rear": self.irradiance_model.vf_ground_rear,
            "mean_bifacial_gain": float(
                hourly_results["bifacial_gain"].replace([np.inf, -np.inf], np.nan).mean()
            ) if "bifacial_gain" in hourly_results.columns else float("nan"),
            "mean_ground_shading_fraction": float(
                hourly_results["ground_shading_fraction"].mean()
            ) if "ground_shading_fraction" in hourly_results.columns else float("nan"),
        }

        config_snapshot: Dict[str, Any] = {
            "site_config": vars(self.site_config) if hasattr(self.site_config, "__dict__") else str(self.site_config),
            "apv_config": vars(self.apv_config) if hasattr(self.apv_config, "__dict__") else str(self.apv_config),
            "collector_config": vars(self.collector_config) if hasattr(self.collector_config, "__dict__") else str(self.collector_config),
            "thermal_config": vars(self.thermal_config) if hasattr(self.thermal_config, "__dict__") else str(self.thermal_config),
            "crop_config": vars(self.crop_config) if hasattr(self.crop_config, "__dict__") else str(self.crop_config),
            "GCR": GCR,
            "par_reduction": par_reduction,
            "estimated_crop_yield_t_per_ha": crop_yield,
            "reference_yield": self.crop_config.reference_yield,
            "LER": LER,
        }

        return SimulationResult(
            config=config_snapshot,
            hourly_results=hourly_results,
            irradiance_kpis=irradiance_kpis,
            thermal_kpis=thermal_kpis,
            reference_thermal_yield_kWh_per_m2=reference_yield_kWh_m2,
        )

    # ------------------------------------------------------------------ #
    def run_reference_thermal(self, weather_df: pd.DataFrame, tilt: float = 35.0) -> float:
        """Simulate a monofacial fixed-tilt reference collector.
        Returns annual thermal yield [kWh/m^2] for LER denominator.
        """
        ref_irr = self.irradiance_model.simulate_reference_tilted(weather_df, tilt=tilt)

        cols = {c.lower(): c for c in weather_df.columns}
        ref_df = pd.DataFrame(index=weather_df.index)
        ref_df["G_front"] = ref_irr["G_poa"].values
        ref_df["G_rear"] = 0.0
        ref_df["temp_air"] = (
            weather_df[cols["temp_air"]].values if "temp_air" in cols else 20.0
        )

        ref_thermal = IGFPCThermalModel(
            eta_0=self.collector_config.eta_0,
            a1=self.collector_config.a_1,
            a2=self.collector_config.a_2,
            collector_area=self.collector_config.area,
            rear_optical_factor=0.0,
        )
        ref_thermal_results = ref_thermal.simulate(ref_df, T_mean_fluid=self.thermal_config.T_mean)
        ref_kpis = ref_thermal.calculate_kpis(ref_thermal_results)
        return float(ref_kpis["annual_yield_bifacial_kWh_per_m2"])


# ============================================================================= #
# 3. DESIGN WINDOW SUMMARY
# ============================================================================= #

def summarize_design_window(
    parametric_results: Dict[str, pd.DataFrame],
    ler_min: float = 1.3,
    crop_yield_min: float = 4.5,
    thermal_yield_min: float = 300.0,
) -> Dict[str, Any]:
    """
    Identify optimal design region meeting LER, crop, and thermal thresholds.
    """
    all_data = []
    for name, df in parametric_results.items():
        df_copy = df.copy()
        df_copy["sweep_name"] = name
        all_data.append(df_copy)

    combined = pd.concat(all_data, ignore_index=True)

    viable = combined[
        (combined["LER"] >= ler_min)
        & (combined["estimated_crop_yield_t_per_ha"] >= crop_yield_min)
        & (combined["annual_thermal_yield_bifacial_kWh_per_m2"] >= thermal_yield_min)
    ]

    if len(viable) > 0:
        optimal_idx = viable["LER"].idxmax()
        optimal = viable.loc[optimal_idx].to_dict()

        base_params = {
            "row_spacing": 4.0,
            "albedo": 0.25,
            "hub_height": 2.0,
            "surface_azimuth": 90.0,
            "module_height": 2.814,
            "surface_tilt": 90.0,
        }
        for param, default_val in base_params.items():
            if param not in optimal or pd.isna(optimal.get(param)):
                optimal[param] = default_val
    else:
        optimal = None

    design_ranges = {}
    param_columns = ["row_spacing", "albedo", "hub_height", "surface_azimuth",
                     "T_mean_fluid", "surface_tilt", "module_height"]

    for param in param_columns:
        if param in viable.columns:
            valid_values = viable[param].dropna()
            if len(valid_values) > 0:
                design_ranges[param] = {
                    "min": float(valid_values.min()),
                    "max": float(valid_values.max()),
                    "values": sorted(valid_values.unique().tolist()),
                }

    summary = {
        "constraints": {"ler_min": ler_min, "crop_yield_min": crop_yield_min, "thermal_yield_min": thermal_yield_min},
        "total_configurations_tested": len(combined),
        "viable_configurations": len(viable),
        "viable_percentage": len(viable) / len(combined) * 100 if len(combined) > 0 else 0,
        "design_ranges": design_ranges,
        "optimal_configuration": optimal,
        "viable_ler_range": {
            "min": float(viable["LER"].min()) if len(viable) > 0 else None,
            "max": float(viable["LER"].max()) if len(viable) > 0 else None,
            "mean": float(viable["LER"].mean()) if len(viable) > 0 else None,
        },
        "viable_thermal_range": {
            "min": float(viable["annual_thermal_yield_bifacial_kWh_per_m2"].min()) if len(viable) > 0 else None,
            "max": float(viable["annual_thermal_yield_bifacial_kWh_per_m2"].max()) if len(viable) > 0 else None,
        },
        "viable_crop_range": {
            "min": float(viable["estimated_crop_yield_t_per_ha"].min()) if len(viable) > 0 else None,
            "max": float(viable["estimated_crop_yield_t_per_ha"].max()) if len(viable) > 0 else None,
        },
    }
    return summary


def print_design_window(summary: Dict[str, Any]) -> None:
    """Print formatted design window summary."""
    print("\n" + "=" * 70)
    print("OPTIMAL DESIGN WINDOW FOR VERTICAL APV-T SYSTEMS")
    print("=" * 70)

    c = summary["constraints"]
    print(f"\nConstraints:")
    print(f"  • LER >= {c['ler_min']}")
    print(f"  • Crop yield >= {c['crop_yield_min']} t/ha")
    print(f"  • Thermal yield >= {c['thermal_yield_min']} kWh/m²")

    print(f"\nResults:")
    print(f"  • Configurations tested: {summary['total_configurations_tested']}")
    print(f"  • Viable configurations: {summary['viable_configurations']} ({summary['viable_percentage']:.1f}%)")

    if summary["viable_configurations"] > 0:
        print(f"\nViable Design Ranges:")
        for param, ranges in summary["design_ranges"].items():
            if ranges["min"] == ranges["max"]:
                print(f"  • {param}: {ranges['min']}")
            else:
                print(f"  • {param}: {ranges['min']} – {ranges['max']}")

        print(f"\nPerformance within viable region:")
        print(f"  • LER: {summary['viable_ler_range']['min']:.3f} – {summary['viable_ler_range']['max']:.3f} (mean: {summary['viable_ler_range']['mean']:.3f})")
        print(f"  • Thermal yield: {summary['viable_thermal_range']['min']:.0f} – {summary['viable_thermal_range']['max']:.0f} kWh/m²")
        print(f"  • Crop yield: {summary['viable_crop_range']['min']:.2f} – {summary['viable_crop_range']['max']:.2f} t/ha")

        if summary["optimal_configuration"]:
            opt = summary["optimal_configuration"]
            print(f"\nOptimal Configuration (max LER):")
            print(f"  • LER: {opt.get('LER', 'N/A'):.3f}")
            print(f"  • Thermal yield: {opt.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A'):.0f} kWh/m²")
            print(f"  • Crop yield: {opt.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")
            if "row_spacing" in opt:
                print(f"  • Row spacing: {opt.get('row_spacing', 'N/A')} m")
            if "albedo" in opt:
                print(f"  • Albedo: {opt.get('albedo', 'N/A')}")
    else:
        print("\n  ⚠️ No configurations meet all constraints. Consider relaxing thresholds.")

    print("=" * 70)


# ============================================================================= #
# DEMO / SELF-TEST
# ============================================================================= #
if __name__ == "__main__":
    import time

    from config import INGOLSTADT, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig, MONTHLY_ALBEDO_INGOLSTADT
    from utils import load_pvgis_tmy

    print("Loading weather data (PVGIS TMY) for Ingolstadt...")
    weather = load_pvgis_tmy(INGOLSTADT.tmy_file)
    print(f"Loaded {len(weather)} hourly weather records.\n")

    sim = APVTSimulation(
        site_config=INGOLSTADT,
        apv_config=APVConfig(surface_azimuth=90.0),
        collector_config=IGFPCConfig(),
        thermal_config=ThermalOperatingConfig(),
        crop_config=CropConfig(),
        monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
    )

    print("Running APV-T simulation (east-facing vertical, azimuth=90°)...")
    t0 = time.time()
    result = sim.run(weather)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f} seconds.\n")

    result.summary()

    print("\n--- Monthly summary ---")
    print(result.monthly_summary().round(2).to_string())

    result.save("results/simulation_east_facing")
    print("\nResults saved to results/simulation_east_facing/")
    print("\nPipeline complete!")
