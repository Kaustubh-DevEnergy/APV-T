"""
Parametric sensitivity analysis for vertical bifacial APV-T systems.

Sweeps over key design variables:
  - Row spacing (2–8 m)
  - Ground albedo (0.15–0.55)
  - Hub height (1.0–2.0 m)
  - Surface azimuth (90–270°)
  - Fluid temperature (40–80°C)
  - Collector tilt (70–90° from horizontal)
  - Module height
  - 2D grid: row spacing × albedo
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    SiteConfig,
    IGFPCConfig,
    APVConfig,
    ThermalOperatingConfig,
    CropConfig,
)
from simulation import APVTSimulation
from economic_model import calculate_lcoe, calculate_npv, calculate_combined_land_npv, EconomicConfig
import dataclasses


@dataclass
class ParametricAnalysis:
    """
    Parametric sensitivity analysis on APV-T system.

    Attributes
    ----------
    site_config : SiteConfig
        Site configuration (location, weather)
    base_collector_config : IGFPCConfig
        Collector configuration (parameters to vary)
    base_apv_config : APVConfig
        APV configuration (parameters to vary)
    base_thermal_config : ThermalOperatingConfig
        Thermal operating configuration
    base_crop_config : CropConfig
        Crop configuration
    weather_df : pd.DataFrame
        Hourly weather data (loaded once, reused)
    monthly_albedo : dict, optional
        Monthly albedo variation
    """
    site_config: SiteConfig
    base_collector_config: IGFPCConfig
    base_apv_config: APVConfig
    base_thermal_config: ThermalOperatingConfig
    base_crop_config: CropConfig
    weather_df: pd.DataFrame
    monthly_albedo: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if self.weather_df is None or len(self.weather_df) == 0:
            raise ValueError("weather_df cannot be empty")

    # ------------------------------------------------------------------ #
    def single_sweep(self, param_name: str, param_values: list) -> pd.DataFrame:
        """Sweep a single parameter while keeping all others at base values."""
        results = []
        for value in param_values:
            print(f"  Running {param_name}={value}...")

            if param_name == "row_spacing":
                modified_collector = dataclasses.replace(
                    self.base_collector_config, row_spacing=value
                )
                modified_apv = self.base_apv_config
            elif param_name == "hub_height":
                modified_collector = dataclasses.replace(
                    self.base_collector_config, hub_height=value
                )
                modified_apv = self.base_apv_config
            elif param_name == "albedo":
                modified_collector = self.base_collector_config
                modified_apv = dataclasses.replace(self.base_apv_config, albedo=value)
            elif param_name == "surface_azimuth":
                modified_collector = self.base_collector_config
                modified_apv = dataclasses.replace(self.base_apv_config, surface_azimuth=value)
            elif param_name == "T_mean_fluid":
                T_mean = value
                T_inlet = T_mean - 15
                T_outlet = T_mean + 15
                modified_collector = self.base_collector_config
                modified_apv = self.base_apv_config
                modified_thermal = dataclasses.replace(
                    self.base_thermal_config, T_inlet=T_inlet, T_outlet=T_outlet
                )
            elif param_name == "surface_tilt":
                modified_collector = self.base_collector_config
                modified_apv = dataclasses.replace(self.base_apv_config, surface_tilt=value)
            elif param_name == "module_height":
                modified_collector = dataclasses.replace(
                    self.base_collector_config, module_height=value
                )
                modified_apv = self.base_apv_config
            else:
                raise ValueError(f"Unknown parameter: {param_name}")

            if param_name != "T_mean_fluid":
                modified_thermal = self.base_thermal_config

            sim = APVTSimulation(
                site_config=self.site_config,
                apv_config=modified_apv,
                collector_config=modified_collector,
                thermal_config=modified_thermal,
                crop_config=self.base_crop_config,
                monthly_albedo=self.monthly_albedo,
            )
            result = sim.run(self.weather_df)
            kpis = result.to_dict()
            kpis[param_name] = value
            results.append(kpis)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    def grid_sweep_2d(
        self,
        param1: str,
        values1: list,
        param2: str,
        values2: list,
    ) -> pd.DataFrame:
        """2D grid sweep over two parameters."""
        results = []
        total = len(values1) * len(values2)
        count = 0

        for val1 in values1:
            for val2 in values2:
                count += 1
                print(f"  Running {param1}={val1}, {param2}={val2}... ({count}/{total})")

                if param1 == "row_spacing":
                    mod_collector = dataclasses.replace(self.base_collector_config, row_spacing=val1)
                else:
                    mod_collector = self.base_collector_config

                if param1 == "albedo":
                    mod_apv = dataclasses.replace(self.base_apv_config, albedo=val1)
                elif param2 == "albedo":
                    mod_apv = dataclasses.replace(self.base_apv_config, albedo=val2)
                else:
                    mod_apv = self.base_apv_config

                sim = APVTSimulation(
                    site_config=self.site_config,
                    apv_config=mod_apv,
                    collector_config=mod_collector,
                    thermal_config=self.base_thermal_config,
                    crop_config=self.base_crop_config,
                    monthly_albedo=self.monthly_albedo,
                )
                result = sim.run(self.weather_df)
                kpis = result.to_dict()
                kpis[param1] = val1
                kpis[param2] = val2
                results.append(kpis)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    def sweep_collector_tilt(self, tilt_values: list = None) -> pd.DataFrame:
        """Sweep collector tilt angle (90 deg = vertical, 70 deg = 20 deg off-vertical)."""
        if tilt_values is None:
            tilt_values = [90.0, 85.0, 80.0, 75.0, 70.0]

        results = []
        for tilt in tilt_values:
            off_vert = 90.0 - tilt if tilt <= 90.0 else tilt - 90.0
            print(f"  Running surface_tilt={tilt}° ({off_vert:.0f} deg from vertical)...")
            modified_config = dataclasses.replace(self.base_apv_config, surface_tilt=tilt)

            sim = APVTSimulation(
                site_config=self.site_config,
                apv_config=modified_config,
                collector_config=self.base_collector_config,
                thermal_config=self.base_thermal_config,
                crop_config=self.base_crop_config,
                monthly_albedo=self.monthly_albedo,
            )
            result = sim.run(self.weather_df)
            kpis = result.to_dict()
            kpis["surface_tilt"] = tilt
            kpis["off_vertical_tilt"] = off_vert
            results.append(kpis)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    def _enrich_economics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add LCOE and combined dual-use land NPV metrics to sweep results."""
        econ_config = EconomicConfig()
        lcoe_list = []
        npv_th_ha_list = []
        npv_crop_ha_list = []
        npv_total_ha_list = []

        for _, row in df.iterrows():
            y_th = float(row.get("annual_thermal_yield_bifacial_kWh_per_m2", 0.0))
            y_crop = float(row.get("estimated_crop_yield_t_per_ha", 0.0))
            spacing = float(row.get("row_spacing", self.base_collector_config.row_spacing))
            h = float(row.get("module_height", self.base_collector_config.module_height))

            lcoe_val = calculate_lcoe(y_th, econ_config)
            land_econ = calculate_combined_land_npv(
                annual_thermal_yield_kwh_per_m2=y_th,
                crop_yield_t_per_ha=y_crop,
                row_spacing=spacing,
                module_height=h,
                economic_config=econ_config,
            )
            lcoe_list.append(lcoe_val)
            npv_th_ha_list.append(land_econ["npv_thermal_eur_per_ha"])
            npv_crop_ha_list.append(land_econ["npv_crop_eur_per_ha"])
            npv_total_ha_list.append(land_econ["npv_total_eur_per_ha"])

        df["lcoe"] = lcoe_list
        df["npv_thermal_eur_per_ha"] = npv_th_ha_list
        df["npv_crop_eur_per_ha"] = npv_crop_ha_list
        df["npv_total_eur_per_ha"] = npv_total_ha_list
        return df

    # ------------------------------------------------------------------ #
    def full_study(self) -> Dict[str, pd.DataFrame]:
        """Run complete parametric study with all sweeps. 55 simulations total."""
        print("\n" + "=" * 62)
        print("FULL PARAMETRIC STUDY")
        print("=" * 62)
        print("Total planned simulations: 55 (22 single + 5 tilt + 4 height + 24 grid)")
        print("Estimated runtime: ~82 s (~1.4 min) at ~1.5 s/run\n")

        all_results = {}

        # 1. Row spacing
        print("--- Sweep: row_spacing (6 values) ---")
        t0 = time.time()
        row_spacing_sweep = self.single_sweep("row_spacing", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
        all_results["row_spacing"] = self._enrich_economics(row_spacing_sweep)
        print(f"  -> row_spacing sweep complete: 6 runs in {time.time()-t0:.1f}s")

        # 2. Albedo
        print("\n--- Sweep: albedo (4 values) ---")
        t0 = time.time()
        albedo_sweep = self.single_sweep("albedo", [0.15, 0.25, 0.4, 0.55])
        all_results["albedo"] = self._enrich_economics(albedo_sweep)
        print(f"  -> albedo sweep complete: 4 runs in {time.time()-t0:.1f}s")

        # 3. Hub height
        print("\n--- Sweep: hub_height (3 values) ---")
        t0 = time.time()
        hub_height_sweep = self.single_sweep("hub_height", [1.0, 1.5, 2.0])
        all_results["hub_height"] = self._enrich_economics(hub_height_sweep)
        print(f"  -> hub_height sweep complete: 3 runs in {time.time()-t0:.1f}s")

        # 4. Surface azimuth
        print("\n--- Sweep: surface_azimuth (5 values) ---")
        t0 = time.time()
        azimuth_sweep = self.single_sweep("surface_azimuth", [90, 135, 180, 225, 270])
        all_results["surface_azimuth"] = self._enrich_economics(azimuth_sweep)
        print(f"  -> surface_azimuth sweep complete: 5 runs in {time.time()-t0:.1f}s")

        # 5. Temperature
        print("\n--- Sweep: T_mean_fluid (4 values) ---")
        t0 = time.time()
        temp_sweep = self.single_sweep("T_mean_fluid", [40, 50, 65, 80])
        all_results["T_mean_fluid"] = self._enrich_economics(temp_sweep)
        print(f"  -> T_mean_fluid sweep complete: 4 runs in {time.time()-t0:.1f}s")

        # 6. Collector tilt
        print("\n--- Sweep: surface_tilt (5 values: 90° down to 70°) ---")
        t0 = time.time()
        tilt_sweep = self.sweep_collector_tilt([90.0, 85.0, 80.0, 75.0, 70.0])
        all_results["surface_tilt"] = self._enrich_economics(tilt_sweep)
        print(f"  -> surface_tilt sweep complete: 5 runs in {time.time()-t0:.1f}s")

        # 7. Module height
        print("\n--- Sweep: module_height (4 values) ---")
        t0 = time.time()
        module_height_sweep = self.single_sweep("module_height", [1.5, 2.0, 2.5, 2.814])
        all_results["module_height"] = self._enrich_economics(module_height_sweep)
        print(f"  -> module_height sweep complete: 4 runs in {time.time()-t0:.1f}s")

        # 8. 2D grid: row_spacing × albedo
        print("\n--- 2-D grid sweep: row_spacing x albedo (6 x 4 = 24 runs) ---")
        t0 = time.time()
        grid_sweep = self.grid_sweep_2d(
            "row_spacing", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            "albedo", [0.15, 0.25, 0.4, 0.55],
        )
        all_results["row_spacing_x_albedo"] = self._enrich_economics(grid_sweep)
        print(f"  -> row_spacing x albedo grid sweep complete: 24 runs in {time.time()-t0:.1f}s")

        print("\n" + "=" * 62)
        print("FULL STUDY COMPLETE: 55 simulations")
        print("=" * 62)

        # Key findings
        print("\n" + "=" * 62)
        print("KEY FINDINGS")
        print("=" * 62)

        for name, table in all_results.items():
            if name == "row_spacing_x_albedo":
                continue
            if "LER" not in table.columns:
                continue

            best_row = table.loc[table["LER"].idxmax()]
            worst_row = table.loc[table["LER"].idxmin()]
            param_col = name if name in table.columns else [
                col for col in table.columns
                if col not in [
                    "annual_ghi_kWh_per_m2", "annual_front_irradiance_kWh_per_m2",
                    "annual_rear_irradiance_kWh_per_m2", "annual_total_irradiance_kWh_per_m2",
                    "bifacial_irradiance_gain_pct", "reference_thermal_yield_kWh_per_m2",
                    "LER", "annual_thermal_yield_bifacial_kWh_per_m2",
                    "bifacial_thermal_gain_pct", "estimated_crop_yield_t_per_ha", "lcoe",
                ]
            ][0]

            print(f"\n  [{name}]")
            print(f"    Best  LER = {best_row['LER']:.3f}  at {param_col}={best_row[param_col]}")
            print(f"      Thermal: {best_row.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A')} kWh/m²")
            print(f"      Crop:    {best_row.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")
            print(f"    Worst LER = {worst_row['LER']:.3f}  at {param_col}={worst_row[param_col]}")

        # Grid summary
        grid = all_results["row_spacing_x_albedo"]
        best_grid = grid.loc[grid["LER"].idxmax()]
        worst_grid = grid.loc[grid["LER"].idxmin()]

        print(f"\n  [row_spacing_x_albedo]")
        print(f"    Best  LER = {best_grid['LER']:.3f}  at row_spacing={best_grid['row_spacing']}, albedo={best_grid['albedo']}")
        print(f"      Thermal: {best_grid.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A')} kWh/m²")
        print(f"      Crop:    {best_grid.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")
        print(f"    Worst LER = {worst_grid['LER']:.3f}  at row_spacing={worst_grid['row_spacing']}, albedo={worst_grid['albedo']}")

        print("\n" + "=" * 62)
        print("OPTIMAL CONFIGURATION")
        print("=" * 62)

        all_data = pd.concat(all_results.values(), ignore_index=True)
        viable = all_data[all_data["estimated_crop_yield_t_per_ha"] > 4.0]
        optimal = viable.loc[viable["LER"].idxmax()].copy()

        opt_spacing = optimal.get("row_spacing", self.base_collector_config.row_spacing)
        opt_albedo = optimal.get("albedo", self.base_apv_config.albedo)
        if pd.isna(opt_spacing):
            opt_spacing = self.base_collector_config.row_spacing
        if pd.isna(opt_albedo):
            opt_albedo = self.base_apv_config.albedo

        print(f"\n  Max LER with crop yield > 4.0 t/ha:")
        print(f"    Row spacing:  {opt_spacing} m")
        print(f"    Albedo:       {opt_albedo}")
        print(f"    LER:          {optimal['LER']:.3f}")
        print(f"    Thermal:      {optimal.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A'):.0f} kWh/m²")
        print(f"    Crop yield:   {optimal.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")
        print(f"    BG (irr):     {optimal.get('bifacial_irradiance_gain_pct', 'N/A'):.1f}%")
        print(f"    BG (thermal): {optimal.get('bifacial_thermal_gain_pct', 'N/A'):.1f}%")

        # Save all results
        self._save_all(all_results)

        return all_results

    # ------------------------------------------------------------------ #
    def _save_all(self, all_results: Dict[str, pd.DataFrame]) -> None:
        """Save all results to CSV files."""
        output_dir = Path("results/parametric_study")
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, table in all_results.items():
            table.to_csv(output_dir / f"{name}.csv", index=False)

        # Summary
        summary_data = []
        for name, table in all_results.items():
            if name != "row_spacing_x_albedo":
                best = table.loc[table["LER"].idxmax()]
                summary_data.append({
                    "sweep": name,
                    "best_ler": best["LER"],
                    "thermal_kwh_m2": best.get("annual_thermal_yield_bifacial_kWh_per_m2", np.nan),
                    "crop_t_ha": best.get("estimated_crop_yield_t_per_ha", np.nan),
                    "lcoe_eur_kwh": best.get("lcoe", np.nan),
                    "npv_total_eur_ha": best.get("npv_total_eur_per_ha", np.nan),
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

        print(f"\nSaved 7 sweep CSV files (+ summary.csv) to '{output_dir}'.")


# ============================================================================= #
# DEMO
# ============================================================================= #
if __name__ == "__main__":
    from utils import load_pvgis_tmy
    from config import (
        INGOLSTADT, IGFPCConfig, APVConfig, ThermalOperatingConfig,
        CropConfig, MONTHLY_ALBEDO_INGOLSTADT,
    )

    print("Loading weather data (PVGIS TMY) for Ingolstadt...")
    weather = load_pvgis_tmy(INGOLSTADT.tmy_file)
    print(f"Loaded {len(weather)} hourly weather records.")

    analysis = ParametricAnalysis(
        site_config=INGOLSTADT,
        base_collector_config=IGFPCConfig(),
        base_apv_config=APVConfig(surface_azimuth=90.0),
        base_thermal_config=ThermalOperatingConfig(),
        base_crop_config=CropConfig(),
        weather_df=weather,
        monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
    )

    results = analysis.full_study()

    print("\n" + "=" * 62)
    print("PARAMETRIC STUDY COMPLETE!")
    print("=" * 62)
