"""
Run a custom simulation with user-defined parameters.

Edit the parameters in the main() function, then run:
    python src/run_custom_simulation.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from config import SiteConfig, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig, MONTHLY_ALBEDO_INGOLSTADT
from simulation import APVTSimulation
from utils import load_pvgis_tmy
from economic_model import calculate_lcoe, EconomicConfig


def main():
    # --- Define your custom scenario --- #
    site = SiteConfig(
        name="Ingolstadt, Germany",
        latitude=48.76,
        longitude=11.42,
        altitude=374,
        timezone="Europe/Berlin",
        tmy_file=Path("data/tmy_ingolstadt.csv"),
    )

    collector = IGFPCConfig(
        module_height=2.814,
        row_spacing=4.0,
        hub_height=2.0,
        eta_0=0.856,
        a_1=4.069,
        a_2=0.009,
    )

    apv = APVConfig(
        surface_azimuth=90.0,  # East-facing vertical
        albedo=0.25,
    )

    thermal = ThermalOperatingConfig(
        T_inlet=50.0,
        T_outlet=80.0,
    )

    crop = CropConfig(
        crop_type="strawberry",
        reference_yield=6.0,
    )

    # --- Load weather and run --- #
    print("Loading weather data...")
    weather = load_pvgis_tmy(site.tmy_file)
    print(f"Loaded {len(weather)} hourly records.\n")

    sim = APVTSimulation(
        site_config=site,
        apv_config=apv,
        collector_config=collector,
        thermal_config=thermal,
        crop_config=crop,
        monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
    )

    print("Running simulation...")
    result = sim.run(weather)

    # --- Print results --- #
    kpis = result.to_dict()
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Annual GHI:                    {kpis['annual_ghi_kWh_per_m2']:.1f} kWh/m²")
    print(f"  Front POA irradiance:          {kpis['annual_front_irradiance_kWh_per_m2']:.1f} kWh/m²")
    print(f"  Rear POA irradiance:           {kpis['annual_rear_irradiance_kWh_per_m2']:.1f} kWh/m²")
    print(f"  Bifacial irradiance gain:      {kpis['bifacial_irradiance_gain_pct']:.1f}%")
    print(f"  Bifacial thermal yield:        {kpis['annual_thermal_yield_bifacial_kWh_per_m2']:.1f} kWh/m²")
    print(f"  Monofacial thermal yield:      {kpis['annual_thermal_yield_monofacial_kWh_per_m2']:.1f} kWh/m²")
    print(f"  Bifacial thermal gain:         {kpis['bifacial_thermal_gain_pct']:.1f}%")
    print(f"  Crop yield:                    {kpis['estimated_crop_yield_t_per_ha']:.2f} t/ha")
    print(f"  LER:                           {kpis['LER']:.3f}")

    # --- Economics --- #
    econ_config = EconomicConfig()
    lcoe = calculate_lcoe(kpis['annual_thermal_yield_bifacial_kWh_per_m2'], econ_config)
    print(f"\n  LCOE:                          {lcoe:.3f} EUR/kWh")

    # --- Save --- #
    result.save(Path("results/custom_simulation"))
    print(f"\nResults saved to results/custom_simulation/")


if __name__ == "__main__":
    main()
