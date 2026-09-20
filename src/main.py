"""
Master APV-T simulation and analysis workflow.
Single command runs: simulation → parametric study → plotting → validation → economics.
"""

from pathlib import Path
import time
from utils import load_pvgis_tmy
from config import (
    INGOLSTADT, IGFPCConfig, APVConfig, ThermalOperatingConfig,
    CropConfig, MONTHLY_ALBEDO_INGOLSTADT,
)
from simulation import APVTSimulation
from parametric_study import ParametricAnalysis
from thermal_model import IGFPCThermalModel
from economic_model import calculate_lcoe, calculate_npv, EconomicConfig
import plotting
import validation
import pandas as pd
import numpy as np


def main():
    print("=" * 80)
    print("APV-T BIFACIAL AGRIVOLTAIC-THERMAL SIMULATION & ANALYSIS WORKFLOW")
    print("=" * 80)

    output_base = Path("results").resolve()
    figures_dir = output_base / "figures"
    t_start = time.time()

    # 1. LOAD WEATHER
    print("\n[1/6] Loading weather data (PVGIS TMY)...")
    weather = load_pvgis_tmy(INGOLSTADT.tmy_file)
    print(f"      ✅ Loaded {len(weather)} hourly records for Ingolstadt")

    # 2. RUN BASE SIMULATION
    print("\n[2/6] Running base APV-T simulation (east-facing, 4m spacing, Tm=65°C)...")
    t0 = time.time()
    sim = APVTSimulation(
        site_config=INGOLSTADT,
        apv_config=APVConfig(surface_azimuth=90.0),
        collector_config=IGFPCConfig(),
        thermal_config=ThermalOperatingConfig(),
        crop_config=CropConfig(),
        monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
    )
    result = sim.run(weather)
    result.save(output_base / "simulation_east_facing")
    print(f"      ✅ Simulation complete ({time.time()-t0:.1f}s)")

    kpis = result.to_dict()
    print(f"         Annual thermal bifacial: {kpis.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A'):.1f} kWh/m²")
    print(f"         Bifacial irradiance gain: {kpis.get('bifacial_irradiance_gain_pct', 'N/A'):.1f}%")
    print(f"         Estimated crop yield: {kpis.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")
    print(f"         LER: {kpis.get('LER', 'N/A'):.3f}")

    # 3. RUN PARAMETRIC STUDY
    print("\n[3/6] Running parametric sensitivity analysis (55 configurations)...")
    t0 = time.time()
    analysis = ParametricAnalysis(
        site_config=INGOLSTADT,
        base_collector_config=IGFPCConfig(),
        base_apv_config=APVConfig(surface_azimuth=90.0),
        base_thermal_config=ThermalOperatingConfig(),
        base_crop_config=CropConfig(),
        weather_df=weather,
        monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
    )
    param_results = analysis.full_study()
    print(f"      ✅ Parametric study complete ({time.time()-t0:.1f}s)")

    # 3b. Generate design window summary
    print("\n[3b/6] Generating design window summary...")
    from simulation import summarize_design_window, print_design_window
    design_window = summarize_design_window(
        param_results,
        ler_min=1.1,
        crop_yield_min=4.0,
        thermal_yield_min=200.0,
    )
    print_design_window(design_window)

    # Save design window to JSON
    import json
    design_window_clean = {
        k: (None if isinstance(v, float) and (np.isnan(v) if hasattr(v, '__float__') else False) else v)
        for k, v in design_window.items()
    }
    with open(output_base / "design_window.json", "w") as f:
        json.dump(design_window_clean, f, indent=2, default=str)

    # 4. GENERATE PLOTS
    print("\n[4/6] Generating publication-quality figures...")
    t0 = time.time()
    figures_dir = output_base / "figures"
    thermal_model = IGFPCThermalModel()

    figures = plotting.generate_all_plots(
        hourly_df=result.hourly_results,
        parametric_results=param_results,
        thermal_model=thermal_model,
        output_dir=figures_dir,
    )
    print(f"      ✅ Generated {len(figures)} figures ({time.time()-t0:.1f}s)")

    # 5. GENERATE VALIDATION REPORT & MULTI-SITE COMPARISON
    print("\n[5/6] Generating literature-comparison and multi-site validation report...")
    t0 = time.time()
    validation_tables = validation.generate_validation_report(
        simulation_result=result,
        parametric_results=param_results,
        output_dir=output_base,
    )
    multisite_summary = validation.run_multisite_comparison(
        output_dir=output_base / "validation"
    )
    print(f"      ✅ Literature & Multi-site validation complete ({time.time()-t0:.1f}s)")

    # 6. RAY-TRACING VALIDATION
    print("\n[6/6] Genuine ray-tracing validation (VF vs bifacial_radiance)...")
    t0 = time.time()
    try:
        from validation_raytracing import generate_raytracing_validation_report
        rt_results = generate_raytracing_validation_report(
            weather_df=weather,
            site_config=INGOLSTADT,
            apv_config=APVConfig(surface_azimuth=90.0),
            collector_config=IGFPCConfig(),
            output_dir=output_base / "validation",
        )
        print(f"      ✅ Ray-tracing validation complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"      ⚠️  Ray-tracing validation failed: {e}")
        print(f"         (This is optional - install bifacial-radiance if needed)")

    # SUMMARY
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nTotal runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  📊 {len(list(figures_dir.glob('*.png')))} figures in {figures_dir}/")
    print(f"  📊 {len(list((output_base / 'validation').glob('*.csv')))} validation CSVs in {output_base}/validation/")
    print(f"  📊 {len(list((output_base / 'validation').glob('*.png')))} validation PNGs in {output_base}/validation/")
    print(f"  📊 {len(list((output_base / 'parametric_study').glob('*.csv')))} parametric sweep CSVs in {output_base}/parametric_study/")

    print(f"\nKey results (base case, east-facing, 4m spacing):")
    print(f"  • Thermal yield (bifacial): {kpis.get('annual_thermal_yield_bifacial_kWh_per_m2', 'N/A'):.1f} kWh/m²")
    print(f"  • Thermal yield (monofacial): {kpis.get('annual_thermal_yield_monofacial_kWh_per_m2', 'N/A'):.1f} kWh/m²")
    print(f"  • Bifacial thermal gain: {kpis.get('bifacial_thermal_gain_pct', 'N/A'):.1f}%")
    print(f"  • Bifacial irradiance gain: {kpis.get('bifacial_irradiance_gain_pct', 'N/A'):.1f}%")
    print(f"  • Land Equivalent Ratio: {kpis.get('LER', 'N/A'):.3f}")
    print(f"  • Estimated crop yield: {kpis.get('estimated_crop_yield_t_per_ha', 'N/A'):.2f} t/ha")

    print(f"\nDesign Window Summary:")
    print(f"  • Viable configurations: {design_window['viable_configurations']}/{design_window['total_configurations_tested']}")
    if design_window['optimal_configuration']:
        opt = design_window['optimal_configuration']
        print(f"  • Optimal LER: {opt.get('LER', 'N/A'):.3f}")

    # Economic performance
    econ_config = EconomicConfig()
    lcoe = calculate_lcoe(kpis.get('annual_thermal_yield_bifacial_kWh_per_m2', 0), econ_config)
    npv = calculate_npv(kpis.get('annual_thermal_yield_bifacial_kWh_per_m2', 0),
                        thermal_price_eur_per_kwh=0.12, economic_config=econ_config)

    print(f"\nEconomic Performance:")
    print(f"  • LCOE (bifacial): {lcoe:.3f} EUR/kWh")
    print(f"  • NPV (25y @ 0.12 EUR/kWh): {npv:.1f} EUR/m²")

    print("\n" + "=" * 80)
    print("Ready for thesis documentation! 🎓")
    print("=" * 80)


if __name__ == "__main__":
    main()
