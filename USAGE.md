# How to Use

## Run the Full Pipeline

```bash
python src/main.py
```

This runs everything in order:
1. Loads weather data for Ingolstadt
2. Runs the base APV-T simulation
3. Performs 55-parametric sensitivity study
4. Generates design window summary
5. Creates publication figures
6. Runs literature validation
7. Runs multi-site comparison (Germany, Belgium, Sweden)
8. Computes economics (LCOE, NPV)

Output goes to `results/`.

## Run a Custom Simulation

```python
from pathlib import Path
from config import SiteConfig, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig, MONTHLY_ALBEDO_INGOLSTADT
from simulation import APVTSimulation
from utils import load_pvgis_tmy

site = SiteConfig(
    name="My Site",
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
)

apv = APVConfig(
    surface_azimuth=90.0,  # East-facing
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

weather = load_pvgis_tmy(site.tmy_file)
sim = APVTSimulation(
    site_config=site,
    apv_config=apv,
    collector_config=collector,
    thermal_config=thermal,
    crop_config=crop,
    monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
)

result = sim.run(weather)
result.save(Path("results/my_simulation"))

# Print KPIs
kpis = result.to_dict()
print(f"Thermal yield: {kpis['annual_thermal_yield_bifacial_kWh_per_m2']:.1f} kWh/m²")
print(f"LER: {kpis['LER']:.3f}")
print(f"Crop yield: {kpis['estimated_crop_yield_t_per_ha']:.2f} t/ha")
```

## Run a Single Parameter Sweep

```python
from parametric_study import ParametricAnalysis
from config import INGOLSTADT, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig
from utils import load_pvgis_tmy

weather = load_pvgis_tmy(INGOLSTADT.tmy_file)

analysis = ParametricAnalysis(
    site_config=INGOLSTADT,
    base_collector_config=IGFPCConfig(),
    base_apv_config=APVConfig(surface_azimuth=90.0),
    base_thermal_config=ThermalOperatingConfig(),
    base_crop_config=CropConfig(),
    weather_df=weather,
    monthly_albedo=MONTHLY_ALBEDO_INGOLSTADT,
)

# Sweep row spacing
results = analysis.single_sweep("row_spacing", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
print(results[["row_spacing", "LER", "annual_thermal_yield_bifacial_kWh_per_m2"]])
```

## Output Files

| Path | Contents |
|------|----------|
| `results/simulation_east_facing/hourly_results.csv` | Hourly time series |
| `results/simulation_east_facing/kpis.json` | Summary KPIs |
| `results/figures/*.png` | Publication plots (300 DPI) |
| `results/parametric_study/*.csv` | Sweep results |
| `results/validation/*.csv` | Validation tables |
| `results/design_window.json` | Optimal design ranges |
