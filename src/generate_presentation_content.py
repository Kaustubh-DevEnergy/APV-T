"""
Generate presentation content from simulation results.

Reads the simulation output and creates slide-ready summaries.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from pathlib import Path


def generate_presentation_content(results_dir: str = "results") -> dict:
    """
    Extract key findings from simulation results for presentation slides.
    """
    results_path = Path(results_dir)

    # Load KPI data
    kpi_files = list(results_path.glob("**/kpis.json"))
    if not kpi_files:
        return {"error": "No kpis.json found. Run main.py first."}

    all_kpis = []
    for kpi_file in kpi_files:
        with open(kpi_file, "r") as f:
            all_kpis.append(json.load(f))

    # Extract summary stats
    base_kpis = all_kpis[0] if all_kpis else {}

    content = {
        "title": "APV-T Bifacial Agrivoltaic-Thermal Simulation",
        "key_findings": {
            "annual_thermal_yield": base_kpis.get("annual_thermal_yield_bifacial_kWh_per_m2", "N/A"),
            "bifacial_gain": base_kpis.get("bifacial_irradiance_gain_pct", "N/A"),
            "thermal_gain": base_kpis.get("bifacial_thermal_gain_pct", "N/A"),
            "crop_yield": base_kpis.get("estimated_crop_yield_t_per_ha", "N/A"),
            "ler": base_kpis.get("LER", "N/A"),
        },
        "configurations_tested": len(list(results_path.glob("parametric_study/*.csv"))),
        "figures_generated": len(list(results_path.glob("figures/*.png"))),
    }

    return content


if __name__ == "__main__":
    content = generate_presentation_content()
    print(json.dumps(content, indent=2, default=str))
