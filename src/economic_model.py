"""
Economic model: LCOE and NPV for vertical bifacial APV-T systems.

Levelized Cost of Energy (LCOE) and Net Present Value (NPV)
over the project lifetime, accounting for capital costs,
operating costs, degradation, and discounting.

Reference: Dufo-López & Bernal-Agustín (2019), German solar thermal cost data.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EconomicConfig:
    """Economic parameters for APV-T system."""
    capex_per_m2: float = 500  # EUR/m² (large-scale)
    opex_annual_pct: float = 0.015  # 1.5% of CAPEX
    discount_rate: float = 0.04  # 4%
    project_lifetime: int = 25
    inflation_rate: float = 0.02
    yield_degradation_pct_per_year: float = 0.3  # Thermal degrades slower than PV


def calculate_lcoe(
    annual_thermal_yield_kwh_per_m2: float,
    economic_config: EconomicConfig = None,
) -> float:
    """
    Levelized Cost of Energy [EUR/kWh].

    LCOE = (CAPEX + NPV of OPEX) / NPV(annual yields over lifetime)
    """
    if economic_config is None:
        economic_config = EconomicConfig()

    npv_yield = 0.0
    for year in range(economic_config.project_lifetime):
        yield_factor = 1.0 - (economic_config.yield_degradation_pct_per_year / 100) * year
        yield_degraded = annual_thermal_yield_kwh_per_m2 * max(yield_factor, 0.5)
        npv_yield += yield_degraded / ((1 + economic_config.discount_rate) ** year)

    npv_opex = 0.0
    for year in range(economic_config.project_lifetime):
        opex = economic_config.capex_per_m2 * economic_config.opex_annual_pct
        opex_inflated = opex * ((1 + economic_config.inflation_rate) ** year)
        npv_opex += opex_inflated / ((1 + economic_config.discount_rate) ** year)

    total_cost = economic_config.capex_per_m2 + npv_opex
    return total_cost / npv_yield if npv_yield > 0 else np.inf


def calculate_npv(
    annual_thermal_yield_kwh_per_m2: float,
    thermal_price_eur_per_kwh: float = 0.12,
    economic_config: EconomicConfig = None,
) -> float:
    """Net Present Value over project lifetime [EUR/m²]."""
    if economic_config is None:
        economic_config = EconomicConfig()

    npv = -economic_config.capex_per_m2

    for year in range(economic_config.project_lifetime):
        yield_factor = 1.0 - (economic_config.yield_degradation_pct_per_year / 100) * year
        yield_degraded = annual_thermal_yield_kwh_per_m2 * max(yield_factor, 0.5)
        revenue = yield_degraded * thermal_price_eur_per_kwh
        opex = economic_config.capex_per_m2 * economic_config.opex_annual_pct
        opex_inflated = opex * ((1 + economic_config.inflation_rate) ** year)
        cash_flow = revenue - opex_inflated
        npv += cash_flow / ((1 + economic_config.discount_rate) ** year)

    return npv


def calculate_combined_land_npv(
    annual_thermal_yield_kwh_per_m2: float,
    crop_yield_t_per_ha: float,
    row_spacing: float,
    module_height: float,
    thermal_price_eur_per_kwh: float = 0.12,
    crop_price_eur_per_t: float = 250.0,
    crop_opex_eur_per_ha: float = 650.0,
    economic_config: EconomicConfig = None,
) -> dict:
    """
    Combined dual-use land NPV [EUR/ha].

    Combines agricultural crop margin with solar thermal revenues.
    """
    if economic_config is None:
        economic_config = EconomicConfig()

    # Thermal NPV per m² of collector
    npv_th_m2 = calculate_npv(
        annual_thermal_yield_kwh_per_m2=annual_thermal_yield_kwh_per_m2,
        thermal_price_eur_per_kwh=thermal_price_eur_per_kwh,
        economic_config=economic_config,
    )

    # Collector aperture area per hectare of land
    coll_density_m2_ha = (module_height / row_spacing) * 10000.0 if row_spacing > 0 else 0.0
    npv_th_ha = npv_th_m2 * coll_density_m2_ha

    # Crop NPV per hectare
    npv_crop_ha = 0.0
    for year in range(economic_config.project_lifetime):
        revenue_crop = crop_yield_t_per_ha * crop_price_eur_per_t
        opex_crop_inf = crop_opex_eur_per_ha * ((1 + economic_config.inflation_rate) ** year)
        margin_crop = revenue_crop - opex_crop_inf
        npv_crop_ha += margin_crop / ((1 + economic_config.discount_rate) ** year)

    npv_total_ha = npv_th_ha + npv_crop_ha

    return {
        "npv_thermal_eur_per_m2": float(npv_th_m2),
        "coll_density_m2_per_ha": float(coll_density_m2_ha),
        "npv_thermal_eur_per_ha": float(npv_th_ha),
        "npv_crop_eur_per_ha": float(npv_crop_ha),
        "npv_total_eur_per_ha": float(npv_total_ha),
    }


# ============================================================================= #
# DEMO
# ============================================================================= #
if __name__ == "__main__":
    config = EconomicConfig()
    annual_yield = 351  # kWh/m²/year (base case bifacial result)

    lcoe = calculate_lcoe(annual_yield, config)
    npv = calculate_npv(annual_yield, thermal_price_eur_per_kwh=0.12, economic_config=config)

    print("=" * 70)
    print("ECONOMIC ANALYSIS: Vertical Bifacial APV-T (Base Case)")
    print("=" * 70)
    print(f"  Annual thermal yield: {annual_yield:.1f} kWh/m²/year")
    print(f"  CAPEX:               {config.capex_per_m2:.0f} EUR/m²")
    print(f"  Project lifetime:    {config.project_lifetime} years")
    print(f"  Discount rate:       {config.discount_rate*100:.0f}%")
    print(f"  LCOE:                {lcoe:.3f} EUR/kWh")
    print(f"  NPV (@ 0.12 EUR/kWh): {npv:.1f} EUR/m²")
    print(f"  Thermal cost:        {lcoe:.3f} EUR/kWh to produce")
    print(f"  25-year profit:      {npv:.0f} EUR/m²")
    print("=" * 70)
