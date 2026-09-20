"""Test: thermal model and efficiency curves."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from thermal_model import IGFPCThermalModel, BifacialThermalAnalysis


def test_steady_state_efficiency():
    """Test efficiency is between 0 and eta_0."""
    model = IGFPCThermalModel()
    G = np.array([200.0, 400.0, 800.0, 1000.0])
    Tm = np.full(4, 65.0)
    Ta = np.full(4, 20.0)
    eta = model.steady_state_efficiency(G, Tm, Ta)

    assert np.all(eta >= 0.0), f"Negative efficiency: {eta}"
    assert np.all(eta <= model.eta_0), f"Efficiency exceeds eta_0: {eta}"
    print(f"  ✅ Efficiency in valid range [0, {model.eta_0}]")
    print(f"     Values: {eta}")


def test_thermal_power():
    """Test thermal power is non-negative."""
    model = IGFPCThermalModel()
    G_front = np.array([300.0, 600.0, 800.0, 400.0])
    G_rear = np.array([250.0, 450.0, 600.0, 300.0])
    Tm = np.full(4, 65.0)
    Ta = np.full(4, 20.0)

    Q = model.thermal_power(G_front, G_rear, Tm, Ta)
    assert np.all(Q >= 0.0), f"Negative thermal power: {Q}"
    print(f"  ✅ Thermal power non-negative")
    print(f"     Values: {Q}")


def test_effective_irradiance():
    """Test G_eff = G_front + rear_optical_factor * G_rear."""
    model = IGFPCThermalModel(rear_optical_factor=0.85)
    G_front = np.array([500.0, 700.0])
    G_rear = np.array([300.0, 400.0])

    G_eff = model.effective_irradiance(G_front, G_rear)
    expected = G_front + 0.85 * G_rear
    assert np.allclose(G_eff, expected), f"G_eff mismatch: {G_eff} vs {expected}"
    print(f"  ✅ Effective irradiance correct")
    print(f"     G_eff: {G_eff}")


def test_simulate():
    """Test full thermal simulation."""
    model = IGFPCThermalModel()
    df = pd.DataFrame({
        "G_front": [300.0, 600.0, 800.0, 400.0],
        "G_rear":  [250.0, 450.0, 600.0, 300.0],
        "temp_air": [15.0, 22.0, 28.0, 20.0],
    })
    results = model.simulate(df, T_mean_fluid=65.0)

    assert "Q_bifacial_W" in results.columns
    assert "Q_monofacial_W" in results.columns
    assert "efficiency" in results.columns
    assert (results["Q_bifacial_W"] >= 0.0).all()
    assert (results["Q_monofacial_W"] >= 0.0).all()

    # Bifacial yield should exceed monofacial
    bifacial_total = results["Q_bifacial_kWh"].sum()
    monofacial_total = results["Q_monofacial_kWh"].sum()
    assert bifacial_total > monofacial_total, "Bifacial should exceed monofacial"

    print(f"  ✅ Thermal simulation valid")
    print(f"     Bifacial total: {bifacial_total:.3f} kWh")
    print(f"     Monofacial total: {monofacial_total:.3f} kWh")


def test_temperature_sensitivity():
    """Test temperature sweep produces sensible results."""
    model = IGFPCThermalModel()
    df = pd.DataFrame({
        "G_front": [500.0, 700.0, 600.0, 800.0, 550.0],
        "G_rear":  [300.0, 400.0, 350.0, 500.0, 380.0],
        "temp_air": [20.0, 22.0, 25.0, 18.0, 21.0],
    })
    analysis = BifacialThermalAnalysis(base_model=model)
    sens = analysis.temperature_sensitivity(df, T_values=[40, 50, 60, 65, 70, 80])

    assert "T_mean" in sens.columns
    assert "yield_bifacial" in sens.columns
    assert len(sens) == 6
    print(f"  ✅ Temperature sensitivity sweep")
    print(sens.round(2).to_string(index=False))


def test_rear_factor_sensitivity():
    """Test rear optical factor sweep."""
    model = IGFPCThermalModel()
    df = pd.DataFrame({
        "G_front": [500.0, 700.0],
        "G_rear":  [300.0, 400.0],
        "temp_air": [20.0, 22.0],
    })
    analysis = BifacialThermalAnalysis(base_model=model)
    sens = analysis.rear_factor_sensitivity(df, factors=[0.5, 0.75, 0.85, 1.0])

    assert "rear_optical_factor" in sens.columns
    assert len(sens) == 4
    print(f"  ✅ Rear factor sensitivity sweep")
    print(sens.round(4).to_string(index=False))


def test_efficiency_curve_data():
    """Test efficiency curve data generation."""
    model = IGFPCThermalModel()
    curve = model.efficiency_curve_data(G_values=[200, 400, 600, 800, 1000], n_points=11)

    assert "G" in curve.columns
    assert "efficiency" in curve.columns
    assert "reduced_temperature" in curve.columns
    print(f"  ✅ Efficiency curve data generated ({len(curve)} rows)")


def test_kpis():
    """Test KPI calculation."""
    model = IGFPCThermalModel()
    df = pd.DataFrame({
        "G_front": [500.0, 700.0, 600.0, 800.0],
        "G_rear":  [300.0, 400.0, 350.0, 500.0],
        "temp_air": [20.0, 22.0, 25.0, 18.0],
    })
    results = model.simulate(df, T_mean_fluid=65.0)
    kpis = model.calculate_kpis(results)

    assert "annual_yield_bifacial_kWh_per_m2" in kpis
    assert "bifacial_thermal_gain_pct" in kpis
    assert kpis["annual_yield_bifacial_kWh_per_m2"] > 0
    print(f"  ✅ KPIs calculated")
    for k, v in kpis.items():
        print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")


if __name__ == "__main__":
    print("=== Thermal Model Tests ===")
    test_steady_state_efficiency()
    test_thermal_power()
    test_effective_irradiance()
    test_simulate()
    test_temperature_sensitivity()
    test_rear_factor_sensitivity()
    test_efficiency_curve_data()
    test_kpis()
    print("\n✅ All thermal model tests passed.")
