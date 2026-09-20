"""Test: plotting functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from plotting import (
    plot_daily_irradiance_profile,
    plot_seasonal_daily_profiles,
    plot_monthly_irradiance,
    plot_monthly_thermal,
    plot_irradiance_heatmap,
    plot_efficiency_curves,
    plot_parametric_sweep,
    plot_parametric_grid_heatmap,
    plot_bifacial_gain_comparison,
    plot_temperature_sensitivity,
    plot_orientation_polar,
    generate_all_plots,
)
from thermal_model import IGFPCThermalModel


def _make_hourly_demo():
    """Create synthetic hourly data for plotting tests."""
    times = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="h", tz="Europe/Berlin")
    hour = times.hour.to_numpy()
    day = times.dayofyear.to_numpy()
    seasonal = 0.30 + 0.70 * np.sin(np.pi * (day - 1) / 365.0) ** 1.2
    solar_shape = np.maximum(np.sin(np.pi * (hour - 5) / 15.0), 0.0)
    morning_weight = np.exp(-0.5 * ((hour - 9.5) / 3.5) ** 2)
    afternoon_weight = np.exp(-0.5 * ((hour - 15.0) / 3.8) ** 2)

    G_front = 680.0 * seasonal * solar_shape * (0.55 + 0.45 * morning_weight)
    G_rear = 430.0 * seasonal * solar_shape * (0.50 + 0.50 * afternoon_weight)
    G_total = G_front + G_rear
    temp_air = 8.0 + 12.0 * np.sin(2 * np.pi * (day - 80) / 365.0)
    efficiency = np.clip(0.55 - 0.0025 * (65.0 - temp_air), 0.05, 0.65)
    Q_bifacial_W = efficiency * G_total * 12.422
    Q_monofacial_W = np.clip(efficiency * G_front * 12.422, 0.0, None)

    return pd.DataFrame(
        {
            "G_front": G_front, "G_rear": G_rear, "G_total": G_total,
            "G_beam_front": 0.65 * G_front, "G_beam_rear": 0.45 * G_rear,
            "G_diff_front": 0.23 * G_front, "G_diff_rear": 0.34 * G_rear,
            "G_gnd_front": 0.12 * G_front, "G_gnd_rear": 0.21 * G_rear,
            "bifacial_gain": np.where(G_front > 1.0, np.divide(G_rear, G_front, where=G_front>1.0), 0.0),
            "solar_elevation": 55.0 * solar_shape,
            "Q_bifacial_W": Q_bifacial_W, "Q_bifacial_kWh": Q_bifacial_W / 1000.0,
            "Q_monofacial_W": Q_monofacial_W, "Q_monofacial_kWh": Q_monofacial_W / 1000.0,
            "efficiency": efficiency, "temp_air": temp_air, "ghi": 0.85 * G_total,
        },
        index=times,
    )


def _make_parametric_demo():
    """Create synthetic parametric sweep results."""
    row_spacing = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    albedo = np.array([0.15, 0.25, 0.40, 0.55])

    spacing_sweep = pd.DataFrame({
        "row_spacing": row_spacing.astype(float),
        "annual_thermal_yield_bifacial_kWh_per_m2": (600 - 22 * row_spacing).astype(float),
        "annual_thermal_yield_monofacial_kWh_per_m2": (390 - 9 * row_spacing).astype(float),
        "LER": (1.15 + 0.08 * row_spacing - 0.008 * row_spacing ** 2).astype(float),
        "estimated_crop_yield_t_per_ha": (5.0 * (1 - 0.5 * 0.35 * 2.0 / row_spacing)).astype(float),
    })
    temp_sweep = pd.DataFrame({
        "T_mean_fluid": [40.0, 50.0, 60.0, 65.0, 70.0, 80.0],
        "annual_thermal_yield_bifacial_kWh_per_m2": [690.0, 620.0, 545.0, 510.0, 475.0, 410.0],
        "annual_thermal_yield_monofacial_kWh_per_m2": [490.0, 435.0, 375.0, 345.0, 320.0, 270.0],
        "LER": [1.64, 1.59, 1.53, 1.49, 1.45, 1.38],
    })
    orientation = pd.DataFrame({
        "surface_azimuth": [90, 135, 180, 225, 270],
        "LER": [1.48, 1.55, 1.50, 1.44, 1.46],
        "annual_total_irradiance_kWh_per_m2": [1410, 1490, 1450, 1380, 1405],
    })
    grid = pd.DataFrame([
        {"row_spacing": spacing, "albedo": alb, "LER": 1.1 + 0.12 * spacing - 0.01 * spacing ** 2 + 0.75 * alb}
        for spacing in row_spacing for alb in albedo
    ])
    return {
        "row_spacing": spacing_sweep, "T_mean_fluid": temp_sweep,
        "surface_azimuth": orientation, "row_spacing_x_albedo": grid,
    }


def test_plot_daily_irradiance_profile():
    hourly = _make_hourly_demo()
    date_str = "2026-06-21"
    fig = plot_daily_irradiance_profile(hourly, date_str)
    assert fig is not None
    print("  ✅ plot_daily_irradiance_profile")


def test_plot_seasonal_daily_profiles():
    hourly = _make_hourly_demo()
    fig = plot_seasonal_daily_profiles(hourly)
    assert fig is not None
    print("  ✅ plot_seasonal_daily_profiles")


def test_plot_monthly_irradiance():
    hourly = _make_hourly_demo()
    fig = plot_monthly_irradiance(hourly)
    assert fig is not None
    print("  ✅ plot_monthly_irradiance")


def test_plot_monthly_thermal():
    hourly = _make_hourly_demo()
    fig = plot_monthly_thermal(hourly)
    assert fig is not None
    print("  ✅ plot_monthly_thermal")


def test_plot_irradiance_heatmap():
    hourly = _make_hourly_demo()
    fig = plot_irradiance_heatmap(hourly, "G_total")
    assert fig is not None
    print("  ✅ plot_irradiance_heatmap")


def test_plot_efficiency_curves():
    thermal_model = IGFPCThermalModel()
    fig = plot_efficiency_curves(thermal_model)
    assert fig is not None
    print("  ✅ plot_efficiency_curves")


def test_plot_parametric_sweep():
    sweep_df = _make_parametric_demo()["row_spacing"]
    fig = plot_parametric_sweep(sweep_df, "row_spacing")
    assert fig is not None
    print("  ✅ plot_parametric_sweep")


def test_plot_parametric_grid_heatmap():
    grid_df = _make_parametric_demo()["row_spacing_x_albedo"]
    fig = plot_parametric_grid_heatmap(grid_df, "row_spacing", "albedo", "LER")
    assert fig is not None
    print("  ✅ plot_parametric_grid_heatmap")


def test_plot_bifacial_gain_comparison():
    hourly = _make_hourly_demo()
    fig = plot_bifacial_gain_comparison(hourly)
    assert fig is not None
    print("  ✅ plot_bifacial_gain_comparison")


def test_plot_temperature_sensitivity():
    temp_sweep = _make_parametric_demo()["T_mean_fluid"].rename(columns={
        "annual_thermal_yield_bifacial_kWh_per_m2": "yield_bifacial",
        "annual_thermal_yield_monofacial_kWh_per_m2": "yield_monofacial",
        "T_mean_fluid": "T_mean",
    })
    fig = plot_temperature_sensitivity(temp_sweep)
    assert fig is not None
    print("  ✅ plot_temperature_sensitivity")


def test_plot_orientation_polar():
    orientation = _make_parametric_demo()["surface_azimuth"]
    fig = plot_orientation_polar(orientation)
    assert fig is not None
    print("  ✅ plot_orientation_polar")


def test_generate_all_plots():
    hourly = _make_hourly_demo()
    parametric = _make_parametric_demo()
    thermal_model = IGFPCThermalModel()
    figures = generate_all_plots(hourly, parametric, thermal_model, "output/test_plots")
    assert len(figures) > 0
    print(f"  ✅ generate_all_plots ({len(figures)} figures)")


if __name__ == "__main__":
    print("=== Plotting Tests ===")
    test_plot_daily_irradiance_profile()
    test_plot_seasonal_daily_profiles()
    test_plot_monthly_irradiance()
    test_plot_monthly_thermal()
    test_plot_irradiance_heatmap()
    test_plot_efficiency_curves()
    test_plot_parametric_sweep()
    test_plot_parametric_grid_heatmap()
    test_plot_bifacial_gain_comparison()
    test_plot_temperature_sensitivity()
    test_plot_orientation_polar()
    test_generate_all_plots()
    print("\n✅ All plotting tests passed.")
