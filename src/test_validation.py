"""Test: validation functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from validation import (
    validate_irradiance_model,
    validate_thermal_model,
    validate_ler,
    validate_crop_yield,
    validate_against_sweden_barley,
    generate_validation_report,
)


def test_validate_irradiance_model():
    """Test irradiance validation table."""
    index = pd.date_range("2025-01-01", periods=8760, freq="h", tz="Europe/Berlin")
    G_front = np.maximum(np.sin(np.pi * (index.hour - 5) / 15), 0.0) * 500
    G_rear = G_front * 0.45
    hourly = pd.DataFrame({"G_front": G_front, "G_rear": G_rear}, index=index)

    table = validate_irradiance_model(hourly)
    assert "Metric" in table.columns
    assert "Value / assessment" in table.columns
    print(f"  ✅ Irradiance validation table")
    print(table.to_string(index=False))


def test_validate_thermal_model():
    """Test thermal validation table."""
    data = {
        "annual_thermal_yield_bifacial_kWh_per_m2": 340.0,
        "annual_thermal_yield_monofacial_kWh_per_m2": 205.0,
        "bifacial_thermal_gain_pct": 65.9,
    }
    table = validate_thermal_model(data)
    assert "Thermal performance" in table.columns
    print(f"  ✅ Thermal validation table")
    print(table.to_string(index=False))


def test_validate_ler():
    """Test LER validation table."""
    row_spacing = pd.DataFrame({
        "row_spacing": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "LER": [1.10, 1.31, 1.35, 1.38, 1.44, 1.42],
        "estimated_crop_yield_t_per_ha": [4.1, 4.5, 4.7, 4.8, 4.85, 4.9],
    })
    parametric = {"row_spacing": row_spacing}

    class DummyCropConfig:
        reference_yield = 6.0

    table = validate_ler(parametric, DummyCropConfig())
    assert "Row spacing (m)" in table.columns
    print(f"  ✅ LER validation table")
    print(table.to_string(index=False))


def test_validate_crop_yield():
    """Test crop yield validation table."""
    class DummyAPVConfig:
        par_reduction = 0.22

    class DummyCropConfig:
        reference_yield = 6.0

    table = validate_crop_yield(DummyAPVConfig(), DummyCropConfig(), estimated_yield_t_ha=4.7)
    assert "Crop-yield metric" in table.columns
    print(f"  ✅ Crop yield validation table")
    print(table.to_string(index=False))


def test_validate_sweden_barley():
    """Test Sweden barley validation."""
    table = validate_against_sweden_barley(
        simulated_crop_yield_t_ha=4.8,
        simulated_yield_reduction_pct=15.0,
        row_spacing_m=4.0,
    )
    assert "Metric" in table.columns
    print(f"  ✅ Sweden barley validation table")
    print(table.to_string(index=False))


def test_generate_validation_report():
    """Test full validation report generation."""
    from dataclasses import dataclass

    @dataclass
    class _DemoAPVConfig:
        albedo: float = 0.25
        surface_azimuth: float = 90.0
        rear_optical_factor: float = 0.85
        par_reduction: float = 0.22

    @dataclass
    class _DemoCropConfig:
        reference_yield: float = 6.0
        light_sensitivity: float = 1.0

    class _DemoResult:
        def __init__(self, hourly_results: pd.DataFrame):
            self.hourly_results = hourly_results
            self.config = {
                "apv_config": {"albedo": 0.25, "par_reduction": 0.22},
                "crop_config": {"reference_yield": 6.0, "light_sensitivity": 1.0},
            }

        def to_dict(self):
            return {
                "annual_thermal_yield_bifacial_kWh_per_m2": 340.0,
                "annual_thermal_yield_monofacial_kWh_per_m2": 205.0,
                "bifacial_thermal_gain_pct": 65.9,
                "estimated_crop_yield_t_per_ha": 4.7,
                "reference_thermal_yield_kWh_per_m2": 300.0,
                "LER": 1.92,
            }

    index = pd.date_range("2025-01-01", periods=8760, freq="h", tz="Europe/Berlin")
    h = index.hour.to_numpy()
    d = index.dayofyear.to_numpy()
    seasonal = 0.3 + 0.7 * np.sin(np.pi * (d - 1) / 365.0) ** 1.2
    daylight = np.maximum(np.sin(np.pi * (h - 5) / 15), 0.0)
    G_front = 550.0 * seasonal * daylight
    G_rear = 0.45 * G_front
    hourly_demo = pd.DataFrame({"G_front": G_front, "G_rear": G_rear}, index=index)

    sweep_demo = pd.DataFrame({
        "row_spacing": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "LER": [1.10, 1.31, 1.35, 1.38, 1.44, 1.42],
        "estimated_crop_yield_t_per_ha": [4.1, 4.5, 4.7, 4.8, 4.85, 4.9],
    })
    demo_result = _DemoResult(hourly_demo)
    demo_params = {"row_spacing": sweep_demo}

    tables = generate_validation_report(demo_result, demo_params, "output/test_validation")
    assert len(tables) >= 4
    print(f"  ✅ Full validation report ({len(tables)} tables)")


if __name__ == "__main__":
    print("=== Validation Tests ===")
    test_validate_irradiance_model()
    test_validate_thermal_model()
    test_validate_ler()
    test_validate_crop_yield()
    test_validate_sweden_barley()
    test_generate_validation_report()
    print("\n✅ All validation tests passed.")
