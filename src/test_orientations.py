"""Test: irradiance model and view-factor geometry."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass
import numpy as np

from irradiance_model import BifacialVerticalIrradiance, ViewFactorCalculator


@dataclass
class _Site:
    latitude: float = 48.7665
    longitude: float = 11.4257
    altitude: float = 365.0
    timezone: str = "Europe/Berlin"


@dataclass
class _APV:
    albedo: float = 0.20
    surface_azimuth: float = 90.0
    reference_surface_azimuth: float = 180.0


@dataclass
class _IGFPC:
    module_height: float = 2.0
    row_spacing: float = 6.0
    hub_height: float = 1.5
    n_ground_segments: int = 100


def test_view_factors():
    """Test static geometric view factors are in valid range."""
    model = BifacialVerticalIrradiance(
        site_config=_Site(), apv_config=_APV(), collector_config=_IGFPC()
    )

    assert 0.0 <= model.vf_sky_front <= 0.5, f"vf_sky_front out of range: {model.vf_sky_front}"
    assert 0.0 <= model.vf_sky_rear <= 0.5, f"vf_sky_rear out of range: {model.vf_sky_rear}"
    assert 0.0 <= model.vf_ground_front <= 1.0, f"vf_ground_front out of range"
    assert 0.0 <= model.vf_ground_rear <= 1.0, f"vf_ground_rear out of range"

    print("  ✅ View factors in valid range [0, 1]")
    print(f"     Sky front:  {model.vf_sky_front:.4f}")
    print(f"     Sky rear:   {model.vf_sky_rear:.4f}")
    print(f"     Ground front: {model.vf_ground_front:.4f}")
    print(f"     Ground rear:  {model.vf_ground_rear:.4f}")


def test_simulate():
    """Test irradiance simulation produces valid outputs."""
    import pandas as pd
    import pvlib

    site = _Site()
    times = pd.date_range("2026-06-21 00:00", "2026-06-21 23:00", freq="1h", tz=site.timezone)
    clearsky = pvlib.location.Location(
        site.latitude, site.longitude, tz=site.timezone, altitude=site.altitude
    ).get_clearsky(times, model="ineichen")

    weather = pd.DataFrame(
        {"ghi": clearsky["ghi"], "dni": clearsky["dni"], "dhi": clearsky["dhi"], "temp_air": 20.0},
        index=times,
    )

    model = BifacialVerticalIrradiance(
        site_config=site, apv_config=_APV(), collector_config=_IGFPC()
    )
    results = model.simulate(weather)

    assert "G_front" in results.columns
    assert "G_rear" in results.columns
    assert "G_total" in results.columns
    assert "bifacial_gain" in results.columns

    assert (results["G_front"] >= 0.0).all(), "Negative front irradiance"
    assert (results["G_rear"] >= 0.0).all(), "Negative rear irradiance"

    daytime = results[results["G_front"] > 1.0]
    if len(daytime) > 0:
        mean_gain = daytime["bifacial_gain"].mean()
        assert mean_gain >= 0.0, f"Negative bifacial gain: {mean_gain}"

    print("  ✅ Irradiance simulation valid")
    print(f"     Front total: {results['G_front'].sum():.0f} W/m²")
    print(f"     Rear total:  {results['G_rear'].sum():.0f} W/m²")
    print(f"     Mean daytime BG: {mean_gain:.3f}")


if __name__ == "__main__":
    print("=== Irradiance Model Tests ===")
    test_view_factors()
    test_simulate()
    print("\n✅ All irradiance model tests passed.")
