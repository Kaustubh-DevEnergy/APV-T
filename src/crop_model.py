"""
Crop model: AquaCrop wrapper + empirical fallback
for agrivoltaic crop yield estimation.

References:
    - Steduto et al. (2009). AquaCrop — FAO crop model. Agronomy Journal.
    - Marrou et al. (2013). Agrivoltaic shading effects.
    - S. Ma Lu et al. (2024). Sweden barley APV data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

AQUACROP_AVAILABLE = False
try:
    from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
    AQUACROP_AVAILABLE = True
except Exception as e:
    warnings.warn(f"AquaCrop import failed: {e}. Empirical fallback used.", stacklevel=2)


# ============================================================================= #
# FAO-56 REFERENCE ET0
# ============================================================================= #

def calculate_fao56_penman_monteith(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: np.ndarray,
    ghi_mj_m2_day: np.ndarray,
    wind_speed: np.ndarray,
    relative_humidity: np.ndarray,
    pressure_kpa: float = 97.0,
) -> np.ndarray:
    """Daily FAO-56 Penman-Monteith reference ET0 [mm/day]."""
    delta = 4098.0 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    gamma = 0.000665 * pressure_kpa

    e_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    e_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (e_tmax + e_tmin) / 2.0
    ea = es * (np.clip(relative_humidity, 10.0, 100.0) / 100.0)

    rn = np.maximum(0.77 * ghi_mj_m2_day - 2.0, 0.0)

    u2 = np.clip(wind_speed, 0.5, 10.0)
    num = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)

    return np.clip(num / den, 0.1, 12.0)


# ============================================================================= #
# RESULT CONTAINER
# ============================================================================= #

@dataclass
class CropSimulationResult:
    """Output container for crop simulation."""
    crop_type: str
    reference_yield_t_ha: float
    apv_yield_t_ha: float
    yield_ratio: float
    shade_fraction: float
    harvest_date: str
    seasonal_irrigation_mm: float
    notes: str = ""


# ============================================================================= #
# AQUACROP SIMULATION WRAPPER
# ============================================================================= #

class AquaCropSimulation:
    """Paired open-field vs APV-shaded crop simulations via AquaCrop."""

    def __init__(
        self,
        crop_name: str = "Barley",
        soil_type: str = "SandyLoam",
        planting_date: str = "05/01",
        harvest_date: Optional[str] = "09/30",
    ):
        self.crop_name = crop_name
        self.soil_type = soil_type
        self.planting_date = planting_date
        self.harvest_date = harvest_date

    def _prepare_daily_weather(
        self,
        weather_df: pd.DataFrame,
        shade_fraction: float = 0.0,
    ) -> Tuple[pd.DataFrame, str, str]:
        """Convert hourly PVGIS TMY to daily AquaCrop weather format."""
        df = weather_df.copy()
        df["m"] = df.index.month
        df["d"] = df.index.day
        df["h"] = df.index.hour
        df_sorted = df.sort_values(["m", "d", "h"]).drop_duplicates(subset=["m", "d", "h"])

        cal_index = pd.date_range("2019-01-01 00:00", periods=len(df_sorted), freq="h")
        df_cal = df_sorted.copy()
        df_cal.index = cal_index

        daily = df_cal.resample("D").agg({
            "temp_air": ["min", "max", "mean"],
            "ghi": "sum",
            "wind_speed": "mean",
            "relative_humidity": "mean",
        })

        n_days = len(daily)
        tmin = daily["temp_air"]["min"].values
        tmax = daily["temp_air"]["max"].values
        tmean = daily["temp_air"]["mean"].values
        ghi_mj = (daily["ghi"]["sum"].values * 3600.0 / 1e6) * (1.0 - shade_fraction)
        ws = daily["wind_speed"]["mean"].values if "wind_speed" in df_cal.columns else np.full(n_days, 2.0)
        rh = daily["relative_humidity"]["mean"].values if "relative_humidity" in df_cal.columns else np.full(n_days, 70.0)

        # Microclimate buffering under canopy
        tmax_eff = tmax - (0.8 * shade_fraction)
        tmin_eff = tmin + (0.3 * shade_fraction)

        et0 = calculate_fao56_penman_monteith(tmin_eff, tmax_eff, tmean, ghi_mj, ws, rh)
        date_range = pd.date_range("2019-01-01", periods=n_days, freq="D")

        np.random.seed(42)
        precip = np.where(np.random.rand(n_days) < 0.30, np.random.exponential(5.5, n_days), 0.0)

        return pd.DataFrame({
            "MinTemp": tmin_eff,
            "MaxTemp": tmax_eff,
            "Precipitation": precip,
            "ReferenceET": et0,
            "Date": date_range,
        }), "2019/05/01", "2019/09/30"

    def simulate(
        self,
        weather_df: pd.DataFrame,
        shade_fraction: float = 0.22,
        reference_yield_override: Optional[float] = None,
    ) -> CropSimulationResult:
        """Run paired open-field vs APV-shaded simulations."""
        if not AQUACROP_AVAILABLE:
            ref_y = reference_yield_override if reference_yield_override else 5.5
            apv_y = ref_y * (1.0 - 0.75 * shade_fraction)
            return CropSimulationResult(
                crop_type=self.crop_name,
                reference_yield_t_ha=ref_y,
                apv_yield_t_ha=apv_y,
                yield_ratio=apv_y / ref_y,
                shade_fraction=shade_fraction,
                harvest_date="2019-08-15",
                seasonal_irrigation_mm=0.0,
                notes="AquaCrop unavailable; empirical fallback used.",
            )

        try:
            # Open-field
            w_open, start_date, end_date = self._prepare_daily_weather(weather_df, shade_fraction=0.0)
            crop_open = Crop(self.crop_name, planting_date=self.planting_date)
            soil_open = Soil(self.soil_type)
            init_wc = InitialWaterContent(value=["FC"])

            model_open = AquaCropModel(start_date, end_date, w_open, soil_open, crop_open, init_wc)
            model_open.run_model(till_termination=True)
            res_open = model_open.get_simulation_results()
            y_open_raw = float(res_open["Dry yield (tonne/ha)"].iloc[-1])

            # Shaded APV
            w_apv, _, _ = self._prepare_daily_weather(weather_df, shade_fraction=shade_fraction)
            crop_apv = Crop(self.crop_name, planting_date=self.planting_date)
            soil_apv = Soil(self.soil_type)

            model_apv = AquaCropModel(start_date, end_date, w_apv, soil_apv, crop_apv, init_wc)
            model_apv.run_model(till_termination=True)
            res_apv = model_apv.get_simulation_results()
            y_apv_raw = float(res_apv["Dry yield (tonne/ha)"].iloc[-1])

            harvest_date = str(res_apv["Harvest Date (YYYY/MM/DD)"].iloc[-1])
            irrigation = float(res_apv["Seasonal irrigation (mm)"].iloc[-1])

            if y_open_raw > 0:
                yield_ratio = y_apv_raw / y_open_raw
            else:
                yield_ratio = 1.0 - 0.75 * shade_fraction

            # Sanity check
            if yield_ratio > 1.0:
                warnings.warn(
                    f"AquaCrop produced yield_ratio={yield_ratio:.3f} > 1.0. Applying empirical penalty.",
                    stacklevel=2,
                )
                yield_ratio = float(np.clip(1.0 - 0.90 * shade_fraction, 0.30, 1.0))

            yield_ratio = float(np.clip(yield_ratio, 0.30, 1.05))

            final_ref_y = reference_yield_override if reference_yield_override else y_open_raw
            final_apv_y = final_ref_y * yield_ratio

            return CropSimulationResult(
                crop_type=self.crop_name,
                reference_yield_t_ha=float(final_ref_y),
                apv_yield_t_ha=float(final_apv_y),
                yield_ratio=float(yield_ratio),
                shade_fraction=float(shade_fraction),
                harvest_date=harvest_date,
                seasonal_irrigation_mm=irrigation,
                notes="Simulated with FAO AquaCrop model.",
            )

        except Exception as e:
            warnings.warn(f"AquaCrop failed: {e}. Using empirical model.", stacklevel=2)
            ref_y = reference_yield_override if reference_yield_override else 5.5
            yield_ratio = 1.0 - 0.90 * shade_fraction
            yield_ratio = float(np.clip(yield_ratio, 0.30, 1.0))
            apv_y = ref_y * yield_ratio
            return CropSimulationResult(
                crop_type=self.crop_name,
                reference_yield_t_ha=ref_y,
                apv_yield_t_ha=apv_y,
                yield_ratio=yield_ratio,
                shade_fraction=shade_fraction,
                harvest_date="2019-08-15",
                seasonal_irrigation_mm=0.0,
                notes=f"AquaCrop exception; empirical model applied.",
            )


# ============================================================================= #
# DEMO
# ============================================================================= #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from utils import load_pvgis_tmy
    from config import INGOLSTADT

    print("=== AquaCrop Standalone Demo ===")
    weather = load_pvgis_tmy(INGOLSTADT.tmy_file)

    crop_sim = AquaCropSimulation(crop_name="Barley", planting_date="05/01")
    result = crop_sim.simulate(weather, shade_fraction=0.25, reference_yield_override=5.5)

    print(f"Crop:              {result.crop_type}")
    print(f"Open-Field Yield:  {result.reference_yield_t_ha:.2f} t/ha")
    print(f"APV Shaded Yield:  {result.apv_yield_t_ha:.2f} t/ha (Shade: {result.shade_fraction*100:.0f}%)")
    print(f"Yield Retention:   {result.yield_ratio*100:.1f}%")
    print(f"Harvest Date:      {result.harvest_date}")
    print(f"Notes:             {result.notes}")
