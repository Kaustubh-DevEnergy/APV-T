"""
Irradiance model: view-factor based bifacial irradiance
for vertical collector rows in agrivoltaic arrays.

Decomposes irradiance on front and rear faces into:
  - Direct beam (with inter-row shading)
  - Sky diffuse (Perez anisotropic model, with sky obstruction)
  - Ground-reflected (shading-weighted, with view factors)

Geometry follows Ernst et al. (2024) and Ledesma et al. (2020).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

import pvlib
from pvlib import solarposition, irradiance, atmosphere

try:
    from config import SiteConfig, APVConfig, IGFPCConfig  # noqa: F401
except ImportError:  # pragma: no cover
    SiteConfig = APVConfig = IGFPCConfig = object


# ============================================================================= #
# 1. SOLAR POSITION
# ============================================================================= #

class SolarPositionCalculator:
    """Wrapper around pvlib solar position utilities."""

    @staticmethod
    def calculate(
        weather_df: pd.DataFrame,
        latitude: float,
        longitude: float,
        altitude: float = 0.0,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """Compute solar position for every timestamp in weather_df."""
        df = weather_df.copy()
        times = df.index
        if times.tz is None:
            times = times.tz_localize(timezone)
        else:
            times = times.tz_convert(timezone)

        solpos = solarposition.get_solarposition(
            time=times,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=df["temp_air"].values if "temp_air" in df.columns else 12.0,
        )
        solpos.index = df.index

        for col in (
            "apparent_zenith", "zenith", "apparent_elevation",
            "elevation", "azimuth", "equation_of_time",
        ):
            df[col] = solpos[col].values

        return df

    @staticmethod
    def get_extraterrestrial(times: pd.DatetimeIndex) -> pd.Series:
        return irradiance.get_extra_radiation(times)


# ============================================================================= #
# 2. VIEW FACTOR GEOMETRY
# ============================================================================= #

class ViewFactorCalculator:
    """
    Analytic view factors for vertical collector rows.

    2-D infinite-row geometry: height H, pitch D, hub height h_hub,
    horizontal ground plane.
    """

    def __init__(
        self,
        module_height: float,
        row_spacing: float,
        hub_height: float,
        n_ground_segments: int = 100,
    ):
        if row_spacing <= 0:
            raise ValueError("row_spacing (D) must be > 0.")
        if module_height <= 0:
            raise ValueError("module_height (H) must be > 0.")

        self.H = float(module_height)
        self.D = float(row_spacing)
        self.h_hub = float(hub_height)
        self.n_ground_segments = int(n_ground_segments)

        self.h_bottom = self.h_hub - self.H / 2.0
        self.h_top = self.h_hub + self.H / 2.0

        if self.h_bottom < 0:
            warnings.warn("h_bottom < 0: module intersects ground plane.", stacklevel=2)

        # Ground segment grid (cached — geometry is static)
        self._x_edges = np.linspace(0.0, self.D, self.n_ground_segments + 1)
        self._x_centers = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        self._dx = np.diff(self._x_edges)

        # Per-segment VF to rear/front surfaces
        self._vf_segments_rear = self._strip_to_vertical_vf(
            self._x_centers, self.h_bottom, self.h_top
        )
        self._vf_segments_front = self._strip_to_vertical_vf(
            self._x_centers, self.h_bottom, self.h_top
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_to_vertical_vf(x: np.ndarray, h_bottom: float, h_top: float) -> np.ndarray:
        """View factor from a ground strip at distance x to a vertical surface
        spanning [h_bottom, h_top] (2-D infinite-row geometry)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_top = np.arctan2(h_top, x)
            alpha_bottom = np.arctan2(h_bottom, x)
        F = 0.5 * (np.sin(alpha_top) - np.sin(alpha_bottom))
        return np.clip(F, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def vf_ground_to_surface(self, surface_side: str = "front") -> float:
        """Area-averaged view factor from inter-row ground to front/rear surface."""
        if surface_side not in ("front", "rear"):
            raise ValueError("surface_side must be 'front' or 'rear'.")
        vf_segments = (
            self._vf_segments_rear if surface_side == "rear" else self._vf_segments_front
        )
        return float(np.clip(np.sum(vf_segments * self._dx) / self.D, 0.0, 1.0))

    def vf_ground_segments(self, surface_side: str = "rear") -> pd.DataFrame:
        """Per-segment ground geometry and view factors."""
        vf_segments = (
            self._vf_segments_rear if surface_side == "rear" else self._vf_segments_front
        )
        return pd.DataFrame({"x_center": self._x_centers, "dx": self._dx, "vf": vf_segments})

    # ------------------------------------------------------------------ #
    def vf_sky_to_surface(self, surface_side: str = "front", array_mode: str = "infinite_array") -> float:
        """Sky view factor accounting for obstruction by neighboring rows."""
        if surface_side not in ("front", "rear"):
            raise ValueError("surface_side must be 'front' or 'rear'.")

        if array_mode == "isolated_row" and surface_side == "front":
            return 0.5

        H, D = self.H, self.D
        h_bottom, h_top = self.h_bottom, self.h_top

        F_exact = (1.0 / (2.0 * H)) * (
            np.sqrt(D**2 + h_top**2) - np.sqrt(D**2 + h_bottom**2)
            + h_bottom - h_top
            + D * (np.arctan2(h_top, D) - np.arctan2(h_bottom, D))
        )
        return float(np.clip(F_exact, 0.0, 0.5))

    def vf_sky_to_surface_simplified(self, surface_side: str = "rear") -> float:
        """Fast approximation of sky view factor."""
        H, D = self.H, self.D
        return float(np.clip(0.5 * (1.0 - H / np.sqrt(H**2 + D**2)), 0.0, 0.5))

    # ------------------------------------------------------------------ #
    def ground_shading_fraction(
        self,
        solar_zenith: float,
        solar_azimuth: float,
        surface_azimuth: float,
    ) -> float:
        """Fraction of inter-row ground strip shaded by the row at given sun position."""
        solar_elevation = 90.0 - solar_zenith
        if solar_elevation <= 0:
            return 1.0

        elev_rad = np.radians(solar_elevation)
        rear_normal_az = (surface_azimuth + 180.0) % 360.0
        delta_az = np.radians(solar_azimuth - rear_normal_az)
        cos_proj = np.cos(delta_az)

        if cos_proj <= 0:
            return 0.0

        L_shadow = (self.h_top / np.tan(elev_rad)) * cos_proj
        return float(np.clip(L_shadow / self.D, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def row_shading_beam(
        self,
        solar_zenith: float,
        solar_azimuth: float,
        surface_azimuth: float,
    ) -> Dict[str, float]:
        """Beam shading fractions on front and rear faces from adjacent row."""
        solar_elevation = 90.0 - solar_zenith
        if solar_elevation <= 0:
            return {"front_shading_fraction": 1.0, "rear_shading_fraction": 1.0}

        elev_rad = np.radians(solar_elevation)
        front_normal_az = surface_azimuth % 360.0
        rear_normal_az = (surface_azimuth + 180.0) % 360.0

        def _face_shading(face_normal_az: float, neighbor_side_cos_sign: float) -> float:
            delta_az = np.radians(solar_azimuth - face_normal_az)
            cos_proj = np.cos(delta_az)
            if neighbor_side_cos_sign * cos_proj <= 0:
                return 0.0
            h_shadow_top = self.H - self.D / np.tan(elev_rad) * abs(cos_proj)
            shaded_height = np.clip(h_shadow_top, 0.0, self.H)
            return float(shaded_height / self.H)

        return {
            "front_shading_fraction": float(np.clip(_face_shading(front_normal_az, -1.0), 0.0, 1.0)),
            "rear_shading_fraction": float(np.clip(_face_shading(rear_normal_az, 1.0), 0.0, 1.0)),
        }


# ============================================================================= #
# 3. PEREZ TRANSPOSITION
# ============================================================================= #

class PerezTranspositionModel:
    """Perez (1990) anisotropic sky diffuse transposition, NaN-safe."""

    @staticmethod
    def transpose(
        DHI: pd.Series,
        DNI: pd.Series,
        GHI: pd.Series,
        solar_zenith: pd.Series,
        solar_azimuth: pd.Series,
        surface_tilt: float,
        surface_azimuth: float,
        airmass_absolute: pd.Series,
        extraterrestrial_radiation: pd.Series,
    ) -> pd.Series:
        poa_sky_diffuse = irradiance.perez(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dhi=DHI,
            dni=DNI,
            dni_extra=extraterrestrial_radiation,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            airmass=airmass_absolute,
            model="allsitescomposite1990",
        )
        poa_sky_diffuse = pd.Series(poa_sky_diffuse, index=DHI.index)
        return poa_sky_diffuse.fillna(0.0).clip(lower=0.0)


# ============================================================================= #
# 4. MAIN SIMULATION CLASS
# ============================================================================= #

class BifacialVerticalIrradiance:
    """
    End-to-end bifacial irradiance simulator for vertical collector rows.

    Combines solar position, analytic view factors, ground self-shading,
    and Perez sky-diffuse transposition.
    """

    def __init__(
        self,
        site_config,
        apv_config,
        collector_config,
        monthly_albedo: Optional[Dict[int, float]] = None,
    ):
        self.site_config = site_config
        self.apv_config = apv_config
        self.collector_config = collector_config
        self.monthly_albedo = monthly_albedo

        self.solar_position_calc = SolarPositionCalculator()
        self.perez_model = PerezTranspositionModel()

        n_seg = getattr(collector_config, "n_ground_segments", 100)
        self.vf_calc = ViewFactorCalculator(
            module_height=collector_config.module_height,
            row_spacing=collector_config.row_spacing,
            hub_height=collector_config.hub_height,
            n_ground_segments=n_seg,
        )

        array_mode = getattr(apv_config, "array_mode", "infinite_array")
        self.vf_sky_front = self.vf_calc.vf_sky_to_surface("front", array_mode=array_mode)
        self.vf_sky_rear = self.vf_calc.vf_sky_to_surface("rear", array_mode=array_mode)
        self.vf_ground_front = self.vf_calc.vf_ground_to_surface("front")
        self.vf_ground_rear = self.vf_calc.vf_ground_to_surface("rear")

        # Ground segments for shading-weighted rear integral
        self._ground_segments_rear = self.vf_calc.vf_ground_segments("rear")
        self._ground_segments_front = self.vf_calc.vf_ground_segments("front")

    # ------------------------------------------------------------------ #
    def _albedo_for_month(self, month: int) -> float:
        if self.monthly_albedo:
            return float(self.monthly_albedo.get(month, self.apv_config.albedo))
        return float(self.apv_config.albedo)

    # ------------------------------------------------------------------ #
    def _ground_reflected_rear(
        self, ghi: float, dhi: float, albedo: float, shading_fraction: float
    ) -> float:
        """Ground reflection on rear face, weighted by shading."""
        seg = self._ground_segments_rear
        D = self.vf_calc.D
        shaded_extent = shading_fraction * D

        x_left = seg["x_center"].values - 0.5 * seg["dx"].values
        x_right = seg["x_center"].values + 0.5 * seg["dx"].values
        overlap = np.clip(np.minimum(x_right, shaded_extent) - np.maximum(x_left, 0.0), 0.0, None)
        shaded_frac = np.where(seg["dx"].values > 0, overlap / seg["dx"].values, 0.0)

        g_shaded = dhi * albedo
        g_unshaded = ghi * albedo
        g_segment = shaded_frac * g_shaded + (1.0 - shaded_frac) * g_unshaded

        return float(max(np.sum(g_segment * seg["vf"].values * seg["dx"].values) / D, 0.0))

    def _ground_reflected_front(self, ghi: float, dhi: float, albedo: float) -> float:
        """Front ground reflection — no shading from this row."""
        return float(max(ghi * albedo * self.vf_ground_front, 0.0))

    # ------------------------------------------------------------------ #
    def simulate(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Run full bifacial vertical irradiance simulation."""
        required_cols = {"ghi", "dni", "dhi"}
        missing = required_cols - set(c.lower() for c in weather_df.columns)
        if missing:
            raise ValueError(f"weather_df missing columns: {missing}")

        cols = {c.lower(): c for c in weather_df.columns}
        ghi = weather_df[cols["ghi"]].astype(float)
        dni = weather_df[cols["dni"]].astype(float)
        dhi = weather_df[cols["dhi"]].astype(float)

        # Solar position
        df_sp = self.solar_position_calc.calculate(
            weather_df,
            latitude=self.site_config.latitude,
            longitude=self.site_config.longitude,
            altitude=getattr(self.site_config, "altitude", 0.0),
            timezone=getattr(self.site_config, "timezone", "UTC"),
        )
        solar_zenith = df_sp["apparent_zenith"]
        solar_azimuth = df_sp["azimuth"]
        solar_elevation = df_sp["apparent_elevation"]

        # Airmass
        airmass_rel = atmosphere.get_relative_airmass(solar_zenith)
        pressure = atmosphere.alt2pres(getattr(self.site_config, "altitude", 0.0))
        airmass_abs = atmosphere.get_absolute_airmass(airmass_rel, pressure)

        # Extraterrestrial radiation
        dni_extra = self.solar_position_calc.get_extraterrestrial(weather_df.index)

        # Monthly albedo
        months = weather_df.index.month
        albedo_used = pd.Series([self._albedo_for_month(m) for m in months], index=weather_df.index)

        surface_azimuth = float(self.apv_config.surface_azimuth)
        raw_tilt = getattr(self.apv_config, "surface_tilt", 90.0)
        surface_tilt = 90.0 if raw_tilt == 0.0 else float(raw_tilt)

        tilt_front = surface_tilt
        az_front = surface_azimuth % 360.0
        tilt_rear = (180.0 - surface_tilt) if surface_tilt != 90.0 else 90.0
        az_rear = (surface_azimuth + 180.0) % 360.0

        # Beam (AOI-based)
        AOI_front = irradiance.aoi(tilt_front, az_front, solar_zenith, solar_azimuth)
        AOI_rear = irradiance.aoi(tilt_rear, az_rear, solar_zenith, solar_azimuth)

        cos_aoi_front = np.cos(np.radians(AOI_front))
        cos_aoi_rear = np.cos(np.radians(AOI_rear))

        G_beam_front_raw = np.where(AOI_front < 90.0, dni.values * cos_aoi_front, 0.0)
        G_beam_rear_raw = np.where(AOI_rear < 90.0, dni.values * cos_aoi_rear, 0.0)

        # Row and ground shading (per-timestep)
        n = len(weather_df)
        row_shade_front = np.zeros(n)
        row_shade_rear = np.zeros(n)
        ground_shade_frac = np.zeros(n)
        zen_vals = solar_zenith.values
        az_vals = solar_azimuth.values

        for i in range(n):
            if zen_vals[i] >= 90.0:
                row_shade_front[i] = 1.0
                row_shade_rear[i] = 1.0
                ground_shade_frac[i] = 1.0
                continue
            shading = self.vf_calc.row_shading_beam(zen_vals[i], az_vals[i], az_front)
            row_shade_front[i] = shading["front_shading_fraction"]
            row_shade_rear[i] = shading["rear_shading_fraction"]
            ground_shade_frac[i] = self.vf_calc.ground_shading_fraction(zen_vals[i], az_vals[i], az_front)

        G_beam_front = G_beam_front_raw * (1.0 - row_shade_front)
        G_beam_rear = G_beam_rear_raw * (1.0 - row_shade_rear)

        # Sky diffuse via Perez
        vf_unobstructed_front = (1.0 + np.cos(np.radians(tilt_front))) / 2.0
        vf_unobstructed_rear = (1.0 + np.cos(np.radians(tilt_rear))) / 2.0

        G_diff_front_perez = self.perez_model.transpose(
            DHI=dhi, DNI=dni, GHI=ghi,
            solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
            surface_tilt=tilt_front, surface_azimuth=az_front,
            airmass_absolute=airmass_abs, extraterrestrial_radiation=dni_extra,
        )
        G_diff_front = G_diff_front_perez * (self.vf_sky_front / max(vf_unobstructed_front, 0.01))

        G_diff_rear_perez = self.perez_model.transpose(
            DHI=dhi, DNI=dni, GHI=ghi,
            solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
            surface_tilt=tilt_rear, surface_azimuth=az_rear,
            airmass_absolute=airmass_abs, extraterrestrial_radiation=dni_extra,
        )
        G_diff_rear = G_diff_rear_perez * (self.vf_sky_rear / max(vf_unobstructed_rear, 0.01))

        # Ground-reflected (shading-weighted)
        G_gnd_front = np.zeros(n)
        G_gnd_rear = np.zeros(n)
        ghi_vals = ghi.values
        dhi_vals = dhi.values
        albedo_vals = albedo_used.values

        for i in range(n):
            G_gnd_front[i] = self._ground_reflected_front(ghi_vals[i], dhi_vals[i], albedo_vals[i])
            G_gnd_rear[i] = self._ground_reflected_rear(
                ghi_vals[i], dhi_vals[i], albedo_vals[i], ground_shade_frac[i]
            )

        # Totals
        G_front = np.clip(G_beam_front + G_diff_front.values + G_gnd_front, 0.0, None)
        G_rear = np.clip(G_beam_rear + G_diff_rear.values + G_gnd_rear, 0.0, None)
        G_total = G_front + G_rear

        with np.errstate(divide="ignore", invalid="ignore"):
            bifacial_gain = np.where(G_front > 1e-6, G_rear / G_front, 0.0)
            bifacial_gain = np.nan_to_num(bifacial_gain, nan=0.0, posinf=0.0, neginf=0.0)

        return pd.DataFrame(
            {
                "solar_zenith": solar_zenith.values,
                "solar_azimuth": solar_azimuth.values,
                "solar_elevation": solar_elevation.values,
                "AOI_front": AOI_front.values,
                "AOI_rear": AOI_rear.values,
                "G_beam_front": G_beam_front,
                "G_diff_front": G_diff_front.values,
                "G_gnd_front": G_gnd_front,
                "G_front": G_front,
                "G_beam_rear": G_beam_rear,
                "G_diff_rear": G_diff_rear.values,
                "G_gnd_rear": G_gnd_rear,
                "G_rear": G_rear,
                "G_total": G_total,
                "bifacial_gain": bifacial_gain,
                "albedo_used": albedo_vals,
                "ground_shading_fraction": ground_shade_frac,
            },
            index=weather_df.index,
        )

    def simulate_reference_tilted(self, weather_df: pd.DataFrame, tilt: float = 35.0) -> pd.DataFrame:
        """Fixed-tilt south-facing reference surface for LER denominator."""
        cols = {c.lower(): c for c in weather_df.columns}
        ghi = weather_df[cols["ghi"]].astype(float)
        dni = weather_df[cols["dni"]].astype(float)
        dhi = weather_df[cols["dhi"]].astype(float)

        df_sp = self.solar_position_calc.calculate(
            weather_df,
            latitude=self.site_config.latitude,
            longitude=self.site_config.longitude,
            altitude=getattr(self.site_config, "altitude", 0.0),
            timezone=getattr(self.site_config, "timezone", "UTC"),
        )
        solar_zenith = df_sp["apparent_zenith"]
        solar_azimuth = df_sp["azimuth"]
        dni_extra = self.solar_position_calc.get_extraterrestrial(weather_df.index)
        airmass_rel = atmosphere.get_relative_airmass(solar_zenith)
        pressure = atmosphere.alt2pres(getattr(self.site_config, "altitude", 0.0))
        airmass_abs = atmosphere.get_absolute_airmass(airmass_rel, pressure)

        surface_azimuth = getattr(self.apv_config, "reference_surface_azimuth", 180.0)
        albedo = self.apv_config.albedo

        total = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            dni_extra=dni_extra,
            airmass=airmass_abs,
            albedo=albedo,
            model="perez",
        )

        result = pd.DataFrame(index=weather_df.index)
        result["G_poa"] = total["poa_global"].clip(lower=0.0)
        result["poa_direct"] = total["poa_direct"].clip(lower=0.0)
        result["poa_diffuse"] = total["poa_diffuse"].clip(lower=0.0)
        result["poa_ground_diffuse"] = total["poa_ground_diffuse"].clip(lower=0.0)
        result["poa_sky_diffuse"] = total["poa_sky_diffuse"].clip(lower=0.0)
        result["solar_zenith"] = solar_zenith.values
        result["solar_azimuth"] = solar_azimuth.values
        return result


# ============================================================================= #
# DEMO / SELF-TEST
# ============================================================================= #
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _DemoSiteConfig:
        latitude: float = 48.7665
        longitude: float = 11.4257
        altitude: float = 365.0
        timezone: str = "Europe/Berlin"

    @dataclass
    class _DemoAPVConfig:
        albedo: float = 0.20
        surface_azimuth: float = 90.0
        reference_surface_azimuth: float = 180.0

    @dataclass
    class _DemoIGFPCConfig:
        module_height: float = 2.0
        row_spacing: float = 6.0
        hub_height: float = 1.5
        n_ground_segments: int = 100

    site = _DemoSiteConfig()
    times = pd.date_range("2026-06-21 00:00", "2026-06-21 23:00", freq="1h", tz=site.timezone)
    solpos_demo = solarposition.get_solarposition(times, site.latitude, site.longitude, altitude=site.altitude)
    clearsky = pvlib.location.Location(
        site.latitude, site.longitude, tz=site.timezone, altitude=site.altitude
    ).get_clearsky(times, model="ineichen")

    weather = pd.DataFrame(
        {"ghi": clearsky["ghi"], "dni": clearsky["dni"], "dhi": clearsky["dhi"], "temp_air": 20.0},
        index=times,
    )

    model = BifacialVerticalIrradiance(
        site_config=site, apv_config=_DemoAPVConfig(), collector_config=_DemoIGFPCConfig()
    )

    print("Static geometric view factors:")
    print(f"  F_sky,front  = {model.vf_sky_front:.4f}")
    print(f"  F_sky,rear   = {model.vf_sky_rear:.4f}")
    print(f"  F_ground,front = {model.vf_ground_front:.4f}")
    print(f"  F_ground,rear  = {model.vf_ground_rear:.4f}")

    results = model.simulate(weather)
    daily_totals = results[["G_front", "G_rear", "G_total"]].sum()
    print("\nDaily insolation totals [Wh/m^2]:")
    print(daily_totals)
    print(f"\nMean daytime bifacial gain: {results.loc[results['G_front'] > 1, 'bifacial_gain'].mean():.3f}")

    ref = model.simulate_reference_tilted(weather, tilt=35.0)
    print(f"Reference tilted (35 deg) daily insolation: {ref['G_poa'].sum():.1f} Wh/m^2")
