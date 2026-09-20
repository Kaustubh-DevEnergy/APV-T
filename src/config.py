"""
Configuration module for APV-T simulation.

Site definitions, collector parameters, APV geometry,
thermal operating conditions, crop settings, monthly albedo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# ============================================================================= #
# SITE DEFINITIONS
# ============================================================================= #

@dataclass
class SiteConfig:
    """Site location and metadata."""
    name: str
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    tmy_file: Path

    def __str__(self) -> str:
        return f"{self.name} ({self.latitude:.2f}°N, {self.longitude:.2f}°E)"


INGOLSTADT = SiteConfig(
    name="Ingolstadt, Germany",
    latitude=48.76,
    longitude=11.42,
    altitude=374,
    timezone="Europe/Berlin",
    tmy_file=Path("data/tmy_ingolstadt.csv"),
)

SWEDEN = SiteConfig(
    name="Sweden (Vidsel)",
    latitude=59.33,
    longitude=18.07,
    altitude=163,
    timezone="Europe/Stockholm",
    tmy_file=Path("data/tmy_sweden.csv"),
)

BELGIUM = SiteConfig(
    name="Ghent, Belgium",
    latitude=51.05,
    longitude=3.72,
    altitude=7,
    timezone="Europe/Brussels",
    tmy_file=Path("data/tmy_belgium.csv"),
)


# ============================================================================= #
# COLLECTOR CONFIGURATION (IGFPC)
# ============================================================================= #

@dataclass
class IGFPCConfig:
    """
    Insulating Glass Flat-Plate Collector (IGFPC).

    Parameters
    ----------
    eta_0 : float
        Optical efficiency. Default 0.856.
    a_1 : float
        Linear heat loss coefficient [W m⁻² K⁻¹]. Default 4.069.
    a_2 : float
        Quadratic heat loss coefficient [W m⁻² K⁻²]. Default 0.009.
    module_height : float
        Collector height (vertical) [m]. Default 2.814.
    module_width : float
        Collector width [m]. Default 4.413.
    module_depth : float
        Collector depth [m]. Default 0.0552.
    mass : float
        Collector mass [kg]. Default 354.9.
    row_spacing : float
        Distance between row centers [m]. Default 4.0.
    hub_height : float
        Module center height above ground [m]. Default 2.0.
    """
    eta_0: float = 0.856
    a_1: float = 4.069
    a_2: float = 0.009

    module_height: float = 2.814
    module_width: float = 4.413
    module_depth: float = 0.0552
    mass: float = 354.9

    row_spacing: float = 4.0
    hub_height: float = 2.0

    @property
    def gross_area(self) -> float:
        return self.module_height * self.module_width

    @property
    def area(self) -> float:
        return self.gross_area

    @property
    def a1(self) -> float:
        return self.a_1

    @property
    def a2(self) -> float:
        return self.a_2


# ============================================================================= #
# APV CONFIGURATION
# ============================================================================= #

@dataclass
class APVConfig:
    """
    Agrivoltaic mounting configuration.

    Attributes
    ----------
    surface_azimuth : float
        Surface azimuth [deg]. 90=East, 180=South, 270=West.
    surface_tilt : float
        Tilt from horizontal [deg]. 90=vertical.
    array_mode : str
        'infinite_array' or 'isolated_row'.
    albedo : float
        Ground reflectivity [0–1]. Default 0.25.
    rear_optical_factor : float
        Rear surface optical factor. Default 0.85.
    iam_b0 : float
        ASHRAE incidence angle modifier coefficient. Default 0.10.
    par_reduction : float
        PAR reduction fraction (shade level). Default 0.22.
    reference_surface_azimuth : float
        Reference monofacial azimuth for LER. Default 180 (South).
    """
    surface_azimuth: float = 90.0
    surface_tilt: float = 90.0
    array_mode: str = "infinite_array"
    albedo: float = 0.25
    rear_optical_factor: float = 0.85
    iam_b0: float = 0.10
    par_reduction: float = 0.22
    reference_surface_azimuth: float = 180.0


# ============================================================================= #
# THERMAL OPERATING CONFIGURATION
# ============================================================================= #

@dataclass
class ThermalOperatingConfig:
    """Thermal operating conditions for the collector fluid."""
    T_inlet: float = 50.0
    T_outlet: float = 80.0
    mass_flow_rate: float = 0.05

    @property
    def T_mean(self) -> float:
        return (self.T_inlet + self.T_outlet) / 2.0


# ============================================================================= #
# CROP CONFIGURATION
# ============================================================================= #

@dataclass
class CropConfig:
    """Crop parameters for agrivoltaic yield estimation."""
    crop_type: str = "strawberry"
    reference_yield: float = 6.0  # Open-field baseline [t/ha]
    light_sensitivity: float = 1.0
    growing_season_start: str = "2018-05-01"
    growing_season_end: str = "2018-09-30"


# ============================================================================= #
# MONTHLY ALBEDO VARIATIONS
# ============================================================================= #

MONTHLY_ALBEDO_INGOLSTADT: Dict[int, float] = {
    1: 0.45, 2: 0.40, 3: 0.30, 4: 0.25, 5: 0.22, 6: 0.20,
    7: 0.20, 8: 0.22, 9: 0.25, 10: 0.28, 11: 0.32, 12: 0.40,
}

MONTHLY_ALBEDO_SWEDEN: Dict[int, float] = {
    1: 0.50, 2: 0.48, 3: 0.35, 4: 0.25, 5: 0.22, 6: 0.20,
    7: 0.20, 8: 0.22, 9: 0.28, 10: 0.35, 11: 0.42, 12: 0.48,
}


# ============================================================================= #
# DEFAULT CONFIGURATIONS
# ============================================================================= #

DEFAULT_IGFPC_CONFIG = IGFPCConfig()
DEFAULT_APV_CONFIG = APVConfig()
DEFAULT_THERMAL_CONFIG = ThermalOperatingConfig()
DEFAULT_CROP_CONFIG = CropConfig()
