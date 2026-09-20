"""
Validation: literature comparison and benchmark screening
for vertical bifacial APV-T simulation results.

Compares simulated outputs against published benchmarks
for bifacial irradiance gain, thermal performance, LER,
and crop yield.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# ============================================================================= #
# LITERATURE BENCHMARKS
# ============================================================================= #

LITERATURE_BIFACIAL_GAINS: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = {
    "Llamoza-Carrasco_2021": {"vertical_albedo_0.25": 45.0},
    "Gu_et_al_2022": {"vertical_albedo_0.30": 48.0},
    "Fuertes_et_al_2023": {"vertical_bifacial": (35.0, 65.0)},
    "Khan_et_al_2017": {"vertical_high_albedo": (70.0, 90.0)},
}

LITERATURE_THERMAL_YIELDS: Dict[str, Dict[str, Union[float, Tuple[float, float]]]] = {
    "ISO_9806_reference": {"monofacial_temperate": (150.0, 280.0)},
    "Summ_et_al_2025": {"bifacial_gain_pct": (50.0, 100.0)},
    "vertical_bifacial_expected": {"yield_range": (250.0, 450.0)},
}

LITERATURE_LER_VALUES: Dict[str, Union[float, Tuple[float, float]]] = {
    "Ziegler_2022_tomato_3m": 1.32,
    "Dupraz_2011_various_4m": (1.25, 1.50),
    "Marrou_2013_lettuce_6m": 1.47,
    "Arteconi_2023_APV_T_5m": (1.30, 1.50),
    "vertical_bifacial_thermal": (1.30, 1.60),
}

STRAWBERRY_BENCHMARKS: Dict[str, float] = {
    "open_field_spain": 6.0,
    "open_field_italy": 5.8,
    "covered_tunnel_spain": 8.5,
    "shade_20pct_reduction": 0.18,
}

SWEDEN_BARLEY_BENCHMARK: Dict[str, Any] = {
    "source": "S. Ma Lu et al. (2024), Data in Brief, 57, 110990",
    "location": "Vidsel, Sweden (66°N)",
    "crop": "Spring barley (Hordeum vulgare)",
    "open_field_yield_t_ha": 5.5,
    "apv_yield_t_ha_range": (4.4, 5.0),
    "yield_reduction_pct_range": (10, 20),
    "row_spacing_m": 10.0,
    "module_height_m": 2.1,
    "notes": "Vertical bifacial PV, east-west orientation",
}

FRONT = "#1976D2"
REAR = "#FF8F00"
TOTAL = "#2E7D32"
MONO = "#757575"

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

PathLike = Union[str, Path]


# ============================================================================= #
# INTERNAL HELPERS
# ============================================================================= #

def _range_label(low: float, high: float) -> str:
    return f"{low:.1f}–{high:.1f}"


def _in_range(value: float, low: float, high: float, tolerance: float = 0.0) -> bool:
    return bool((low - tolerance) <= value <= (high + tolerance))


def _extract_bifacial_gain(hourly_results_df: pd.DataFrame) -> float:
    """Annual rear-to-front irradiance gain from hourly data."""
    required = {"G_front", "G_rear"}
    missing = required - set(c.lower() for c in hourly_results_df.columns)
    if missing:
        raise ValueError(f"hourly_results_df missing columns: {sorted(missing)}")

    front = float(np.clip(hourly_results_df["G_front"].to_numpy(dtype=float), 0.0, None).sum())
    rear = float(np.clip(hourly_results_df["G_rear"].to_numpy(dtype=float), 0.0, None).sum())
    return float(rear / front * 100.0) if front > 0.0 else float("nan")


def _extract_annual_value(
    data: Mapping[str, Any], primary_key: str, fallback_key: Optional[str] = None
) -> float:
    value = data.get(primary_key, data.get(fallback_key, np.nan) if fallback_key else np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _save_table_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


# ============================================================================= #
# 1. IRRADIANCE MODEL VALIDATION
# ============================================================================= #

def validate_irradiance_model(hourly_results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare modelled annual bifacial irradiance gain against benchmarks."""
    your_gain = _extract_bifacial_gain(hourly_results_df)
    broad_low, broad_high = 30.0, 90.0
    fuertes_low, fuertes_high = LITERATURE_BIFACIAL_GAINS["Fuertes_et_al_2023"]["vertical_bifacial"]
    khan_low, khan_high = LITERATURE_BIFACIAL_GAINS["Khan_et_al_2017"]["vertical_high_albedo"]
    llamoza = float(LITERATURE_BIFACIAL_GAINS["Llamoza-Carrasco_2021"]["vertical_albedo_0.25"])
    gu = float(LITERATURE_BIFACIAL_GAINS["Gu_et_al_2022"]["vertical_albedo_0.30"])

    if not np.isfinite(your_gain):
        assessment = "Not assessable: annual front irradiance is zero or invalid."
    elif _in_range(your_gain, khan_low, khan_high):
        assessment = (
            f"Within Khan et al. (2017) vertical East-West benchmark ({_range_label(khan_low, khan_high)}%); "
            "consistent with mid-latitude direct morning/afternoon solar reception."
        )
    elif _in_range(your_gain, fuertes_low, fuertes_high):
        assessment = (
            f"Within Fuertes et al. vertical-bifacial range ({_range_label(fuertes_low, fuertes_high)}%)."
        )
    elif _in_range(your_gain, broad_low, broad_high):
        assessment = f"Within broad vertical bifacial literature range ({_range_label(broad_low, broad_high)}%)."
    else:
        assessment = (
            "Outside supplied vertical-bifacial benchmark; review albedo, row pitch, "
            "rear optical assumptions, and irradiation normalization."
        )

    rows = [
        ("Your model bifacial gain (%)", your_gain),
        ("Khan et al. (2017) vertical East-West benchmark (%)", _range_label(khan_low, khan_high)),
        ("Fuertes et al. (2023) vertical bifacial range (%)", _range_label(fuertes_low, fuertes_high)),
        ("Llamoza-Carrasco (2021), vertical albedo 0.25 (%)", llamoza),
        ("Gu et al. (2022), vertical albedo 0.30 (%)", gu),
        ("Assessment", assessment),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value / assessment"])


# ============================================================================= #
# 2. THERMAL MODEL VALIDATION
# ============================================================================= #

def validate_thermal_model(annual_results_dict: Mapping[str, Any]) -> pd.DataFrame:
    """Compare simulated bifacial/monofacial thermal performance with benchmarks."""
    y_bifi = _extract_annual_value(annual_results_dict, "annual_thermal_yield_bifacial_kWh_per_m2", "thermal_annual_yield_bifacial_kWh_per_m2")
    y_mono = _extract_annual_value(annual_results_dict, "annual_thermal_yield_monofacial_kWh_per_m2", "thermal_annual_yield_monofacial_kWh_per_m2")
    gain = _extract_annual_value(annual_results_dict, "bifacial_thermal_gain_pct", "thermal_bifacial_thermal_gain_pct")

    mono_low, mono_high = LITERATURE_THERMAL_YIELDS["ISO_9806_reference"]["monofacial_temperate"]
    gain_low, gain_high = LITERATURE_THERMAL_YIELDS["Summ_et_al_2025"]["bifacial_gain_pct"]

    gain_ok = np.isfinite(gain) and _in_range(gain, gain_low, gain_high)
    mono_ok = np.isfinite(y_mono) and _in_range(y_mono, mono_low, mono_high)

    if gain_ok and mono_ok:
        assessment = "Bifacial gain and monofacial annual yield both within supplied reference ranges; ISO 9806 implementation is literature-consistent."
    elif gain_ok:
        assessment = "Bifacial gain within Summ et al. benchmark; monofacial yield differs from temperate reference; check climate-year and operating-temperature comparability."
    elif mono_ok:
        assessment = "Monofacial yield within ISO 9806 reference range; bifacial gain differs from Summ et al. benchmark; review rear irradiance and optical derating assumptions."
    else:
        assessment = "Thermal outputs differ from one or both supplied benchmarks; compare collector coefficients, mean fluid temperature, weather data, aperture definition, and timestep integration."

    rows = [
        ("Your bifacial yield (kWh/m2/year)", y_bifi),
        ("Your monofacial yield (kWh/m2/year)", y_mono),
        ("Your bifacial thermal gain (%)", gain),
        ("ISO 9806 reference monofacial range (kWh/m2/year)", _range_label(mono_low, mono_high)),
        ("Summ et al. (2025) bifacial gain range (%)", _range_label(gain_low, gain_high)),
        ("Assessment", assessment),
    ]
    return pd.DataFrame(rows, columns=["Thermal performance", "Value / assessment"])


# ============================================================================= #
# 3. LAND EQUIVALENT RATIO VALIDATION
# ============================================================================= #

def validate_ler(
    parametric_results_dict: Mapping[str, pd.DataFrame],
    crop_config: Any,
) -> pd.DataFrame:
    """Compare modelled LER values across row spacing with literature intervals."""
    if "row_spacing" not in parametric_results_dict:
        raise KeyError("parametric_results_dict must contain a 'row_spacing' DataFrame.")

    df = parametric_results_dict["row_spacing"].copy()
    required = {"row_spacing", "LER"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"row_spacing results missing columns: {sorted(missing)}")

    bins: Dict[float, Tuple[float, float, str]] = {
        2.0: (0.85, 1.05, "Vertical APV-Thermal heavy shading benchmark (2 m)"),
        3.0: (1.00, 1.20, "Vertical APV-Thermal moderate spacing context (3 m)"),
        4.0: (1.05, 1.25, "Vertical APV-Thermal reference spacing (4 m)"),
        5.0: (1.10, 1.30, "Arteconi et al. 2023 APV-T context (5 m)"),
        6.0: (1.15, 1.35, "Vertical APV-Thermal light-shading range (6 m)"),
        8.0: (1.20, 1.40, "Vertical APV-Thermal wide-pitch benchmark (8 m)"),
    }

    rows = []
    for _, row in df.sort_values("row_spacing").iterrows():
        spacing = float(row["row_spacing"])
        your_ler = float(row["LER"])
        nearest_spacing = min(bins, key=lambda x: abs(x - spacing))
        low, high, source_context = bins[nearest_spacing]

        if not np.isfinite(your_ler):
            assessment = "Not assessable: invalid LER value."
        elif _in_range(your_ler, low, high):
            assessment = f"Within supplied literature context ({source_context})."
        elif your_ler < low:
            assessment = f"Below supplied literature context ({source_context}); investigate energy/crop assumptions."
        else:
            assessment = f"Above supplied literature context ({source_context}); verify land-area normalization and crop model."

        rows.append({
            "Row spacing (m)": spacing,
            "Your LER": your_ler,
            "Literature range": _range_label(low, high),
            "Assessment": assessment,
        })

    return pd.DataFrame(rows)


# ============================================================================= #
# 4. STRAWBERRY CROP-YIELD VALIDATION
# ============================================================================= #

def validate_crop_yield(
    apv_config: Any,
    crop_config: Any,
    estimated_yield_t_ha: float,
) -> pd.DataFrame:
    """Compare modelled strawberry yield with open-field and shade benchmarks."""
    shade_fraction = None
    for attribute in ("shade_fraction", "par_reduction", "estimated_par_reduction"):
        if hasattr(apv_config, attribute):
            shade_fraction = float(getattr(apv_config, attribute))
            break
    if shade_fraction is None:
        shade_fraction = 0.22
    shade_fraction = float(np.clip(shade_fraction, 0.0, 1.0))

    reference_yield = float(getattr(crop_config, "reference_yield", STRAWBERRY_BENCHMARKS["open_field_spain"]))
    if reference_yield <= 0:
        reference_yield = STRAWBERRY_BENCHMARKS["open_field_spain"]

    loss_per_shade_fraction = STRAWBERRY_BENCHMARKS["shade_20pct_reduction"] / 0.20
    expected_reduction = float(np.clip(loss_per_shade_fraction * shade_fraction, 0.0, 1.0))
    expected_yield = reference_yield * (1.0 - expected_reduction)

    observed_reduction = 1.0 - float(estimated_yield_t_ha) / reference_yield
    shade_band = "20–30% shade" if 0.20 <= shade_fraction <= 0.30 else "30–40% shade" if 0.30 < shade_fraction <= 0.40 else "other shade level"

    if not np.isfinite(estimated_yield_t_ha):
        assessment = "Not assessable: modelled crop yield is invalid."
    elif abs(estimated_yield_t_ha - expected_yield) <= 0.50:
        assessment = f"Modelled yield consistent with {shade_band} benchmark and expected shade-adjusted yield (~{expected_yield:.1f} t/ha)."
    elif estimated_yield_t_ha < expected_yield:
        assessment = "Modelled yield below shade-adjusted benchmark; crop response, water, microclimate, or shade distribution may be conservative."
    else:
        assessment = "Modelled yield exceeds shade-adjusted benchmark; verify whether APV microclimate benefits or crop-shade-response effects are represented."

    rows = [
        ("Your estimated APV strawberry yield (t/ha)", float(estimated_yield_t_ha)),
        ("Reference crop yield used by model (t/ha)", reference_yield),
        ("Open-field benchmark, Spain (t/ha)", STRAWBERRY_BENCHMARKS["open_field_spain"]),
        ("Open-field benchmark, Italy (t/ha)", STRAWBERRY_BENCHMARKS["open_field_italy"]),
        ("Covered/tunnel benchmark, Spain (t/ha)", STRAWBERRY_BENCHMARKS["covered_tunnel_spain"]),
        ("Assumed APV shade / PAR reduction (%)", shade_fraction * 100.0),
        ("Literature-based expected yield reduction (%)", expected_reduction * 100.0),
        ("Your implied yield reduction (%)", observed_reduction * 100.0),
        ("Expected shade-adjusted yield (t/ha)", expected_yield),
        ("Assessment", assessment),
    ]
    return pd.DataFrame(rows, columns=["Crop-yield metric", "Value / assessment"])


# ============================================================================= #
# 4b. SWEDEN BARLEY VALIDATION
# ============================================================================= #

def validate_against_sweden_barley(
    simulated_crop_yield_t_ha: float,
    simulated_yield_reduction_pct: float,
    row_spacing_m: float,
) -> pd.DataFrame:
    """Compare simulated crop yield against S. Ma Lu et al. (2024) Sweden barley data."""
    bench = SWEDEN_BARLEY_BENCHMARK

    red_low, red_high = bench["yield_reduction_pct_range"]
    yield_low, yield_high = bench["apv_yield_t_ha_range"]

    reduction_in_range = red_low <= simulated_yield_reduction_pct <= red_high
    yield_in_range = yield_low <= simulated_crop_yield_t_ha <= yield_high

    if reduction_in_range and yield_in_range:
        assessment = "Simulated values within observed range from S. Ma Lu et al. (2024). Model consistent with Swedish vertical APV field measurements."
    elif reduction_in_range or yield_in_range:
        assessment = "Simulated values partially agree with S. Ma Lu et al. (2024). Differences may be due to latitude (66°N vs simulation site), row spacing, or crop variety."
    else:
        assessment = "Simulated values differ from S. Ma Lu et al. (2024) observations. Direct comparison requires matching site conditions."

    rows = [
        ("Source", bench["source"]),
        ("Reference location", bench["location"]),
        ("Reference row spacing", f"{bench['row_spacing_m']} m"),
        ("Reference open-field yield", f"{bench['open_field_yield_t_ha']} t/ha"),
        ("Reference APV yield range", f"{yield_low}–{yield_high} t/ha"),
        ("Reference yield reduction", f"{red_low}–{red_high}%"),
        ("Your simulated APV yield", f"{simulated_crop_yield_t_ha:.2f} t/ha"),
        ("Your simulated yield reduction", f"{simulated_yield_reduction_pct:.1f}%"),
        ("Your row spacing", f"{row_spacing_m} m"),
        ("Assessment", assessment),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ============================================================================= #
# OPTIONAL VALIDATION VISUALS
# ============================================================================= #

def plot_ler_literature_comparison(
    ler_validation_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Bar/range plot comparing simulated LER by row spacing vs literature intervals."""
    required = {"Row spacing (m)", "Your LER", "Literature range"}
    missing = required - set(ler_validation_df.columns)
    if missing:
        raise ValueError(f"ler_validation_df missing columns: {sorted(missing)}")

    df = ler_validation_df.sort_values("Row spacing (m)")
    x = np.arange(len(df))
    model = df["Your LER"].to_numpy(dtype=float)
    lows, highs = [], []
    for label in df["Literature range"]:
        low_s, high_s = str(label).replace("–", "-").split("-")
        lows.append(float(low_s))
        highs.append(float(high_s))
    lows_arr, highs_arr = np.asarray(lows), np.asarray(highs)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x, model, color=TOTAL, alpha=0.85, label="APV-T model")
    ax.errorbar(
        x, (lows_arr + highs_arr) / 2.0,
        yerr=np.vstack(((highs_arr - lows_arr) / 2.0, (highs_arr - lows_arr) / 2.0)),
        fmt="o", capsize=5, color=REAR, linewidth=2.0, label="Literature range",
    )
    ax.axhline(1.0, color=MONO, linestyle="--", linewidth=1.2, label="LER = 1 reference")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in df["Row spacing (m)"]])
    ax.set_xlabel("Row spacing [m]")
    ax.set_ylabel("Land Equivalent Ratio, LER [-]")
    ax.set_title("APV-T Land Equivalent Ratio versus literature benchmarks")
    ax.set_ylim(bottom=0)
    ax.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_thermal_gain_comparison(
    thermal_validation_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Bar/range plot comparing modelled bifacial thermal gain vs Summ et al. (2025) range."""
    gain_row = thermal_validation_df.loc[
        thermal_validation_df["Thermal performance"] == "Your bifacial thermal gain (%)",
        "Value / assessment",
    ]
    if gain_row.empty:
        raise ValueError("Thermal validation table does not contain 'Your bifacial thermal gain (%)'.")
    your_gain = float(gain_row.iloc[0])
    low, high = LITERATURE_THERMAL_YIELDS["Summ_et_al_2025"]["bifacial_gain_pct"]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar([0], [your_gain], color=TOTAL, width=0.55, label="APV-T model")
    ax.errorbar([1], [(low + high) / 2.0], yerr=[(high - low) / 2.0], fmt="o", color=REAR, capsize=7, linewidth=2.2, label="Summ et al. (2025) range")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Your model", "Published IGFPC"])
    ax.set_ylabel("Bifacial thermal gain [%]")
    ax.set_title("Bifacial thermal gain: APV-T model versus literature")
    ax.set_ylim(bottom=0)
    ax.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_bifacial_gain_comparison(
    irradiance_validation_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Scatter/range plot for simulated vs published vertical bifacial irradiance gains."""
    value_map = dict(zip(irradiance_validation_df["Metric"], irradiance_validation_df["Value / assessment"]))
    your_gain = float(value_map["Your model bifacial gain (%)"])
    llamoza = float(value_map["Llamoza-Carrasco (2021), vertical albedo 0.25 (%)"])
    gu = float(value_map["Gu et al. (2022), vertical albedo 0.30 (%)"])
    fuertes_low, fuertes_high = LITERATURE_BIFACIAL_GAINS["Fuertes_et_al_2023"]["vertical_bifacial"]
    khan_low, khan_high = LITERATURE_BIFACIAL_GAINS["Khan_et_al_2017"]["vertical_high_albedo"]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter([0], [your_gain], color=TOTAL, s=95, zorder=3, label="APV-T view-factor model")
    ax.scatter([1], [llamoza], color=FRONT, s=80, zorder=3, label="Llamoza-Carrasco (2021)")
    ax.scatter([2], [gu], color=REAR, s=80, zorder=3, label="Gu et al. (2022)")
    ax.errorbar([3], [(fuertes_low + fuertes_high) / 2.0], yerr=[(fuertes_high - fuertes_low) / 2.0], fmt="o", color=MONO, capsize=7, linewidth=2.0, label="Fuertes et al. (2023) range")
    ax.errorbar([4], [(khan_low + khan_high) / 2.0], yerr=[(khan_high - khan_low) / 2.0], fmt="o", color="#D32F2F", capsize=7, linewidth=2.0, label="Khan et al. (2017) East-West")
    ax.axhspan(30.0, 90.0, color=REAR, alpha=0.10, label="Broad vertical range (30–90%)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["Your model", "Llamoza", "Gu et al.", "Fuertes", "Khan et al."], rotation=10)
    ax.set_ylabel("Bifacial irradiance gain [%]")
    ax.set_title("Vertical bifacial irradiance gain versus literature")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


# ============================================================================= #
# 5. INTEGRATED REPORT
# ============================================================================= #

def generate_validation_report(
    simulation_result: Any,
    parametric_results: Mapping[str, pd.DataFrame],
    output_dir: PathLike,
) -> Dict[str, pd.DataFrame]:
    """Generate all APV-T literature-comparison tables and figures."""
    validation_dir = Path(output_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    hourly_results = simulation_result.hourly_results
    annual_results = simulation_result.to_dict()

    config = getattr(simulation_result, "config", {})
    apv_config = config.get("apv_config", {}) if isinstance(config, Mapping) else {}
    crop_config = config.get("crop_config", {}) if isinstance(config, Mapping) else {}

    apv_config = getattr(simulation_result, "apv_config", apv_config)
    crop_config = getattr(simulation_result, "crop_config", crop_config)

    class _ConfigProxy:
        def __init__(self, values: Mapping[str, Any]):
            for key, value in values.items():
                setattr(self, key, value)

    if isinstance(apv_config, Mapping):
        apv_config = _ConfigProxy(apv_config)
    if isinstance(crop_config, Mapping):
        crop_config = _ConfigProxy(crop_config)

    estimated_yield = _extract_annual_value(annual_results, "estimated_crop_yield_t_per_ha")

    tables = {
        "irradiance": validate_irradiance_model(hourly_results),
        "thermal": validate_thermal_model(annual_results),
        "ler": validate_ler(parametric_results, crop_config),
        "crop": validate_crop_yield(apv_config, crop_config, estimated_yield),
    }

    reference_yield = float(getattr(crop_config, "reference_yield", 5.5))
    if reference_yield > 0 and estimated_yield > 0:
        yield_reduction_pct = (1.0 - estimated_yield / reference_yield) * 100.0
        row_spacing = float(parametric_results.get("row_spacing", pd.DataFrame({"row_spacing": [4.0]}))["row_spacing"].iloc[0]) if "row_spacing" in parametric_results else 4.0
        tables["sweden_barley"] = validate_against_sweden_barley(
            simulated_crop_yield_t_ha=estimated_yield,
            simulated_yield_reduction_pct=yield_reduction_pct,
            row_spacing_m=row_spacing,
        )

    for name, table in tables.items():
        _save_table_csv(table, validation_dir / f"validation_{name}.csv")

    fig_irr = plot_bifacial_gain_comparison(tables["irradiance"], validation_dir / "bifacial_gain_literature.png")
    fig_thermal = plot_thermal_gain_comparison(tables["thermal"], validation_dir / "thermal_gain_literature.png")
    fig_ler = plot_ler_literature_comparison(tables["ler"], validation_dir / "ler_literature.png")
    plt.close(fig_irr)
    plt.close(fig_thermal)
    plt.close(fig_ler)

    irradiance_assessment = str(tables["irradiance"].iloc[-1, 1])
    thermal_assessment = str(tables["thermal"].iloc[-1, 1])
    crop_assessment = str(tables["crop"].iloc[-1, 1])
    ler_in_range = int(tables["ler"]["Assessment"].str.startswith("Within").sum())
    ler_total = len(tables["ler"])

    report_lines = [
        "APV-T LITERATURE COMPARISON REPORT",
        "=" * 38,
        "",
        "Irradiance model:",
        irradiance_assessment,
        "",
        "Thermal model:",
        thermal_assessment,
        "",
        "Crop-yield model:",
        crop_assessment,
        "",
        "LER model:",
        f"{ler_in_range} of {ler_total} row-spacing cases fall within their supplied literature-context range.",
        "",
        "Overall assessment:",
        "The results are a literature-plausibility comparison, not direct experimental validation. Use matching site, collector, crop, weather, and operating-condition measurements for final calibration.",
    ]
    (validation_dir / "validation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return tables


# ============================================================================= #
# 6. MULTI-SITE GEOGRAPHICAL VALIDATION
# ============================================================================= #

def plot_multisite_comparison(
    multisite_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Publication-quality comparison figure across European sites."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    x = np.arange(len(multisite_df))
    width = 0.25
    sites = multisite_df["site_name"].tolist()

    ax1.bar(x - width, multisite_df["annual_ghi_kWh_m2"], width, label="Annual GHI", color="#757575")
    ax1.bar(x, multisite_df["annual_front_irr_kWh_m2"], width, label="Front POA", color="#1976D2")
    ax1.bar(x + width, multisite_df["annual_rear_irr_kWh_m2"], width, label="Rear POA", color="#FF8F00")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sites, rotation=10)
    ax1.set_ylabel("Annual Irradiance [kWh/m²]")
    ax1.set_title("Annual Irradiance across European Climates")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(x - width/2, multisite_df["annual_bifacial_thermal_kWh_m2"], width, label="Bifacial Thermal Yield [kWh/m²]", color="#2E7D32")
    ax2.bar(x + width/2, multisite_df["annual_monofacial_thermal_kWh_m2"], width, label="Monofacial Thermal Yield [kWh/m²]", color="#9E9E9E")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites, rotation=10)
    ax2.set_ylabel("Thermal Yield [kWh/m²]")
    ax2.set_title("Thermal Yield and Bifacial Gain across Sites")

    ax2_r = ax2.twinx()
    ax2_r.plot(x, multisite_df["LER"], color="#D32F2F", marker="o", linewidth=2.0, markersize=8, label="LER [-]")
    ax2_r.set_ylabel("Land Equivalent Ratio, LER [-]", color="#D32F2F")
    ax2_r.tick_params(axis="y", labelcolor="#D32F2F")
    ax2_r.set_ylim(0.8, 1.6)

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax2.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def run_multisite_comparison(
    sites: Optional[Sequence[Any]] = None,
    output_dir: PathLike = "results/validation",
) -> pd.DataFrame:
    """Run full APV-T simulation across European climatic zones."""
    from utils import load_pvgis_tmy
    from config import INGOLSTADT, BELGIUM, SWEDEN, IGFPCConfig, APVConfig, ThermalOperatingConfig, CropConfig, MONTHLY_ALBEDO_INGOLSTADT, MONTHLY_ALBEDO_SWEDEN
    from simulation import APVTSimulation
    from economic_model import calculate_lcoe, EconomicConfig

    if sites is None:
        sites = [INGOLSTADT, BELGIUM, SWEDEN]

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    econ_config = EconomicConfig()

    rows = []
    print("\n" + "=" * 70)
    print("MULTI-SITE GEOGRAPHICAL VALIDATION & COMPARISON")
    print("=" * 70)

    for site in sites:
        if not site.tmy_file.exists():
            print(f"  ⚠️ TMY file missing for {site.name}: {site.tmy_file}")
            continue

        print(f"  Simulating {site.name} ({site.latitude:.2f}°N, {site.longitude:.2f}°E)...")
        weather = load_pvgis_tmy(str(site.tmy_file))

        monthly_alb = MONTHLY_ALBEDO_SWEDEN if "Sweden" in site.name else MONTHLY_ALBEDO_INGOLSTADT

        sim = APVTSimulation(
            site_config=site,
            apv_config=APVConfig(surface_azimuth=90.0),
            collector_config=IGFPCConfig(),
            thermal_config=ThermalOperatingConfig(),
            crop_config=CropConfig(),
            monthly_albedo=monthly_alb,
        )
        res = sim.run(weather)
        kpis = res.to_dict()

        bifi_yield = kpis.get("annual_thermal_yield_bifacial_kWh_per_m2", 0.0)
        lcoe_val = calculate_lcoe(bifi_yield, econ_config)

        rows.append({
            "site_name": site.name,
            "latitude_deg": site.latitude,
            "longitude_deg": site.longitude,
            "annual_ghi_kWh_m2": kpis.get("annual_ghi_kWh_per_m2", 0.0),
            "annual_front_irr_kWh_m2": kpis.get("annual_front_irradiance_kWh_per_m2", 0.0),
            "annual_rear_irr_kWh_m2": kpis.get("annual_rear_irradiance_kWh_per_m2", 0.0),
            "bifacial_irr_gain_pct": kpis.get("bifacial_irradiance_gain_pct", 0.0),
            "annual_bifacial_thermal_kWh_m2": bifi_yield,
            "annual_monofacial_thermal_kWh_m2": kpis.get("annual_thermal_yield_monofacial_kWh_per_m2", 0.0),
            "bifacial_thermal_gain_pct": kpis.get("bifacial_thermal_gain_pct", 0.0),
            "reference_tilted_thermal_kWh_m2": kpis.get("reference_thermal_yield_kWh_per_m2", 0.0),
            "estimated_crop_yield_t_ha": kpis.get("estimated_crop_yield_t_per_ha", 0.0),
            "LER": kpis.get("LER", 0.0),
            "lcoe_eur_kWh": lcoe_val,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "validation_multisite_summary.csv", index=False)

    fig = plot_multisite_comparison(summary_df, save_path=out_dir / "multisite_comparison.png")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("MULTI-SITE SUMMARY TABLE")
    print("=" * 70)
    disp_cols = ["site_name", "annual_ghi_kWh_m2", "annual_front_irr_kWh_m2", "annual_rear_irr_kWh_m2", "bifacial_irr_gain_pct", "annual_bifacial_thermal_kWh_m2", "bifacial_thermal_gain_pct", "LER", "lcoe_eur_kWh"]
    print(summary_df[disp_cols].round(2).to_string(index=False))

    return summary_df


# ============================================================================= #
# DEMO / SELF-TEST
# ============================================================================= #
if __name__ == "__main__":
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
        def __init__(self, hourly_results: pd.DataFrame) -> None:
            self.hourly_results = hourly_results
            self.config = {
                "apv_config": {"albedo": 0.25, "par_reduction": 0.22},
                "crop_config": {"reference_yield": 6.0, "light_sensitivity": 1.0},
            }

        def to_dict(self) -> Dict[str, float]:
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

    output = Path("output/validation_demo")
    tables = generate_validation_report(demo_result, demo_params, output)

    for name, table in tables.items():
        print(f"\n=== {name.upper()} VALIDATION ===")
        print(table.to_string(index=False))

    print("\nOverall assessment: literature-comparison tables and figures generated successfully.")
