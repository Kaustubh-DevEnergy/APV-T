"""
Plotting utilities for APV-T simulation study.

Generates publication-quality figures (300 DPI) for:
- Daily and seasonal irradiance profiles
- Monthly irradiance and thermal yield
- Irradiance heatmaps
- ISO 9806 efficiency curves
- Parametric sensitivity sweeps
- LER breakdown
- Bifacial gain comparison
- Temperature sensitivity
- Orientation polar plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# ============================================================================= #
# STYLE AND COLOUR CONSTANTS
# ============================================================================= #
FRONT = "#1976D2"
REAR = "#FF8F00"
TOTAL = "#2E7D32"
MONO = "#757575"

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PathLike = Union[str, Path]


# ============================================================================= #
# INTERNAL HELPERS
# ============================================================================= #

def _save_figure(fig: Figure, save_path: Optional[PathLike]) -> None:
    """Save figure at 300 DPI if path is given."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], function_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{function_name} requires missing columns: {missing}")


def _monthly_energy(series: pd.Series) -> pd.Series:
    """Aggregate hourly power/irradiance samples (W/m2 or W) to monthly energy."""
    return series.groupby(series.index.month).sum() / 1000.0


def _all_months(monthly: pd.Series) -> pd.Series:
    """Reindex a monthly Series to calendar months 1–12, filling absent months with zero."""
    return monthly.reindex(np.arange(1, 13), fill_value=0.0)


def _coerce_curve_data(thermal_model_or_curve_data: Any) -> pd.DataFrame:
    """Accept either a thermal-model object or a precomputed curve DataFrame."""
    if isinstance(thermal_model_or_curve_data, pd.DataFrame):
        curve_df = thermal_model_or_curve_data.copy()
    elif hasattr(thermal_model_or_curve_data, "efficiency_curve_data"):
        curve_df = thermal_model_or_curve_data.efficiency_curve_data()
    else:
        raise TypeError(
            "thermal_model_or_curve_data must be a pandas DataFrame or an "
            "object exposing efficiency_curve_data()."
        )
    _require_columns(curve_df, ["G", "T_mean", "reduced_temperature", "efficiency"], "plot_efficiency_curves")
    return curve_df


# ============================================================================= #
# 1. DAILY IRRADIANCE PROFILE
# ============================================================================= #

def plot_daily_irradiance_profile(
    hourly_df: pd.DataFrame,
    date_str: str,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Plot front/rear/total irradiance plus component detail for one day."""
    required = [
        "G_front", "G_rear", "G_total",
        "G_beam_front", "G_beam_rear",
        "G_diff_front", "G_diff_rear",
        "G_gnd_front", "G_gnd_rear",
    ]
    _require_columns(hourly_df, required, "plot_daily_irradiance_profile")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    try:
        day = hourly_df.loc[date_str]
    except KeyError as exc:
        raise ValueError(f"No records found for date '{date_str}'.") from exc
    if isinstance(day, pd.Series):
        day = day.to_frame().T
    if day.empty:
        raise ValueError(f"No records found for date '{date_str}'.")

    hours = day.index.hour
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    ax_top.fill_between(hours, day["G_total"], color=TOTAL, alpha=0.18, label="Total (front + rear)")
    ax_top.plot(hours, day["G_total"], color=TOTAL, linewidth=2.5, label="Total irradiance")
    ax_top.plot(hours, day["G_front"], color=FRONT, linewidth=2.0, label="Front")
    ax_top.plot(hours, day["G_rear"], color=REAR, linewidth=2.0, label="Rear")
    ax_top.set_title(f"Bifacial irradiance profile — {pd.Timestamp(date_str).date()}")
    ax_top.set_ylabel("Irradiance [W/m2]")
    ax_top.legend(ncol=3, loc="upper left")
    ax_top.set_ylim(bottom=0)

    components = [
        ("G_beam_front", FRONT, "-", "Front beam"),
        ("G_diff_front", FRONT, "--", "Front diffuse"),
        ("G_gnd_front", FRONT, ":", "Front ground-reflected"),
        ("G_beam_rear", REAR, "-", "Rear beam"),
        ("G_diff_rear", REAR, "--", "Rear diffuse"),
        ("G_gnd_rear", REAR, ":", "Rear ground-reflected"),
    ]
    for column, color, style, label in components:
        ax_bottom.plot(hours, day[column], color=color, linestyle=style, linewidth=1.8, label=label)
    ax_bottom.set_xlabel("Hour of day [local time]")
    ax_bottom.set_ylabel("Irradiance component [W/m2]")
    ax_bottom.set_ylim(bottom=0)
    ax_bottom.set_xticks(np.arange(0, 24, 2))
    ax_bottom.legend(ncol=2, loc="upper left")

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 2. SEASONAL DAILY PROFILES
# ============================================================================= #

def plot_seasonal_daily_profiles(
    hourly_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Plot front, rear, total irradiance profiles for solstice/equinox days (2x2)."""
    _require_columns(hourly_df, ["G_front", "G_rear", "G_total"], "plot_seasonal_daily_profiles")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    year = int(hourly_df.index[0].year)
    targets = [
        (f"{year}-12-21", "Winter solstice (Dec 21)"),
        (f"{year}-03-21", "Spring equinox (Mar 21)"),
        (f"{year}-06-21", "Summer solstice (Jun 21)"),
        (f"{year}-09-21", "Autumn equinox (Sep 21)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()

    unique_dates = pd.DatetimeIndex(hourly_df.index.normalize().unique())
    for ax, (target_string, label) in zip(axes_flat, targets):
        target = pd.Timestamp(target_string, tz=hourly_df.index.tz)
        nearest_date = unique_dates[np.argmin(np.abs(unique_dates - target))]
        day = hourly_df.loc[hourly_df.index.normalize() == nearest_date]
        hours = day.index.hour

        ax.fill_between(hours, day["G_total"], color=TOTAL, alpha=0.16)
        ax.plot(hours, day["G_total"], color=TOTAL, linewidth=2.2, label="Total")
        ax.plot(hours, day["G_front"], color=FRONT, linewidth=1.8, label="Front")
        ax.plot(hours, day["G_rear"], color=REAR, linewidth=1.8, label="Rear")
        ax.set_title(f"{label}\n({nearest_date.date()})")
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, 24, 4))

    for ax in axes[:, 0]:
        ax.set_ylabel("Irradiance [W/m2]")
    for ax in axes[-1, :]:
        ax.set_xlabel("Hour of day [local time]")
    axes_flat[0].legend(ncol=3, loc="upper left")

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 3. MONTHLY IRRADIANCE
# ============================================================================= #

def plot_monthly_irradiance(
    hourly_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Grouped bar chart of monthly front/rear POA irradiance."""
    _require_columns(hourly_df, ["G_front", "G_rear"], "plot_monthly_irradiance")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    front = _all_months(_monthly_energy(hourly_df["G_front"]))
    rear = _all_months(_monthly_energy(hourly_df["G_rear"]))
    months = np.arange(1, 13)
    x = np.arange(len(months))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x - width / 2, front.values, width=width, color=FRONT, label="Front")
    ax.bar(x + width / 2, rear.values, width=width, color=REAR, label="Rear")
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Plane-of-array irradiance [kWh/m2/month]")
    ax.set_title("Monthly bifacial collector irradiance")
    ax.legend()
    ax.set_ylim(bottom=0)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 4. MONTHLY THERMAL YIELD
# ============================================================================= #

def plot_monthly_thermal(
    hourly_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Monthly bifacial/monofacial thermal energy + bifacial gain on secondary axis."""
    _require_columns(hourly_df, ["Q_bifacial_kWh", "Q_monofacial_kWh"], "plot_monthly_thermal")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    bifi = _all_months(hourly_df["Q_bifacial_kWh"].groupby(hourly_df.index.month).sum())
    mono = _all_months(hourly_df["Q_monofacial_kWh"].groupby(hourly_df.index.month).sum())
    gain = np.where(mono.values > 0, (bifi.values - mono.values) / mono.values * 100.0, np.nan)

    months = np.arange(1, 13)
    x = np.arange(len(months))
    width = 0.36

    fig, ax_left = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars_bifi = ax_left.bar(x - width / 2, bifi.values, width=width, color=TOTAL, label="Bifacial")
    bars_mono = ax_left.bar(x + width / 2, mono.values, width=width, color=MONO, label="Monofacial")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax_left.set_xlabel("Month")
    ax_left.set_ylabel("Useful thermal energy [kWh/month]")
    ax_left.set_ylim(bottom=0)
    ax_left.set_title("Monthly thermal yield and bifacial thermal gain")

    ax_right = ax_left.twinx()
    line = ax_right.plot(x, gain, color=REAR, marker="o", linewidth=2.0, label="Bifacial thermal gain")
    ax_right.set_ylabel("Bifacial thermal gain [%]", color=REAR)
    ax_right.tick_params(axis="y", labelcolor=REAR)

    handles = [bars_bifi, bars_mono, line[0]]
    ax_left.legend(handles, [h.get_label() for h in handles], loc="upper left")

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 5. IRRADIANCE HEATMAP
# ============================================================================= #

def plot_irradiance_heatmap(
    hourly_df: pd.DataFrame,
    column: str = "G_total",
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Month x hour-of-day heatmap of mean irradiance."""
    _require_columns(hourly_df, [column], "plot_irradiance_heatmap")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    working = pd.DataFrame({"month": hourly_df.index.month, "hour": hourly_df.index.hour, "value": hourly_df[column].values})
    pivot = working.pivot_table(index="hour", columns="month", values="value", aggfunc="mean")
    pivot = pivot.reindex(index=np.arange(24), columns=np.arange(1, 13))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean irradiance [W/m2]"}, linewidths=0.15, linecolor="white")
    ax.set_title(f"Mean hourly {column.replace('_', ' ')} by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day [local time]")
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=0)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 6. ISO 9806 EFFICIENCY CURVES
# ============================================================================= #

def plot_efficiency_curves(
    thermal_model_or_curve_data: Any,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Efficiency vs reduced-temperature curves at several irradiance levels."""
    curve_df = _coerce_curve_data(thermal_model_or_curve_data)
    levels = sorted(curve_df["G"].dropna().unique())
    palette = sns.color_palette("viridis", n_colors=max(len(levels), 2))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for color, G in zip(palette, levels):
        subset = curve_df.loc[curve_df["G"] == G].sort_values("reduced_temperature")
        ax.plot(subset["reduced_temperature"], subset["efficiency"], color=color, linewidth=2.0, label=f"G = {G:g} W/m2")

    nominal_Tm, nominal_Ta, nominal_G = 65.0, 20.0, 800.0
    nominal_redT = (nominal_Tm - nominal_Ta) / nominal_G
    if hasattr(thermal_model_or_curve_data, "steady_state_efficiency"):
        nominal_eta = float(np.asarray(
            thermal_model_or_curve_data.steady_state_efficiency(nominal_G, nominal_Tm, nominal_Ta)
        ))
        ax.scatter(nominal_redT, nominal_eta, s=65, color="black", zorder=5, label="Nominal IGFPC point (65C, 20C, 800 W/m2)")

    ax.set_xlabel("Reduced temperature difference, (T_m - T_a) / G [m2*K/W]")
    ax.set_ylabel("Thermal efficiency, eta [-]")
    ax.set_title("ISO 9806:2017 IGFPC steady-state efficiency curves")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.0)
    ax.legend(loc="upper right")

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 7. SINGLE-PARAMETER SWEEP
# ============================================================================= #

def plot_parametric_sweep(
    sweep_df: pd.DataFrame,
    param_name: str,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Bifacial thermal yield bars + LER line against swept parameter."""
    required = [param_name, "LER", "annual_thermal_yield_bifacial_kWh_per_m2"]
    _require_columns(sweep_df, required, "plot_parametric_sweep")
    df = sweep_df.sort_values(param_name)

    x = np.arange(len(df))
    labels = [f"{value:g}" if isinstance(value, (float, np.floating, int, np.integer)) else str(value) for value in df[param_name]]

    fig, ax_left = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax_left.bar(x, df["annual_thermal_yield_bifacial_kWh_per_m2"], color=TOTAL, alpha=0.88, label="Bifacial thermal yield")
    ax_left.set_ylabel("Bifacial thermal yield [kWh/m2]")
    ax_left.set_xlabel(param_name.replace("_", " "))
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylim(bottom=0)
    ax_left.set_title(f"APV-T sensitivity to {param_name.replace('_', ' ')}")

    ax_right = ax_left.twinx()
    line = ax_right.plot(x, df["LER"], color=FRONT, marker="o", linewidth=2.2, markersize=6, label="Land Equivalent Ratio")
    ax_right.set_ylabel("Land Equivalent Ratio, LER [-]", color=FRONT)
    ax_right.tick_params(axis="y", labelcolor=FRONT)
    ax_right.axhline(1.0, color=MONO, linestyle="--", linewidth=1.0, alpha=0.8)

    ax_left.legend([bars, line[0]], [bars.get_label(), line[0].get_label()], loc="upper left")

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 8. TWO-DIMENSIONAL GRID HEATMAP
# ============================================================================= #

def plot_parametric_grid_heatmap(
    grid_df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Annotated heatmap for 2-D parameter grid sweep."""
    _require_columns(grid_df, [param1, param2, metric], "plot_parametric_grid_heatmap")
    pivot = grid_df.pivot_table(index=param1, columns=param2, values=metric, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".3g", cmap="YlGnBu", linewidths=0.5, linecolor="white", cbar_kws={"label": metric.replace("_", " ")}, ax=ax)
    ax.set_title(title if title is not None else f"{metric.replace('_', ' ')} by {param1.replace('_', ' ')} and {param2.replace('_', ' ')}")
    ax.set_xlabel(param2.replace("_", " "))
    ax.set_ylabel(param1.replace("_", " "))

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 9. LER BREAKDOWN
# ============================================================================= #

def plot_LER_breakdown(
    results_list: Sequence[Union[Mapping[str, Any], Any]],
    labels: Sequence[str],
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Stacked LER breakdown into crop and thermal contributions."""
    if len(results_list) != len(labels):
        raise ValueError("results_list and labels must have the same length.")
    if not results_list:
        raise ValueError("results_list must contain at least one result.")

    crop_ratios: List[float] = []
    energy_ratios: List[float] = []
    for result in results_list:
        if hasattr(result, "to_dict"):
            data = result.to_dict()
            config = getattr(result, "config", {})
            reference_yield = data.get("reference_yield", config.get("reference_yield"))
        else:
            data = dict(result)
            reference_yield = data.get("reference_yield")

        if reference_yield is None:
            raise ValueError("Each LER result needs a reference_yield value.")
        crop_yield = float(data["estimated_crop_yield_t_per_ha"])
        thermal_yield = float(data["annual_thermal_yield_bifacial_kWh_per_m2"])
        reference_thermal = float(data["reference_thermal_yield_kWh_per_m2"])

        crop_ratios.append(crop_yield / float(reference_yield))
        energy_ratios.append(thermal_yield / reference_thermal if reference_thermal > 0 else np.nan)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x, crop_ratios, color="#8BC34A", label="Crop contribution")
    ax.bar(x, energy_ratios, bottom=crop_ratios, color=TOTAL, label="Thermal energy contribution")
    ax.axhline(1.0, color=MONO, linestyle="--", linewidth=1.5, label="LER = 1 reference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("APV-T configuration")
    ax.set_ylabel("Land Equivalent Ratio, LER [-]")
    ax.set_title("Land Equivalent Ratio breakdown")
    ax.legend()
    ax.set_ylim(bottom=0)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 10. BIFACIAL GAIN COMPARISON
# ============================================================================= #

def plot_bifacial_gain_comparison(
    hourly_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Monthly irradiance and thermal bifacial gain comparison."""
    _require_columns(hourly_df, ["G_front", "G_rear", "Q_bifacial_kWh", "Q_monofacial_kWh"], "plot_bifacial_gain_comparison")
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("hourly_df must have a DatetimeIndex.")

    front = _all_months(_monthly_energy(hourly_df["G_front"]))
    rear = _all_months(_monthly_energy(hourly_df["G_rear"]))
    bifi_q = _all_months(hourly_df["Q_bifacial_kWh"].groupby(hourly_df.index.month).sum())
    mono_q = _all_months(hourly_df["Q_monofacial_kWh"].groupby(hourly_df.index.month).sum())

    irr_gain = np.where(front.values > 0, rear.values / front.values * 100.0, np.nan)
    thermal_gain = np.where(mono_q.values > 0, (bifi_q.values - mono_q.values) / mono_q.values * 100.0, np.nan)

    x = np.arange(12)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x - width / 2, irr_gain, width=width, color=REAR, label="Irradiance bifacial gain")
    ax.bar(x + width / 2, thermal_gain, width=width, color=TOTAL, label="Thermal bifacial gain")
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Bifacial gain [%]")
    ax.set_title("Monthly bifacial irradiance and thermal gains")
    ax.legend()
    ax.set_ylim(bottom=0)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 11. TEMPERATURE SENSITIVITY
# ============================================================================= #

def plot_temperature_sensitivity(
    temp_sweep_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Thermal yield vs mean fluid temperature with bifacial advantage shaded."""
    _require_columns(temp_sweep_df, ["T_mean", "yield_bifacial", "yield_monofacial"], "plot_temperature_sensitivity")
    df = temp_sweep_df.sort_values("T_mean")

    x = df["T_mean"].to_numpy(dtype=float)
    y_bifi = df["yield_bifacial"].to_numpy(dtype=float)
    y_mono = df["yield_monofacial"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(x, y_bifi, color=TOTAL, marker="o", linewidth=2.4, label="Bifacial IGFPC")
    ax.plot(x, y_mono, color=MONO, marker="s", linewidth=2.1, linestyle="--", label="Monofacial reference")
    ax.fill_between(x, y_mono, y_bifi, where=y_bifi >= y_mono, color=TOTAL, alpha=0.18, label="Bifacial thermal advantage")
    ax.set_xlabel("Mean fluid temperature, T_m [degC]")
    ax.set_ylabel("Thermal yield [kWh/m2]")
    ax.set_title("Thermal-yield sensitivity to fluid temperature")
    ax.legend()
    ax.set_ylim(bottom=0)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# 12. ORIENTATION POLAR PLOT
# ============================================================================= #

def plot_orientation_polar(
    orientation_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> Figure:
    """Orientation sensitivity on a polar axis (North at top, CW)."""
    _require_columns(orientation_df, ["surface_azimuth"], "plot_orientation_polar")
    metric = "LER" if "LER" in orientation_df.columns else "annual_total_irradiance_kWh_per_m2"
    _require_columns(orientation_df, [metric], "plot_orientation_polar")

    df = orientation_df.sort_values("surface_azimuth")
    theta = np.radians(df["surface_azimuth"].to_numpy(dtype=float) % 360.0)
    radius = df[metric].to_numpy(dtype=float)

    if len(theta) > 2:
        theta = np.r_[theta, theta[0]]
        radius = np.r_[radius, radius[0]]

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"}, constrained_layout=True)
    ax.plot(theta, radius, color=TOTAL, marker="o", linewidth=2.2)
    ax.fill(theta, radius, color=TOTAL, alpha=0.14)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title(f"Orientation sensitivity — {metric.replace('_', ' ')}", pad=18)
    ax.set_rlabel_position(135)

    _save_figure(fig, save_path)
    return fig


# ============================================================================= #
# MASTER FIGURE GENERATOR
# ============================================================================= #

def generate_all_plots(
    hourly_df: pd.DataFrame,
    parametric_results: Mapping[str, pd.DataFrame],
    thermal_model: Any,
    output_dir: PathLike,
) -> Dict[str, Figure]:
    """Generate and save the complete standard APV-T figure set."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, Figure] = {}

    dates = pd.DatetimeIndex(hourly_df.index.normalize().unique())
    first_date = dates[0].strftime("%Y-%m-%d")
    year = dates[0].year
    target_date = pd.Timestamp(f"{year}-06-21", tz=hourly_df.index.tz)
    daily_date = dates[np.argmin(np.abs(dates - target_date))].strftime("%Y-%m-%d")

    figures["daily_irradiance_profile"] = plot_daily_irradiance_profile(hourly_df, daily_date, out_dir / "daily_irradiance_profile.png")
    figures["seasonal_daily_profiles"] = plot_seasonal_daily_profiles(hourly_df, out_dir / "seasonal_daily_profiles.png")
    figures["monthly_irradiance"] = plot_monthly_irradiance(hourly_df, out_dir / "monthly_irradiance.png")
    figures["monthly_thermal"] = plot_monthly_thermal(hourly_df, out_dir / "monthly_thermal.png")
    figures["irradiance_heatmap"] = plot_irradiance_heatmap(hourly_df, "G_total", out_dir / "irradiance_heatmap.png")
    figures["efficiency_curves"] = plot_efficiency_curves(thermal_model, out_dir / "efficiency_curves.png")
    figures["bifacial_gain_comparison"] = plot_bifacial_gain_comparison(hourly_df, out_dir / "bifacial_gain_comparison.png")

    for key, sweep_df in parametric_results.items():
        if key == "row_spacing_x_albedo":
            figures["row_spacing_x_albedo_heatmap"] = plot_parametric_grid_heatmap(
                sweep_df, "row_spacing", "albedo", "LER",
                title="Land Equivalent Ratio: row spacing x albedo",
                save_path=out_dir / "row_spacing_x_albedo_LER.png",
            )
        elif key == "T_mean_fluid":
            renamed = sweep_df.rename(columns={
                "annual_thermal_yield_bifacial_kWh_per_m2": "yield_bifacial",
                "annual_thermal_yield_monofacial_kWh_per_m2": "yield_monofacial",
                "T_mean_fluid": "T_mean",
            })
            figures["temperature_sensitivity"] = plot_temperature_sensitivity(renamed, out_dir / "temperature_sensitivity.png")
        elif key in sweep_df.columns and "LER" in sweep_df.columns and "annual_thermal_yield_bifacial_kWh_per_m2" in sweep_df.columns:
            figures[f"sweep_{key}"] = plot_parametric_sweep(sweep_df, key, out_dir / f"sweep_{key}.png")

    if "surface_azimuth" in parametric_results:
        figures["orientation_polar"] = plot_orientation_polar(parametric_results["surface_azimuth"], out_dir / "orientation_polar.png")

    return figures


# ============================================================================= #
# DEMO / SELF-TEST
# ============================================================================= #
if __name__ == "__main__":
    from pathlib import Path

    class _DemoThermalModel:
        eta_0 = 0.856
        a1 = 4.069
        a2 = 0.009

        def steady_state_efficiency(self, G: float, Tm: float, Ta: float) -> np.ndarray:
            G_arr = np.asarray(G, dtype=float)
            dT = np.asarray(Tm, dtype=float) - np.asarray(Ta, dtype=float)
            valid = G_arr > 0
            G_safe = np.where(valid, G_arr, 1.0)
            eta = self.eta_0 - self.a1 * dT / G_safe - self.a2 * dT ** 2 / G_safe
            return np.clip(np.where(valid, eta, 0.0), 0.0, self.eta_0)

        def efficiency_curve_data(self) -> pd.DataFrame:
            Tm = np.linspace(20.0, 120.0, 101)
            frames = []
            for G in (200.0, 400.0, 600.0, 800.0, 1000.0):
                frames.append(pd.DataFrame({"G": G, "T_mean": Tm, "reduced_temperature": (Tm - 20.0) / G, "efficiency": self.steady_state_efficiency(G, Tm, 20.0)}))
            return pd.concat(frames, ignore_index=True)

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

    hourly_demo = pd.DataFrame(
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
    parametric_demo = {
        "row_spacing": spacing_sweep, "T_mean_fluid": temp_sweep,
        "surface_azimuth": orientation, "row_spacing_x_albedo": grid,
    }

    output = Path("output/plots_demo")
    figures = generate_all_plots(hourly_demo, parametric_demo, _DemoThermalModel(), output)
    print(f"Generated {len(figures)} publication-style figures in: {output.resolve()}")
    plt.close("all")
