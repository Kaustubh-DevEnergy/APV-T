"""
Utility functions: PVGIS TMY loader, weather summary,
validation, error metrics, and helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_pvgis_tmy(filepath: str) -> pd.DataFrame:
    """
    Load and parse a PVGIS TMY CSV file with hourly data.

    Handles the standard PVGIS text format: header lines,
    8760 (or 8784) hourly rows, and unit metadata at the end.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Locate the hourly data header
    header_idx = None
    footer_idx = None
    for idx, line in enumerate(lines):
        clean = line.strip().lower()
        if clean.startswith("time(utc)") or clean.startswith("time"):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"Could not find hourly data header in {filepath}")

    # Find where hourly rows end
    for idx in range(header_idx + 1, len(lines)):
        s = lines[idx].strip()
        if not s or s.startswith("#"):
            footer_idx = idx
            break

    nrows = (footer_idx - header_idx - 1) if footer_idx else None

    df = pd.read_csv(
        filepath,
        skiprows=header_idx,
        nrows=nrows,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]

    # Map PVGIS column names to standard names
    col_map = {
        "T2m": "T_ambient",
        "G(h)": "GHI",
        "Gb(n)": "DNI",
        "Gd(h)": "DHI",
        "WS10m": "wind_speed",
        "RH": "relative_humidity",
        "SP": "pressure",
        "IR(h)": "infra_red",
    }
    df = df.rename(columns=col_map)

    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if not time_col:
        raise KeyError("Could not identify datetime column")

    # Parse datetime (handles both '20200101:0010' and ISO formats)
    ts = df[time_col].astype(str).str.strip()
    parsed = pd.to_datetime(ts, format="%Y%m%d:%H%M", errors="coerce", utc=True)
    if parsed.isnull().all():
        parsed = pd.to_datetime(ts, errors="coerce", utc=True)

    df.index = pd.DatetimeIndex(parsed)
    df = df.drop(columns=[time_col], errors="ignore")
    df = df[df.index.notnull()]

    # Convert numeric fields
    for col in ["T_ambient", "GHI", "DNI", "DHI", "wind_speed", "relative_humidity", "pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clamp irradiance to non-negative
    for col in ["GHI", "DNI", "DHI"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # Lowercase aliases for convenience
    if "GHI" in df.columns:
        df["ghi"] = df["GHI"]
        df["dni"] = df["DNI"]
        df["dhi"] = df["DHI"]
    if "T_ambient" in df.columns:
        df["temp_air"] = df["T_ambient"]

    return df


def weather_summary(df: pd.DataFrame) -> dict:
    """Print and return basic weather statistics."""
    summary = {}
    print("=" * 50)
    print("WEATHER DATA SUMMARY")
    print("=" * 50)

    for col in ["ghi", "dni", "dhi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            annual = df[col].sum() / 1000.0
            summary[f"annual_{col}_kWh_m2"] = annual
            print(f"  Annual {col.upper():>3s}: {annual:>8.0f} kWh/m2")

    if "ghi" in df.columns and "dhi" in df.columns:
        ghi_mean = df["ghi"].mean()
        if ghi_mean > 0:
            ratio = df["dhi"].mean() / ghi_mean
            summary["dhi_ghi_ratio"] = ratio
            label = "cloudy" if ratio > 0.55 else "moderate" if ratio > 0.40 else "sunny"
            print(f"  DHI/GHI ratio:   {ratio:.2f}  ({label})")

    if "temp_air" in df.columns:
        summary["temp_mean"] = df["temp_air"].mean()
        summary["temp_min"] = df["temp_air"].min()
        summary["temp_max"] = df["temp_air"].max()
        print(f"  Temperature: mean={summary['temp_mean']:.1f}C, "
              f"min={summary['temp_min']:.1f}C, max={summary['temp_max']:.1f}C")

    if "ghi" in df.columns:
        summary["sunshine_hours"] = int((df["ghi"] > 120).sum())
        summary["peak_ghi"] = float(df["ghi"].max())
        print(f"  Sunshine hours:  {summary['sunshine_hours']}")
        print(f"  Peak GHI:        {summary['peak_ghi']:.0f} W/m2")

    summary["n_records"] = len(df)
    print(f"  Records:         {summary['n_records']}")
    print("=" * 50)
    return summary


def validate_weather_data(df: pd.DataFrame) -> dict:
    """Basic validation of weather DataFrame ranges and completeness."""
    warnings_list = []
    valid = True

    ranges = {
        "ghi": (0, 1500),
        "dni": (0, 1200),
        "dhi": (0, 800),
        "temp_air": (-40, 50),
        "wind_speed": (0, 30),
    }
    out_of_range = {}
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            out_of_range[col] = n_bad
            if n_bad > 0:
                warnings_list.append(f"{col}: {n_bad} values outside [{lo}, {hi}]")

    if "ghi" in df.columns:
        annual_ghi = df["ghi"].sum() / 1000.0
        if not (800 <= annual_ghi <= 2500):
            valid = False
            warnings_list.append(f"Annual GHI = {annual_ghi:.0f} kWh/m2, outside expected range")

    return {
        "valid": valid,
        "n_records": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "out_of_range": out_of_range,
        "warnings": warnings_list,
    }


def calculate_error_metrics(measured: np.ndarray, modeled: np.ndarray) -> dict:
    """Error metrics between measured and modeled arrays."""
    measured = np.asarray(measured, dtype=float)
    modeled = np.asarray(modeled, dtype=float)

    mask = np.isfinite(measured) & np.isfinite(modeled) & (measured != 0)
    m = measured[mask]
    p = modeled[mask]

    if len(m) == 0:
        return {"rmsd": np.nan, "mbd": np.nan, "nrmsd": np.nan, "r2": np.nan, "n_points": 0}

    diff = p - m
    rmsd = float(np.sqrt(np.mean(diff ** 2)))
    mbd = float(np.mean(diff))
    mean_m = np.mean(m)
    nrmsd = float(rmsd / mean_m * 100) if mean_m != 0 else np.nan
    mae = float(np.mean(np.abs(diff)))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((m - mean_m) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {
        "rmsd": rmsd,
        "mbd": mbd,
        "nrmsd": nrmsd,
        "mae": mae,
        "r2": r2,
        "max_error": float(np.max(np.abs(diff))),
        "n_points": len(m),
    }


def get_monthly_albedo(timestamp_or_month, albedo_dict=None, default=0.25):
    """Return albedo for a given month or timestamp."""
    if albedo_dict is None:
        return default
    if isinstance(timestamp_or_month, (pd.Timestamp,)):
        month = timestamp_or_month.month
    else:
        month = int(timestamp_or_month)
    return albedo_dict.get(month, default)
