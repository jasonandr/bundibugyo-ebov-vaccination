from pathlib import Path

import pandas as pd

from paths import PROJECT_ROOT, result_path


OVERRIDE_PATH = result_path("sitrep_2026_07_05_override.csv")


def _read_cumulative(path, value_name):
    df = pd.read_csv(path, header=None, names=["Country", "Date", value_name])
    df["Date"] = pd.to_datetime(df["Date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


def _append_override(df, value_name, override_column, country="DRC"):
    if not OVERRIDE_PATH.exists():
        return df
    overrides = pd.read_csv(OVERRIDE_PATH)
    overrides = overrides.loc[overrides["country"] == country].copy()
    if overrides.empty or override_column not in overrides.columns:
        return df
    overrides["Date"] = pd.to_datetime(overrides["date"])
    overrides["Country"] = overrides["country"]
    overrides[value_name] = pd.to_numeric(overrides[override_column], errors="coerce")
    overrides = overrides[["Country", "Date", value_name]].dropna(subset=[value_name])
    out = pd.concat([df, overrides], ignore_index=True)
    out = out.sort_values(["Country", "Date"]).drop_duplicates(["Country", "Date"], keep="last")
    return out


def cumulative_confirmed_cases(country="DRC"):
    df = _read_cumulative(
        PROJECT_ROOT / "BDBV2026-Data" / "build" / "long" / "insp_sitrep__national_cumulative_confirmed_cases.csv",
        "Cases",
    )
    return _append_override(df, "Cases", "confirmed_cases", country=country)


def cumulative_confirmed_deaths(country="DRC"):
    df = _read_cumulative(
        PROJECT_ROOT / "BDBV2026-Data" / "build" / "long" / "insp_sitrep__national_cumulative_confirmed_deaths.csv",
        "Deaths",
    )
    return _append_override(df, "Deaths", "confirmed_deaths", country=country)


def cumulative_recovered(country="DRC"):
    df = _read_cumulative(
        PROJECT_ROOT / "BDBV2026-Data" / "build" / "long" / "insp_sitrep__national_cumulative_recovered_cases.csv",
        "Recovered",
    )
    return _append_override(df, "Recovered", "recovered", country=country)


def latest_sitrep(country="DRC"):
    if not OVERRIDE_PATH.exists():
        return None
    overrides = pd.read_csv(OVERRIDE_PATH)
    row = overrides.loc[overrides["country"] == country]
    if row.empty:
        return None
    return row.iloc[-1].to_dict()
