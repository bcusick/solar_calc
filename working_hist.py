"""
solar_budget.py  (Phase 1: keep Python + pvlib; binary-search budget; return data for Android charting)

Designed to be called from Android (Chaquopy) like:
    import solar_budget
    result = solar_budget.run(params_dict)

No matplotlib. All outputs are JSON-serializable (lists, floats, strings, dicts).

Dependencies (same as your script, minus matplotlib):
  - openmeteo_requests
  - requests_cache
  - retry_requests
  - pandas
  - numpy
  - pvlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import math

import numpy as np
import pandas as pd

import openmeteo_requests
import requests_cache
from retry_requests import retry

import pvlib
from pvlib.location import Location
from pvlib import irradiance


# ----------------------------
# Core SOC simulation + search
# ----------------------------

def simulate_soc_daily(
    forecast_wh: np.ndarray,
    battery_wh: float,
    soc0: float,
    daily_budget_wh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate daily battery energy with clamping [0, battery_wh].

    Returns:
      energy_wh:  array len N (end-of-day energy after each day)
      soc_pct:    array len N
      spill_wh:   array len N (energy that would have exceeded max)
    """
    F = np.asarray(forecast_wh, dtype=float)
    n = F.size

    e = float(soc0) * float(battery_wh)
    e_min = 0.0
    e_max = float(battery_wh)

    energy = np.empty(n, dtype=float)
    soc_pct = np.empty(n, dtype=float)
    spill = np.empty(n, dtype=float)

    for i in range(n):
        e_next = e + F[i] - float(daily_budget_wh)

        if e_next > e_max:
            spill[i] = e_next - e_max
            e_next = e_max
        else:
            spill[i] = 0.0

        if e_next < e_min:
            e_next = e_min

        e = e_next
        energy[i] = e
        soc_pct[i] = 100.0 * e / e_max

    return energy, soc_pct, spill


def min_energy_over_horizon(
    forecast_wh: np.ndarray,
    battery_wh: float,
    soc0: float,
    daily_budget_wh: int,
) -> float:
    """
    Fast feasibility helper: returns minimum energy reached over horizon
    (including end-of-day values) for a given integer daily budget.
    """
    F = np.asarray(forecast_wh, dtype=float)
    e = float(soc0) * float(battery_wh)
    e_min = 0.0
    e_max = float(battery_wh)

    min_e = e
    B = float(daily_budget_wh)

    for Fd in F:
        e = e + float(Fd) - B
        if e > e_max:
            e = e_max
        if e < e_min:
            e = e_min
        if e < min_e:
            min_e = e

    return float(min_e)


def find_max_budget_binary(
    forecast_wh: np.ndarray,
    battery_wh: float,
    soc0: float,
    reserve_wh: float,
    *,
    high_start: int = 250,
    hard_cap: int = 200_000,
) -> int:
    """
    Find the maximum integer daily budget (Wh/day) such that the simulated
    battery energy never drops below reserve_wh over the forecast horizon.

    Uses:
      - doubling search to find an infeasible high bound
      - integer binary search to find max feasible budget
    """
    F = np.asarray(forecast_wh, dtype=float)
    battery_wh = float(battery_wh)
    reserve_wh = float(reserve_wh)

    if battery_wh <= 0:
        return 0

    # If reserve is >= starting energy, only 0 might be feasible (or none).
    # We'll still compute normally.
    def feasible(B: int) -> bool:
        return min_energy_over_horizon(F, battery_wh, soc0, B) >= reserve_wh

    low = 0
    if not feasible(low):
        # Even with 0 load, you violate reserve (e.g., reserve > initial energy and no sun).
        return 0

    high = int(high_start)

    # Find first infeasible high by doubling
    while high < hard_cap and feasible(high):
        high *= 2
        print(high)
    # If we never found infeasible within cap, clamp answer to hard_cap - 1
    if high >= hard_cap and feasible(hard_cap):
        return int(hard_cap)

    # Now invariant: feasible(low)=True, feasible(high)=False
    # If high is still feasible but == hard_cap, treat hard_cap as answer.
    if feasible(high):
        return int(high)

    while (high - low) > 1:
        print(f"high: {high}, low: {low}")
        mid = (low + high) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid

    return int(low)


# ----------------------------
# Open-Meteo + pvlib pipeline
# ----------------------------

def fetch_openmeteo_hourly(
    latitude: float,
    longitude: float,
    days: int,
    *,
    cache_dir: Optional[str] = ".cache",
    cache_expire_s: int = 3600,
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> Tuple[pd.DataFrame, str, float]:
    """
    Fetch hourly DNI/DHI/GHI from Open-Meteo Forecast API and return:
      - hourly_dataframe indexed by localized tz
      - timezone string
      - dt_hours (sample interval hours)

    DataFrame columns:
      direct_normal_irradiance (dni) [W/m^2]
      diffuse_radiation (dhi)        [W/m^2]
      shortwave_radiation (ghi)      [W/m^2]
    """
    # Cache: on Android you may want to point cache_dir to app files dir.
    # If cache_dir is None, skip disk caching.
    if cache_dir is None:
        session = requests_cache.CachedSession(backend="memory", expire_after=cache_expire_s)
    else:
        session = requests_cache.CachedSession(cache_dir, expire_after=cache_expire_s)

    retry_session = retry(session, retries=retries, backoff_factor=backoff_factor)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2022-01-01",
        "end_date": "2025-12-31",
        "hourly": ["direct_normal_irradiance", "shortwave_radiation", "diffuse_radiation"],
        "timezone": "auto",
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    timezone = response.Timezone().decode("utf-8")

    dni = hourly.Variables(0).ValuesAsNumpy()
    dhi = hourly.Variables(1).ValuesAsNumpy()
    ghi = hourly.Variables(2).ValuesAsNumpy()

    dt_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(timezone)

    df = pd.DataFrame(
        {
            "dni": dni.astype(float),
            "dhi": dhi.astype(float),
            "ghi": ghi.astype(float),
        },
        index=dt_index,
    )

    # Sample interval hours
    dt_hours = float(hourly.Interval()) / 3600.0

    return df, timezone, dt_hours


def compute_daily_energy(
    hourly_df: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone: str,
    tilt_deg: float,
    azimuth_deg: float,
    panel_watts: float,
    eta: float,
    alpha: float,
) -> pd.DataFrame:
    """
    Use pvlib to compute POA (forecast and clearsky), convert to power and energy,
    and aggregate to daily Wh.

    Returns daily DataFrame indexed by day with:
      forecast_Wh, clear_Wh, percent, sun_hours
    """
    lat = float(latitude)
    lon = float(longitude)

    dni = hourly_df["dni"].to_numpy(dtype=float)
    dhi = hourly_df["dhi"].to_numpy(dtype=float)
    ghi = hourly_df["ghi"].to_numpy(dtype=float)

    # pvlib solar position + clear sky
    location = Location(lat, lon, tz=timezone)
    solpos = pvlib.solarposition.get_solarposition(hourly_df.index, lat, lon)

    clearsky = location.get_clearsky(hourly_df.index, model="ineichen")
    clearsky = clearsky.clip(lower=0)

    # POA from forecast irradiance
    poa = irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(azimuth_deg),
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
    )

    # POA from clearsky irradiance
    poa_clear = irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(azimuth_deg),
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )

    # Convert POA irradiance -> estimated panel power
    # Your original: forecast uses eta * alpha; clear uses alpha only
    power_w = (poa["poa_global"].to_numpy(dtype=float) / 1000.0) * float(panel_watts) * float(eta) * float(alpha)
    clear_power_w = (poa_clear["poa_global"].to_numpy(dtype=float) / 1000.0) * float(panel_watts) * float(alpha)

    power_w = np.clip(power_w, 0.0, None)
    clear_power_w = np.clip(clear_power_w, 0.0, None)

    # Energy per sample
    # hourly_df index spacing is constant per Open-Meteo interval, so derive dt from index
    # If you used hourly interval, dt=1 hour; still safe:
    # Use first two timestamps if available; fall back to 1.
    if len(hourly_df.index) >= 2:
        dt_hours = (hourly_df.index[1] - hourly_df.index[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0

    hourly_df = hourly_df.copy()
    hourly_df["energy_Wh"] = power_w * dt_hours
    hourly_df["clear_energy_Wh"] = clear_power_w * dt_hours

    daily = hourly_df["energy_Wh"].resample("D").sum().to_frame(name="forecast_Wh")
    daily["clear_Wh"] = hourly_df["clear_energy_Wh"].resample("D").sum()

    # Derived metrics (avoid divide-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (daily["forecast_Wh"] / daily["clear_Wh"] * 100.0)
    pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily["percent"] = pct.astype(int)

    # "sun_hours" defined as clear_Wh / panel_watts
    if float(panel_watts) > 0:
        daily["sun_hours"] = daily["clear_Wh"] / float(panel_watts)
    else:
        daily["sun_hours"] = 0.0

    return daily


# ----------------------------
# Phase 1 public entrypoint
# ----------------------------

def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for Android.

    Required params:
      latitude: float
      longitude: float
      tilt: float (deg)
      azimuth: float (deg)
      panel_watts: float
      battery_wh: float
      eta: float
      alpha: float
      reserve_wh: float
      soc0: float (0..1)
      days: int

    Optional:
      cache_dir: str or None
      low_frac: float (default 0.50)
      mid_frac: float (default 0.75)

    Returns JSON-serializable dict with:
      meta, budgets, daily, curves
    """
    lat = float(params["latitude"])
    lon = float(params["longitude"])
    tilt = float(params.get("tilt", 30.0))
    azimuth = float(params.get("azimuth", 180.0))
    panel_watts = float(params.get("panel_watts", 1000.0))
    battery_wh = float(params["battery_wh"])
    eta = float(params.get("eta", 1.0))
    alpha = float(params.get("alpha", 0.9))
    reserve_wh = float(params.get("reserve_wh", 0.0))
    soc0 = float(params.get("soc0", 0.8))
    days = int(params.get("days", 16))

    cache_dir = params.get("cache_dir", ".cache")
    low_frac = float(params.get("low_frac", 0.50))
    mid_frac = float(params.get("mid_frac", 0.75))

    # 1) Fetch Open-Meteo hourly
    hourly_df, timezone, _dt_hours = fetch_openmeteo_hourly(
        lat, lon, days, cache_dir=cache_dir
    )

    # 2) Compute daily energy using pvlib POA + clearsky
    daily_energy = compute_daily_energy(
        hourly_df,
        lat, lon, timezone,
        tilt_deg=tilt,
        azimuth_deg=azimuth,
        panel_watts=panel_watts,
        eta=eta,
        alpha=alpha,
    )

    forecast = daily_energy["forecast_Wh"].to_numpy(dtype=float)
    mean_forecast = float(np.mean(forecast)) if forecast.size else 0.0

    # 3) Binary-search max sustainable budget with reserve constraint
    budget_max = find_max_budget_binary(
        forecast,
        battery_wh=battery_wh,
        soc0=soc0,
        reserve_wh=reserve_wh,
        high_start=mean_forecast,
        hard_cap=200_000,
    )

    budget_low = int(math.floor(budget_max * low_frac))
    budget_mid = int(math.floor(budget_max * mid_frac))
    budget_mean = int(round(mean_forecast))

    # 4) Generate SOC curves for plotting (one simulation each)
    e_low, soc_low, spill_low = simulate_soc_daily(forecast, battery_wh, soc0, budget_low)
    e_mid, soc_mid, spill_mid = simulate_soc_daily(forecast, battery_wh, soc0, budget_mid)
    e_max, soc_max, spill_max = simulate_soc_daily(forecast, battery_wh, soc0, budget_max)

    # "Mean" curve: your original used reserve=0, budget=mean (not searched).
    e_mean, soc_mean, spill_mean = simulate_soc_daily(forecast, battery_wh, soc0, budget_mean)

    # 5) Package output as JSON-serializable
    dates = [d.date().isoformat() for d in daily_energy.index]  # "YYYY-MM-DD"
    out = {
        "meta": {
            "timezone": timezone,
            "latitude": lat,
            "longitude": lon,
            "tilt_deg": tilt,
            "azimuth_deg": azimuth,
            "panel_watts": panel_watts,
            "battery_wh": battery_wh,
            "eta": eta,
            "alpha": alpha,
            "reserve_wh": reserve_wh,
            "soc0": soc0,
            "days": days,
        },
        "budgets_wh_per_day": {
            "mean": budget_mean,
            "low": budget_low,
            "mid": budget_mid,
            "max": budget_max,
        },
        "daily": {
            "dates": dates,
            "forecast_Wh": daily_energy["forecast_Wh"].astype(float).round(3).tolist(),
            "clear_Wh": daily_energy["clear_Wh"].astype(float).round(3).tolist(),
            "percent": daily_energy["percent"].astype(int).tolist(),
            "sun_hours": daily_energy["sun_hours"].astype(float).round(3).tolist(),
        },
        "curves": {
            "soc_pct": {
                "low": soc_low.round(3).tolist(),
                "mid": soc_mid.round(3).tolist(),
                "max": soc_max.round(3).tolist(),
                "mean": soc_mean.round(3).tolist(),
            },
            "energy_wh": {
                "low": e_low.round(3).tolist(),
                "mid": e_mid.round(3).tolist(),
                "max": e_max.round(3).tolist(),
                "mean": e_mean.round(3).tolist(),
            },
            "spill_wh": {
                "low": spill_low.round(3).tolist(),
                "mid": spill_mid.round(3).tolist(),
                "max": spill_max.round(3).tolist(),
                "mean": spill_mean.round(3).tolist(),
            },
        },
    }

    return out


# ----------------------------
# Optional: desktop quick test
# ----------------------------

if __name__ == "__main__":
    # Example usage mirroring your original constants (Xalapa defaults)
    params = {
        "latitude": 19.54234,
        "longitude": -96.92520,
        "tilt": 38,
        "azimuth": 180,
        "panel_watts": 650 * 10,
        "battery_wh": 305 * 3.2 * 16 * 0.8 * 3,  # your expression
        "eta": 1.0,
        "alpha": 0.9,
        "reserve_wh": 2000.0,
        "soc0": 1.0,
        "days": 16,
        # "cache_dir": None,  # use in-memory cache if desired
    }

    result = run(params)
    print("Budgets (Wh/day):", result["budgets_wh_per_day"])

