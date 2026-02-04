'''
Tool to forecast expected energy received for a given Lat/Long location.  Uses OpenMeteo API for forecast solar data and
PVlib to model PV panel behavior.
This data is used to model battery bank performance and determine allowable Wh energy use for the next n days.
'''
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import pvlib
from pvlib import irradiance
import matplotlib.pyplot as plt
from pvlib.location import Location
import numpy as np

###
# INPUTS
###
# mills river
#latitude = 35.37786
#longitude = -82.58089
# Xalapa
#latitude = 19.54234
#longitude = -96.92520
#la paz
latitude = 24.13277 
longitude = -110.32305

max_watts = 1200
min_watts = 100  # W
#panel_cost = 400

max_batteries = 6
battery_wh_per_battery = 2000  # Wh
#battery_cost = 1000

eta = 1      # percent insolation captured
alpha = 0.9  # accounts for electrical losses to/from battery

reserve_wh = 500  # Wh min allowable in the whole battery bank
SOC = 1            # day 1 starting charge


def daily_soc(forecast_wh, battery_wh, SOC, reserve_pct, limit=1e6):
    """
    forecast_wh: 1D array-like (Wh/day)
    battery_wh: total bank capacity (Wh)
    reserve_pct: minimum SOC (%) allowed
    """
    F = np.asarray(forecast_wh, dtype=float)

    E_min = 0
    E_max = battery_wh
    min_reserve = 100
    budget = 200  # starting guess

    while (min_reserve > reserve_pct) and (budget < limit):
        if reserve_pct == 0:  # special case to show dead battery
            budget = limit
        else:
            budget += 10
            #print(budget)

        e = SOC * battery_wh
        E = []
        spill = []

        for Fd in F:
            e_next = e + Fd - budget

            # clamp and track spill
            if e_next > E_max:
                spill.append(e_next - E_max)
                e_next = E_max
            else:
                spill.append(0.0)

            e_next = max(E_min, e_next)
            e = e_next
            E.append(e)

        df = pd.DataFrame({
            "battery_wh": E,
            "soc_pct": [100 * e / battery_wh for e in E],
            "spill_wh": spill,
        })

        min_reserve = df["soc_pct"].min()

    return df, budget


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "hourly": ["direct_normal_irradiance", "shortwave_radiation", "diffuse_radiation"],
    "timezone": "auto",
}
responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Process hourly data
hourly = response.Hourly()
timezone = response.Timezone().decode('utf-8')

dni = hourly.Variables(0).ValuesAsNumpy()
dhi = hourly.Variables(1).ValuesAsNumpy()
ghi = hourly.Variables(2).ValuesAsNumpy()

dt_index = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left",
).tz_convert(timezone)

hourly_df = pd.DataFrame(
    {
        "direct_normal_irradiance": dni,
        "diffuse_radiation": dhi,
        "shortwave_radiation": ghi,
    },
    index=dt_index,
)

# pvlib solar position
location = Location(latitude, longitude, tz=timezone)
solpos = pvlib.solarposition.get_solarposition(hourly_df.index, latitude, longitude)

# approx. best tilt per day
daily_tilt = (
    solpos["apparent_zenith"]
    .clip(lower=5, upper=60)
    .resample("D")
    .min()
)

# map that daily value onto each hour
tilt_hourly = daily_tilt.reindex(hourly_df.index.floor("D")).to_numpy()

surface_azimuth = 180  # south

poa = irradiance.get_total_irradiance(
    surface_tilt=0,
    surface_azimuth=surface_azimuth,
    solar_zenith=solpos["apparent_zenith"].to_numpy(),
    solar_azimuth=solpos["azimuth"].to_numpy(),
    dni=dni,
    ghi=ghi,
    dhi=dhi,
)

# Convert POA irradiance (W/m^2) -> estimated panel power (W)
hourly_df["power_W"] = poa["poa_global"] / 1000.0 * eta * alpha
hourly_df["power_W"] = hourly_df["power_W"].clip(lower=0)

# Convert power -> energy per sample (Wh)
dt_hours = hourly.Interval() / 3600.0
hourly_df["energy_Wh"] = hourly_df["power_W"] * dt_hours
daily_df = hourly_df["energy_Wh"].resample("D").sum().to_frame(name="historical_Wh")

sizes = np.arange(min_watts, max_watts + min_watts, min_watts)

# ---- NEW: outer loop over battery count ----
results = {}  # batt_count -> list of budgets across panels
#total_cost = {}

for batt_count in range(1, max_batteries + 1):
    battery_total_wh = batt_count * battery_wh_per_battery
    reserve_pct = (reserve_wh / battery_total_wh) * 100.0

    use_limits = []
    #costs = []
    for size in sizes:
        print(size)
        array_out = (daily_df["historical_Wh"] * size).to_numpy()  # 1D Wh/day
        charge_max, daily_budget_max = daily_soc(array_out, battery_total_wh, SOC, reserve_pct)
        #cost = panel * panel_cost + batt_count * battery_cost
        use_limits.append(daily_budget_max)
        #costs.append(cost)

    results[batt_count] = use_limits
    #total_cost[batt_count] = costs

energies = pd.DataFrame(results)
#monies = pd.DataFrame(total_cost)

print(energies)
#print(monies)



# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6))

for batt_count, use_limits in results.items():
    ax.plot(
        sizes,
        use_limits,
        linestyle="-",
        linewidth=2,
        label=f"{batt_count} batt ({batt_count * battery_wh_per_battery:.0f} Wh)",
    )

ax.set_xlabel("watts")
ax.set_ylabel("Max daily budget (Wh/day)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
