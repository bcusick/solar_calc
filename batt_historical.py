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
#INPUTS
###

# mills river
#latitude = 35.37786
#longitude = -82.58089
#Richmond
#latitude = 37.51159749662755
#longitude = -77.47304198592848
#Xalapa
latitude = 19.54234
longitude = -96.92520
#la paz
#latitude = 24.13277 
#longitude = -110.32305

panel_watts = 650*10
battery = 305*3.2*16*.8 * 2 #wh
eta = 1 #percent insolation captured
alpha = 0.9 #accounts for electrical losses to/from battery
reserve = 2000     #wh min allowable in battery bank
SOC = 1 #day 1 starting charge
reserve = reserve / battery * 100 #%


def daily_soc(forecast_wh, battery, SOC, reserve, limit=1e6):
    F = np.asarray(forecast_wh, dtype=float)

    E_min = 0
    E_max = battery
    min_reserve = 100
    budget = 14000
    
    while(min_reserve > reserve) and (budget < limit):
        if reserve == 0: #special case to show dead battery
            budget = limit
        else:
            budget += 1
            print(budget)
        e = SOC * battery
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
            "soc_pct": [100*e/battery for e in E],
            "spill_wh": spill,
        })
        min_reserve = df["soc_pct"].min()
        
    return df, budget

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
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
    surface_tilt=38,
    surface_azimuth=surface_azimuth,
    solar_zenith=solpos["apparent_zenith"].to_numpy(),
    solar_azimuth=solpos["azimuth"].to_numpy(),
    dni=dni,
    ghi=ghi,
    dhi=dhi,
)

# Convert POA irradiance (W/m^2) -> estimated panel power (W)
hourly_df["power_W"] = poa["poa_global"] / 1000.0 * panel_watts * eta * alpha
hourly_df["power_W"] = hourly_df["power_W"].clip(lower=0)

# Convert power -> energy per sample (Wh)
dt_hours = hourly.Interval() / 3600.0
hourly_df["energy_Wh"] = hourly_df["power_W"] * dt_hours
daily_df = hourly_df["energy_Wh"].resample("D").sum().to_frame(name="historical_Wh")

charge_max, daily_budget_max = daily_soc(daily_df, battery, SOC, reserve)
charge_max.index = daily_df.index

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    charge_max["soc_pct"],
    linestyle="-",
    linewidth=1,
    color="grey",
    label=f"Max ({daily_budget_max:.0f} Wh)",
)

ax.set_ylabel("State of Charge (%)")
ax.set_ylim(0, 105)

# ---- Combined legend ----
lines1, labels1 = ax.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 , labels1 , loc="upper left")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

