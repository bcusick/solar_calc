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
tilt = 0  # tilt angle of the panel in degrees
azimuth = 180  # azimuth angle (south-facing)
panel_watts = 480
battery = 4300 #wh
eta = 1 #percent insolation captured
alpha = 0.9 #accounts for electrical losses to/from battery
reserve = 500     #wh
SOC = 0.8
#reserve_frac = 0.5
reserve_frac = reserve / battery
days = 16
run_to_limit = 0  #set true to run to min during set days, otherwise mean energy will be used
daily_limit = 2000

def daily_soc(forecast_wh, battery, SOC, reserve_frac, limit):
    F = np.asarray(forecast_wh, dtype=float)

    E_min = 0
    E_max = battery
    min_reserve = 100
    budget = 0
    print (f"limit = {limit}")
    
    while(min_reserve > reserve_frac*100) and (budget < limit):
        budget += 1
        # ABSOLUTE starting energy (kWh)
        e = SOC * battery
        E = []
        spill = []  
        for Fd in F:
            #Id = alpha * eta * Fd  # match budget assumptions

            # Update absolute battery energy
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
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": ["direct_normal_irradiance", "diffuse_radiation", "shortwave_radiation"],
    "models": "best_match",
    "timezone": "auto",
    "forecast_days": days,
    "past_days": 0
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

hourly_dataframe = pd.DataFrame(
    {
        "direct_normal_irradiance": dni,
        "diffuse_radiation": dhi,
        "shortwave_radiation": ghi,
    },
    index=dt_index,
)

# pvlib solar position
location = Location(latitude, longitude, tz=timezone)
solar_position = pvlib.solarposition.get_solarposition(hourly_dataframe.index, latitude, longitude)
clearsky = location.get_clearsky(hourly_dataframe.index, model="ineichen")  # GHI, DNI, DHI in W/m^2
clearsky = clearsky.clip(lower=0)
poa = irradiance.get_total_irradiance(
    #surface_tilt=solar_position["apparent_zenith"].clip(lower=0, upper=90),
    surface_tilt = tilt,
    #surface_azimuth=solar_position["azimuth"],
    surface_azimuth=azimuth,
    solar_zenith=solar_position["apparent_zenith"],
    solar_azimuth=solar_position["azimuth"],
    dni=dni,
    ghi=ghi,
    dhi=dhi,
)

poa_clear = irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solar_position["apparent_zenith"],
    solar_azimuth=solar_position["azimuth"],
    dni=clearsky['dni'],
    ghi=clearsky['ghi'],
    dhi=clearsky['dhi'],
)

# Convert POA irradiance (W/m^2) -> estimated panel power (W)
hourly_dataframe["power_W"] = poa["poa_global"] / 1000.0 * panel_watts * eta * alpha
hourly_dataframe["clear_power_W"] = poa_clear["poa_global"] / 1000.0 * panel_watts * alpha
hourly_dataframe["power_W"] = hourly_dataframe["power_W"].clip(lower=0)
hourly_dataframe["clear_power_W"] = hourly_dataframe["clear_power_W"].clip(lower=0)

# Convert power -> energy per sample (Wh)
dt_hours = hourly.Interval() / 3600.0
hourly_dataframe["energy_Wh"] = hourly_dataframe["power_W"] * dt_hours
hourly_dataframe["clear_energy_Wh"] = hourly_dataframe["clear_power_W"] * dt_hours

daily_energy = hourly_dataframe["energy_Wh"].resample("D").sum().to_frame(name="forecast_Wh")
daily_energy["clear_Wh"] = hourly_dataframe["clear_energy_Wh"].resample("D").sum()
daily_energy['percent'] = (daily_energy['forecast_Wh'] / daily_energy['clear_Wh'] * 100).astype(int)
daily_energy['sun_hours'] = daily_energy["clear_Wh"] / panel_watts

forecast = daily_energy["forecast_Wh"] 
mean_forecast = daily_energy["forecast_Wh"].mean()
print (f"mean = {int(mean_forecast)}")
if run_to_limit:
    mean_forecast = daily_limit

charge, daily_budget = daily_soc(forecast, battery, SOC, reserve_frac, mean_forecast)
charge.index = daily_energy.index

print(charge, daily_budget)

fig, ax = plt.subplots(figsize=(10, 6))

# Forecast energy as bars
ax.bar(
    daily_energy.index,
    daily_energy["forecast_Wh"],
    label="Forecast",
)

# Clear-sky upper limit as line
ax.plot(
    daily_energy.index,
    daily_energy["clear_Wh"],
    linewidth=2,
    label="Clear-sky",
)

# Daily energy budget
ax.axhline(
    daily_budget,
    linestyle="--",
    linewidth=2,
    color="grey",
    label=f"Daily budget ({daily_budget:.0f} Wh)",
)

ax.set_ylabel("Energy (Wh / day)")
ax.set_title("Daily Solar Energy Forecast")
ax.grid(axis="y", alpha=0.3)

# ---- Secondary axis for SOC ----
ax2 = ax.twinx()

ax2.plot(
    charge["soc_pct"],
    linestyle="-",
    linewidth=2,
)

ax2.set_ylabel("State of Charge (%)")
ax2.set_ylim(0, 100)

# ---- Combined legend ----
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

