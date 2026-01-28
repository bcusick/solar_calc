import pandas as pd
import numpy as np
import pvlib
from pvlib import irradiance
from pvlib.location import Location
import matplotlib.pyplot as plt

#Xalapa
latitude = 19.54234
longitude = -96.92520
timezone = 'America/Mexico_City'

array_p = 650*6/1000 #kW
alpha = 0.9 #accounts for electrical losses to/from battery
eta = 0.6 #account for weather and storage

times = pd.date_range(
    start=pd.Timestamp("2026-01-01 00:00:00", tz=timezone),
    end=pd.Timestamp("2027-01-01 00:00:00", tz=timezone),
    freq=pd.Timedelta(seconds=3600),
    inclusive="left",
)

# Solar position
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
location = Location(latitude, longitude, tz=timezone)

# Clear sky
cs = location.get_clearsky(times)

# approx. best tilt per day
daily_tilt = (
    solpos["apparent_zenith"]
    .clip(lower=5, upper=60)
    .resample("D")
    .min()
)

# map that daily value onto each hour
tilt_hourly = daily_tilt.reindex(times.floor("D")).to_numpy()

surface_azimuth = 180  # south

poa = irradiance.get_total_irradiance(
    surface_tilt=tilt_hourly,
    surface_azimuth=surface_azimuth,
    solar_zenith=solpos["apparent_zenith"].to_numpy(),
    solar_azimuth=solpos["azimuth"].to_numpy(),
    dni=cs["dni"].clip(lower=0).to_numpy(),
    ghi=cs["ghi"].clip(lower=0).to_numpy(),
    dhi=cs["dhi"].clip(lower=0).to_numpy(),
)

poa_wm2 = pd.Series(poa["poa_global"], index=times).clip(lower=0)

hourly_kwh = poa_wm2 / 1000.0 * array_p * alpha * eta
daily_kwh = hourly_kwh.resample("D").sum().to_frame("kwh")

fig, ax1 = plt.subplots(figsize=(10, 4))
daily_kwh.plot(ax=ax1, label="kWh")
ax1.set_ylabel("Daily Energy kWh")

ax2 = ax1.twinx()
daily_tilt.plot(ax=ax2, color="orange", label="Daily Tilt")
ax2.set_ylabel("Tilt")

plt.title("Yearly Energy (Clearsky)")
ax1.grid(True)
plt.tight_layout()
plt.show()

print(daily_kwh.describe())

