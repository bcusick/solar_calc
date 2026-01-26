import numpy as np
import pandas as pd
import pvlib
from pvlib import irradiance
from pvlib.location import Location

#################
tilt_min=0
tilt_max=90
tilt_step=1

# mills river
#latitude = 35.37786
#longitude = -82.58089
#timezone = 'America/New_York'
#Xalapa
latitude = 19.54234
longitude = -96.92520
timezone = 'America/Mexico_City'

azimuth = 180  # azimuth angle (south-facing)


days = 120
start = pd.Timestamp.now(tz=timezone).normalize()
times = pd.date_range(
    start=start,
    periods=days * 24,
    freq="1h",
    tz=timezone
)

# Solar position
solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
location = Location(latitude, longitude, tz=timezone)

# Clear sky
cs = location.get_clearsky(times)
tilts = np.arange(tilt_min, tilt_max + tilt_step, tilt_step)
daily_by_tilt = {}

for tilt in tilts:
    poa = irradiance.get_total_irradiance(
        surface_tilt=float(tilt),
        surface_azimuth=float(azimuth),
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
    )
    hourly = poa["poa_global"].clip(lower=0) 
    daily = hourly.resample("D").sum()
    daily_by_tilt[tilt] = daily

energy_table = pd.DataFrame(daily_by_tilt)  # index: day, columns: tilt

best_tilt = energy_table.idxmax(axis=1).rename("best_tilt_deg")
best_energy = energy_table.max(axis=1).rename("sun hrs").astype(int) / 1000

best_by_day = pd.concat([best_tilt, best_energy], axis=1)

print(best_by_day) 