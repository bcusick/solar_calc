import pandas as pd
import matplotlib.pyplot as plt
import pvlib

# ----------------------------
# 1) Time index (hourly, midnight start)
# ----------------------------
tz = "America/Mexico_City"
days = 1
times = pd.date_range(
    start=pd.Timestamp.now(tz=tz).normalize(),
    periods= days* 24 *60,
    freq="1min",
    tz=tz
)

# ----------------------------
# 2) Site + clearsky irradiance
# ----------------------------
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

panel_watts = 1000
n_sys = 0.9

fixed_tilt = 0      # degrees (edit)
fixed_azimuth = 180  # degrees (180 = south in N hemisphere)
max_tilt = 90

site = pvlib.location.Location(latitude, longitude, tz=tz)
solpos = site.get_solarposition(times)

# Clearsky model (Ineichen). Returns GHI, DNI, DHI
cs = site.get_clearsky(times, model="ineichen")  # columns: ghi, dni, dhi

# Optional: zero out night numerical noise
cs = cs.clip(lower=0)


# ----------------------------
# 3) zero-tilt POA
# ----------------------------


poa_flat = pvlib.irradiance.get_total_irradiance(
    surface_tilt=fixed_tilt,
    surface_azimuth=fixed_azimuth,
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=cs["dni"],
    ghi=cs["ghi"],
    dhi=cs["dhi"],
)["poa_global"]

# ----------------------------
# 3) Fixed-tilt POA
# ----------------------------


poa_fixed = pvlib.irradiance.get_total_irradiance(
    surface_tilt=solpos["apparent_zenith"].clip(lower=0, upper=max_tilt).min(),
    surface_azimuth=fixed_azimuth,
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=cs["dni"],
    ghi=cs["ghi"],
    dhi=cs["dhi"],
)["poa_global"]

# ----------------------------
# 4) Single-axis tracking POA
# ----------------------------

poa_1axis = pvlib.irradiance.get_total_irradiance(
    surface_tilt=solpos["apparent_zenith"].clip(lower=0, upper=max_tilt),
    surface_azimuth=fixed_azimuth,
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=cs["dni"],
    ghi=cs["ghi"],
    dhi=cs["dhi"],
)["poa_global"]

# ----------------------------
# 5) Dual-axis tracking POA
# ----------------------------
# Dual-axis: panel normal points at sun.
# surface tilt = 90 - elevation = zenith (with some conventions)
# In pvlib tilt is from horizontal: tilt = solar zenith

poa_2axis = pvlib.irradiance.get_total_irradiance(
    surface_tilt=solpos["apparent_zenith"].clip(lower=0, upper=max_tilt),
    surface_azimuth=solpos["azimuth"],
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    dni=cs["dni"],
    ghi=cs["ghi"],
    dhi=cs["dhi"],
)["poa_global"]

# ----------------------------
# 6) Assemble + plot
# ----------------------------
df = pd.DataFrame(
    {
        "poa_flat": poa_flat,
        "poa_fixed": poa_fixed,
        "poa_1axis": poa_1axis,
        "poa_2axis": poa_2axis,
    },
    index=times
).clip(lower=0)

#energy outputs
energy = ((df["poa_fixed"].clip(lower=0) / 1000.0) * panel_watts * n_sys / 1).to_frame(name="fixed")
energy["flat"] = (df["poa_flat"].clip(lower=0) / 1000.0) * panel_watts * n_sys / 1
energy["1axis"] = (df["poa_1axis"].clip(lower=0) / 1000.0) * panel_watts * n_sys / 1
energy["2axis"] = (df["poa_2axis"].clip(lower=0) / 1000.0) * panel_watts * n_sys / 1

#daily sums
daily = (energy["flat"].resample("D").sum() / 60).to_frame(name="flat")
daily["fixed"] = (energy["fixed"].resample("D").sum() / 60)
daily["1axis"] = (energy["1axis"].resample("D").sum() / 60)
daily["2axis"] = (energy["2axis"].resample("D").sum() / 60)
#daily["2axis%"] = daily['2axis'] / daily['flat'] * panel_watts
percent_of_flat = (daily["flat"] / daily["flat"] * 100).to_frame(name="flat")
percent_of_flat["1axis"] = daily["1axis"] / daily["flat"] * 100
percent_of_flat["2axis"] = daily["2axis"] / daily["flat"] * 100

ax = energy.plot(figsize=(10, 6) 
    )
ax.set_ylabel("Power (W)")
ax.set_title("Power: Fixed vs Single-Axis vs Dual-Axis (Clearsky)")
ax.grid(True)
plt.tight_layout()
plt.show()
print(daily)
