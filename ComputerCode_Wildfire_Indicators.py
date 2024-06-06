# Use this code to calculate Fire Weather Index (FWI) seen in Tafrate, J.(2024). Arctic Climate Threat Indicator Set (ArCTISt): A municipal decision making tool. 
# Reccomended to first identify the highest accuracy model using "ComputerCode_Model_Selection".

# import necessary packages and install if needed, xarray version 2023.4.1
#reccomended
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import gcsfs
import geopandas as gpd
import xclim
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# see for information on CMIP6 model codes
#https://docs.google.com/document/d/1yUx6jr9EdedCOLd--CPdTfGDwEwzPpCF6p1jRmqx-0Q/edit#heading=h.sqoyqmabpjai
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
# filter by desired prameters: daily values, tas = air temperature, ssp126, and select model
df_var = df.query(" table_id == 'day' & variable_id == 'tasmax' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datatmax126 = df_var
 # this code loads xarray dataset from the zarr store
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datatmax126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst = xr.open_zarr(mapper, consolidated=True)
tasmax126 = dst

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'pr' & experiment_id == 'ssp126' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
datapr126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datapr126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dsp = xr.open_zarr(mapper, consolidated=True)
pr126 = dsp

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'sfcWind' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datawind126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datawind126.zstore.values[0]
mapper = gcs.get_mapper(zstore)
dsw = xr.open_zarr(mapper, consolidated=True)
wind126 = dsw

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'hursmin' & experiment_id == 'ssp126' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
datahurs126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datahurs126.zstore.values[0]
mapper = gcs.get_mapper(zstore)
dshurs126 = xr.open_zarr(mapper, consolidated=True)
hum126 = dshurs126

# remove leap year observations for consistency between models
def time_convert(time64M):
  leap_year_entries = (time64M['time.month'] == 2) & (time64M['time.day'] == 29)
  model_no_leap = time64M.where(~leap_year_entries, drop=True)
  return model_no_leap

tasmaxNL = time_convert(tasmax126)
prNL = time_convert(pr126)
windNL = time_convert(wind126)
humNL = time_convert(hum126)

climate_data = {
    'tasmax': tasmaxNL,
    'pr': prNL,
    'wind': windNL,
    'hum': humNL
}

# if you want to check variables can plot here to easily see unit and begin to understand the datasets better
data = climate_data['hum']
data.hursmin.sel(time='2023-02-28').squeeze().plot()

def convert_lon(ds):
    # stores origional attributes to call back
    OG_attrs = ds.attrs
    # rolls prime meridian 0
    ds_rolled = ds.roll(lon=(ds.dims['lon'] // 2), roll_coords=True)
    ds_rolled.attrs = OG_attrs
    # Converts 0 to 360 to -180 to -180
    ds_rolled['lon'] = (ds_rolled['lon'] + 180) % 360 - 180
    return ds_rolled

def city_select(ds180):
  #set lat and long to the study city
  lat = 65.584816 # Luleå example
  lon =  22.156704 # Luleå example
  #save attribute data in variable attrs to keep info
  attrs = ds180.attrs
  # select nearest cell based off lat and long from station
  loc = ds180.sel(lat=lat, lon=lon, method='nearest')
  # rewrite attributes
  loc.attrs = attrs
  return loc

converted_datasets = {}  # Create an empty dictionary to store the converted datasets
# Iterate through each model in the datasets dictionary
for name, data in climate_data.items():
    # Apply the convert_lon function to the model's data
    converted_data = convert_lon(data)
    # Store the converted data in the new dictionary with the same model names
    converted_datasets[name] = converted_data

city_data = {}
for name, data in converted_datasets.items():
    selected_data = city_select(data)
    city_data[name] = selected_data

#check again everything is good
for name, data in city_data.items():
    print(f"Model Name: {name}")
    print(f"Latitude (lat): {data.lat.values}")
    print(f"Longitude (lon): {data.lon.values}")
    print("\n")

#isolate data arrays
tasmax = city_data['tasmax']
pr = city_data['pr']
hurs = city_data['hum']
sfcWind = city_data['wind']

#unchunk time for calculation
tasmax = tasmax.chunk({'time': -1})
pr = pr.chunk({'time': -1})
hurs = hurs.chunk({'time': -1})
sfcWind = sfcWind.chunk({'time': -1})

#drop unneeded variables
tasmax = tasmax.drop_vars('height')
hurs = hurs.drop_vars('height')
sfcWind = sfcWind.drop_vars('height')
tasmax = tasmax.drop_vars('lat_bnds')
pr = pr.drop_vars('lat_bnds')
hurs = hurs.drop_vars('lat_bnds')
sfcWind = sfcWind.drop_vars('lat_bnds')

#recombine cleaned data arrays to dataset
climate_data = xr.merge([tasmax,pr,hurs,sfcWind], compat='override')

# Define season mask based on CFS usage, this can be edited depending on data and use case
season_mask = xclim.indices.fire_season(
    tas=climate_data['tasmax'],
    method="WF93",
    freq="YS",
    temp_start_thresh="12 degC",
    temp_end_thresh="5 degC",
    temp_condition_days=3,
)

# Calculate CFWI for ssp126
FWI_result126 = xclim.indices.cffwis_indices(
    tas=climate_data['tasmax'],
    pr=climate_data['pr'],
    hurs=climate_data['hursmin'],
    sfcWind=climate_data['sfcWind'],
    lat=climate_data['lat'],
    season_mask=season_mask,
    overwintering=True,
    dry_start="CFS",
    prec_thresh="1.5 mm/d",
    dmc_dry_factor=1.2,
    carry_over_fraction=0.75,
    wetting_efficiency_fraction=0.75,
    dc_start=15,
    dmc_start=6,
    ffmc_start=85,
)

# Print individual components and isolate final FWI score
DC, DMC, FFMC, ISI, BUI, FWI = FWI_result126
FWI126 = FWI
print("Drought Code (DC):")
print(DC)

print("Duff Moisture Code (DMC):")
print(DMC)

print("Fine Fuel Moisture Code (FFMC):")
print(FFMC)

print("Initial Spread Index (ISI):")
print(ISI)

print("Buildup Index (BUI):")
print(BUI)

print("Canadian Forest Fire Weather Index (CFFWIS):")
print(FWI)

# NOW REPEAT PROCESS FOR SSP58

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tasmax' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datatmax = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datatmax.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst = xr.open_zarr(mapper, consolidated=True)
tasmax585 = dst

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'pr' & experiment_id == 'ssp585' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
datapr = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datapr.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dsp = xr.open_zarr(mapper, consolidated=True)
pr585 = dsp

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'sfcWind' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datawind = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datawind.zstore.values[0]
mapper = gcs.get_mapper(zstore)
dsw = xr.open_zarr(mapper, consolidated=True)
wind585 = dsw

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'hursmin' & experiment_id == 'ssp585' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
datahurs = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datahurs.zstore.values[0]
mapper = gcs.get_mapper(zstore)
dshurs = xr.open_zarr(mapper, consolidated=True)
hum585 = dshurs

tasmaxNL = time_convert(tasmax585)
prNL = time_convert(pr585)
windNL = time_convert(wind585)
humNL = time_convert(hum585)

climate_data = {
    'tasmax': tasmaxNL,
    'pr': prNL,
    'wind': windNL,
    'hum': humNL
}

converted_datasets = {}  # Create an empty dictionary to store the converted datasets
# Iterate through each model in the datasets dictionary
for name, data in climate_data.items():
    # Apply the convert_lon function to the model's data
    converted_data = convert_lon(data)
    # Store the converted data in the new dictionary with the same model names
    converted_datasets[name] = converted_data

city_data = {}
for name, data in converted_datasets.items():
    selected_data = city_select(data)
    city_data[name] = selected_data

#check again everything is good
for name, data in city_data.items():
    print(f"Model Name: {name}")
    print(f"Latitude (lat): {data.lat.values}")
    print(f"Longitude (lon): {data.lon.values}")
    print("\n")

"""#SSP585ModelCalc"""

tasmax = city_data['tasmax']
pr = city_data['pr']
hurs = city_data['hum']
sfcWind = city_data['wind']

tasmax = tasmax.chunk({'time': -1})
pr = pr.chunk({'time': -1})
hurs = hurs.chunk({'time': -1})
sfcWind = sfcWind.chunk({'time': -1})

tasmax = tasmax.drop_vars('height')
hurs = hurs.drop_vars('height')
sfcWind = sfcWind.drop_vars('height')

tasmax = tasmax.drop_vars('lat_bnds')
pr = pr.drop_vars('lat_bnds')
hurs = hurs.drop_vars('lat_bnds')
sfcWind = sfcWind.drop_vars('lat_bnds')

tasmax = tasmax.drop_vars('lon_bnds')
pr = pr.drop_vars('lon_bnds')
hurs = hurs.drop_vars('lon_bnds')
sfcWind = sfcWind.drop_vars('lon_bnds')

climate_data = xr.merge([tasmax,pr,hurs,sfcWind], compat='override')
climate_data

FWI_result585 = xclim.indices.cffwis_indices(
    tas=climate_data['tasmax'],
    pr=climate_data['pr'],
    hurs=climate_data['hursmin'],
    sfcWind=climate_data['sfcWind'],
    lat=climate_data['lat'],
    season_mask=season_mask,
    overwintering=True,
    dry_start="CFS",
    prec_thresh="1.5 mm/d",
    dmc_dry_factor=1.2,
    carry_over_fraction=0.75,
    wetting_efficiency_fraction=0.75,
    dc_start=15,
    dmc_start=6,
    ffmc_start=85,
)

DC, DMC, FFMC, ISI, BUI, FWI = FWI_result585
FWI585 = FWI
print("Drought Code (DC):")
print(DC)

print("Duff Moisture Code (DMC):")
print(DMC)

print("Fine Fuel Moisture Code (FFMC):")
print(FFMC)

print("Initial Spread Index (ISI):")
print(ISI)

print("Buildup Index (BUI):")
print(BUI)

print("Canadian Forest Fire Weather Index (CFFWIS):")
print(FWI)

#take average yearly value
yearly126 = FWI126.groupby('time.year').mean(dim='time')
yearly585 = FWI585.groupby('time.year').mean(dim='time')

#convert outputs to data frames
yearly126.name = '126_FWI'
df126 = yearly126.to_dataframe()
yearly585.name = '585_FWI'
df585 = yearly585.to_dataframe()
FireWeather = pd.concat([df126, df585['585_FWI']], axis=1)

FireWeather.reset_index(inplace=True)
FireWeather.rename(columns={'index': 'year'}, inplace=True)

FireWeather

# custom ArCTISt plots
def plot_data(SSP126, SSP585, years, window_size=5):
    # Calculate the rolling average for '126_Summer' and '585_Summer'
    SSP126_smoothed = SSP126.rolling(window=window_size, min_periods=1).mean()
    SSP585_smoothed = SSP585.rolling(window=window_size, min_periods=1).mean()

    # Create figure axis
    fig, ax = plt.subplots(figsize=(5, 3))

    fig.patch.set_facecolor('#484848')
    # Use a smoothing spline to fit curves to the smoothed data
    x_new = np.linspace(years.min(), years.max(), 300)  # Create a new x-axis for smoother lines

    spline_126 = make_interp_spline(years, SSP126_smoothed, k=3) #use k=3, increasing will smooth further
    spline_585 = make_interp_spline(years, SSP585_smoothed, k=3)

    # Plot the smoothed curves
    ax.plot(x_new, spline_126(x_new), label='SSP126', color='#006C84', linestyle='-', linewidth=3, alpha=0.7)  # Blue
    ax.plot(x_new, spline_585(x_new), label='SSP585', color='#AD0047', linestyle='-', linewidth=3, alpha=0.7)  # Red

    z = np.polyfit(years, SSP126, 1) #fit trend lines
    p = np.poly1d(z)
    ax.plot(years, p(years), label='SSP126 Trend', color='#006c84', linestyle='--', linewidth=2)

    z = np.polyfit(years, SSP585, 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), label='SSP 585 Trend', color='#AD0047', linestyle='--', linewidth=2)

    # Set the y-axis limits and ticks
    y_min = min(SSP126.min(), SSP585.min()) - 5
    y_max = max(SSP126.max(), SSP585.max()) + 5
    ax.set_ylim(bottom=int(round(y_min)), top=int(round(y_max)))
    ax.set_yticks(np.arange(int(round(y_min)), int(round(y_max)) + 1, 10))
    ax.tick_params(axis='x', colors='#ffffff')
    ax.tick_params(axis='y', colors='#ffffff')

    ax.spines['bottom'].set_color('#ffffff')
    ax.spines['top'].set_color('#ffffff')
    ax.spines['right'].set_color('#ffffff')
    ax.spines['left'].set_color('#ffffff')

    # Set labels and title
    ax.set_xlabel('Year', color='white')

    # Choose to display legend
    #ax.legend()

    # Show the plot
    plt.show()

plot_data(FireWeather['126_FWI'], FireWeather['585_FWI'], FireWeather['year']) # FWI Plot
