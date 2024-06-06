# Use this code to calculate Extreme Precipitation Indicators seen in Tafrate, J.(2024). Arctic Climate Threat Indicator Set (ArCTISt): A municipal decision making tool. 
# 1. Extreme Precipitation Days (EPD)
# 2. Maximum Precipitation Period (MPP)
# 3. Heavy Snow Fall (HSF)
# 4. Rain on Frozen Ground (RFG)
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
# filter by desired prameters: daily values, pr = precipitation, ssp126, and select model
df_var = df.query(" table_id == 'day' & variable_id == 'pr' & experiment_id == 'ssp126' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
data126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = data126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst126 = xr.open_zarr(mapper, consolidated=True)
pr126 = dst126
pr126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'pr' & experiment_id == 'ssp585' & source_id=='EC-Earth3-Veg-LR' & member_id=='r1i1p1f1'")
data585 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = data585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst585 = xr.open_zarr(mapper, consolidated=True)
pr585 = dst585
pr585

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'snw' & experiment_id == 'ssp126' & source_id=='MRI-ESM2-0' & member_id=='r1i1p1f1'")
dstsn126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dstsn126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstsn126 = xr.open_zarr(mapper, consolidated=True)
snw126 = dstsn126
snw126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'snw' & experiment_id == 'ssp585' & source_id=='MRI-ESM2-0' & member_id=='r1i1p1f1'")
dstsn585 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dstsn585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstsn585 = xr.open_zarr(mapper, consolidated=True)
snw585 = dstsn585
snw585

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dst126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dst126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst126 = xr.open_zarr(mapper, consolidated=True)
tas126 = dst126
tas126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dst585 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dst585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst585 = xr.open_zarr(mapper, consolidated=True)
tas585 = dst585
tas585

#Functions which will be used later
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

def time_filter(station):
  # Selects the the station date column
  station['DATE'] = pd.to_datetime(station['DATE'])
  # Add in start and end of time filter
  start = pd.to_datetime('1981-01', format='%Y-%m')
  end = pd.to_datetime('2010-12', format='%Y-%m')
  # Creates a new df that contains only time between specified start and end
  df_filtered = station[(station['DATE'] >= start) & (station['DATE'] <= end)]
  df_filtered['DATE'] = df_filtered['DATE'].dt.strftime('%Y-%m')
  return df_filtered

def time_convert(time64M): # remove leap year observations for consistency between models
  leap_year_entries = (time64M['time.month'] == 2) & (time64M['time.day'] == 29)
  model_no_leap = time64M.where(~leap_year_entries, drop=True)
  return model_no_leap

end_date = '2100-12-31'
snw126_2100 = snw126.sel(time=slice(None, end_date))
snw585_2100 = snw585.sel(time=slice(None, end_date))

climate_data = {
    'tas126': tas126,
    'pr126': pr126,
    'snw126' : snw126_2100,
    'tas585': tas585,
    'pr585': pr585,
    'snw585' : snw585_2100}

climate_data

converted_datasets = {}  # Create an empty dictionary to store the converted datasets
# Iterate through each model in the datasets dictionary
for name, data in climate_data.items():
    # Apply the convert_lon function to the model's data
    converted_data = convert_lon(data)
    # Store the converted data in the new dictionary with the same model names
    converted_datasets[name] = converted_data

NL_datasets = {}  # NL = No Leap Year
for name, data in converted_datasets.items():
    NL_data =  time_convert(data)
    NL_datasets[name] = NL_data

city_data = {}
for name, data in NL_datasets.items():
    selected_data = city_select(data)
    city_data[name] = selected_data

for name, data in city_data.items(): #check here for any errors based on desired lat / lon and model name
    print(f"Model Name: {name}")
    print(f"Latitude (lat): {data.lat.values}")
    print(f"Longitude (lon): {data.lon.values}")
    print(f"Time: {data.time.values}")
    print("\n")

#remove time chunking for subsequent calculation and rename
tas126 = city_data['tas126'].chunk({'time': -1})
pr126 =  city_data['pr126'].chunk({'time': -1})
snw126 = city_data['snw126'].chunk({'time': -1})
tas585 = city_data['tas585'].chunk({'time': -1})
pr585 =  city_data['pr585'].chunk({'time': -1})
snw585 = city_data['snw585'].chunk({'time': -1})

#merge data arrays to one Dataset based on ssp126 and ssp585
clim_data126 = xr.merge([tas126, pr126, snw126], compat='override')
clim_data585 = xr.merge([tas585, pr585, snw585],  compat='override')

#reset time variable to datetime64[ns] data type
clim_data126['time'] = tas126['time']
clim_data585['time'] = tas126['time']

# Extreme Precipitation Days
EPD126 = xclim.indices.wetdays(clim_data126['pr'], thresh='20 mm/day', freq='YS', op='>')
EPD585 = xclim.indices.wetdays(clim_data585['pr'], thresh='20 mm/day', freq='YS', op='>')

# Maximum Precipitation Period
MPP126 = xclim.indices.maximum_consecutive_wet_days(clim_data126['pr'], thresh='1 mm/day', freq='YS', resample_before_rl=True)
MPP585 = xclim.indices.maximum_consecutive_wet_days(clim_data585['pr'], thresh='1 mm/day', freq='YS', resample_before_rl=True)

# Heavy Snow Fall
HSF126 = xclim.indices.snw_max(clim_data126['snw'], freq='YS')
HSF585 = xclim.indices.snw_max(clim_data585['snw'], freq='YS')

# Rain on Frozen Ground
RFG126 = xclim.indices.rain_on_frozen_ground_days(clim_data126['pr'], clim_data126['tas'], thresh='1 mm/d', freq='YS')
RFG585 = xclim.indices.rain_on_frozen_ground_days(clim_data585['pr'], clim_data585['tas'], thresh='1 mm/d', freq='YS')

#convert outputs to data frames
EPD126.name = 'EPD_126'
dfEPD126 = EPD126.to_dataframe()
EPD585.name = 'EPD_585'
dfEPD585 = EPD585.to_dataframe()

MPP126.name = 'MPP_126'
dfMPP126 = MPP126.to_dataframe()
MPP585.name = 'MPP_585'
dfMPP585 = MPP585.to_dataframe()

HSF126.name = 'HSF_126'
dfHSF126 = HSF126.to_dataframe()
HSF585.name = 'HSF_585'
dfHSF585 = HSF585.to_dataframe()

RFG126.name = 'RFG_126'
dfRFG126 = RFG126.to_dataframe()
RFG585.name = 'RFG_585'
dfRFG585 = RFG585.to_dataframe()

EPDdf = pd.concat([dfEPD126, dfEPD585['EPD_585']], axis=1)
MPPdf = pd.concat([dfMPP126, dfMPP585['MPP_585']], axis=1)
HSFdf = pd.concat([dfHSF126, dfHSF585['HSF_585']], axis=1)
RFGdf = pd.concat([dfRFG126, dfRFG585['RFG_585']], axis=1)

#reset index to make year column
EPDdf['year'] = EPDdf.index.year
MPPdf['year'] = MPPdf.index.year
HSFdf['year'] = HSFdf.index.year
RFGdf['year'] = RFGdf.index.year

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

plot_data(EPDdf['EPD_126'], EPDdf['EPD_585'], EPDdf['year']) # EPD plot

plot_data(MPPdf['MPP_126'], MPPdf['MPP_585'], MPPdf['year']) # MPP plot

plot_data(HSFdf['HSF_126'], HSFdf['HSF_585'], HSFdf['year']) # HSF plot

plot_data(RFGdf['RFG_126'], RFGdf['RFG_585'], RFGdf['year']) # RFG plot
