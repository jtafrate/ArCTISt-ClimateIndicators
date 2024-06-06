# Use this code to calculate Extreme Temperature Indicators seen in Tafrate, J.(2024). Arctic Climate Threat Indicator Set (ArCTISt): A municipal decision making tool. 
# 1. Extreme Temperature Range (ETR)
# 2. Heat Wave Frequency (HWF)
# 3. Cold Wave Fequency (CWF)
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

# see for information on CMIP6 model codes
#https://docs.google.com/document/d/1yUx6jr9EdedCOLd--CPdTfGDwEwzPpCF6p1jRmqx-0Q/edit#heading=h.sqoyqmabpjai
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: daily values, tasmax = maximum air temp, ssp126, and select model
df_var = df.query(" table_id == 'day' & variable_id == 'tasmax' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dstmax126 = df_var
#load in model and output it
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dstmax126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstmax126 = xr.open_zarr(mapper, consolidated=True)
tasmax126 = dstmax126

#repeat this code for all desired variables and scenarios
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tasmin' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dstmin126 = df_var

gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dstmin126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstmin126 = xr.open_zarr(mapper, consolidated=True)
tasmin126 = dstmin126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tasmax' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datatmax585 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')

zstore = datatmax585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstmax585 = xr.open_zarr(mapper, consolidated=True)
tasmax585 = dstmax585
tasmax585

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tasmin' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
datatmin585 = df_var

gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datatmin585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dstmin585 = xr.open_zarr(mapper, consolidated=True)
tasmin585 = dstmin585
tasmin585

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp126' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dst126 = df_var

gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dst126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst126 = xr.open_zarr(mapper, consolidated=True)
tas126 = dst126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp585' & source_id=='MPI-ESM1-2-HR' & member_id=='r1i1p1f1'")
dst585 = df_var

gcs = gcsfs.GCSFileSystem(token='anon')
zstore = dst585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst585 = xr.open_zarr(mapper, consolidated=True)
tas585 = dst585

# put all climate variables in a new dictionary
climate_data = {
    'tasmax126': tasmax126,
    'tasmin126': tasmin126,
    'tas126' : tas126,
    'tasmax585': tasmax585,
    'tasmin585': tasmin585,
    'tas585' : tas585}

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

def station_data(path):
    #use pd.read_csv to read the data into pandas dataframe
    df = pd.read_csv(path)
    return df

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

path = '/content/drive/MyDrive/Chapter3Coding/Data/Lulea_stationCSV.csv'#update path to station data path
station = station_data(path)
stationdf = time_filter(station) # filter to desired time frame
stationdf

p_90 = stationdf['TAVG'].quantile(0.90) # calculate 90th percentile from station data
p_10 = stationdf['TAVG'].quantile(0.1) # calculate 10th percentile from station data
print("10th Percentile Values:")
print(p_10)
print("90th Percentile Values:")
print(p_90)

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
    print("\n")

#remove time chunking for subsequent calculation and rename
tasmax126 = city_data['tasmax126'].chunk({'time': -1})
tasmin126 =  city_data['tasmin126'].chunk({'time': -1})
tas126 = city_data['tas126'].chunk({'time': -1})
tasmax585 = city_data['tasmax585'].chunk({'time': -1})
tasmin585 = city_data['tasmin585'].chunk({'time': -1})
tas585 = city_data['tas585'].chunk({'time': -1})

#merge data arrays to one Dataset based on ssp126 and ssp585
clim_data126 = xr.merge([tasmax126, tasmin126, tas126])
clim_data585 = xr.merge([tasmax585, tasmin585, tas585])

#calculate indicators using xclim see https://xclim.readthedocs.io/en/stable/indices.html#indices-library for more information
ETR126 = xclim.indices.extreme_temperature_range(clim_data126['tasmin'],clim_data126['tasmax'], freq='YS')
ETR585 = xclim.indices.extreme_temperature_range(clim_data585['tasmin'],clim_data585['tasmax'], freq='YS')

HWF126 = xclim.indices.heat_wave_index(clim_data126['tas'], thresh= f'{p_90} degC', window=5, freq='YS', op='>', resample_before_rl=True)
HWF585 = xclim.indices.heat_wave_index(clim_data585['tas'], thresh=  f'{p_90} degC', window=5, freq='YS', op='>', resample_before_rl=True)

CWF126 = xclim.indices.cold_spell_days(clim_data126['tas'], thresh= f'{p_10} degC', window=5, freq='YS', op='<', resample_before_rl=True)
CWF585 = xclim.indices.cold_spell_days(clim_data585['tas'], thresh= f'{p_10} degC', window=5, freq='YS', op='<', resample_before_rl=True)

#convert outputs to data frames
ETR126.name = 'ETR_126'
dfETR126 = ETR126.to_dataframe()
ETR585.name = 'ETR_585'
dfETR585 = ETR585.to_dataframe()

HWF126.name = 'HWF_126'
dfHWF126 = HWF126.to_dataframe()
HWF585.name = 'HWF_585'
dfHWF585 = HWF585.to_dataframe()

dfHWF585

CWF126.name = 'CWF_126'
dfCWF126 = CWF126.to_dataframe()
CWF585.name = 'CWF_585'
dfCWF585 = CWF585.to_dataframe()

#merge ssp outputs to one new dataframe
ETRdf = pd.concat([dfETR126, dfETR585['ETR_585']], axis=1)
HWFdf = pd.concat([dfHWF126, dfHWF585['HWF_585']], axis=1)
CWFdf = pd.concat([dfCWF126, dfCWF585['CWF_585']], axis=1)

#create new year column based on index values
ETRdf['year'] = ETRdf.index.year
HWFdf['year'] = HWFdf.index.year
CWFdf['year'] = CWFdf.index.year

ETRdf

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

plot_data( ETRdf['ETR_126'], ETRdf['ETR_585'], ETRdf['year']) # ETR plot

plot_data(HWFdf['HWF_126'], HWFdf['HWF_585'], HWFdf['year']) #HWF plot

plot_data(CWFdf['CWF_126'], CWFdf['CWF_585'], CWFdf['year']) #CWF plot
