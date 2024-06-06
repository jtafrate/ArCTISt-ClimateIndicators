#Use this code to calculate Adjusted Thaw Potential (ATP) seen in Tafrate, J.(2024). Arctic Climate Threat Indicator Set (ArCTISt): A municipal decision making tool. 
#Reccomended to first identify the highest accuracy model using "ComputerCode_Model_Selection". 

#  Adjusted Thaw Potential Based on Permafrost coefficients from
#  "Brown, J., O. Ferrians, J. A. Heginbottom, and E. Melnikov. 2002. Circum-Arctic Map of Permafrost
#  and Ground-Ice Conditions, Version 2. [Indicate subset used]. Boulder, Colorado USA. NASA
#  National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.7265/skbgkf16. [Date Accessed]."
continuous = 1.0
discontinuous = 0.9
sporadic = 0.5
isolated = 0.1
none = 0

# import necessary packages and install if needed, xarray version 2023.4.1 reccomended
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
# filter by desired prameters: daily values, tas = air temperature, ssp126, and select model
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp126' & source_id=='EC-Earth3' & member_id=='r1i1p1f1'")
datatas126 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datatas126.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst126 = xr.open_zarr(mapper, consolidated=True)
tas126 = dst126

df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
df_var = df.query(" table_id == 'day' & variable_id == 'tas' & experiment_id == 'ssp585' & source_id=='EC-Earth3' & member_id=='r1i1p1f1'")
datatas585 = df_var
gcs = gcsfs.GCSFileSystem(token='anon')
zstore = datatas585.zstore.values[-1]
mapper = gcs.get_mapper(zstore)
dst585 = xr.open_zarr(mapper, consolidated=True)
tas585 = dst585

def time_convert(time64M): # remove leap year observations for consistency between models
  leap_year_entries = (time64M['time.month'] == 2) & (time64M['time.day'] == 29)
  model_no_leap = time64M.where(~leap_year_entries, drop=True)
  return model_no_leap

tas126NL = time_convert(tas126)
tas585NL = time_convert(tas585)

def convert_lon(ds):
    # stores attributes to call back in
    OG_attrs = ds.attrs
    # rolls prime meridian 0
    ds_rolled = ds.roll(lon=(ds.dims['lon'] // 2), roll_coords=True)
    ds_rolled.attrs = OG_attrs
    # Converts 0 to 360 to -180 to -180
    ds_rolled['lon'] = (ds_rolled['lon'] + 180) % 360 - 180
    return ds_rolled
def city_select(ds180):
  lat = 62.453972 #Yellowknife example
  lon = -114.371788 #Yellowknife example
  attrs = ds180.attrs
  loc = ds180.sel(lat=lat, lon=lon, method='nearest')
  loc.attrs = attrs
  return loc

tas126_180 = convert_lon(tas126NL)
tas126_city = city_select(tas126_180)
tas585_180 = convert_lon(tas585NL)
tas585_city = city_select(tas585_180)

tas126_array = tas126_city['tas']
tas585_array = tas585_city['tas']

# calculate freezing degree days usisng xarray heating degree days
FDD126 = xclim.indices.heating_degree_days(tas126_array, thresh='0.0 degC', freq='YS')
FDD585 = xclim.indices.heating_degree_days(tas585_array, thresh='0.0 degC', freq='YS')

# calculate warming degree days usisng xarray cooling degree days
WDD126 = xclim.indices.cooling_degree_days(tas126_array, thresh='0.0 degC', freq='YS')
WDD585 = xclim.indices.cooling_degree_days(tas585_array, thresh='0.0 degC', freq='YS')

#convert outputs to data frames
FDD126.name = 'FDD_126'
dfFDD126 = FDD126.to_dataframe()
FDD585.name = 'FDD_585'
dfFDD585 = FDD585.to_dataframe()
WDD126.name = 'WDD_126'
dfWDD126 = WDD126.to_dataframe()
WDD585.name = 'WDD_585'
dfWDD585 = WDD585.to_dataframe()

#combine dataframes to one
TPdf = pd.concat([dfFDD126, dfFDD585['FDD_585'], dfWDD126['WDD_126'], dfWDD585['WDD_585']], axis=1)
TPdf['year'] = TPdf.index.year #add year column from index
TPdf

# calculate adjusted TP by (warming degree days / freezing degree days) * permafrost coefficient
TPdf['TP_126'] = (TPdf['WDD_126'] / TPdf['FDD_126']) * discontinuous
TPdf['TP_585'] = (TPdf['WDD_585'] / TPdf['FDD_585'] )* discontinuous
TPdf

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

plot_data(TPdf['TP_126'], TPdf['TP_585'], TPdf['year']) # ETR plot
