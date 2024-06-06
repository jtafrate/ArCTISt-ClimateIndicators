# Use this code to identify the highest accuracy model using station recrods seen in Tafrate, J.(2024). Arctic Climate Threat Indicator Set (ArCTISt): A municipal decision making tool.
# Outputs include correlation, root mean square error, and Taylor Plots for each model. After identifying the most accurate model use 
# ComputerCode_Precipitation_Indicators.py
# ComputerCode_Temperature_Indicators.py
# ComputerCode_Wildfire_Indicators.py
# ComputerCode_Permafrost_Indicators.py
# to calculate ArCTISt indicator values

# import necessary packages and install if needed, xarray version 2023.4.1
#reccomended
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import gcsfs
import geopandas as gpd
import seaborn as sns
import scipy.stats as stats
import matplotlib.ticker as mticker
from easy_mpl import taylor_plot
from easy_mpl.utils import version_info

# see for information on CMIP6 model codes
#https://docs.google.com/document/d/1yUx6jr9EdedCOLd--CPdTfGDwEwzPpCF6p1jRmqx-0Q/edit#heading=h.sqoyqmabpjai

#list of all model names for comparison
model_name = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2',
              'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3',
              'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6',
              'MPI-ESM1-2-HR','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
              'NorESM2-MM']
datasets = {}

# function creates a dictionary with the model names and corresponding data
def model_select(model):
  # read large google csv of all climate models and experiments
  df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  # filter by desired prameters: CMIP=historic datasest, Amon=monthly, tas= air temperature
  # for precipitation use  table_id == 'day' & variable_id == 'pr'
  df_ta = df.query("activity_id=='CMIP' & table_id == 'Amon' & variable_id == 'tas' & experiment_id == 'historical' & member_id=='r1i1p1f1'")
  # here use source_id from the variable model
  df_model = df_ta.query(f'source_id == "{model}"')
  data = df_model
  # this code loads xarray dataset from the zarr store
  gcs = gcsfs.GCSFileSystem(token='anon')
  zstore = data.zstore.values[0]
  mapper = gcs.get_mapper(zstore)
  ds = xr.open_zarr(mapper, consolidated=True)
  #Return the dictionary
  return {model: ds}

# Loop through the list of model names and get dictionary output using the above function
for model in model_name:
    datasets.update(model_select(model))

# read in station data
def station_data(path):
    df = pd.read_csv(path)
    return df

#put your path to station data here
path = "/content/drive/MyDrive/Chapter3Coding/Data/Lulea_stationCSV.csv" #example path
station_data = station_data(path)
station_data

#select city by lat long, will select nearest cell
def city_select(ds180):
  #save attribute data in variable attrs to keep info
  attrs = ds180.attrs
  # select tas (temperature) variable based off lat and long from station
  loc = ds180.tas.sel(lat=65.5848, lon=22.1567, method='nearest') # put signed decimal degree coordinates here,Luleå example shown
  # rewrite attributes
  loc.attrs = attrs
  return loc

# Takes in climate mode and converts lon from 0-to-360 to -180-to+180
def convert_lon(ds):
    # stores attributes to call back in
    OG_attrs = ds.attrs
    # rolls prime meridian 0
    ds_rolled = ds.roll(lon=(ds.dims['lon'] // 2), roll_coords=True)
    ds_rolled.attrs = OG_attrs
    # Converts 0 to 360 to -180 to -180
    ds_rolled['lon'] = (ds_rolled['lon'] + 180) % 360 - 180
    return ds_rolled

# filter the station data to the appropriate time period for the study (1981-2010)
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

# use this code to convert pr to mm/day and then monthly values before model comparison
# converted_pr = {}  # Create an empty dictionary to store the converted datasets

# for model_name, ds in LUL_data.items():
# # Create a copy of the dataset to avoid modifying the original
#     converted_ds = ds.copy()

# #Multiply the 'pr' variable by 86400 to convert
#     converted_ds = converted_ds * 86400

# # Add the converted dataset to the dictionary with the model name as the key
#     converted_pr[model_name] = converted_ds

# # use this code to convert pr data daily values to monthly
# def groupby_month_sum(dataset):
#     monthly = dataset.resample(time='M').sum('time')
#     return monthly

# # Create an empty dictionary to store the results
# monthly_sums = {}

# # Iterate through the converted_pr dictionary
# for model_name, ds in converted_pr.items():
# #Apply the groupby_month_sum function to each dataset
#     monthly_sums[model_name] = groupby_month_sum(ds)

# Applt time filet
filt_station = time_filter(station_data)
filt_station

converted_datasets = {}  # Create an empty dictionary to store the converted datasets

# Iterate through each model in the datasets dictionary
for model_name, model_data in datasets.items():
    # Apply the convert_lon function to the model's data
    converted_data = convert_lon(model_data)
    # Store the converted data in the new dictionary with the same model name
    converted_datasets[model_name] = converted_data

keys = converted_datasets.keys()
print(keys)

#create empty dictionary to store the resulting data for nearest model cell to city
data_update = {}

# Iterate through the converted_datasets dictionary
for model_name, dataset in converted_datasets.items():
    # Apply the city_select function to the dataset for each model
    selected_data = city_select(dataset)
    # Store the result in thedictionary with the model name as the key
    data_update[model_name] = selected_data

#check the different latitude and longitude values for each model in the list
for model_name, data in data_update.items():
    print(f"Model Name: {model_name}")
    print(f"Latitude (lat): {data.lat.values}")
    print(f"Longitude (lon): {data.lon.values}")
    print("\n")

#Update each mode to have the appropriate date time index
#this might be a little tricky and require some tinkering

time64 = data_update['ACCESS-CM2'].indexes['time']

data_update['CanESM5'] = data_update['CanESM5'].assign_coords(time=time64)
data_update['CMCC-ESM2'] = data_update['CMCC-ESM2'].assign_coords(time=time64)
#WARNING: FGOALS-g3this has extra data to 2016 and yeilds an error, first cut by desired
#length and then reassign time
data_update['FGOALS-g3'] = data_update['FGOALS-g3'].isel(time=slice(1980))
data_update['FGOALS-g3'] = data_update['FGOALS-g3'].assign_coords(time=time64)
data_update['GFDL-ESM4'] = data_update['GFDL-ESM4'].assign_coords(time=time64)
data_update['INM-CM4-8'] = data_update['INM-CM4-8'].assign_coords(time=time64)
data_update['INM-CM5-0'] = data_update['INM-CM5-0'].assign_coords(time=time64)
data_update['NorESM2-LM'] = data_update['NorESM2-LM'].assign_coords(time=time64)
data_update['NorESM2-MM'] = data_update['NorESM2-MM'].assign_coords(time=time64)

import pandas as pd
df = pd.DataFrame()
# Iterate through the data dictionary and add data for each model as a column
for model_name, data in data_update.items():
# Extract the data as a series and add it as a column with the model name as the header
    df[model_name] = data.to_series()
# Display the resulting DataFrame to compare raw model values, note °K
df

# check to see if any columns have missing values, address above if so
Mvals = df.isnull()
count = Mvals.sum()
count

# reset index so time is stand alone column
df = df.reset_index()
df

# select time range of study for models
def select_time_range(df, start_year, end_year):
    mask = (df['time'].dt.year >= start_year) & (df['time'].dt.year <= end_year)
    selected_data = df[mask]
    return selected_data

df_1981_2010 = select_time_range(df, start_year=1981, end_year=2010)

# convert values to celsius
column_to_exclude = 'time'
# Apply subtraction to all columns except the specified one
dfCelsius = df_1981_2010.apply(lambda col: col - 273.15 if col.name != column_to_exclude else col)
models_df = dfCelsius.reset_index()
models_df

#reset indexes for them to match
station_df = filt_station.reset_index()
station_df

# merge the dataframes
combined_df = pd.concat([models_df, station_df['TAVG'], station_df['DATE']], axis=1)
combined_df

"""# Model comparison"""

# select all temperature data but in "wide" form
df_wide = combined_df[['DATE','TAVG',
  'ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2',
  'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3',
  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6',
  'MPI-ESM1-2-HR','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
  'NorESM2-MM']]
df_wide

# check all temp statistics features, this is a good place to check for errors
df_wide.describe()

# select all temperature columns and convert from wide to long for seaborn plot
# Re-sorts values in specified model to plot
import pandas as pd
df_melt = pd.melt(df_wide[['DATE','TAVG',
  'ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2',
  'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3',
  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6',
  'MPI-ESM1-2-HR','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
  'NorESM2-MM']],['DATE'])
df_melt

# compare temperature time series
import matplotlib.ticker as mticker
# Plot station average with historical data, not seasonally-adjusted
fig, ax = plt.subplots(figsize = (18,8))
sns.lineplot(data=df_melt, x='DATE', y='value', hue='variable')
plt.ylabel('Temperature')
plt.title('Temperature Comparison of Sation to Models, 1981-2010')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
myLocator = mticker.MultipleLocator(29)
ax.xaxis.set_major_locator(myLocator)
fig.autofmt_xdate()

# Plot 1-year moving average for each model
x_time = pd.to_datetime(df_wide['DATE'])
plt.figure(figsize=(15,12))
for model_name in df_wide.columns[1:]:
    plt.subplot(3, 1, 1)
    plt.plot(x_time, df_wide[model_name].rolling(12).mean().tolist(), label=model_name)
plt.grid(False)
plt.title('1 Year Moving Average, 1981-2010')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

for model_name in df_wide.columns[1:]:
    plt.subplot(3, 1, 2)
    plt.plot(x_time, df_wide[model_name].rolling(60).mean().tolist(), label=model_name)
plt.grid(False)
plt.ylabel('Temperature')
plt.title('5 Year Moving Average, 1981-2010')

for model_name in df_wide.columns[1:]:
    plt.subplot(3, 1, 3)
    plt.plot(x_time, df_wide[model_name].rolling(120).mean().tolist(), label=model_name)
plt.grid(False)
plt.ylabel('Temperature')
plt.title('10 Year Moving Average, 1981-2010')
plt.show()

"""Taylor Plot"""

#creare list of model names again
model_names = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2',
              'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3',
              'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6',
              'MPI-ESM1-2-HR','MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
              'NorESM2-MM']
results = []
target_data = df_wide['TAVG'] #define station data

# calculate RMSE
def rmse_cal(y0, y):
    rmse = np.sqrt(np.mean((y0 - y) ** 2))
    return rmse

for model_name in model_names:
    model_data = df_wide[model_name]
    # Perform linear regression using stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(target_data, model_data)
    # Calculate R-squared
    correlation = r_value
    rmse = rmse_cal(target_data, model_data)
    results.append({'Model': model_name, 'correlation': correlation, 'RMSE': rmse})
eval_df = pd.DataFrame(results)
eval_df

#create list of correlations and rmse values
correlation = eval_df['correlation'].tolist()
rmse_list = eval_df['RMSE'].tolist()
rmse_list

# Create a dictionary to store model names and their standard deviations
std_dict = {}

# Calculate standard deviation and normalized deviation for each model
for model_name in model_names:
    model_data = df_wide[model_name]
    std = model_data.std()
    normalized_std = std / target_data.std()
    std_dict[model_name] = {'Standard Deviation': std, 'Normalized Standard Deviation': normalized_std}

# Create a new DataFrame from the dictionary
std_df = pd.DataFrame.from_dict(std_dict, orient='index')

# Display the new DataFrame
std_df

#print out best model based off correlations
best_r2 = eval_df.loc[eval_df['correlation'].idxmax()]
best_r2

#print out best model based off rmse
best_rmse = eval_df.loc[eval_df['RMSE'].idxmin()]
best_rmse

"""# Taylor Diagram

TAYLOR PLOT 2
"""

eval_df

std_reset = std_df.reset_index()
eval_df2 = pd.concat([eval_df, std_reset], axis=1)
new_eval_df = eval_df2.drop(columns= ['index'])
new_eval_df

#define color pallette
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087',
    '#f95d6a', '#ff7c43', '#ffa600']
observations = target_data #ensure this is station data defined earlier
simulations =  {"ACCESS-CM2": df_wide['ACCESS-CM2'],
            "ACCESS-ESM1-5": df_wide['ACCESS-ESM1-5'],
               "CanESM5" : df_wide['CanESM5'],
                "CMCC-ESM2" : df_wide['CMCC-ESM2'],
                "EC-Earth3" : df_wide['EC-Earth3'],
                "EC-Earth3-Veg-LR" : df_wide['EC-Earth3-Veg-LR'],
                "EC-Earth3-Veg" : df_wide['EC-Earth3-Veg'],
               "FGOALS-g3" : df_wide['FGOALS-g3'],
                "GFDL-ESM4" : df_wide['GFDL-ESM4'],
                "INM-CM4-8" : df_wide['INM-CM4-8'],
                "INM-CM5-0" : df_wide['INM-CM5-0'],
                "IPSL-CM6A-LR" : df_wide['IPSL-CM6A-LR'],
                "MIROC6" : df_wide['MIROC6'],
                "MPI-ESM1-2-HR" : df_wide['MPI-ESM1-2-HR'],
                "MPI-ESM1-2-LR" : df_wide['MPI-ESM1-2-LR'],
                "MRI-ESM2-0" : df_wide['MRI-ESM2-0'],
                "NorESM2-LM" : df_wide['NorESM2-LM'],
                "NorESM2-MM" : df_wide['NorESM2-MM']}

_ = taylor_plot(
    observations=observations,
    simulations=simulations,
    ref_color='k',

    colors=colors,  # Specify colors here
    #custom markers for each model
    sim_marker={'ACCESS-CM2': '$1$', 'ACCESS-ESM1-5': '$2$', 'CanESM5': '$3$',
    'CMCC-ESM2': '$4$', 'EC-Earth3': '$5$', 'EC-Earth3-Veg-LR': '$6$',
    'EC-Earth3-Veg': '$7$', 'FGOALS-g3': '$8$', 'GFDL-ESM4': '$9$',
    'INM-CM4-8': '$10$', 'INM-CM5-0': '$11$', 'IPSL-CM6A-LR': '$12$',
    'MIROC6': '$13$', 'MPI-ESM1-2-HR': '$14$', 'MPI-ESM1-2-LR': '$15$',
    'MRI-ESM2-0': '$16$', 'NorESM2-LM': '$17$', 'NorESM2-MM': '$18$'},
    leg_kws={'bbox_to_anchor': (1.15, 0.95), 'numpoints': 1, 'fontsize': 12,
             'markerscale': 1},
    axis_fontdict={'left': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10},
                  'bottom': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10},
                  'top': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10},},
    cont_kws={'colors': 'grey', 'linewidths': 1.0, 'linestyles': 'dotted'},
    grid_kws={'axis': 'x', 'color': 'darkgrey', 'lw': 1.0},
    title='Air Temperature')

new_eval_df

#example code for taylor plots with normalized std values, use new_eval_df values
observations = {'std': 1}
simulations = {
        'ACCESS-CM2': {'std': 0.870062	, 'corr_coeff': 0.930936}, #1
        'ACCESS-ESM1-5': {'std': 0.530927, 'corr_coeff': 0.839986}, #2
        'CanESM5': {'std': 0.917265	, 'corr_coeff': 0.927434}, #3
        'CMCC-ESM2': {'std': 0.766937	, 'corr_coeff': 0.902185}, #4
        'EC-Earth3': {'std': 0.914352, 'corr_coeff': 0.909387}, #5
        'EC-Earth3-Veg-LR':{'std': 0.989215		, 'corr_coeff': 0.913018 }, #6
        'EC-Earth3-Veg':{'std': 0.881364, 'corr_coeff': 0.915699}, #7
        'FGOALS-g3':{'std': 1.572036, 'corr_coeff': 0.933046}, #8
        'GFDL-ESM4':{'std': 1.027108, 'corr_coeff': 0.925809}, #9
        'INM-CM4-8':{'std': 1.042218	, 'corr_coeff': 0.863745}, #10
        'INM-CM5-0':{'std': 0.995392	, 'corr_coeff': 0.830144}, #11
        'IPSL-CM6A-LR':{'std': 1.134495	, 'corr_coeff': 0.938218}, #12
        'MIROC6':{'std': 1.014492	, 'corr_coeff': 0.934658}, #13
        'MPI-ESM1-2-HR':{'std': 1.134507, 'corr_coeff': 0.940722}, #14
        'MPI-ESM1-2-LR':{'std': 1.038514, 'corr_coeff': 0.930410}, #15
        'MRI-ESM2-0':{'std': 0.788965, 'corr_coeff': 0.915170}, #16
        'NorESM2-LM':{'std': 0.723834	, 'corr_coeff': 0.897462}, #17
        'NorESM2-MM':{'std': 0.806130	, 'corr_coeff': 0.894740}, #18
}

figure = taylor_plot(
    observations=observations,
    simulations=simulations,
    ref_color='k',
    x_lim=(0, 35),  # Set the x-axis limits
    y_lim=(0, 35),
    #ms =90,
    #marker_kws={'ms': 30, 'lw': 0.0},  # Set marker size
    colors=colors,  # Specify colors here
    sim_marker={'ACCESS-CM2': '$1$', 'ACCESS-ESM1-5': '$2$', 'CanESM5': '$3$',
    'CMCC-ESM2': '$4$', 'EC-Earth3': '$5$', 'EC-Earth3-Veg-LR': '$6$',
    'EC-Earth3-Veg': '$7$', 'FGOALS-g3': '$8$', 'GFDL-ESM4': '$9$',
    'INM-CM4-8': '$10$', 'INM-CM5-0': '$11$', 'IPSL-CM6A-LR': '$12$',
    'MIROC6': '$13$', 'MPI-ESM1-2-LR': '$15$',  'MPI-ESM1-2-HR': '$14$',
    'MRI-ESM2-0': '$16$', 'NorESM2-LM': '$17$', 'NorESM2-MM': '$18$'},
    leg_kws={'bbox_to_anchor': (1.15, 0.95), 'numpoints': 1, 'fontsize': 12,
             'markerscale': 1},
    axis_fontdict={'left': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10},
                  'bottom': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10},
                  'top': {'fontsize': 12, 'color': 'k', 'ticklabel_fs': 10}},
    add_ith_interval = True,
   cont_kws={'colors': 'grey', 'linewidths': 1.0, 'linestyles': 'dotted', 'level': 3},
   grid_kws={'axis': 'x', 'color': 'darkgrey', 'lw': 1.0},
   title='Precipitation')
