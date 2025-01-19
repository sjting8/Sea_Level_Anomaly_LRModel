# %% [markdown]
# # EDA 
# Getting the data organized into a master data sheet

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import xarray as xr
import numpy as np
import netCDF4
from matplotlib import pyplot as plt
from collections import defaultdict
import sklearn
from sklearn import linear_model
import numpy
import gzip
import math

# %%
MIN_YEAR=1993
MIN_DAY=1
MIN_MONTH=1

MAX_YEAR=2013
MAX_DAY=31
MAX_MONTH=12

# %%
def preprocess_date(data_str):
    try:
        if len(data_str) != 10:
            raise ValueError(f"Invalid date format: {data_str}")
        year = int(data_str[:4])
        month = int(data_str[5:7])
        day = int(data_str[8:10])
        year_norm = (year - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        day_sin = np.sin(2 * np.pi * (day - 1) / 31)
        day_cos = np.cos(2 * np.pi * (day - 1) / 31)
        return np.array([year_norm, month_sin, month_cos, day_sin, day_cos])
    except ValueError as e:
        print(e)
        return None

# %%
folder_path = 'dataset/Training_Anomalies_Station Data'

# Process each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Ensure it's a CSV file
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        
        # Drop the longitude and latitude columns if they exist
        df = df.drop(columns=['longitude', 'latitude'], errors='ignore')
        
        # Save the updated DataFrame back to the same file (or modify as needed)
        df.to_csv(file_path, index=False)

# %%
folder_path = 'dataset/Training_Anomalies_Station Data'
all_data = []

# Process each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Ensure it's a CSV file
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df = df[['t', 'anomaly', 'location']]  # Ensure the required columns
        all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Handle duplicates by aggregating using mean or another function
combined_df = combined_df.groupby(['t', 'location']).mean().reset_index()

# Pivot the DataFrame so each location is a column
result = combined_df.pivot(index='t', columns='location', values='anomaly')

# Reset the index name to "time"
result.index.name = 'Date'

result.to_csv('Station_Anomaly.csv')


Station_Anomaly = pd.read_csv('Station_Anomaly.csv')
Station_Anomaly.head()

# %%
directory = "dataset/Copernicus_ENA_Satelite_Maps_Training_Data"
data = []

for filename in os.listdir(directory):
    if filename.endswith(".nc"):
        file_path = os.path.join(directory, filename)

        # Extract the date part from the filename and format it
        date_str = filename.split("_")[2]
        if len(date_str) == 8:
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # Open the .nc file
            dataset = netCDF4.Dataset(file_path, mode="r")

            # Extract the 'sla' variable
            sla = dataset.variables["sla"][:][0]
            sla = sla.filled(-10)

            data.append({'Date': formatted_date, 'Map': sla})
            
Map_df = pd.DataFrame(data)
Map_df.head()

# %%
Raw_Master_df = Station_Anomaly.merge(Map_df[['Date', 'Map']], on = 'Date')
print(Raw_Master_df.shape)
Raw_Master_df.head()

# %%
Raw_Master_df['Date'] = Raw_Master_df['Date'].apply(preprocess_date)

# %%
def LogRegProc(City_df, City_name):
    x = City_df[['Date', 'Map']].to_numpy()
    y = City_df[[City_name]].to_numpy()
    X = []
    for row in x:
        X.append(numpy.concatenate((row[0], row[1].flatten())))
    X = np.array(X)
    return X, y

# %%
def scoreStation(df, name):
    X, y = LogRegProc(df, name)
    xTrain = X[:int(len(X)*0.8)]
    yTrain = y[:int(len(y)*0.8)]
    xTest = X[int(len(X)*0.8):]
    yTest = y[int(len(y)*0.8):]
    mod = sklearn.linear_model.LogisticRegression(fit_intercept=False)
    mod.fit(xTrain,yTrain)
    return mod.score(xTest, yTest)

# %%
Atlantic_City = Raw_Master_df[['Date', 'Atlantic City', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Atlantic_City.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Atlantic_City.csv")
Atlantic_City.to_csv(output_file, index=False)


# %%
Baltimore = Raw_Master_df[['Date', 'Baltimore', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Baltimore.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Baltimore.csv")
Baltimore.to_csv(output_file, index=False)

# %%
Eastport = Raw_Master_df[['Date', 'Eastport', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Eastport.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Eastport.csv")
Eastport.to_csv(output_file, index=False)

# %%
Fort_Pulaski = Raw_Master_df[['Date', 'Fort Pulaski', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Fort_Pulaski.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Fort Pulaski.csv")
Fort_Pulaski.to_csv(output_file, index=False)

# %%
Lewes = Raw_Master_df[['Date', 'Lewes', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Lewes.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Lewes.csv")
Lewes.to_csv(output_file, index=False)

# %%
New_London = Raw_Master_df[['Date', 'New London', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
New_London.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "New_London.csv")
New_London.to_csv(output_file, index=False)

# %%
Newport = Raw_Master_df[['Date', 'Newport', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Newport.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Newport.csv")
Newport.to_csv(output_file, index=False)

# %%
Portland = Raw_Master_df[['Date', 'Portland', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Portland.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Portland.csv")
Portland.to_csv(output_file, index=False)

# %%
Sandy_Hook = Raw_Master_df[['Date', 'Sandy Hook', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Sandy_Hook.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Sandy_Hook.csv")
Sandy_Hook.to_csv(output_file, index=False)

# %%
Sewells_Point = Raw_Master_df[['Date', 'Sewells Point', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Sewells_Point.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Sewells_Point.csv")
Sewells_Point.to_csv(output_file, index=False)

# %%
The_Battery = Raw_Master_df[['Date', 'The Battery', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
The_Battery.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "The_Battery.csv")
The_Battery.to_csv(output_file, index=False)

# %%
Washington = Raw_Master_df[['Date', 'Washington', 'Map']]
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
Washington.dropna(inplace=True)
output_dir = 'cleaned_dataset'
output_file = os.path.join(output_dir, "Washington.csv")
Washington.to_csv(output_file, index=False)
print(Washington.shape)
Washington.head()

# %%
stations_dict = {
    'Atlantic City': Atlantic_City,
    'Baltimore': Baltimore,
    'Eastport': Eastport,
    'Fort Pulaski': Fort_Pulaski,
    'Lewes': Lewes,
    'New London': New_London,
    'Newport': Newport,
    'Portland': Portland,
    'Sandy Hook': Sandy_Hook,
    'Sewells Point': Sewells_Point,
    'The Battery': The_Battery,
    'Washington': Washington
}
score_dict = {}

# %%
for k, v in stations_dict.items():
    scoreStation(v, k)
    score_dict[k] = scoreStation(v, k)
print(score_dict)


