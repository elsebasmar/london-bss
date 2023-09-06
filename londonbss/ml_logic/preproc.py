from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from colorama import Fore, Style

# ENCODE EVENTS

##London_zones --> Should that go into params ?? if we use them somewhere else?
# South East limits
se_limit_lat = 51.523811 # everthing lower than this value is south east
se_limit_lon = -0.101342 # everything greater than this value is south east

# East limits
e_limit_lat = 51.523811 # everything greater than this value is east
e_limit_lon = -0.022306 # everything greater than this is east

# South West limits
sw_limit_lat = 51.501631 # everthing lower than this value is south east
sw_limit_lon = -0.101342 # everything lower than this value is south east

# West limits
w_limit_lat = 51.501072  # everthing greater than this value is south east
w_limit_lon = -0.182406  # everthing lower than this value is south east

# Central limits
c_toplimit_lat = 51.501072 # everthing greater than this value is central
c_lowlimit_lat = 51.523811 # everything lower than this value is central

c_toplimit_lon = -0.101342 # everything lower than this value is central
c_lowlimit_lon = -0.182406 # everthing greater than this value is central

# North limits
n_toplimit_lon = -0.022306 # everything lower than this value is central
n_lowlimit_lon = -0.182406 # everthing greater than this value is central

n_lowlimit_lat = 51.523811 # everthing greater than this value is central

## Encode zone based on events lat and long
def encode_events_zone(X: pd.DataFrame):
    X = X.fillna(0)
    london_loc = []
    for i in range(len(X.index)):
        if X['event_latitude'][i] == 0:
            london_loc.append('No_Event')
        elif X['event_latitude'][i] == "London-wide":
            london_loc.append('London_all') # London wide
        elif float(X['event_latitude'][i]) > c_toplimit_lat and float(X['event_latitude'][i]) < c_lowlimit_lat and float(X['event_longitude'][i])< c_toplimit_lon and float(X['event_longitude'][i])>c_lowlimit_lon:
                london_loc.append('Central') # Central
        elif float(X['event_longitude'][i]) < n_toplimit_lon and float(X['event_longitude'][i]) > n_lowlimit_lon and float(X['event_latitude'][i]) > n_lowlimit_lat:
            london_loc.append('North') # North
        elif float(X['event_longitude'][i]) < w_limit_lon and float(X['event_latitude'][i]) > w_limit_lat:
            london_loc.append('West') # West
        elif float(X['event_longitude'][i]) < sw_limit_lon and float(X['event_latitude'][i]) < sw_limit_lat:
            london_loc.append('South_West')  # South_West
        elif float(X['event_longitude'][i]) > e_limit_lon and float(X['event_latitude'][i]) > e_limit_lat:
            london_loc.append('East') # East
        elif float(X['event_longitude'][i]) > se_limit_lon and float(X['event_latitude'][i]) < se_limit_lat:
            london_loc.append('South_East') # South East
        else:
            london_loc.append('Other')

    X["London_zone"] = london_loc
    X.drop(columns=["event_latitude", "event_longitude", "event_location", "event_start_date", "event_end_date"], inplace=True)

    return X

    ## OHE for London zone
def London_zone_manual_ohe(X: pd.DataFrame):
    X["London_zone_Central"] = 0
    X["London_zone_North"] = 0
    X["London_zone_West"] = 0
    X["London_zone_South_West"] = 0
    X["London_zone_South_East"] = 0
    X["London_zone_East"] = 0
    X["London_all"] = 0
    X["Event"] = 0
    X["London_zone_Central"] = X.apply(lambda x: 1.0 if x['London_zone'] == "Central" else 0.0, axis=1)
    X["London_zone_North"] = X.apply(lambda x: 1.0 if x['London_zone'] == "North" else 0.0, axis=1)
    X["London_zone_West"] = X.apply(lambda x: 1.0 if x['London_zone'] == "West" else 0.0, axis=1)
    X["London_zone_South_West"] = X.apply(lambda x: 1.0 if x['London_zone'] == "South_West" else 0.0, axis=1)
    X["London_zone_South_East"] = X.apply(lambda x: 1.0 if x['London_zone'] == "South_East" else 0.0, axis=1)
    X["London_zone_East"] = X.apply(lambda x: 1.0 if x['London_zone'] == "East" else 0.0, axis=1)
    X["London_all"] = X.apply(lambda x: 1.0 if x['London_zone'] == "London_all" else 0.0, axis=1)
    X["Event"] = X.apply(lambda x: 1.0 if x['London_zone'] != "No_Event" else 0.0, axis=1)

    return X

def london_all_encoding(X: pd.DataFrame):
    X["London_zone_Central"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_Central'], axis=1)
    X["London_zone_North"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_North'], axis=1)
    X["London_zone_West"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_West'], axis=1)
    X["London_zone_South_West"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_South_West'], axis=1)
    X["London_zone_South_East"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_South_East'], axis=1)
    X["London_zone_East"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_East'], axis=1)
    X.drop(columns= ["London_all", "event_title", "London_zone"], inplace = True)

    X = X.groupby("startdate").sum()

    return X

    ## Encode event_title
def encoding_strings(text):
    if text == 0:
        pass
    else:
        text = 1
    return text

events_pipeline = make_pipeline(
    FunctionTransformer(encode_events_zone),
    FunctionTransformer(London_zone_manual_ohe),
    FunctionTransformer(london_all_encoding),
    )

# ENCODE OTHER COLUMNS

    ## Encode elisabeth line and lockdownm strikes and school holidays
def bool_to_int(X: pd.DataFrame):

    X['elisabeth_line'] = (X['elisabeth_line'] >0).astype(int)
    X['lockdown'] = (X['lockdown'] >0).astype(int)

    X['strike'] = X['strike'].fillna('NoSTRIKE')
    X['strike'] = (X['strike'] != 'NoSTRIKE').astype(int)

    X['school_holidays'] = X['school_holidays'].fillna('NoHOL')
    X['school_holidays'] = (X['school_holidays'] != 'NoHOL').astype(int)

    X['daytime'] = (X['daytime'] == 'daytime').astype(int)

    X = X.groupby("startdate").sum()

    return X

    ## Add additional dates details

def turns_into_onetwos(X: pd.DataFrame):
    for col in X.columns:
        X[col] = (X[col] > 0).astype(int)
    return X

additional_pipeline = make_pipeline(
        FunctionTransformer(bool_to_int),
        FunctionTransformer(turns_into_onetwos),
)

# ENCODE WEATHER COLUMNS
def weather_drop_duplicates(X: pd.DataFrame):
    X = X.groupby("startdate").agg(pd.Series.mode)
    # print("weather", X.shape)
    return X

weather_scaler = make_pipeline(FunctionTransformer(weather_drop_duplicates),
                               MinMaxScaler())
weather_pipeline = make_column_transformer(
    (weather_scaler, ['temperature', 'rainfall', 'snowfall', 'cloudcover','wind_speed', 'wind_direction']),
    remainder='passthrough'
)

untouched_columns_pipeline = make_pipeline(FunctionTransformer(weather_drop_duplicates))

# FINAL FULL PREPROCESSOR
weather_columns = ['temperature', 'rainfall', 'snowfall', 'cloudcover','wind_speed', 'wind_direction']
events_columns = ['event_title','event_start_date', 'event_end_date', 'event_location','event_latitude', 'event_longitude']
other_columns = ['elisabeth_line', 'lockdown','school_holidays', 'strike', 'daytime']
untouched_columns = ['year', 'month', 'day', 'hour', 'weekday']

additional_col_list = ['elisabeth_line', 'lockdown', 'strike', 'school_holidays', 'daytime']
events_col_list = ['London_zone_Central', 'London_zone_North', 'London_zone_West',
       'London_zone_South_West', 'London_zone_South_East', 'London_zone_East',
       'Event']
final_col_list =  additional_col_list + events_col_list + weather_columns + untouched_columns

final_preprocessor = make_column_transformer(
        (additional_pipeline, other_columns),
        (events_pipeline, events_columns),
        (weather_pipeline, weather_columns),
        (untouched_columns_pipeline, untouched_columns))


# FIT
def fit_transform_features(X: pd.DataFrame, stage):
    if stage == "train":
        X_processed = final_preprocessor.fit_transform(X)
    elif stage == "val":
        X_processed = final_preprocessor.transform(X)
    elif stage == "test":
        X_processed = final_preprocessor.transform(X)
    else:
        print("You didn't provide a stage(train, val or test)")

    return pd.DataFrame(X_processed), final_col_list
