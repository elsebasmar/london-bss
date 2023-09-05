from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from colorama import Fore, Style

# TRAIN / VAL /TEST split
X = pd.read_csv("~/code/elsebasmar/london-bss/raw_data/final_features_df.csv")
final_main_index = []

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
    # X['event_latitude'].astype("float")
    # X['event_longitude'].astype("float")
    london_loc = []
    for i in range(len(X.index)):
        if X['event_latitude'][i] == "London-wide":
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
    print("london_loc", X.shape)
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
    X["London_zone_Central"] = X.apply(lambda x: 1.0 if x['London_zone'] == "Central" else 0.0, axis=1)
    X["London_zone_North"] = X.apply(lambda x: 1.0 if x['London_zone'] == "North" else 0.0, axis=1)
    X["London_zone_West"] = X.apply(lambda x: 1.0 if x['London_zone'] == "West" else 0.0, axis=1)
    X["London_zone_South_West"] = X.apply(lambda x: 1.0 if x['London_zone'] == "South_West" else 0.0, axis=1)
    X["London_zone_South_East"] = X.apply(lambda x: 1.0 if x['London_zone'] == "South_East" else 0.0, axis=1)
    X["London_zone_East"] = X.apply(lambda x: 1.0 if x['London_zone'] == "East" else 0.0, axis=1)
    X["London_all"] = X.apply(lambda x: 1.0 if x['London_zone'] == "London_all" else 0.0, axis=1)
    print("london_zone_ohe", X.shape)
    return X

def london_all_encoding(X: pd.DataFrame):
    X["London_zone_Central"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_Central'], axis=1)
    X["London_zone_North"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_North'], axis=1)
    X["London_zone_West"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_West'], axis=1)
    X["London_zone_South_West"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_South_West'], axis=1)
    X["London_zone_South_East"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_South_East'], axis=1)
    X["London_zone_East"] = X.apply(lambda x: 1.0 if x['London_all'] == 1.0 else x['London_zone_East'], axis=1)
    X.drop(columns= ["London_all"], inplace = True)
    print("london_all_encoding", X.shape)
    return X

    ## Encode event_title
def encoding_strings(text):
    if text == 0:
        pass
    else:
        text = 1
    return text

X = X.reset_index()
def encode_event_title(X: pd.DataFrame):
    # X = X.reset_index()
    # X = X.set_index('index')
    X = X.groupby("timestamp").sum()
    X['event_title'] = X['event_title'].fillna(0)
    X['event_title'] = X['event_title'].apply(encoding_strings)
    # X.drop(columns = ["event_title"], inplace = True)
    print("event_encoding", X.shape)
    final_main_index = X.index
    print("initial main index", final_main_index)
    return X



events_pipeline = make_pipeline(
    FunctionTransformer(encode_events_zone),
    FunctionTransformer(London_zone_manual_ohe),
    FunctionTransformer(london_all_encoding),
    FunctionTransformer(encode_event_title),
)

# ENCODE OTHER COLUMNS

    ## Encode elisabeth line and lockdown
def bool_to_int(X: pd.DataFrame):
    print('bootoint', X.head())
    # X.groupby("timestamp").sum()
    X['elisabeth_line'] = np.where(X['elisabeth_line'] >0, 1, 0)
    X['lockdown'] = np.where(X['lockdown']>0, 1, 0)
    # X1 = X1[~X1.index.duplicated(keep="first")
    # print(X1.shape)
    print("booltoint", X.shape)
    print('bootoint', X.head())
    #print(np.unique(X.index).tolist())
    return X

    ## Encode strikes and school holidays
def encode_strikes_holidays(X: pd.DataFrame):
    print("strikes", X.head())
    X['strike'] = X['strike'].fillna(0)
    X['strike'] = X['strike'].apply(encoding_strings)
    X['school_holidays'] = X['school_holidays'].fillna(0)
    X['school_holidays'] = X['school_holidays'].apply(encoding_strings)
    #X = X.drop_duplicates()
    print("strikes_hols", X.shape)
    return X

    ## Add additional dates details

def encode_day_nighttime(X: pd.DataFrame):
    #print("daytime", X.head())
    print("daynighttime", X.shape)
    X["daytime"] = X["daytime"].replace("daytime", "1")
    X["daytime"] = X["daytime"].replace("nighttime", "0")
    #X = X.drop_duplicates()
    X = X.groupby("timestamp").sum()
    print("daynighttime", X.shape)
    return X

additional_pipeline = make_pipeline(
        FunctionTransformer(bool_to_int),
        FunctionTransformer(encode_strikes_holidays),
        FunctionTransformer(encode_day_nighttime),
)

# ENCODE WEATHER COLUMNS
def weather_drop_duplicates(X: pd.DataFrame):
    # X_index = pd.DataFrame(index = final_main_index)
    # print("final X_index", final_main_index)
    # X2 = X_index.join(X, how = "left")
    #X = X.drop_duplicates()
    X = X.groupby("timestamp").agg(pd.Series.mode)
    print("weather", X.shape)
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

    return pd.DataFrame(X_processed)

# DROP COLUMNS


# put scaling outside of the pipe and fit_transform specifically on X_train only
# then .transform on X_val and X_test at a later stage
# create a function if X_train then whatever, if X_val only transform and if X_test only transform --> call the function at a later stage wh
# when we have the final total DF
