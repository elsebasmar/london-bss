import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import schedule
from londonbss.ml_logic.data import load_data_to_bq
from londonbss import params

def thing_you_wanna_do():
    response= requests.get('https://api.tfl.gov.uk/BikePoint/')
    stations = response.json()
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data =  pd.DataFrame()
    s_names = []
    s_lat = []
    s_lon = []

    NbBikes = []
    NbEmptyDocks = []
    NbDocks = []
    NbStandardBikes = []
    NbEBikes = []
    s_id = []

    len_lat = 0
    for station in stations:
        s_names.append(station['commonName'])
        s_lat.append(float(station['lat']))
        s_lon.append(float(station['lon']))
        for add_property in station['additionalProperties']:
            if add_property['key'] == 'NbBikes':
                NbBikes.append(int(add_property['value']))
            if add_property['key'] == 'NbEmptyDocks':
                NbEmptyDocks.append(int(add_property['value']))
            if add_property['key'] == 'NbDocks':
                NbDocks.append(int(add_property['value']))
            if add_property['key'] == 'NbStandardBikes':
                NbStandardBikes.append(int(add_property['value']))
            if add_property['key'] == 'NbEBikes':
                NbEBikes.append(int(add_property['value']))
            if add_property['key'] == 'TerminalName':
                s_id.append(int(add_property['value']))

    data['_Station_name'] = s_names
    data['_s_lat'] = s_lat
    data['_s_lon'] = s_lon
    data['_s_num_bikes'] = NbBikes
    data['_s_num_empty_docks'] = NbEmptyDocks
    data['_s_num_docks'] = data['_s_num_bikes'] + data['_s_num_empty_docks']
    data['_s_id'] = s_id
    data['_time_api'] = time_now

    load_data_to_bq(data,  params.GCP_PROJECT, params.BQ_DATASET_VM
                    , params.BQ_TABLE_VM
                    , truncate= False
    )

    return data

# schedule.every().hour.do(thing_you_wanna_do)

# while True:
#     schedule.run_pending()
