import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

def get_status():

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

    data['Station_name'] = s_names
    data['s_lat'] = s_lat
    data['s_lon'] = s_lon
    data['s_num_bikes'] = NbBikes
    data['s_num_empty_docks'] = NbEmptyDocks
    data['s_num_docks'] = data['s_num_bikes'] + data['s_num_empty_docks']
    data['s_id'] = s_id
    data['time_api'] = time_now

    return data
