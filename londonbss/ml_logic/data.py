import pandas as pd
import requests
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
from londonbss.params import *

def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)
    print(f"Data loaded, with shape {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    # Changing names

    df = df.rename(columns={
        '_StartDate':'startdate',
        '_StartStationName':'startstation',
        '_EndStationName':'endstation',
        '_Nooftrips':'nooftrips'
    })

    print("✅ data cleaned")

    return df

def get_net_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Getting the total trips that started from each station
    at each hour of the data series
    """

    # Getting total origin and destination
    origin = pd.pivot_table(df, values='nooftrips', index=['startdate'],columns=['startstation'], aggfunc="sum").fillna(0)*-1
    destination = pd.pivot_table(df, values='nooftrips', index=['startdate'],columns=['endstation'], aggfunc="sum").fillna(0)

    # Lower case Stations
    origin.columns= origin.columns.str.strip().str.lower().str.replace(',',' ').str.replace('.','').str.replace('(','').str.replace(')','').str.replace(' ','_').str.replace("'","")
    destination.columns= destination.columns.str.strip().str.lower().str.replace(',',' ').str.replace('.','').str.replace('(','').str.replace(')','').str.replace(' ','_').str.replace("'","")

    # Eliminating double station
    ori_uniq_stations = pd.DataFrame(pd.DataFrame(origin.columns)['startstation'].value_counts()).reset_index()
    ori_uniq_stations = ori_uniq_stations[ori_uniq_stations['count']==1]
    origin = origin[ori_uniq_stations['startstation']]

    des_uniq_stations = pd.DataFrame(pd.DataFrame(destination.columns)['endstation'].value_counts()).reset_index()
    des_uniq_stations = des_uniq_stations[des_uniq_stations['count']==1]
    destination = destination[des_uniq_stations['endstation']]

    # Calculating the net balance
    net_balance = origin.add(destination, fill_value=0)

    # Changing type of Index
    net_balance.index = pd.DatetimeIndex(net_balance.index)

    # Sort by Date
    net_balance.sort_index(ascending=True,inplace=True)
    # net_balance = net_balance.astype('int8')

    print("✅ Balance matrix created")

    return net_balance

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    columns_new=[]
    for column in range(len(data.columns)):
        columns_new.append(str(data.columns[column]))

    data.columns=columns_new

    PROJECT = gcp_project
    DATASET = bq_dataset
    TABLE = table

    full_table_name = f"{PROJECT}.{DATASET}.{TABLE}"

    if truncate==True:
        client = bigquery.Client()
        write_mode = "WRITE_TRUNCATE"
    else:
        client = bigquery.Client()
        write_mode = "WRITE_APPEND"

    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)

    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

    return result

def get_stations_info():
    response= requests.get('https://api.tfl.gov.uk/BikePoint/')
    stations = response.json()

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
    data['s_num_std_bikes'] = NbStandardBikes
    data['s_num_e-bikes'] = NbEBikes
    data['s_num_docks'] = data['s_num_bikes'] + data['s_num_empty_docks']
    data['s_id'] = s_id
