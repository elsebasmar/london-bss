import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_DATASET_VM = os.environ.get("BQ_DATASET_VM")
BQ_TABLE_VM = os.environ.get("BQ_TABLE_VM")
BQ_TABLE = os.environ.get("BQ_TABLE")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GCR_IMAGE = os.environ.get("GCR_IMAGE")
GCR_REGION = os.environ.get("GCR_REGION")
GCR_MEMORY = os.environ.get("GCR_MEMORY")

##################  CONSTANTS  #####################
COLUMN_NAMES_RAW = ['_StartDate','_StartStationName', '_EndStationName', '_Nooftrips']
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "elsebasmar", "london-bss","raw_data")

DTYPES_RAW = {
    "_StartStationName": "object",
    "_EndStationName": "object",
    "_Nooftrips": "int8"
}

FEATURES_ADDED = ['temperature', 'rainfall', 'snowfall', 'cloudcover', 'wind_speed',
       'wind_direction', 'date', 'year', 'month', 'day', 'hour', 'weekday',
       'daytime', 'event_title', 'event_start_date', 'event_end_date',
       'event_location', 'event_latitude', 'event_longitude', 'elisabeth_line',
       'lockdown', 'school_holidays', 'strike']

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "elsebasmar", "london-bss","training_outputs")
