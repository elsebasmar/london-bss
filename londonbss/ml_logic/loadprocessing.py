import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from londonbss.params import *
from londonbss.ml_logic.data import get_data_with_cache, clean_data, get_net_balance, load_data_to_bq
from londonbss.ml_logic.preproc import fit_transform_features, get_preprocessor


# This function gets the raw data from BigQuery database

def get_rawdata(min_date:str = '2022-01-01', max_date:str = '2023-01-01') -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
        WHERE _StartDate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY _StartDate
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("bq", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Process data
    data_clean = clean_data(data_query)

    # Creating matrix of Origins and Destinations by station
    y, X = get_net_balance(data_clean, min_date, max_date)

    y.index.name = 'startdate'
    X.index.name = 'startdate'

    load_data_to_bq(
        y,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_trips_{DATA_SIZE}',
        truncate=True
    )

    load_data_to_bq(
        X,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_features_{DATA_SIZE}',
        truncate=True
    )

# This function process all data and loads it to BigQuery database

def data_processing(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        station = 'abbey_orchard_street__westminster'
    ):

    # Transforming features
    tranformer, final_col_list = get_preprocessor()

    print(Fore.MAGENTA + "\n⭐️ Processing Data Starting ..." + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Check if val, test and train files exist
    df_train_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed",
                                                         f"processed_all_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")

    if df_train_cache_path.is_file():

        print(Fore.BLUE + "\nData stored locally in CSV..." + Style.RESET_ALL)

    else:

        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        # Load processed data using `get_data_with_cache` in chronological order
        # Try it out manually on console.cloud.google.com first!

        # Loading the y
        query = f"""
            SELECT *
            FROM {GCP_PROJECT}.{BQ_DATASET}.processed_trips_{DATA_SIZE}
            WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
            ORDER BY startdate ASC
        """

        y_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processedy_{min_date}_{max_date}_{DATA_SIZE}.csv")
        y_data_processed = get_data_with_cache(
            gcp_project=GCP_PROJECT,
            query=query,
            cache_path=y_cache_path,
            data_has_header=True
        )

        # Loading the X
        query = f"""
            SELECT *
            FROM {GCP_PROJECT}.{BQ_DATASET}.processed_features_{DATA_SIZE}
            WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
            ORDER BY startdate ASC
        """

        X_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processedX_{min_date}_{max_date}_{DATA_SIZE}.csv")
        X_data_processed = get_data_with_cache(
            gcp_project=GCP_PROJECT,
            query=query,
            cache_path=X_cache_path,
            data_has_header=True
        )

        print(Fore.BLUE + "\nStaring processing data..." + Style.RESET_ALL)

        # Create my y and X
        y = y_data_processed[[station]]
        y.set_index(y_data_processed['startdate'], inplace=True)
        X = X_data_processed
        X.set_index('startdate', inplace=True)

        X_processed = pd.DataFrame(tranformer.fit_transform(X))
        X_processed.index = X.index.unique().sort_values()
        X_processed.columns = final_col_list

        # Save transformer
        transformer_cache_path = Path(LOCAL_DATA_PATH).joinpath("transformer","transformer.pkl")
        joblib.dump(tranformer,transformer_cache_path)

        print("✅ Processing done")

        # Merging

        df_train = X_processed.join(y).fillna(0)

        load_data_to_bq(
            df_train,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f'processed_all_{station}_{DATA_SIZE}',
            truncate=True
        )

        print("\n✅ Processed data loaded to BG Done")

# This function gets the processed data from BigQuery or locally

def get_processed_data(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        station = 'abbey_orchard_street__westminster'
    ) -> pd.DataFrame:

    """
    - Download raw data from BQ based on the min and max dates
    - Process the data based on the station
    - Store the data to BQ based on the min, max dates and station

    Return flow of stations and features as DataFrame
    """

    # Calls the function of raw data
    get_rawdata(min_date=min_date, max_date=max_date)

    # Calls the function of processing
    data_processing(min_date=min_date, max_date=max_date,station=station)

    print(Fore.MAGENTA + "\n⭐️ Getting processed Data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Loading the train
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_all_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    df_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed",
                                                         f"processed_all_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")

    df_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=df_cache_path,
        data_has_header=True
    )

    print(Fore.BLUE + "\nProcessed data loaded and a copy was saved locally" + Style.RESET_ALL)

    return df_processed


########### TO-DOs
# CHANGE this functions

def processing_pred(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        station = 'abbey_orchard_street__westminster'
    ):

    print(Fore.MAGENTA + "\n⭐️ Processing Data for prediction Starting ..." + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Check if val, test and train files exist
    x_pred_cache_path = Path(LOCAL_DATA_PATH).joinpath("x_pred",
                                                         f"processed_x_pred_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")

    if x_pred_cache_path.is_file():

        print(Fore.BLUE + "\Data stored in local CSV..." + Style.RESET_ALL)

    else:

        print(Fore.BLUE + "\nLoad data from csv file..." + Style.RESET_ALL)
        # Load processed data using `get_data_with_cache` in chronological order
        # Try it out manually on console.cloud.google.com first!


        file_path = f'{LOCAL_DATA_PATH}/X_pred.csv'

        X_pred=pd.read_csv(file_path)

        X_pred.fillna(value=0,inplace=True)

        # Transforming features
        X_pred_processed, X_pred_columns = fit_transform_features(X_pred,'test')

        # # Recovering index for X
        # X_pred_processed.index = X_pred.index.unique().sort_values()
        # X_pred_processed.columns = X_pred_columns

        # print(Fore.MAGENTA + "\n✅ X_pred is processed" + Style.RESET_ALL)

        # X_pred_processed.to_csv(x_pred_cache_path, header=True, index=False)

    return X_pred
