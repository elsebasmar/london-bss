import numpy as np
import pandas as pd
import pickle
import joblib

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
import math

from londonbss.params import *
from londonbss.ml_logic.data import get_data_with_cache, clean_data, get_net_balance, load_data_to_bq
from londonbss.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model, test_adf
from londonbss.ml_logic.model import initialize_model_series, train_model_series, evaluate_model_series
from londonbss.ml_logic.preproc import fit_transform_features, get_preprocessor

from sklearn.preprocessing import RobustScaler
# from taxifare.ml_logic.preprocessor import preprocess_features
from londonbss.ml_logic.registry import load_model, save_model, save_results
from statsmodels.tools.eval_measures import rmse
# from taxifare.ml_logic.registry import mlflow_run, mlflow_transition_model

def preprocess(min_date:str = '2022-01-01', max_date:str = '2023-01-01') -> None:
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

    print("✅ preprocess() done \n")

    return y, X

def processing(
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

        if y_data_processed.shape[0] < 10:
            print("❌ Not enough processed data retrieved to train on")
            return None

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

        if X_data_processed.shape[0] < 10:
            print("❌ Not enough processed data retrieved to train on")
            return None

        # Create my y and X
        y = y_data_processed[[station]]
        y.set_index(y_data_processed['startdate'], inplace=True)
        X = X_data_processed
        X.set_index('startdate', inplace=True)

        X_processed = pd.DataFrame(tranformer.fit_transform(X))
        X_processed.index = X.index.unique().sort_values()
        X_processed.columns = final_col_list

        joblib.dump(tranformer, 'tranformer.pkl')

        print(Fore.MAGENTA + "\n✅ Xs processed" + Style.RESET_ALL)

        # Merging

        df_train = X_processed.join(y).fillna(0)

        load_data_to_bq(
            df_train,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f'processed_all_{station}_{DATA_SIZE}',
            truncate=True
        )

        print(Fore.MAGENTA + "\n✅ Data Merged Done" + Style.RESET_ALL)

    return None


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




def train(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.01,
        batch_size = 20,
        patience = 10,
        station = 'abbey_orchard_street__westminster'
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset based on the station (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Training Starting ..." + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

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

    df_train_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed",
                                                         f"processed_all_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")

    df_train_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=df_train_cache_path,
        data_has_header=True
    )

    if df_train_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    print(Fore.MAGENTA + "\n✅ Train dataset loaded" + Style.RESET_ALL)


def evaluate(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        stage: str = "Production",
        station = 'abbey_orchard_street__westminster'
    ):
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage, n_station=station)

    print("✅ load_model() done \n")

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_test_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_test_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_test = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_train_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_train_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_train = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_val_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_val_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_val = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )


    df_test.set_index('startdate',inplace=True)
    df_train.set_index('startdate',inplace=True)
    df_val.set_index('startdate',inplace=True)

    y_test_processed = df_test[[df_test.columns[0]]]
    X_test_processed = df_test.drop(columns=[df_test.columns[0]])

    if df_test.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    train_size = df_train.shape[0] + df_val.shape[0]
    test_size = df_test.shape[0]
    steps = -24


    predictions = evaluate_model_series(model=model, X=X_test_processed,
                                        y=y_test_processed,start =train_size,
                                        end=train_size+test_size+(steps)-1)

    y_test_processed = y_test_processed.iloc[:-21]

    error=rmse(pd.DataFrame(predictions), y_test_processed)


    print(f"✅ evaluate() done with a error of {round(error[0],2)} bikes\n")

    return predictions , y_test_processed

def pred(X_pred: pd.DataFrame, n_station:str) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.startdate("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model(stage='Production', n_station=n_station)

    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
