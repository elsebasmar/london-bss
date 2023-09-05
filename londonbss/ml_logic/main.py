import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from londonbss.params import *
from londonbss.ml_logic.data import get_data_with_cache, clean_data, get_net_balance, load_data_to_bq
from londonbss.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
# from taxifare.ml_logic.preprocessor import preprocess_features
# from taxifare.ml_logic.registry import load_model, save_model, save_results
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
    data_processed = get_net_balance(data_clean)

    # TO-DO Add the preprocessing

    # TO-DO Not working the BigQuery

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_trips_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")

def train(
        min_date:str = '2022-01-01',
        max_date:str = '2023-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.01,
        batch_size = 20,
        patience = 10
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_trips_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    train_length = int(len(data_processed)*(1-split_ratio))

    data_processed_train = data_processed.iloc[:train_length, :].sample(frac=1).to_numpy()
    data_processed_val = data_processed.iloc[train_length:, :].sample(frac=1).to_numpy()

    X_train_processed = data_processed_train[:, :-1]
    y_train = data_processed_train[:, -1]

    X_val_processed = data_processed_val[:, :-1]
    y_val = data_processed_val[:, -1]

    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=X_train_processed.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # # Save model weight on the hard drive (and optionally on GCS too!)
    # save_model(model=model)

    # # The latest model should be moved to staging
    # if MODEL_TARGET == 'mlflow':
    #     mlflow_transition_model(current_stage="None", new_stage="Staging")

    # print("✅ train() done \n")

    return data_processed
