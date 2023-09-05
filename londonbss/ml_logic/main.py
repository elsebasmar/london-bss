import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
import math

from londonbss.params import *
from londonbss.ml_logic.data import get_data_with_cache, clean_data, get_net_balance, load_data_to_bq
from londonbss.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from sklearn.preprocessing import RobustScaler
# from taxifare.ml_logic.preprocessor import preprocess_features
from londonbss.ml_logic.registry import load_model, save_model, save_results
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

#Define the station

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
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create data frame to train
    station_list = [station]
    columns_add = station_list + FEATURES_ADDED
    df = data_processed[columns_add]

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    # Get/Compute the number of rows to train the model on
    training_data_len = math.ceil(len(df) *.8) # taking 90% of data to train and 10% of data to test
    testing_data_len = len(df) - training_data_len

    time_steps = 24
    train, test = df.iloc[0:training_data_len], df.iloc[(training_data_len-time_steps):len(df)]

    print(Fore.MAGENTA + "\n✅ Train and test sets created" + Style.RESET_ALL)
    print(Fore.BLUE + "\nShape of Entire dataset : "+ str(df.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Training dataset : "+ str(train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Test dataset : "+ str(test.shape) + Style.RESET_ALL)

    # Get/Compute the number of rows to train the model on
    val_training_data_len = math.ceil(len(train) *.8) # taking 90% of data to train and 10% of data to test
    val_testing_data_len = len(train) - val_training_data_len

    time_steps = 24
    val_train, val_test = train.iloc[0:val_training_data_len], train.iloc[(val_training_data_len-time_steps):len(train)]

    print(Fore.MAGENTA + "\n✅ Validation train and test sets created" + Style.RESET_ALL)
    print(Fore.BLUE + "\nShape of Validation dataset : "+ str(train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Val. Train dataset : "+ str(val_train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Val. Test dataset : "+ str(val_test.shape) + Style.RESET_ALL)

    #Scale the all of the data from columns ['nooftrips']
    Robust_scale = RobustScaler().fit(val_train[[station]])
    val_train[station] = Robust_scale.transform(val_train[[station]])
    val_test[station] = Robust_scale.transform(val_test[[station]])
    test[station] = Robust_scale.transform(test[[station]])

    print(Fore.MAGENTA + "\n✅ y scaled" + Style.RESET_ALL)

    train.to_numpy()
    test.to_numpy()

    #Split the data into x_train and y_train data sets
    X_val_train = []
    y_val_train = []

    for i in range(len(val_train) - time_steps):
        X_val_train.append(val_train.drop(columns=station).iloc[i:i + time_steps].to_numpy())
        y_val_train.append(val_train.loc[:,station].iloc[i + time_steps])

    #Convert x_train and y_train to numpy arrays
    X_val_train = np.array(X_val_train)
    y_val_train = np.array(y_val_train)

    #Create the x_test and y_test data sets
    X_val_test = []
    y_val_test = train.loc[:,station].iloc[val_training_data_len:len(train)]

    for i in range(len(val_test) - time_steps):
        X_val_test.append(val_test.drop(columns=station).iloc[i:i + time_steps].to_numpy())

    #Convert x_test and y_test to numpy arrays
    X_val_test = np.array(X_val_test)
    y_val_test = np.array(y_val_test)

    #Create the x_test and y_test data sets
    X_test = []
    y_test = df.loc[:,station].iloc[training_data_len:len(df)]

    for i in range(len(test) - time_steps):
        X_test.append(test.drop(columns=station).iloc[i:i + time_steps].to_numpy())
        #y_test.append(test.loc[:,'cnt'].iloc[i + time_steps])

    #Convert x_test and y_test to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(Fore.MAGENTA + "\n✅ X and y created" + Style.RESET_ALL)
    print(Fore.BLUE +'Validation Train data size:'+ Style.RESET_ALL)
    print(X_val_train.shape, y_val_train.shape)
    print(Fore.BLUE +'Validation Test data size:'+ Style.RESET_ALL)
    print(X_val_test.shape, y_val_test.shape)
    print(Fore.BLUE +'Test data size:'+ Style.RESET_ALL)
    print(X_test.shape, y_test.shape)

    # Train model using `model.py`
    model = load_model(stage='Production', n_station=station)

    if model is None:
        model = initialize_model(input_shape=X_val_train.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_val_train, y_val_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_test, y_val_test)
    )

    history_model = history

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_val_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae), n_station=station)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model,n_station=station)

    # # The latest model should be moved to staging
    # if MODEL_TARGET == 'mlflow':
    #     mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done for f{station} \n")

    return val_mae


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
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_trips_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    # Create data frame to train
    station_list = [station]
    columns_add = station_list + FEATURES_ADDED
    df = data_processed[columns_add]

    # Get/Compute the number of rows to train the model on
    training_data_len = math.ceil(len(df) *.8) # taking 90% of data to train and 10% of data to test
    testing_data_len = len(df) - training_data_len

    time_steps = 24
    train, test = df.iloc[0:training_data_len], df.iloc[(training_data_len-time_steps):len(df)]

    print(Fore.MAGENTA + "\n✅ Train and test sets created" + Style.RESET_ALL)
    print(Fore.BLUE + "\nShape of Entire dataset : "+ str(df.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Training dataset : "+ str(train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Test dataset : "+ str(test.shape) + Style.RESET_ALL)

    # Get/Compute the number of rows to train the model on
    val_training_data_len = math.ceil(len(train) *.8) # taking 90% of data to train and 10% of data to test
    val_testing_data_len = len(train) - val_training_data_len

    time_steps = 24
    val_train, val_test = train.iloc[0:val_training_data_len], train.iloc[(val_training_data_len-time_steps):len(train)]

    print(Fore.MAGENTA + "\n✅ Validation train and test sets created" + Style.RESET_ALL)
    print(Fore.BLUE + "\nShape of Validation dataset : "+ str(train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Val. Train dataset : "+ str(val_train.shape) + Style.RESET_ALL)
    print(Fore.BLUE + "Shape of Val. Test dataset : "+ str(val_test.shape) + Style.RESET_ALL)

    #Scale the all of the data from columns ['nooftrips']
    Robust_scale = RobustScaler().fit(val_train[[station]])
    val_train[station] = Robust_scale.transform(val_train[[station]])
    val_test[station] = Robust_scale.transform(val_test[[station]])
    test[station] = Robust_scale.transform(test[[station]])

    print(Fore.MAGENTA + "\n✅ y scaled" + Style.RESET_ALL)

    train.to_numpy()
    test.to_numpy()

    #Split the data into x_train and y_train data sets
    X_val_train = []
    y_val_train = []

    for i in range(len(val_train) - time_steps):
        X_val_train.append(val_train.drop(columns=station).iloc[i:i + time_steps].to_numpy())
        y_val_train.append(val_train.loc[:,station].iloc[i + time_steps])

    #Convert x_train and y_train to numpy arrays
    X_val_train = np.array(X_val_train)
    y_val_train = np.array(y_val_train)

    #Create the x_test and y_test data sets
    X_val_test = []
    y_val_test = train.loc[:,station].iloc[val_training_data_len:len(train)]

    for i in range(len(val_test) - time_steps):
        X_val_test.append(val_test.drop(columns=station).iloc[i:i + time_steps].to_numpy())

    #Convert x_test and y_test to numpy arrays
    X_val_test = np.array(X_val_test)
    y_val_test = np.array(y_val_test)

    #Create the x_test and y_test data sets
    X_test = []
    y_test = df.loc[:,station].iloc[training_data_len:len(df)]

    for i in range(len(test) - time_steps):
        X_test.append(test.drop(columns=station).iloc[i:i + time_steps].to_numpy())

    #Convert x_test and y_test to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(Fore.MAGENTA + "\n✅ X and y created" + Style.RESET_ALL)
    print(Fore.BLUE +'Validation Train data size:'+ Style.RESET_ALL)
    print(X_val_train.shape, y_val_train.shape)
    print(Fore.BLUE +'Validation Test data size:'+ Style.RESET_ALL)
    print(X_val_test.shape, y_val_test.shape)
    print(Fore.BLUE +'Test data size:'+ Style.RESET_ALL)
    print(X_test.shape, y_test.shape)

    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_test)
    )

    save_results(params=params, metrics=metrics_dict, n_station=station)

    print("✅ evaluate() done \n")

    return X_test

def pred(X_pred: pd.DataFrame, n_station:str) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
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
