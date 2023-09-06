import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
import math

from londonbss.params import *
from londonbss.ml_logic.data import get_data_with_cache, clean_data, get_net_balance, load_data_to_bq
from londonbss.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model, test_adf
from londonbss.ml_logic.model import initialize_model_series, train_model_series, evaluate_model_series
from londonbss.ml_logic.preproc import fit_transform_features

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

    # Load data from Train, Val and Test

    # Loading the Train
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_train_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    X_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_train_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_train = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=X_cache_path,
        data_has_header=True
    )

    df_train.set_index('startdate',inplace=True)

    # Loading the validation
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_val_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    X_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_val_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_val = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=X_cache_path,
        data_has_header=True
    )

    df_val.set_index('startdate',inplace=True)

    # Loading the test
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_test_{station}_{DATA_SIZE}
        WHERE startdate BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY startdate ASC
    """

    X_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_test_{station}_{min_date}_{max_date}_{DATA_SIZE}.csv")
    df_test = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=X_cache_path,
        data_has_header=True
    )

    df_test.set_index('startdate',inplace=True)

    if df_train.shape[0] == 0:

        # Create my y and X
        y = y_data_processed[[station]]
        y.set_index(y_data_processed['startdate'], inplace=True)
        X = X_data_processed
        X.set_index('startdate', inplace=True)

        X_train = X['2020-01-01':'2022-01-01']
        X_val = X['2022-01-02':'2022-07-01'] # Time step of one day for Validation
        X_test = X['2022-07-02':'2023-01-01'] # Time step of one day for Test

        y_train = y['2020-01-01':'2022-01-01']
        y_val = y['2022-01-02':'2022-07-01'] # Time step of one day for Validation
        y_test = y['2022-07-02':'2023-01-01'] # Time step of one day for Test

        # Transforming our y values
        Robust_scale = RobustScaler().fit(y_train)
        y_train_processed = pd.DataFrame(Robust_scale.transform(y_train))
        y_val_processed = pd.DataFrame(Robust_scale.transform(y_val))
        y_test_processed = pd.DataFrame(Robust_scale.transform(y_test))

        # Recovering index for y
        y_train_processed.index = y_train.index
        y_train_processed.columns = y_train.columns

        y_val_processed.index = y_val.index
        y_val_processed.columns = y_val.columns

        y_test_processed.index = y_test.index
        y_test_processed.columns = y_test.columns

        print(Fore.MAGENTA + "\n✅ ys processed" + Style.RESET_ALL)

        # Transforming features
        X_train_processed, X_train_columns = fit_transform_features(X_train,'train')
        X_val_processed, X_val_columns = fit_transform_features(X_val,'val')
        X_test_processed, X_test_columns = fit_transform_features(X_test,'test')

        # Recovering index for X
        X_train_processed.index = X_train.index.unique().sort_values()
        X_train_processed.columns = X_val_columns

        X_val_processed.index = X_val.index.unique().sort_values()
        X_val_processed.columns = X_train_columns

        X_test_processed.index = X_test.index.unique().sort_values()
        X_test_processed.columns = X_test_columns

        print(Fore.MAGENTA + "\n✅ Xs processed" + Style.RESET_ALL)

        # Creating Data Set for Prediction
        steps = -24

        y_train_processed_pred = y_train_processed.copy()
        y_train_processed_pred[y_train_processed_pred.columns[0]]=y_train_processed_pred[[y_train_processed_pred.columns[0]]].shift(steps)

        y_val_processed_pred = y_val_processed.copy()
        y_val_processed_pred[y_val_processed_pred.columns[0]]=y_val_processed_pred[[y_val_processed_pred.columns[0]]].shift(steps)

        y_test_processed_pred = y_test_processed.copy()
        y_test_processed_pred[y_test_processed_pred.columns[0]]=y_test_processed_pred[[y_test_processed_pred.columns[0]]].shift(steps)

        # Merging
        df_train = y_train_processed_pred.join(X_train_processed).dropna()

        load_data_to_bq(
            df_train,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f'processed_train_{station}_{DATA_SIZE}',
            truncate=True
        )

        df_val = y_val_processed_pred.join(X_val_processed).dropna()

        load_data_to_bq(
            df_val,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f'processed_val_{station}_{DATA_SIZE}',
            truncate=True
        )

        df_test = y_test_processed_pred.join(X_test_processed).dropna()

        load_data_to_bq(
            df_test,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f'processed_test_{station}_{DATA_SIZE}',
            truncate=True
        )

        print(Fore.MAGENTA + "\n✅ Date Merged Done" + Style.RESET_ALL)

        print(Fore.BLUE + "Shape of Training dataset : X = "+
                        str(y_train_processed.shape) + " y = "+
                        str(X_train_processed.shape)+Style.RESET_ALL)

        print(Fore.BLUE + "Shape of Validation dataset : X = "+
                        str(y_val_processed.shape) + " y = "+
                        str(X_val_processed.shape)+Style.RESET_ALL)

        print(Fore.BLUE + "Shape of Test dataset : X = "+
                        str(y_test_processed.shape) + " y = "+
                        str(X_test_processed.shape)+Style.RESET_ALL)

    #Creating set of X an Y
    y_train_processed = df_train[[df_train.columns[0]]]
    X_train_processed = df_train.drop(columns=[df_train.columns[0]])

    y_val_processed = df_val[[df_val.columns[0]]]
    X_val_processed = df_val.drop(columns=[df_val.columns[0]])

    y_test_processed = df_test[[df_test.columns[0]]]
    X_test_processed = df_test.drop(columns=[df_test.columns[0]])


    test_adf(y_test_processed[y_test_processed.columns[0]], "Bike status")

    # Train model using `model.py`
    model = load_model(stage='Production', n_station=station)

    if model is None:
        model = initialize_model_series(y=y_train_processed, X=X_train_processed)
        model_one, results = train_model_series(best_model=model, X=X_val_processed, y=y_val_processed)
        # Save model weight on the hard drive (and optionally on GCS too!)
        save_model(model=results,n_station=station)

    print("✅ train() done for f{station} \n")


    '''
    # # Create the x_test and y_test data sets

    # # Training
    # X_train_processed_steps = []
    # y_train_processed_steps = []

    # for i in range(len(X_train_processed) - time_steps):
    #     X_train_processed_steps.append(X_train_processed.iloc[i:i + time_steps].to_numpy())
    #     y_train_processed_steps.append(y_train_processed.iloc[i + time_steps])

    # X_train_processed_steps = np.array(X_train_processed_steps)
    # y_train_processed_steps = np.array(y_train_processed_steps)

    # # Validation
    # X_val_processed_steps = []
    # y_val_processed_steps = y_val_processed.iloc[:len(X_val_processed) - time_steps]


    # for i in range(len(X_val_processed) - time_steps):
    #     X_val_processed_steps.append(X_val_processed.iloc[i:i + time_steps].to_numpy())

    # X_val_processed_steps = np.array(X_val_processed_steps)
    # y_val_processed_steps = np.array(y_val_processed_steps)

    # # Testing

    # X_test_processed_steps = []
    # y_test_processed_steps = y_test_processed.iloc[:len(X_test_processed) - time_steps]

    # for i in range(len(X_test_processed) - time_steps):
    #     X_test_processed_steps.append(X_test_processed.iloc[i:i + time_steps].to_numpy())

    # X_test_processed_steps = np.array(X_test_processed_steps)
    # y_test_processed_steps = np.array(y_test_processed_steps)

    # print(Fore.MAGENTA + "\n✅ Steps Done" + Style.RESET_ALL)

    # print(Fore.BLUE + "Shape of Training dataset : X = "+
    #                 str(y_train_processed_steps.shape) + " y = "+
    #                 str(X_train_processed_steps.shape)+Style.RESET_ALL)

    # print(Fore.BLUE + "Shape of Validation dataset : X = "+
    #                 str(y_val_processed_steps.shape) + " y = "+
    #                 str(X_val_processed_steps.shape)+Style.RESET_ALL)

    # print(Fore.BLUE + "Shape of Test dataset : X = "+
    #                 str(y_test_processed_steps.shape) + " y = "+
    #                 str(X_test_processed_steps.shape)+Style.RESET_ALL)














    # # Create data frame to train
    # station_list = [station]
    # columns_add = station_list + FEATURES_ADDED
    # df = data_processed[columns_add]

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    # Get/Compute the number of rows to train the model on
    # training_data_len = math.ceil(len(df) *.8) # taking 90% of data to train and 10% of data to test
    # testing_data_len = len(df) - training_data_len

    # time_steps = 24
    # train, test = df.iloc[0:training_data_len], df.iloc[(training_data_len-time_steps):len(df)]

    # print(Fore.MAGENTA + "\n✅ Train and test sets created" + Style.RESET_ALL)
    # print(Fore.BLUE + "\nShape of Entire dataset : "+ str(df.shape) + Style.RESET_ALL)
    # print(Fore.BLUE + "Shape of Training dataset : "+ str(train.shape) + Style.RESET_ALL)
    # print(Fore.BLUE + "Shape of Test dataset : "+ str(test.shape) + Style.RESET_ALL)

    # # Get/Compute the number of rows to train the model on
    # val_training_data_len = math.ceil(len(train) *.8) # taking 90% of data to train and 10% of data to test
    # val_testing_data_len = len(train) - val_training_data_len

    # time_steps = 24
    # val_train, val_test = train.iloc[0:val_training_data_len], train.iloc[(val_training_data_len-time_steps):len(train)]

    # print(Fore.MAGENTA + "\n✅ Validation train and test sets created" + Style.RESET_ALL)
    # print(Fore.BLUE + "\nShape of Validation dataset : "+ str(train.shape) + Style.RESET_ALL)
    # print(Fore.BLUE + "Shape of Val. Train dataset : "+ str(val_train.shape) + Style.RESET_ALL)
    # print(Fore.BLUE + "Shape of Val. Test dataset : "+ str(val_test.shape) + Style.RESET_ALL)

    # #Scale the all of the data from columns ['nooftrips']
    # Robust_scale = RobustScaler().fit(val_train[[station]])
    # val_train[station] = Robust_scale.transform(val_train[[station]])
    # val_test[station] = Robust_scale.transform(val_test[[station]])
    # test[station] = Robust_scale.transform(test[[station]])

    # print(Fore.MAGENTA + "\n✅ y scaled" + Style.RESET_ALL)

    # train.to_numpy()
    # test.to_numpy()

    # #Split the data into x_train and y_train data sets
    # X_val_train = []
    # y_val_train = []

    # for i in range(len(val_train) - time_steps):
    #     X_val_train.append(val_train.drop(columns=station).iloc[i:i + time_steps].to_numpy())
    #     y_val_train.append(val_train.loc[:,station].iloc[i + time_steps])

    # #Convert x_train and y_train to numpy arrays
    # X_val_train = np.array(X_val_train)
    # y_val_train = np.array(y_val_train)

    # #Create the x_test and y_test data sets
    # X_val_test = []
    # y_val_test = train.loc[:,station].iloc[val_training_data_len:len(train)]

    # for i in range(len(val_test) - time_steps):
    #     X_val_test.append(val_test.drop(columns=station).iloc[i:i + time_steps].to_numpy())

    # #Convert x_test and y_test to numpy arrays
    # X_val_test = np.array(X_val_test)
    # y_val_test = np.array(y_val_test)

    # #Create the x_test and y_test data sets
    # X_test = []
    # y_test = df.loc[:,station].iloc[training_data_len:len(df)]

    # for i in range(len(test) - time_steps):
    #     X_test.append(test.drop(columns=station).iloc[i:i + time_steps].to_numpy())
    #     #y_test.append(test.loc[:,'cnt'].iloc[i + time_steps])

    # #Convert x_test and y_test to numpy arrays
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # print(Fore.MAGENTA + "\n✅ X and y created" + Style.RESET_ALL)
    # print(Fore.BLUE +'Validation Train data size:'+ Style.RESET_ALL)
    # print(X_val_train.shape, y_val_train.shape)
    # print(Fore.BLUE +'Validation Test data size:'+ Style.RESET_ALL)
    # print(X_val_test.shape, y_val_test.shape)
    # print(Fore.BLUE +'Test data size:'+ Style.RESET_ALL)
    # print(X_test.shape, y_test.shape)

    # Preprocess Features


    # # Train model using `model.py`
    # model = load_model(stage='Production', n_station=station)

    # if model is None:
    #     model = initialize_model(input_shape=X_val_train.shape[1:])

    # model = compile_model(model, learning_rate=learning_rate)
    # model, history = train_model(
    #     model, X_val_train, y_val_train,
    #     batch_size=batch_size,
    #     patience=patience,
    #     validation_data=(X_val_test, y_val_test)
    # )

    # history_model = history

    # val_mae = np.min(history.history['val_mae'])

    # params = dict(
    #     context="train",
    #     training_set_size=DATA_SIZE,
    #     row_count=len(X_val_train),
    # )

    # # Save results on the hard drive using taxifare.ml_logic.registry
    # save_results(params=params, metrics=dict(mae=val_mae), n_station=station)

    # # Save model weight on the hard drive (and optionally on GCS too!)
    # save_model(model=model,n_station=station)

    # # # The latest model should be moved to staging
    # # if MODEL_TARGET == 'mlflow':
    # #     mlflow_transition_model(current_stage="None", new_stage="Staging")

    # print("✅ train() done for f{station} \n")

    '''

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

    df_test.set_index('startdate',inplace=True)
    y_test_processed = df_test[[df_test.columns[0]]]
    X_test_processed = df_test.drop(columns=[df_test.columns[0]])

    if df_test.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    predictions = evaluate_model_series(model=model, X=X_test_processed, y=y_test_processed)

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
