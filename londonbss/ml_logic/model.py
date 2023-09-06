import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

from statsmodels.tsa.statespace.sarimax import SARIMAX

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


################ TIME SERIES #####################

from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

def test_adf(series, title=''):
    dfout = {}
    dftest = adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05:
        print(Fore.MAGENTA + "\n✅ Strong evidence against Null Hypothesis"+ Style.RESET_ALL)
        print(Fore.BLUE +"\nReject Null Hypothesis - Data is Stationary"+ Style.RESET_ALL)
        print(Fore.BLUE +"Data is Stationary "+ title + Style.RESET_ALL)
    else:
        print("\n❌ Strong evidence for  Null Hypothesis")
        print("\nAccept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for ", title)

def initialize_model_series(y, X) -> Model:
    """
    Initialize the time series
    """
    print(Fore.MAGENTA +'\n Initializing Model'+ Style.RESET_ALL)

    # SARIMAX Model
    SARIMAX_model = auto_arima(y, exogenous=X,
                            start_p=1, start_q=1,
                            max_p=7, max_q=7,
                            d=1, max_d=7,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True
                            )

    print("\n✅ Model initialized")

    return SARIMAX_model

def train_model_series(
        best_model: Model,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model= SARIMAX(y,
                    exog=X,
                    order=best_model.get_params()['order'],
                    enforce_invertibility=False, enforce_stationarity=False,
                    seasonal_order=best_model.get_params()['seasonal_order'])

    results= model.fit()

    print(f"\n✅ Model trained")

    return model , results

def evaluate_model_series(
        model: Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    test_size = X.shape[0]

    predictions = model.predict(exog=X)
    # loss = metrics["loss"]
    # mae = metrics["mae"]

    print(f"✅ Model evaluated")

    return predictions

##############  LSTM - MODEL ########################
def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    #Build the LSTM network model
    model = Sequential()
    model.add(layers.Bidirectional(
        layers.LSTM(units=50, activation='tanh',input_shape=input_shape)))
    model.add(layers.Dense(35))
    model.add(layers.Dense(20))
    model.add(layers.Dense(15))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation="linear"))

    print("\n✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.01) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("\n✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=20,
        patience=10,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=50,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"\n✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
