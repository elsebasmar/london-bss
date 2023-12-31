import glob
import os
import sys
import time
import pickle

from colorama import Fore, Style
from google.cloud import storage
from darts.models import NBEATSModel , ExponentialSmoothing

from londonbss.params import *

def save_model(model , n_station:str) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}"+"-"+n_station+".pkl")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    # if MODEL_TARGET == "mlflow":
    #     mlflow.tensorflow.log_model(
    #         model=model,
    #         artifact_path="model",
    #         registered_model_name=MLFLOW_MODEL_NAME
    #     )

    #     print("✅ Model saved to MLflow")

    #    return None

    return None

def load_model(stage:str, n_station:str) :
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        file_path = os.path.dirname(os.path.realpath(sys.argv[1]))
        local_model_directory = os.path.join(file_path, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        list_models = [x for x in local_model_paths if x.split('-')[-1].split('.')[0]==n_station]
        most_recent_model_path_on_disk = sorted(list_models)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = ExponentialSmoothing.load(most_recent_model_path_on_disk) ### Wrong Model

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        try:
            files = [x for x in blobs if x.name.split('-')[-1].split('.')[0]==n_station]
            latest_blob = max(files, key=lambda x: x.updated)
            print(os.getcwd())
            latest_model_path_to_save = os.path.join(os.getcwd(),'training_outputs', latest_blob.name)
            print(latest_model_path_to_save)

            latest_blob.download_to_filename(latest_model_path_to_save)
            print("✅ Done download")

            latest_model = ExponentialSmoothing.load(latest_model_path_to_save) ## Wrong Model

            print("✅ Latest model downloaded from cloud storage")

            return latest_model

        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    # elif MODEL_TARGET == "mlflow":
    #     print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

    #     # Load model from MLflow
    #     model = None
    #     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #     client = MlflowClient()

    #     try:
    #         model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
    #         model_uri = model_versions[0].source

    #         assert model_uri is not None
    #     except:
    #         print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

    #         return None

    #     model = mlflow.tensorflow.load_model(model_uri=model_uri)

    #     print("✅ Model loaded from MLflow")
    #     return model

    else:
        return None

def save_results(params: dict, metrics: dict, n_station:str) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    # if MODEL_TARGET == "mlflow":
    #     if params is not None:
    #         mlflow.log_params(params)
    #     if metrics is not None:
    #         mlflow.log_metrics(metrics)
    #     print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + "-"+n_station+".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + "-"+n_station+ ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def get_local_model(station_name='eagle_wharf_road__hoxton',n=24):

    local_model_directory = os.path.join(LOCAL_DATA_PATH, "models")
    model_loaded = NBEATSModel.load(local_model_directory+f"/{station_name}_model.pkl")

    pred = model_loaded.predict(n=n)

    values = pred.all_values().ravel()

    return values
