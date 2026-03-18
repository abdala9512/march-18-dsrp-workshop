import os

import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "")
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

EXPERIMENT_NAME = "uber-fare-prediction"
CHAMPION_MODEL_NAME = "uber-fare-champion"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def setup_mlflow():
    """Inicializa la conexion con DagsHub y configura MLflow."""
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow configurado: {MLFLOW_TRACKING_URI}")
    print(f"Experimento: {EXPERIMENT_NAME}")
