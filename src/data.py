import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "uber.csv")


def load_uber_data() -> pd.DataFrame:
    """Carga el dataset de tarifas de Uber desde data/uber.csv."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontro el dataset en {DATA_PATH}.\n"
            "Descargalo de https://www.kaggle.com/datasets/yasserh/uber-fares-dataset\n"
            "y coloca el archivo 'uber.csv' en la carpeta data/"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia en km entre dos puntos usando la formula de Haversine."""
    R = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica ingenieria de features al dataset de Uber."""
    df = df.copy()

    # Eliminar columna key (identificador)
    if "key" in df.columns:
        df = df.drop("key", axis=1)

    # Convertir fecha a datetime y extraer features temporales
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hora"] = df["pickup_datetime"].dt.hour
    df["dia_semana"] = df["pickup_datetime"].dt.dayofweek
    df["mes"] = df["pickup_datetime"].dt.month
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["es_hora_pico"] = df["hora"].apply(lambda h: 1 if h in [7, 8, 9, 17, 18, 19] else 0)
    df = df.drop("pickup_datetime", axis=1)

    # Calcular distancia entre pickup y dropoff
    df["distancia_km"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )

    # Limpiar datos anomalos
    df = df.dropna()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
    df = df[(df["distancia_km"] > 0) & (df["distancia_km"] < 200)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    df = df[(df["pickup_latitude"].between(-90, 90)) & (df["pickup_longitude"].between(-180, 180))]
    df = df[(df["dropoff_latitude"].between(-90, 90)) & (df["dropoff_longitude"].between(-180, 180))]

    # Eliminar coordenadas crudas (ya tenemos la distancia)
    df = df.drop(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"], axis=1)

    print(f"Dataset procesado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def prepare_data(df: pd.DataFrame):
    """Separa features y target, y divide en train/test."""
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
    return X_train, X_test, y_train, y_test
