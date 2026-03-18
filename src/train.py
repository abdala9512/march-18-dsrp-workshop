import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

DEFAULT_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
    },
}

MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def train_model(X_train, y_train, model_type="random_forest", **hyperparams):
    """Entrena un modelo de regresion y registra parametros en MLflow."""
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Modelo no soportado: {model_type}. Opciones: {list(MODEL_CLASSES.keys())}")

    params = {**DEFAULT_PARAMS[model_type], **hyperparams}
    model_class = MODEL_CLASSES[model_type]
    model = model_class(**params)

    mlflow.log_param("model_type", model_type)
    mlflow.log_params(params)

    print(f"Entrenando {model_type} con parametros: {params}")
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    return model
