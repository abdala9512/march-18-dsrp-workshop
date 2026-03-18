import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test) -> dict:
    """Evalua el modelo y retorna las metricas."""
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    print(f"Metricas: RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")
    return metrics


def log_metrics_to_mlflow(metrics: dict):
    """Registra las metricas en MLflow."""
    mlflow.log_metrics(metrics)
    print("Metricas registradas en MLflow.")


def create_evaluation_plots(y_test, y_pred, output_dir="."):
    """Genera graficos de evaluacion y los registra en MLflow."""
    os.makedirs(output_dir, exist_ok=True)

    # Predicho vs Real
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Valor Real")
    axes[0].set_ylabel("Valor Predicho")
    axes[0].set_title("Predicho vs Real")

    # Residuos
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=30, edgecolor="k", alpha=0.7)
    axes[1].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribucion de Residuos")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    mlflow.log_artifact(plot_path)
    print(f"Graficos guardados en {plot_path}")
    return plot_path
