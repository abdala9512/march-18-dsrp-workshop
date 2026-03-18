"""
Pipeline de entrenamiento completo para prediccion de tarifas de Uber.
Uso: uv run python scripts/train_pipeline.py [--model-type random_forest|gradient_boosting] [--promote]
"""

import argparse
import sys

import mlflow

from src.champion_challenger import (
    compare_models,
    generate_report,
    get_champion_metrics,
    promote_to_champion,
)
from src.config import setup_mlflow
from src.data import feature_engineering, load_uber_data, prepare_data
from src.evaluate import (
    create_evaluation_plots,
    evaluate_model,
    log_metrics_to_mlflow,
)
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento MLOps - Uber Fares")
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="Tipo de modelo a entrenar",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promover automaticamente si el Challenger es mejor",
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Hiperparametros opcionales
    hyperparams = {}
    if args.n_estimators is not None:
        hyperparams["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        hyperparams["max_depth"] = args.max_depth
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate

    # 1. Configurar MLflow
    setup_mlflow()

    # 2. Cargar y preparar datos
    df = load_uber_data()
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 3. Entrenar y evaluar dentro de un run de MLflow
    with mlflow.start_run() as run:
        print(f"\nRun ID: {run.info.run_id}")

        # Entrenar
        model = train_model(X_train, y_train, model_type=args.model_type, **hyperparams)

        # Evaluar
        challenger_metrics = evaluate_model(model, X_test, y_test)
        log_metrics_to_mlflow(challenger_metrics)

        # Graficos
        y_pred = model.predict(X_test)
        create_evaluation_plots(y_test, y_pred)

        # Registrar modelo como artefacto
        mlflow.sklearn.log_model(model, "model")

        # 4. Comparar con Champion
        print("\n--- Comparacion Champion vs Challenger ---")
        champion_metrics = get_champion_metrics()
        comparison = compare_models(challenger_metrics, champion_metrics)
        report = generate_report(comparison, challenger_metrics, champion_metrics)

        # 5. Guardar reporte para CML
        with open("report.md", "w") as f:
            f.write(report)
        mlflow.log_artifact("report.md")
        print("\nReporte guardado en report.md")

        # 6. Mostrar resultado
        print(f"\n{'=' * 50}")
        print(f"Resultado: {comparison['reason']}")
        print(f"{'=' * 50}")

        # 7. Promover si corresponde
        if comparison["is_better"] and args.promote:
            print("\nPromoviendo Challenger a Champion...")
            promote_to_champion(run.info.run_id)
        elif comparison["is_better"] and not args.promote:
            print("\nEl Challenger es mejor. Usa --promote para promoverlo.")
        else:
            print("\nEl Champion actual sigue siendo mejor. No se realizan cambios.")

    return 0 if comparison["is_better"] else 1


if __name__ == "__main__":
    sys.exit(main())
