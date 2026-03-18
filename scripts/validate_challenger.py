"""
Valida un modelo Challenger contra el Champion actual sin reentrenar.
Obtiene las metricas del run de MLflow y genera un reporte de comparacion.

Uso: uv run python scripts/validate_challenger.py --challenger-run-id <RUN_ID>

Exit codes:
  0 - El Challenger es mejor que el Champion
  1 - El Champion sigue siendo mejor (o error)
"""

import argparse
import json
import sys

import mlflow

from src.champion_challenger import (
    compare_models,
    generate_report,
    get_champion_metrics,
)
from src.config import setup_mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="Validar Challenger vs Champion")
    parser.add_argument(
        "--challenger-run-id",
        required=True,
        help="Run ID de MLflow del modelo Challenger a evaluar",
    )
    return parser.parse_args()


def get_challenger_metrics(run_id: str) -> dict:
    """Obtiene las metricas de un run de MLflow por su ID."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params

    print(f"Challenger encontrado (run_id={run_id[:8]}...):")
    print(f"  Modelo: {params.get('model_type', 'desconocido')}")
    print(f"  RMSE: {metrics.get('rmse', 'N/A')}")
    print(f"  MAE: {metrics.get('mae', 'N/A')}")
    print(f"  R2: {metrics.get('r2', 'N/A')}")

    return {
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "run_id": run_id,
        "model_type": params.get("model_type", "desconocido"),
    }


def main():
    args = parse_args()

    # 1. Configurar MLflow
    setup_mlflow()

    # 2. Obtener metricas del Challenger
    print(f"\n--- Obteniendo metricas del Challenger ---")
    challenger_metrics = get_challenger_metrics(args.challenger_run_id)

    # 3. Obtener metricas del Champion
    print(f"\n--- Obteniendo metricas del Champion ---")
    champion_metrics = get_champion_metrics()

    # 4. Comparar
    print(f"\n--- Comparacion Champion vs Challenger ---")
    comparison = compare_models(challenger_metrics, champion_metrics)

    # 5. Generar reporte
    report = generate_report(comparison, challenger_metrics, champion_metrics)

    with open("report.md", "w") as f:
        f.write(report)
    print("\nReporte guardado en report.md")

    # 6. Guardar resultado como JSON para otros scripts
    result = {
        "is_better": comparison["is_better"],
        "challenger_run_id": args.challenger_run_id,
        "challenger_rmse": comparison["challenger_rmse"],
        "challenger_r2": comparison["challenger_r2"],
        "champion_rmse": comparison["champion_rmse"],
        "champion_r2": comparison["champion_r2"],
        "reason": comparison["reason"],
    }
    with open("validation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # 7. Mostrar resultado
    print(f"\n{'=' * 50}")
    print(f"Resultado: {comparison['reason']}")
    print(f"{'=' * 50}")

    return 0 if comparison["is_better"] else 1


if __name__ == "__main__":
    sys.exit(main())
