"""
Promueve un modelo Challenger a Champion en el Model Registry de MLflow
y registra la informacion en champion_info.json.

Uso: uv run python scripts/promote_challenger.py --run-id <RUN_ID>
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import mlflow

from src.champion_challenger import promote_to_champion
from src.config import CHAMPION_MODEL_NAME, MLFLOW_TRACKING_URI, setup_mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="Promover Challenger a Champion")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID de MLflow del modelo a promover",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configurar MLflow
    setup_mlflow()

    # 2. Obtener info del run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(args.run_id)
    metrics = run.data.metrics
    params = run.data.params

    # 3. Promover en el Model Registry
    print(f"\nPromoviendo run {args.run_id[:8]}... a Champion")
    promote_to_champion(args.run_id)

    # 4. Escribir champion_info.json para tracking en git
    champion_info = {
        "champion_run_id": args.run_id,
        "model_name": CHAMPION_MODEL_NAME,
        "model_type": params.get("model_type", "desconocido"),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": MLFLOW_TRACKING_URI,
        "metrics": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
        },
        "params": dict(params),
    }

    with open("champion_info.json", "w") as f:
        json.dump(champion_info, f, indent=2)

    print(f"\nchampion_info.json actualizado:")
    print(json.dumps(champion_info, indent=2))
    print("\nPromocion completada.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
