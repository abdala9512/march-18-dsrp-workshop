import mlflow
from mlflow.exceptions import MlflowException

from src.config import CHAMPION_MODEL_NAME


def get_champion_metrics() -> dict | None:
    """Obtiene las metricas del modelo Champion actual desde el Model Registry."""
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(CHAMPION_MODEL_NAME, "champion")
        run = client.get_run(model_version.run_id)
        metrics = run.data.metrics
        print(f"Champion encontrado (version {model_version.version}): RMSE={metrics.get('rmse', 'N/A')}")
        return {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "run_id": model_version.run_id,
            "version": model_version.version,
        }
    except MlflowException:
        print("No se encontro un modelo Champion registrado.")
        return None


def compare_models(challenger_metrics: dict, champion_metrics: dict | None) -> dict:
    """Compara el Challenger contra el Champion."""
    if champion_metrics is None:
        return {
            "is_better": True,
            "reason": "No existe un Champion previo. El Challenger sera el primer Champion.",
            "challenger_rmse": challenger_metrics["rmse"],
            "challenger_r2": challenger_metrics["r2"],
            "champion_rmse": None,
            "champion_r2": None,
            "rmse_improvement_pct": None,
        }

    rmse_improvement = champion_metrics["rmse"] - challenger_metrics["rmse"]
    rmse_improvement_pct = (rmse_improvement / champion_metrics["rmse"]) * 100
    is_better = challenger_metrics["rmse"] < champion_metrics["rmse"]

    return {
        "is_better": is_better,
        "reason": (
            f"Challenger es {'mejor' if is_better else 'peor'}: "
            f"RMSE {'disminuyo' if is_better else 'aumento'} en {abs(rmse_improvement_pct):.2f}%"
        ),
        "challenger_rmse": challenger_metrics["rmse"],
        "challenger_r2": challenger_metrics["r2"],
        "champion_rmse": champion_metrics["rmse"],
        "champion_r2": champion_metrics["r2"],
        "rmse_improvement_pct": rmse_improvement_pct,
    }


def promote_to_champion(run_id: str):
    """Registra el modelo del run dado y lo promueve a Champion."""
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    try:
        client.get_registered_model(CHAMPION_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(CHAMPION_MODEL_NAME)

    mv = client.create_model_version(
        name=CHAMPION_MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )
    client.set_registered_model_alias(CHAMPION_MODEL_NAME, "champion", mv.version)
    print(f"Modelo promovido a Champion: version {mv.version}, run_id={run_id}")


def generate_report(comparison: dict, challenger_metrics: dict, champion_metrics: dict | None) -> str:
    """Genera un reporte Markdown para CML."""
    lines = [
        "# Reporte de Entrenamiento: Champion vs Challenger",
        "",
        f"## Resultado: {'✅ Challenger es MEJOR' if comparison['is_better'] else '❌ Champion sigue siendo mejor'}",
        "",
        f"**{comparison['reason']}**",
        "",
        "## Metricas",
        "",
        "| Metrica | Champion | Challenger | Diferencia |",
        "|---------|----------|------------|------------|",
    ]

    if champion_metrics:
        for metric in ["rmse", "mae", "r2"]:
            champ_val = champion_metrics.get(metric, "N/A")
            chall_val = challenger_metrics.get(metric, "N/A")
            if isinstance(champ_val, float) and isinstance(chall_val, float):
                diff = chall_val - champ_val
                arrow = "↓" if (diff < 0 and metric in ["rmse", "mae"]) or (diff > 0 and metric == "r2") else "↑"
                lines.append(f"| {metric.upper()} | {champ_val:.4f} | {chall_val:.4f} | {arrow} {abs(diff):.4f} |")
            else:
                lines.append(f"| {metric.upper()} | {champ_val} | {chall_val} | - |")
    else:
        for metric in ["rmse", "mae", "r2"]:
            chall_val = challenger_metrics.get(metric, "N/A")
            val_str = f"{chall_val:.4f}" if isinstance(chall_val, float) else str(chall_val)
            lines.append(f"| {metric.upper()} | N/A | {val_str} | Primer modelo |")

    lines.extend([
        "",
        "## Graficos",
        "",
        "![Evaluacion](evaluation_plots.png)",
        "",
        "---",
        "*Reporte generado automaticamente por el pipeline de MLOps*",
    ])

    return "\n".join(lines)
