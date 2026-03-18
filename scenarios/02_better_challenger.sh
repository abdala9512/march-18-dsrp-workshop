#!/bin/bash
# =============================================================
# Escenario 2: Challenger MEJOR que el Champion
# =============================================================
#
# Entrena un Gradient Boosting con hiperparametros optimizados.
# GradientBoosting con mas estimadores y learning rate ajustado
# tipicamente supera al Random Forest base en datos tabulares
# como las tarifas de Uber.
#
# El reporte mostrara que el RMSE del Challenger es MENOR
# y el R2 es MAYOR que el Champion.
#
# Metricas esperadas (aproximadas):
#   RMSE: ~4.0 - 5.0  (mejor que el Champion ~5.0-6.0)
#   MAE:  ~2.5 - 3.5
#   R2:   ~0.72 - 0.80 (mejor que el Champion ~0.60-0.70)
# =============================================================

set -e

echo "🥊 Entrenando Challenger (Gradient Boosting optimizado)..."
echo ""

uv run python scripts/train_pipeline.py \
    --model-type gradient_boosting \
    --n-estimators 200 \
    --max-depth 6 \
    --learning-rate 0.1

echo ""
echo "📊 Revisa report.md para ver la comparacion."
echo ""
echo "Si el Challenger es mejor, puedes:"
echo "  1. Promover manualmente:  uv run python scripts/train_pipeline.py --model-type gradient_boosting --n-estimators 200 --max-depth 6 --learning-rate 0.1 --promote"
echo "  2. Usar el workflow:      gh workflow run validate_and_promote.yml -f challenger_run_id=<RUN_ID> -f auto_promote=true"
echo "  3. Usar Claude Code:      /validate"
