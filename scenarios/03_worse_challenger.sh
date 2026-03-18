#!/bin/bash
# =============================================================
# Escenario 3: Challenger PEOR que el Champion
# =============================================================
#
# Entrena un Random Forest deliberadamente debil:
# - Solo 10 arboles (pocos estimadores)
# - Profundidad maxima de 2 (arboles muy simples)
#
# Este modelo subajusta (underfits) severamente, produciendo
# metricas peores que cualquier Champion razonable.
#
# El reporte mostrara que el RMSE del Challenger es MAYOR
# y el R2 es MENOR que el Champion.
#
# Metricas esperadas (aproximadas):
#   RMSE: ~7.0 - 9.0  (peor que el Champion ~5.0-6.0)
#   MAE:  ~5.0 - 6.0
#   R2:   ~0.20 - 0.40 (peor que el Champion ~0.60-0.70)
# =============================================================

set -e

echo "🥊 Entrenando Challenger debil (Random Forest limitado)..."
echo ""

uv run python scripts/train_pipeline.py \
    --model-type random_forest \
    --n-estimators 10 \
    --max-depth 2

echo ""
echo "📊 Revisa report.md para ver la comparacion."
echo "   El Champion deberia seguir siendo mejor."
echo "   El workflow de validacion NO creara un PR de promocion."
