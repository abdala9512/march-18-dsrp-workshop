#!/bin/bash
# =============================================================
# Escenario 1: Establecer el modelo Champion base
# =============================================================
#
# Entrena un Random Forest con parametros por defecto y lo
# promueve como el primer Champion en el Model Registry.
#
# Este es el punto de partida: todos los Challengers futuros
# se compararan contra este modelo.
#
# Metricas esperadas (aproximadas):
#   RMSE: ~5.0 - 6.0
#   MAE:  ~3.0 - 4.0
#   R2:   ~0.60 - 0.70
# =============================================================

set -e

echo "🏆 Entrenando modelo Champion base (Random Forest)..."
echo ""

uv run python scripts/train_pipeline.py \
    --model-type random_forest \
    --n-estimators 100 \
    --max-depth 10 \
    --promote

echo ""
echo "✅ Champion base establecido."
echo "   Revisa las metricas en DagsHub y en report.md"
