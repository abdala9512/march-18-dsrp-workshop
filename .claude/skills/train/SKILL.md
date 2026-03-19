---
name: train
description: Ejecuta el pipeline de entrenamiento de ML para prediccion de tarifas de Uber, registra metricas en MLflow/DagsHub, y compara contra el modelo champion actual.
---

# Skill: Entrenar Modelo

Ejecuta el pipeline de entrenamiento del modelo de prediccion de tarifas de Uber.

## Instrucciones

1. Verifica que exista el archivo `data/uber.csv`. Si no existe, indica al usuario que lo descargue de https://www.kaggle.com/datasets/yasserh/uber-fares-dataset y lo coloque en `data/uber.csv`.

2. Verifica que exista un archivo `.env` en la raiz del proyecto con las variables `DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME` y `DAGSHUB_USER_TOKEN`. Si no existe, pregunta al usuario por estos valores y crea el archivo.

3. Ejecuta el pipeline de entrenamiento:
   ```bash
   uv run python scripts/train_pipeline.py --model-type random_forest
   ```

4. Lee el archivo `report.md` generado y muestra los resultados al usuario en formato legible.

5. Si el modelo Challenger es mejor que el Champion, pregunta al usuario si desea promoverlo. Si acepta, ejecuta:
   ```bash
   uv run python scripts/train_pipeline.py --model-type random_forest --promote
   ```

6. Muestra un resumen final con:
   - Metricas del modelo (RMSE, MAE, R2)
   - Comparacion con el Champion actual
   - Link al experimento en DagsHub: `https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow`
