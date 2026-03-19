---
name: validate
description: Valida un modelo Challenger contra el Champion actual en MLflow y dispara el workflow de GitHub Actions para generar reporte CML y crear PR de promocion si es mejor.
---

# Skill: Validar Challenger

Valida un modelo Challenger contra el Champion actual y, si es mejor, crea un PR de promocion via GitHub Actions.

## Instrucciones

1. Pregunta al usuario por el **Run ID de MLflow** del modelo Challenger que quiere validar. Si no lo tiene, sugerile:
   - Revisar el dashboard de DagsHub
   - Ejecutar `/train` para entrenar un nuevo modelo
   - Listar runs recientes con:
     ```bash
     uv run python -c "
     from src.config import setup_mlflow
     import mlflow
     setup_mlflow()
     runs = mlflow.search_runs(max_results=5, order_by=['start_time DESC'])
     print(runs[['run_id', 'metrics.rmse', 'metrics.r2', 'params.model_type', 'start_time']].to_string())
     "
     ```

2. Verifica que el repositorio tenga un remote de GitHub configurado:
   ```bash
   git remote -v
   ```

3. Dispara el workflow de validacion y promocion:
   ```bash
   gh workflow run validate_and_promote.yml \
     -f challenger_run_id="<RUN_ID>" \
     -f auto_promote=true
   ```

4. Monitorea el workflow:
   ```bash
   gh run list --workflow=validate_and_promote.yml --limit 1
   ```
   Espera a que termine:
   ```bash
   gh run watch $(gh run list --workflow=validate_and_promote.yml --limit 1 --json databaseId -q '.[0].databaseId')
   ```

5. Muestra los logs del resultado:
   ```bash
   gh run view $(gh run list --workflow=validate_and_promote.yml --limit 1 --json databaseId -q '.[0].databaseId') --log
   ```

6. Si se creo un PR de promocion, muestra el link:
   ```bash
   gh pr list --search "promote/challenger" --limit 1
   ```

7. Pregunta al usuario si quiere hacer merge del PR.
