# Escenarios de Entrenamiento

Estos scripts demuestran el patron Champion/Challenger en diferentes situaciones.

## Orden de ejecucion

Ejecuta los escenarios en orden para ver el flujo completo:

### Paso 1: Establecer el Champion base

```bash
bash scenarios/01_champion_baseline.sh
```

Entrena un Random Forest con parametros estandar y lo promueve como Champion.
Este sera el modelo contra el que se comparen todos los Challengers.

### Paso 2: Entrenar un Challenger MEJOR

```bash
bash scenarios/02_better_challenger.sh
```

Entrena un Gradient Boosting optimizado que deberia superar al Champion.
El reporte mostrara una mejora en RMSE y R2.

Despues puedes validar y promover via GitHub Actions:
```bash
# Obtener el run_id del output o de DagsHub
gh workflow run validate_and_promote.yml \
  -f challenger_run_id="<RUN_ID_DEL_PASO_2>" \
  -f auto_promote=true
```

### Paso 3: Entrenar un Challenger PEOR

```bash
bash scenarios/03_worse_challenger.sh
```

Entrena un modelo deliberadamente debil. El reporte confirmara que el
Champion actual sigue siendo mejor. No se creara PR de promocion.

## Que observar en cada escenario

| Escenario | RMSE | R2 | Resultado esperado |
|-----------|------|-----|-------------------|
| 01 Champion base (RF) | ~5.0-6.0 | ~0.60-0.70 | Se promueve como primer Champion |
| 02 Mejor Challenger (GB) | ~4.0-5.0 | ~0.72-0.80 | Challenger es mejor → PR de promocion |
| 03 Peor Challenger (RF debil) | ~7.0-9.0 | ~0.20-0.40 | Champion sigue siendo mejor → sin PR |
