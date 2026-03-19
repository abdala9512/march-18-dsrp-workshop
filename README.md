# MLOps Workshop: Champion/Challenger con MLflow, DagsHub y GitHub Actions

Workshop practico para implementar un pipeline de MLOps completo con patron Champion/Challenger, tracking de experimentos en DagsHub y reportes automaticos con CML.

**Caso de uso:** Prediccion de tarifas de Uber usando el [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset).

## Requisitos previos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado
- Cuenta en [DagsHub](https://dagshub.com)
- Cuenta en GitHub

## Configuracion inicial

### 1. Clonar e instalar dependencias

```bash
git clone <repo-url>
cd march-18-dsrp-workshop
uv sync
```

### 2. Configurar DagsHub

Crear un repositorio en DagsHub (puedes conectarlo directamente desde el navegador) y configurar las variables de entorno:

```bash
cp .env.example .env
# Editar .env con tus datos de DagsHub
```

Solo necesitas 3 variables:
- `DAGSHUB_REPO_OWNER` - Tu usuario de DagsHub
- `DAGSHUB_REPO_NAME` - Nombre del repo en DagsHub
- `DAGSHUB_USER_TOKEN` - Token de API (obtener en https://dagshub.com/user/settings/tokens)

`dagshub.init()` se encarga de configurar MLflow automaticamente con ese token.

### 3. Configurar secretos en GitHub

En tu repositorio de GitHub, agregar:

**Secrets:**
- `DAGSHUB_USER_TOKEN` - Token de API de DagsHub

**Variables:**
- `DAGSHUB_REPO_OWNER` - Dueno del repo en DagsHub
- `DAGSHUB_REPO_NAME` - Nombre del repo en DagsHub

## Estructura del proyecto

```
├── .claude/skills/
│   ├── train.md               # Skill /train para entrenar modelos
│   └── validate.md            # Skill /validate para validar y promover
├── .github/workflows/
│   ├── train_and_report.yml   # CI: entrena en PR y publica reporte CML
│   └── validate_and_promote.yml # CD: valida challenger y crea PR de promocion
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb  # EDA del dataset
│   └── 02_entrenamiento_uber.ipynb     # Pipeline de entrenamiento
├── src/
│   ├── config.py              # Configuracion y conexion MLflow
│   ├── data.py                # Carga, limpieza y feature engineering
│   ├── train.py               # Entrenamiento de modelos
│   ├── evaluate.py            # Evaluacion y graficos
│   └── champion_challenger.py # Logica Champion vs Challenger
├── scripts/
│   ├── train_pipeline.py      # Pipeline de entrenamiento (CLI)
│   ├── validate_challenger.py # Validar challenger vs champion
│   └── promote_challenger.py  # Promover modelo a champion
├── scenarios/                 # Escenarios de ejemplo
├── data/
│   └── uber.csv               # Dataset incluido en el repo
└── pyproject.toml             # Dependencias (uv)
```

## Uso

### Entrenar localmente

```bash
# Random Forest (default)
uv run python scripts/train_pipeline.py

# Gradient Boosting
uv run python scripts/train_pipeline.py --model-type gradient_boosting

# Con hiperparametros personalizados
uv run python scripts/train_pipeline.py --model-type random_forest --n-estimators 200 --max-depth 15

# Promover automaticamente si es mejor
uv run python scripts/train_pipeline.py --promote
```

### Con Claude Code

```
/train      # Entrenar un modelo interactivamente
/validate   # Validar un challenger y disparar workflow de promocion
```

### Flujo CI/CD

1. Crear una rama con cambios en `src/` o `scripts/`
2. Abrir un Pull Request hacia `main`
3. GitHub Actions entrena el modelo y publica un reporte CML en el PR
4. Revisar el reporte y decidir si hacer merge

### Validar y Promover

```bash
# Validar manualmente
uv run python scripts/validate_challenger.py --challenger-run-id <RUN_ID>

# Disparar workflow de validacion + PR automatico
gh workflow run validate_and_promote.yml -f challenger_run_id=<RUN_ID> -f auto_promote=true
```

## Escenarios de ejemplo

Ver `scenarios/README.md` para instrucciones paso a paso:

```bash
bash scenarios/01_champion_baseline.sh    # Establecer Champion base
bash scenarios/02_better_challenger.sh    # Challenger que supera al Champion
bash scenarios/03_worse_challenger.sh     # Challenger que pierde vs Champion
```

## Feature Engineering

El pipeline transforma los datos crudos en features utiles:

| Feature | Descripcion |
|---------|-------------|
| `passenger_count` | Numero de pasajeros |
| `hora` | Hora del dia (0-23) |
| `dia_semana` | Dia de la semana (0=Lunes, 6=Domingo) |
| `mes` | Mes del ano (1-12) |
| `es_fin_de_semana` | 1 si es sabado/domingo |
| `es_hora_pico` | 1 si es hora pico (7-9 AM, 5-7 PM) |
| `distancia_km` | Distancia Haversine entre pickup y dropoff |
