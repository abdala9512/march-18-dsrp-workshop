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

### 2. Descargar el dataset

Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) y coloca el archivo `uber.csv` en la carpeta `data/`:

```
data/uber.csv
```

### 3. Configurar DagsHub

Crear un repositorio en DagsHub y configurar las variables de entorno:

```bash
cp .env.example .env
# Editar .env con tus datos de DagsHub
```

### 4. Configurar secretos en GitHub

En tu repositorio de GitHub, agregar:

**Secrets:**
- `DAGSHUB_TOKEN` - Token de API de DagsHub
- `DAGSHUB_USERNAME` - Tu usuario de DagsHub

**Variables:**
- `DAGSHUB_REPO_OWNER` - Dueno del repo en DagsHub
- `DAGSHUB_REPO_NAME` - Nombre del repo en DagsHub

## Estructura del proyecto

```
├── .claude/skills/train.md    # Skill de Claude Code para entrenar
├── .github/workflows/         # GitHub Actions (CI/CD)
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
│   └── train_pipeline.py      # Pipeline principal (CLI)
├── data/
│   └── uber.csv               # Dataset (descargar de Kaggle)
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
/train
```

### Flujo CI/CD

1. Crear una rama con cambios en `src/` o `scripts/`
2. Abrir un Pull Request hacia `main`
3. GitHub Actions entrena el modelo y publica un reporte CML en el PR
4. Revisar el reporte y decidir si hacer merge

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
