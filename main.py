"""
main.py
-------
API REST que sirve el modelo entrenado (Sesión 3 del curso de MLOps).

Flujo del curso:
  - Sesión 1: train.py entrena y guarda 'modelo.pkl'.
  - Sesión 2: CI valida el F1 macro en cada push.
  - Sesión 3 (este archivo): el modelo se expone como API REST y se empaqueta
                             en una imagen Docker.
  - Sesión 4: la imagen se despliega en Azure Container Apps.

Endpoints:
  GET  /         -> health check. Confirma que la API está viva. NO predice.
  POST /predict  -> recibe las 4 features de Iris y devuelve la predicción.
  GET  /docs     -> interfaz Swagger (la genera FastAPI sola) para probar sin curl.

Correr localmente:
    uvicorn main:app --reload
    y abrir http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API de predicción - Iris MLOps")

# El modelo se carga UNA sola vez, al arrancar la API (no en cada petición).
# Si se regenera modelo.pkl, hay que reiniciar uvicorn para que lo recargue.
modelo = joblib.load("modelo.pkl")

# El pipeline recuerda con qué columnas fue entrenado: lo usamos como ÚNICA
# fuente de verdad. Así, si train.py cambia sus FEATURES, esta API se adapta
# sola y evitamos el desajuste entrenamiento/servicio (training-serving skew).
FEATURES_MODELO = list(modelo.feature_names_in_)


class IrisInput(BaseModel):
    """Contrato de entrada: el usuario envía SIEMPRE las 4 features de Iris.

    Internamente solo se usan las que el modelo entrenó (ver FEATURES_MODELO),
    pero el contrato de la API se mantiene estable aunque el modelo cambie.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    """Health check: útil para confirmar que el contenedor levantó."""
    return {
        "mensaje": "API de predicción Iris activa",
        #"features_que_usa_el_modelo": FEATURES_MODELO,
        "como_predecir": "POST /predict  ·  interfaz interactiva en /docs",
    }


@app.post("/predict")
def predict(datos: IrisInput):
    """Recibe las 4 features y devuelve la clase predicha."""
    # DataFrame con nombres de columna: el modelo se entrenó con un DataFrame,
    # así que pasarle nombres evita warnings de sklearn y errores de orden.
    entrada = pd.DataFrame([datos.model_dump()])

    # Seleccionamos SOLO las columnas con las que el modelo fue entrenado.
    X = entrada[FEATURES_MODELO]

    pred = modelo.predict(X)

    return {
        "prediccion": int(pred[0]),
        #"features_usadas": FEATURES_MODELO,
    }
