"""
train.py
--------
Script de entrenamiento del curso de MLOps.

Está diseñado para funcionar SIN CAMBIOS en todo el flujo del curso:

  - Sesión 1: se ejecuta localmente (autenticación vía `az login`).
  - Sesión 2 (CI): GitHub Actions lo importa y valida el F1 con test_model.py.
                   La autenticación usa un Service Principal (variables de entorno /
                   GitHub Secrets) que DefaultAzureCredential detecta automáticamente.
  - Sesión 3/4: el artefacto 'modelo.pkl' que genera se empaqueta en la imagen Docker
                y lo consume la API (main.py) sin volver a entrenar.
  - Sesión 5: al reejecutarse, lee TODOS los datos de la tabla (original + lotes con
              drift), por lo que reentrena con la información más reciente.

Los datos se leen desde Azure Table Storage (no desde load_iris local), de modo que
el entrenamiento refleje siempre el estado real del dataset en la nube.

--------------------------------------------------------------------------------
¿DÓNDE EJECUTAR ESTE SCRIPT? GitHub Actions vs Azure Container Apps Jobs
--------------------------------------------------------------------------------
En este curso `train.py` corre en GitHub Actions. Como alternativa podría correr en
Azure Container Apps Jobs (un "job" = tarea que arranca, entrena y termina, a
diferencia de un "app" que corre siempre, como la API).

GitHub Actions
  + Ya integrado con el repositorio: un `git push` dispara el entrenamiento (CI).
  + Cero infraestructura extra que gestionar; capa gratuita generosa.
  + El modelo.pkl se usa dentro del mismo pipeline (se hornea en la imagen Docker),
    así que no hay que persistirlo aparte.
  - Cómputo limitado (runners estándar, sin GPU); malo para datasets/modelos grandes.
  - El entorno de ejecución vive en GitHub, no dentro de tu red/recursos de Azure.

Azure Container Apps Jobs
  + Corre en cómputo de Azure, junto a tus demás recursos (misma red, mismo entorno
    que la API de la Sesión 4, reutiliza la misma imagen del ACR).
  + Escala a más CPU/memoria y admite ejecución programada (cron) o por eventos.
  - Más complejidad: otra imagen Docker, manejo de secrets en el job.
  - El sistema de archivos es efímero: modelo.pkl se PIERDE al terminar, por lo que
    el script debería subirlo a Blob Storage (paso adicional que aquí no hace falta).

Regla práctica: GitHub Actions para este curso introductorio; Container Apps Jobs
cuando el entrenamiento necesite más cómputo del que da un runner de GitHub.
"""

import os
import joblib
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient
from azure.data.tables import TableServiceClient

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --- CONFIGURACIÓN ---
# Se leen de variables de entorno para no exponer IDs en el repositorio.
# Localmente puedes exportarlas; en CI se inyectan como GitHub Secrets.
SUBSCRIPTION_ID      = os.environ.get("AZURE_SUBSCRIPTION_ID", "<tu-subscription-id>")
RESOURCE_GROUP_NAME  = os.environ.get("AZURE_RESOURCE_GROUP", "mlops-intro-rg")
STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT", "mlopsintrostorage")
TABLE_NAME           = os.environ.get("AZURE_TABLE_NAME", "irisdata")

FEATURES = ["sepal_length", "petal_length"]
TARGET   = "target"
MODEL_PATH = "modelo.pkl"
# ---------------------


def conectar_tabla():
    """Conecta a la tabla existente en Azure Table Storage y devuelve su cliente."""
    credential = DefaultAzureCredential()
    storage_client = StorageManagementClient(credential, SUBSCRIPTION_ID)

    keys_result = storage_client.storage_accounts.list_keys(
        RESOURCE_GROUP_NAME, STORAGE_ACCOUNT_NAME
    )
    account_key = keys_result['keys'][0].value
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};"
        f"AccountKey={account_key};EndpointSuffix=core.windows.net"
    )

    table_service = TableServiceClient.from_connection_string(conn_str)
    return table_service.get_table_client(TABLE_NAME)


def cargar_datos():
    """Descarga TODAS las filas de la tabla (original + lotes con drift) como DataFrame."""
    table_client = conectar_tabla()
    entidades = list(table_client.list_entities())

    if not entidades:
        raise ValueError(
            f"La tabla '{TABLE_NAME}' está vacía. "
            "Ejecuta primero la celda que carga el dataset Iris."
        )

    df = pd.DataFrame(entidades)

    # Aseguramos los tipos correctos (Table Storage puede devolver valores como texto).
    df[FEATURES] = df[FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(int)

    return df


def entrenar(df):
    """Entrena un pipeline (escalado + modelo) y devuelve el pipeline y el F1 en test."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: el escalado viaja DENTRO del modelo guardado, de modo que la API
    # (main.py) obtiene predicciones consistentes sin escalar manualmente la entrada.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # 'macro' porque Iris es multiclase (3 especies); promedia el F1 por clase
    f1 = f1_score(y_test, y_pred, average="macro")

    return pipeline, f1


def main():
    """Orquesta carga, entrenamiento y guardado. Devuelve el F1 para el test de CI."""
    print("Cargando datos desde Azure Table Storage...")
    df = cargar_datos()
    print(f"{len(df)} registros cargados (original + lotes con drift, si existen).")

    pipeline, f1 = entrenar(df)
    print(f"F1-score (macro) en test: {f1:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado como '{MODEL_PATH}'.")

    return f1


if __name__ == "__main__":
    main()
