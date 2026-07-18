"""
monitor.py — Monitor automático de F1 para iris-api-2.

Lee las últimas `ventana_horas` filas etiquetadas desde Azure Table Storage
(tabla `irisdata`), llama al endpoint /predict de iris-api-2 en Container Apps,
calcula el F1 weighted (Iris tiene 3 clases) y publica dos métricas
personalizadas a Application Insights vía OpenTelemetry:

    iris_f1_score   — el valor del F1
    iris_breach     — 1 si F1 < umbral, 0 si está sano

Azure Monitor observa `customMetrics/iris_breach` y notifica al Action Group
cuando dispara.

Variables de entorno requeridas:
    AZURE_STORAGE_CONNECTION_STRING
    IRIS_API_URL                       (p.ej. https://iris-api-2.<...>.azurecontainerapps.io)
    APPLICATIONINSIGHTS_CONNECTION_STRING
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests
from sklearn.metrics import f1_score
from azure.data.tables import TableServiceClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import metrics


UMBRAL_POR_DEFECTO = 0.700
MIN_MUESTRAS = 30
CARACTERISTICAS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def leer_lote(cadena_conexion: str, tabla: str, ventana_horas: int) -> pd.DataFrame:
    """Lee de Table Storage las filas cuyo Timestamp cae en la ventana.

    Nota: usamos el `Timestamp` que Table Storage escribe automáticamente en
    cada entidad. Si tu esquema tiene otra columna de tiempo, ajústalo aquí.
    """
    tsc = TableServiceClient.from_connection_string(cadena_conexion)
    tabla_cliente = tsc.get_table_client(tabla)
    corte = datetime.now(timezone.utc) - timedelta(hours=ventana_horas)
    filtro = f"Timestamp ge datetime'{corte.isoformat()}'"
    filas = list(tabla_cliente.query_entities(filtro))
    return pd.DataFrame(filas)


def predecir_lote(url_api: str, X: pd.DataFrame) -> list:
    """Llama al endpoint /predict de iris-api-2, una fila a la vez."""
    predicciones = []
    for _, fila in X.iterrows():
        payload = fila.to_dict()
        r = requests.post(f"{url_api}/predict", json=payload, timeout=30)
        r.raise_for_status()
        pred = r.json()
        predicciones.append(pred["prediccion"])
    return predicciones


def enviar_a_app_insights(f1: float, breach: int, n_muestras: int) -> None:
    """Emite las 3 métricas personalizadas a Application Insights.

    Usa `create_histogram` en lugar de `create_gauge` porque histograma
    existe en todas las versiones estables del SDK. Para una alerta con
    "max en 15 minutos", el comportamiento es equivalente.
    """
    configure_azure_monitor(
        connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    m = metrics.get_meter("iris.monitor")

    hist_f1 = m.create_histogram("iris_f1_score", description="F1 macro por corrida")
    hist_breach = m.create_histogram("iris_breach", description="1 si F1 < umbral")
    hist_n = m.create_histogram("iris_n_muestras", description="tamaño del lote evaluado")

    hist_f1.record(f1)
    hist_breach.record(breach)
    hist_n.record(n_muestras)


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor de F1 para iris-api-2.")
    ap.add_argument("--ventana-horas", type=int, default=1,
                    help="Ventana en horas para el cálculo de F1.")
    ap.add_argument("--umbral", type=float, default=UMBRAL_POR_DEFECTO,
                    help="Umbral bajo el cual se marca breach = 1.")
    args = ap.parse_args()

    # 1. RECOLECTAR
    df = leer_lote(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"],
        "irisdata",
        args.ventana_horas,
    )
    if len(df) < MIN_MUESTRAS:
        print(
            f"lote con muy pocas filas ({len(df)} < {MIN_MUESTRAS}); "
            f"no se calcula F1 en esta corrida."
        )
        return 0

    # 2. CALCULAR — llama a iris-api-2 y computa el F1 Macro
    y_pred = predecir_lote(os.environ["IRIS_API_URL"], df[CARACTERISTICAS])
    f1 = float(f1_score(df["target"], y_pred, average="macro"))
    breach = int(f1 < args.umbral)

    print(
        f"F1 Macro en las últimas {args.ventana_horas}h: {f1:.4f} "
        f"(n={len(df)})  breach={breach}"
    )

    # 3 y 4. COMPARAR + ACTUAR — Azure Monitor evalúa la regla sobre esta métrica.
    enviar_a_app_insights(f1=f1, breach=breach, n_muestras=len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
