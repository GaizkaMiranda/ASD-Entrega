"""
main.py
Servicio FastAPI para el backend de Machine Learning del
Sistema de Mantenimiento Predictivo 4.0.

Endpoints:
    GET  /health   -- verificacion de estado del servicio
    POST /predict  -- inferencia de probabilidad de fallo
"""

import os
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import MODEL_PATH, train_and_save_model

MODEL = None


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicacion
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar y libera recursos al detener el servidor."""
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado. Iniciando entrenamiento automatico...")
        train_and_save_model(MODEL_PATH)
    MODEL = joblib.load(MODEL_PATH)
    print("Modelo cargado correctamente.")
    yield
    MODEL = None


app = FastAPI(
    title="Predictive Maintenance 4.0 - ML Backend",
    description="API REST para inferencia de fallos en equipos industriales.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas de datos
# ---------------------------------------------------------------------------

class TelemetriaInput(BaseModel):
    temperatura: float = Field(..., description="Temperatura del equipo en grados Celsius", example=75.5)
    vibracion: float = Field(..., description="Nivel de vibracion del equipo en mm/s", example=12.2)


class PrediccionOutput(BaseModel):
    probabilidad_fallo: float = Field(..., description="Probabilidad de fallo entre 0.0 y 1.0")
    estado: str = Field(..., description="Estado del equipo: Normal, Advertencia o Critico")


# ---------------------------------------------------------------------------
# Logica de negocio
# ---------------------------------------------------------------------------

UMBRAL_ADVERTENCIA = 0.40
UMBRAL_CRITICO = 0.70


def clasificar_estado(probabilidad: float) -> str:
    if probabilidad >= UMBRAL_CRITICO:
        return "Critico"
    if probabilidad >= UMBRAL_ADVERTENCIA:
        return "Advertencia"
    return "Normal"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Verificacion de estado del servicio")
def health_check():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    return {"status": "ok", "modelo_cargado": True}


@app.post(
    "/predict",
    response_model=PrediccionOutput,
    summary="Predice la probabilidad de fallo a partir de telemetria",
)
def predict(telemetria: TelemetriaInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    features = np.array([[telemetria.temperatura, telemetria.vibracion]])

    try:
        probabilidad = float(MODEL.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}")

    estado = clasificar_estado(probabilidad)
    return PrediccionOutput(probabilidad_fallo=round(probabilidad, 4), estado=estado)
