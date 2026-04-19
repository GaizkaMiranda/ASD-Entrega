"""
main.py
Servicio FastAPI para el backend de Machine Learning del
Sistema de Mantenimiento Predictivo 4.0.

Endpoints:
    GET  /health                          -- verificacion de estado del servicio
    POST /predict                         -- inferencia puntual de probabilidad de fallo
    POST /equipos/{equipo_id}/predict     -- inferencia + registro en MongoDB
    GET  /equipos/{equipo_id}/historial   -- historial de predicciones de un equipo
    GET  /estadisticas                    -- resumen global por estado para el dashboard
"""

import os
from datetime import datetime, timezone
from typing import Dict, List

import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from train_model import MODEL_PATH, train_and_save_model
from database import get_db

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
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


class EntradaHistorial(BaseModel):
    timestamp: str = Field(..., description="Marca de tiempo ISO 8601 de la prediccion")
    temperatura: float
    vibracion: float
    probabilidad_fallo: float
    estado: str


class HistorialOutput(BaseModel):
    equipo_id: str
    total_registros: int
    registros: List[EntradaHistorial]


class EstadisticasOutput(BaseModel):
    total_predicciones: int
    equipos_monitorizados: int
    conteo_por_estado: Dict[str, int]
    porcentaje_criticos: float


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


def _inferir(telemetria: TelemetriaInput) -> PrediccionOutput:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    features = np.array([[telemetria.temperatura, telemetria.vibracion]])
    try:
        probabilidad = float(MODEL.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}")
    estado = clasificar_estado(probabilidad)
    return PrediccionOutput(probabilidad_fallo=round(probabilidad, 4), estado=estado)


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
    summary="Predice la probabilidad de fallo (sin registrar historial)",
)
def predict(telemetria: TelemetriaInput):
    return _inferir(telemetria)


@app.post(
    "/equipos/{equipo_id}/predict",
    response_model=PrediccionOutput,
    summary="Predice y persiste el resultado en MongoDB",
)
def predict_equipo(equipo_id: str, telemetria: TelemetriaInput):
    resultado = _inferir(telemetria)
    db = get_db()
    db.historial.insert_one({
        "equipo_id": equipo_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperatura": telemetria.temperatura,
        "vibracion": telemetria.vibracion,
        "probabilidad_fallo": resultado.probabilidad_fallo,
        "estado": resultado.estado,
    })
    return resultado


@app.get(
    "/equipos/{equipo_id}/historial",
    response_model=HistorialOutput,
    summary="Devuelve el historial de predicciones de un equipo",
)
def get_historial(equipo_id: str):
    db = get_db()
    registros = list(db.historial.find({"equipo_id": equipo_id}, {"_id": 0}))
    return HistorialOutput(
        equipo_id=equipo_id,
        total_registros=len(registros),
        registros=[EntradaHistorial(**r) for r in registros],
    )


@app.get(
    "/estadisticas",
    response_model=EstadisticasOutput,
    summary="Resumen global del sistema para el dashboard del frontend",
)
def get_estadisticas():
    db = get_db()
    docs = list(db.historial.find({}, {"_id": 0}))
    conteo: Dict[str, int] = {"Normal": 0, "Advertencia": 0, "Critico": 0}
    for doc in docs:
        conteo[doc["estado"]] = conteo.get(doc["estado"], 0) + 1
    total = len(docs)
    equipos = len(db.historial.distinct("equipo_id"))
    porcentaje = round((conteo["Critico"] / total * 100), 2) if total > 0 else 0.0
    return EstadisticasOutput(
        total_predicciones=total,
        equipos_monitorizados=equipos,
        conteo_por_estado=conteo,
        porcentaje_criticos=porcentaje,
    )
