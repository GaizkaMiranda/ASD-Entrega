"""
tests/test_main.py
Suite de pruebas unitarias y de componentes para el backend de ML.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fake MongoDB en memoria (sin dependencias externas)
# ---------------------------------------------------------------------------

class FakeCollection:
    """Simula una coleccion MongoDB usando una lista Python."""

    def __init__(self):
        self._docs = []

    def insert_one(self, doc):
        self._docs.append({k: v for k, v in doc.items()})

    def find(self, filter=None, projection=None):
        result = []
        for doc in self._docs:
            if filter is None or all(doc.get(k) == v for k, v in filter.items()):
                d = {k: v for k, v in doc.items()}
                if projection and projection.get("_id") == 0:
                    d.pop("_id", None)
                result.append(d)
        return result

    def count_documents(self, filter=None):
        if not filter:
            return len(self._docs)
        return sum(1 for d in self._docs if all(d.get(k) == v for k, v in filter.items()))

    def distinct(self, field):
        return list({d[field] for d in self._docs if field in d})


class FakeDB:
    def __init__(self):
        self.historial = FakeCollection()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Mock del modelo ML para evitar dependencia del artefacto en disco."""
    import main as app_module
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.15, 0.85]])
    monkeypatch.setattr(app_module, "MODEL", mock)
    monkeypatch.setattr("joblib.load", lambda path: mock)
    return mock


@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """Sustituye get_db por una base de datos en memoria para cada test."""
    db = FakeDB()
    monkeypatch.setattr("main.get_db", lambda: db)
    return db


@pytest.fixture()
def client(mock_model, mock_db):
    """Cliente de prueba sincrono de FastAPI."""
    from main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Pruebas del endpoint /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["modelo_cargado"] is True

    def test_health_returns_503_when_model_not_loaded(self, client, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        assert client.get("/health").status_code == 503


# ---------------------------------------------------------------------------
# Pruebas del endpoint POST /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_input(self, client):
        assert client.post("/predict", json={"temperatura": 85.0, "vibracion": 15.0}).status_code == 200

    def test_predict_response_contains_required_fields(self, client):
        data = client.post("/predict", json={"temperatura": 85.0, "vibracion": 15.0}).json()
        assert "probabilidad_fallo" in data
        assert "estado" in data

    def test_predict_probabilidad_critico(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        data = client.post("/predict", json={"temperatura": 90.0, "vibracion": 18.0}).json()
        assert data["estado"] == "Critico"
        assert data["probabilidad_fallo"] >= 0.70

    def test_predict_probabilidad_advertencia(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])
        data = client.post("/predict", json={"temperatura": 70.0, "vibracion": 9.0}).json()
        assert data["estado"] == "Advertencia"
        assert 0.40 <= data["probabilidad_fallo"] < 0.70

    def test_predict_probabilidad_normal(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        data = client.post("/predict", json={"temperatura": 50.0, "vibracion": 3.0}).json()
        assert data["estado"] == "Normal"
        assert data["probabilidad_fallo"] < 0.40

    def test_predict_probabilidad_entre_0_y_1(self, client):
        data = client.post("/predict", json={"temperatura": 75.0, "vibracion": 12.0}).json()
        assert 0.0 <= data["probabilidad_fallo"] <= 1.0

    def test_predict_falla_sin_campo_temperatura(self, client):
        assert client.post("/predict", json={"vibracion": 12.0}).status_code == 422

    def test_predict_falla_sin_campo_vibracion(self, client):
        assert client.post("/predict", json={"temperatura": 75.0}).status_code == 422

    def test_predict_falla_con_cuerpo_vacio(self, client):
        assert client.post("/predict", json={}).status_code == 422

    def test_predict_falla_con_tipo_incorrecto(self, client):
        assert client.post("/predict", json={"temperatura": "caliente", "vibracion": 12.0}).status_code == 422

    def test_predict_503_cuando_modelo_no_cargado(self, client, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        assert client.post("/predict", json={"temperatura": 75.0, "vibracion": 12.0}).status_code == 503


# ---------------------------------------------------------------------------
# Pruebas del endpoint POST /equipos/{equipo_id}/predict
# ---------------------------------------------------------------------------

class TestPredictEquipoEndpoint:
    def test_predict_equipo_returns_200(self, client):
        assert client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0}).status_code == 200

    def test_predict_equipo_response_contains_required_fields(self, client):
        data = client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0}).json()
        assert "probabilidad_fallo" in data
        assert "estado" in data

    def test_predict_equipo_registra_en_db(self, client, mock_db):
        client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        assert mock_db.historial.count_documents({"equipo_id": "EQ-01"}) == 1

    def test_predict_equipo_multiples_registros_acumulan(self, client, mock_db):
        for _ in range(3):
            client.post("/equipos/EQ-02/predict", json={"temperatura": 70.0, "vibracion": 10.0})
        assert mock_db.historial.count_documents({"equipo_id": "EQ-02"}) == 3

    def test_predict_equipo_registro_contiene_campos_correctos(self, client, mock_db):
        client.post("/equipos/EQ-03/predict", json={"temperatura": 65.0, "vibracion": 8.0})
        docs = mock_db.historial.find({"equipo_id": "EQ-03"})
        assert len(docs) == 1
        doc = docs[0]
        assert doc["temperatura"] == 65.0
        assert doc["vibracion"] == 8.0
        assert "probabilidad_fallo" in doc
        assert "estado" in doc
        assert "timestamp" in doc

    def test_predict_equipo_503_cuando_modelo_no_cargado(self, client, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        assert client.post("/equipos/EQ-01/predict", json={"temperatura": 75.0, "vibracion": 12.0}).status_code == 503

    def test_predict_equipo_ids_distintos_no_se_mezclan(self, client, mock_db):
        client.post("/equipos/EQ-A/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 60.0, "vibracion": 5.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 61.0, "vibracion": 5.5})
        assert mock_db.historial.count_documents({"equipo_id": "EQ-A"}) == 1
        assert mock_db.historial.count_documents({"equipo_id": "EQ-B"}) == 2


# ---------------------------------------------------------------------------
# Pruebas del endpoint GET /equipos/{equipo_id}/historial
# ---------------------------------------------------------------------------

class TestHistorialEndpoint:
    def test_historial_equipo_sin_registros_devuelve_lista_vacia(self, client):
        data = client.get("/equipos/NUEVO/historial").json()
        assert data["total_registros"] == 0
        assert data["registros"] == []

    def test_historial_equipo_devuelve_registros_correctos(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        client.post("/equipos/EQ-10/predict", json={"temperatura": 88.0, "vibracion": 16.0})
        data = client.get("/equipos/EQ-10/historial").json()
        assert data["equipo_id"] == "EQ-10"
        assert data["total_registros"] == 1
        assert data["registros"][0]["temperatura"] == 88.0
        assert data["registros"][0]["estado"] == "Critico"

    def test_historial_total_registros_coincide_con_longitud(self, client):
        for i in range(4):
            client.post("/equipos/EQ-11/predict", json={"temperatura": 60.0 + i, "vibracion": 5.0})
        data = client.get("/equipos/EQ-11/historial").json()
        assert data["total_registros"] == len(data["registros"]) == 4

    def test_historial_contiene_timestamp_valido(self, client):
        client.post("/equipos/EQ-12/predict", json={"temperatura": 70.0, "vibracion": 9.0})
        data = client.get("/equipos/EQ-12/historial").json()
        from datetime import datetime
        datetime.fromisoformat(data["registros"][0]["timestamp"])


# ---------------------------------------------------------------------------
# Pruebas del endpoint GET /estadisticas
# ---------------------------------------------------------------------------

class TestEstadisticasEndpoint:
    def test_estadisticas_sin_datos_devuelve_ceros(self, client):
        data = client.get("/estadisticas").json()
        assert data["total_predicciones"] == 0
        assert data["equipos_monitorizados"] == 0
        assert data["porcentaje_criticos"] == 0.0

    def test_estadisticas_conteo_por_estado_correcto(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.10, 0.90]])
        client.post("/equipos/EQ-20/predict", json={"temperatura": 90.0, "vibracion": 18.0})
        client.post("/equipos/EQ-21/predict", json={"temperatura": 91.0, "vibracion": 19.0})
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        client.post("/equipos/EQ-22/predict", json={"temperatura": 40.0, "vibracion": 2.0})
        data = client.get("/estadisticas").json()
        assert data["conteo_por_estado"]["Critico"] == 2
        assert data["conteo_por_estado"]["Normal"] == 1
        assert data["total_predicciones"] == 3

    def test_estadisticas_equipos_monitorizados_correctos(self, client):
        client.post("/equipos/EQ-30/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        client.post("/equipos/EQ-31/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        data = client.get("/estadisticas").json()
        assert data["equipos_monitorizados"] == 2

    def test_estadisticas_porcentaje_criticos_correcto(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.10, 0.90]])
        client.post("/equipos/EQ-40/predict", json={"temperatura": 90.0, "vibracion": 18.0})
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        for i in range(3):
            client.post(f"/equipos/EQ-4{i+1}/predict", json={"temperatura": 40.0, "vibracion": 2.0})
        data = client.get("/estadisticas").json()
        assert data["porcentaje_criticos"] == 25.0
