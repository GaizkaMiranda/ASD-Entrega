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
# Fixture: modelo simulado para evitar dependencia de artefacto en disco
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """
    Reemplaza la variable global MODEL con un mock que devuelve
    probabilidades predefinidas para cada caso de prueba.
    Tambien parchea joblib.load para que el lifespan no cargue el modelo real.
    """
    import main as app_module

    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.15, 0.85]])
    monkeypatch.setattr(app_module, "MODEL", mock)
    monkeypatch.setattr("joblib.load", lambda path: mock)
    return mock


@pytest.fixture()
def client(mock_model):
    """Cliente de prueba sincrono de FastAPI con historial limpio en cada test."""
    import main as app_module
    app_module.HISTORIAL.clear()
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
        response = client.get("/health")
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Pruebas del endpoint POST /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_input(self, client):
        payload = {"temperatura": 85.0, "vibracion": 15.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_contains_required_fields(self, client):
        payload = {"temperatura": 85.0, "vibracion": 15.0}
        data = client.post("/predict", json=payload).json()
        assert "probabilidad_fallo" in data
        assert "estado" in data

    def test_predict_probabilidad_critico(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        payload = {"temperatura": 90.0, "vibracion": 18.0}
        data = client.post("/predict", json=payload).json()
        assert data["estado"] == "Critico"
        assert data["probabilidad_fallo"] >= 0.70

    def test_predict_probabilidad_advertencia(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])
        payload = {"temperatura": 70.0, "vibracion": 9.0}
        data = client.post("/predict", json=payload).json()
        assert data["estado"] == "Advertencia"
        assert 0.40 <= data["probabilidad_fallo"] < 0.70

    def test_predict_probabilidad_normal(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        payload = {"temperatura": 50.0, "vibracion": 3.0}
        data = client.post("/predict", json=payload).json()
        assert data["estado"] == "Normal"
        assert data["probabilidad_fallo"] < 0.40

    def test_predict_probabilidad_entre_0_y_1(self, client):
        payload = {"temperatura": 75.0, "vibracion": 12.0}
        data = client.post("/predict", json=payload).json()
        assert 0.0 <= data["probabilidad_fallo"] <= 1.0

    def test_predict_falla_sin_campo_temperatura(self, client):
        response = client.post("/predict", json={"vibracion": 12.0})
        assert response.status_code == 422

    def test_predict_falla_sin_campo_vibracion(self, client):
        response = client.post("/predict", json={"temperatura": 75.0})
        assert response.status_code == 422

    def test_predict_falla_con_cuerpo_vacio(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_falla_con_tipo_incorrecto(self, client):
        response = client.post("/predict", json={"temperatura": "caliente", "vibracion": 12.0})
        assert response.status_code == 422

    def test_predict_503_cuando_modelo_no_cargado(self, client, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        response = client.post("/predict", json={"temperatura": 75.0, "vibracion": 12.0})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Pruebas del endpoint POST /equipos/{equipo_id}/predict
# ---------------------------------------------------------------------------

class TestPredictEquipoEndpoint:
    def test_predict_equipo_returns_200(self, client):
        response = client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        assert response.status_code == 200

    def test_predict_equipo_response_contains_required_fields(self, client):
        data = client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0}).json()
        assert "probabilidad_fallo" in data
        assert "estado" in data

    def test_predict_equipo_registra_en_historial(self, client):
        client.post("/equipos/EQ-01/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        import main as app_module
        assert "EQ-01" in app_module.HISTORIAL
        assert len(app_module.HISTORIAL["EQ-01"]) == 1

    def test_predict_equipo_multiples_registros_acumulan(self, client):
        for _ in range(3):
            client.post("/equipos/EQ-02/predict", json={"temperatura": 70.0, "vibracion": 10.0})
        import main as app_module
        assert len(app_module.HISTORIAL["EQ-02"]) == 3

    def test_predict_equipo_registro_contiene_campos_correctos(self, client):
        client.post("/equipos/EQ-03/predict", json={"temperatura": 65.0, "vibracion": 8.0})
        import main as app_module
        entrada = app_module.HISTORIAL["EQ-03"][0]
        assert "timestamp" in entrada
        assert entrada["temperatura"] == 65.0
        assert entrada["vibracion"] == 8.0
        assert "probabilidad_fallo" in entrada
        assert "estado" in entrada

    def test_predict_equipo_503_cuando_modelo_no_cargado(self, client, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        response = client.post("/equipos/EQ-01/predict", json={"temperatura": 75.0, "vibracion": 12.0})
        assert response.status_code == 503

    def test_predict_equipo_ids_distintos_no_se_mezclan(self, client):
        client.post("/equipos/EQ-A/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 60.0, "vibracion": 5.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 61.0, "vibracion": 5.5})
        import main as app_module
        assert len(app_module.HISTORIAL["EQ-A"]) == 1
        assert len(app_module.HISTORIAL["EQ-B"]) == 2


# ---------------------------------------------------------------------------
# Pruebas del endpoint GET /equipos/{equipo_id}/historial
# ---------------------------------------------------------------------------

class TestHistorialEndpoint:
    def test_historial_equipo_sin_registros_devuelve_lista_vacia(self, client):
        response = client.get("/equipos/NUEVO/historial")
        assert response.status_code == 200
        data = response.json()
        assert data["total_registros"] == 0
        assert data["registros"] == []

    def test_historial_equipo_devuelve_registros_correctos(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        client.post("/equipos/EQ-10/predict", json={"temperatura": 88.0, "vibracion": 16.0})
        response = client.get("/equipos/EQ-10/historial")
        assert response.status_code == 200
        data = response.json()
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
        ts = data["registros"][0]["timestamp"]
        # No debe lanzar excepcion si el formato es ISO valido
        datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# Pruebas del endpoint GET /estadisticas
# ---------------------------------------------------------------------------

class TestEstadisticasEndpoint:
    def test_estadisticas_sin_datos_devuelve_ceros(self, client):
        response = client.get("/estadisticas")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predicciones"] == 0
        assert data["equipos_monitorizados"] == 0
        assert data["porcentaje_criticos"] == 0.0

    def test_estadisticas_conteo_por_estado_correcto(self, client, mock_model):
        # 2 criticos
        mock_model.predict_proba.return_value = np.array([[0.10, 0.90]])
        client.post("/equipos/EQ-20/predict", json={"temperatura": 90.0, "vibracion": 18.0})
        client.post("/equipos/EQ-21/predict", json={"temperatura": 91.0, "vibracion": 19.0})
        # 1 normal
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
        client.post("/equipos/EQ-41/predict", json={"temperatura": 40.0, "vibracion": 2.0})
        client.post("/equipos/EQ-42/predict", json={"temperatura": 41.0, "vibracion": 2.0})
        client.post("/equipos/EQ-43/predict", json={"temperatura": 42.0, "vibracion": 2.0})
        data = client.get("/estadisticas").json()
        assert data["porcentaje_criticos"] == 25.0
