"""
tests/test_main.py
Suite de pruebas unitarias para el backend de ML usando pytest y httpx.
"""

import os
import sys
import pytest
import joblib
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Asegura que el directorio raiz del backend este en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixture: modelo simulado para evitar dependencia de artefacto en disco
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """
    Reemplaza la variable global MODEL con un mock que devuelve
    probabilidades predefinidas para cada caso de prueba.
    """
    import main as app_module

    mock = MagicMock()
    # predict_proba devuelve [[prob_clase_0, prob_clase_1]]
    mock.predict_proba.return_value = np.array([[0.15, 0.85]])
    monkeypatch.setattr(app_module, "MODEL", mock)
    return mock


@pytest.fixture()
def client(mock_model):
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

    def test_health_returns_503_when_model_not_loaded(self, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get("/health")
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

    def test_predict_503_cuando_modelo_no_cargado(self, monkeypatch):
        import main as app_module
        monkeypatch.setattr(app_module, "MODEL", None)
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post("/predict", json={"temperatura": 75.0, "vibracion": 12.0})
        assert response.status_code == 503
