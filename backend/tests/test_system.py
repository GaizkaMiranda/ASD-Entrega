"""
tests/test_system.py
Pruebas de sistema: flujos end-to-end del sistema integrado de mantenimiento predictivo.
Verifican que los subsistemas backend ML y frontend se integran correctamente.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fake MongoDB en memoria (igual que en test_main.py, sin dependencias externas)
# ---------------------------------------------------------------------------

class FakeCollection:
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

@pytest.fixture()
def mock_model():
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.15, 0.85]])
    return mock


@pytest.fixture()
def system_client(mock_model, monkeypatch):
    """Cliente con todos los componentes integrados (solo el modelo ML es mock)."""
    import main as app_module
    db = FakeDB()
    monkeypatch.setattr(app_module, "MODEL", mock_model)
    monkeypatch.setattr("main.get_db", lambda: db)
    monkeypatch.setattr("joblib.load", lambda path: mock_model)
    from main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, db, mock_model


# ---------------------------------------------------------------------------
# SYS-01: Ciclo completo de monitorización de un equipo
# ---------------------------------------------------------------------------

class TestCicloCompletoEquipo:
    def test_ciclo_prediccion_historial_estadisticas(self, system_client):
        """El sistema registra una predicción y la refleja en historial y estadísticas."""
        client, db, model = system_client
        model.predict_proba.return_value = np.array([[0.10, 0.90]])

        r = client.post("/equipos/EQ-SYS-01/predict", json={"temperatura": 92.0, "vibracion": 19.0})
        assert r.status_code == 200
        assert r.json()["estado"] == "Critico"

        historial = client.get("/equipos/EQ-SYS-01/historial").json()
        assert historial["total_registros"] == 1
        assert historial["registros"][0]["estado"] == "Critico"

        stats = client.get("/estadisticas").json()
        assert stats["total_predicciones"] == 1
        assert stats["equipos_monitorizados"] == 1
        assert stats["conteo_por_estado"]["Critico"] == 1

    def test_multiples_predicciones_mismo_equipo_acumulan_en_historial(self, system_client):
        client, db, model = system_client
        for i in range(5):
            client.post("/equipos/EQ-SYS-02/predict", json={"temperatura": 60.0 + i, "vibracion": 5.0})
        historial = client.get("/equipos/EQ-SYS-02/historial").json()
        assert historial["total_registros"] == 5
        assert len(historial["registros"]) == 5

    def test_equipos_independientes_no_comparten_historial(self, system_client):
        client, db, model = system_client
        client.post("/equipos/EQ-A/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        client.post("/equipos/EQ-B/predict", json={"temperatura": 81.0, "vibracion": 14.0})

        assert client.get("/equipos/EQ-A/historial").json()["total_registros"] == 1
        assert client.get("/equipos/EQ-B/historial").json()["total_registros"] == 2


# ---------------------------------------------------------------------------
# SYS-02: Consistencia de datos entre endpoints
# ---------------------------------------------------------------------------

class TestConsistenciaDatos:
    def test_estadisticas_reflejan_todos_los_equipos(self, system_client):
        client, db, model = system_client
        equipos = ["EQ-10", "EQ-11", "EQ-12"]
        for eq in equipos:
            client.post(f"/equipos/{eq}/predict", json={"temperatura": 70.0, "vibracion": 10.0})
        stats = client.get("/estadisticas").json()
        assert stats["equipos_monitorizados"] == 3
        assert stats["total_predicciones"] == 3

    def test_porcentaje_criticos_consistente_con_conteo(self, system_client):
        client, db, model = system_client
        model.predict_proba.return_value = np.array([[0.10, 0.90]])
        client.post("/equipos/EQ-C1/predict", json={"temperatura": 90.0, "vibracion": 18.0})
        client.post("/equipos/EQ-C2/predict", json={"temperatura": 90.0, "vibracion": 18.0})
        model.predict_proba.return_value = np.array([[0.95, 0.05]])
        client.post("/equipos/EQ-N1/predict", json={"temperatura": 40.0, "vibracion": 2.0})
        client.post("/equipos/EQ-N2/predict", json={"temperatura": 40.0, "vibracion": 2.0})

        stats = client.get("/estadisticas").json()
        criticos = stats["conteo_por_estado"]["Critico"]
        total = stats["total_predicciones"]
        esperado = round(criticos / total * 100, 2)
        assert stats["porcentaje_criticos"] == esperado

    def test_campos_historial_coinciden_con_datos_enviados(self, system_client):
        client, db, model = system_client
        model.predict_proba.return_value = np.array([[0.45, 0.55]])
        client.post("/equipos/EQ-D1/predict", json={"temperatura": 73.5, "vibracion": 11.2})
        registro = client.get("/equipos/EQ-D1/historial").json()["registros"][0]
        assert registro["temperatura"] == 73.5
        assert registro["vibracion"] == 11.2
        assert registro["estado"] == "Advertencia"
        assert 0.0 <= registro["probabilidad_fallo"] <= 1.0


# ---------------------------------------------------------------------------
# SYS-03: Integración frontend ↔ backend (contrato de API)
# ---------------------------------------------------------------------------

class TestIntegracionFrontendBackend:
    def test_cors_headers_presentes_para_frontend(self, system_client):
        """El backend permite peticiones desde cualquier origen (requerido por el frontend)."""
        client, db, model = system_client
        r = client.options(
            "/estadisticas",
            headers={"Origin": "http://localhost", "Access-Control-Request-Method": "GET"},
        )
        assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost")

    def test_endpoint_estadisticas_devuelve_estructura_esperada_por_dashboard(self, system_client):
        """La respuesta de /estadisticas tiene todos los campos que el frontend necesita."""
        client, db, model = system_client
        data = client.get("/estadisticas").json()
        assert "total_predicciones" in data
        assert "equipos_monitorizados" in data
        assert "conteo_por_estado" in data
        assert "porcentaje_criticos" in data
        assert isinstance(data["conteo_por_estado"], dict)

    def test_endpoint_historial_devuelve_estructura_esperada_por_frontend(self, system_client):
        """La respuesta de /historial tiene los campos que el frontend renderiza."""
        client, db, model = system_client
        client.post("/equipos/EQ-F1/predict", json={"temperatura": 75.0, "vibracion": 12.0})
        data = client.get("/equipos/EQ-F1/historial").json()
        assert "equipo_id" in data
        assert "total_registros" in data
        assert "registros" in data
        registro = data["registros"][0]
        for campo in ("timestamp", "temperatura", "vibracion", "probabilidad_fallo", "estado"):
            assert campo in registro, f"Campo '{campo}' requerido por el frontend no encontrado"

    def test_predict_puntual_no_persiste_en_historial(self, system_client):
        """/predict no guarda en DB; solo /equipos/{id}/predict lo hace."""
        client, db, model = system_client
        client.post("/predict", json={"temperatura": 80.0, "vibracion": 14.0})
        stats = client.get("/estadisticas").json()
        assert stats["total_predicciones"] == 0


# ---------------------------------------------------------------------------
# SYS-04: Robustez del sistema ante entradas inválidas
# ---------------------------------------------------------------------------

class TestRobustezSistema:
    def test_equipo_inexistente_devuelve_historial_vacio(self, system_client):
        client, db, model = system_client
        data = client.get("/equipos/EQUIPO-INEXISTENTE/historial").json()
        assert data["total_registros"] == 0
        assert data["registros"] == []

    def test_predict_rechaza_temperatura_no_numerica(self, system_client):
        client, db, model = system_client
        assert client.post("/predict", json={"temperatura": "alta", "vibracion": 10.0}).status_code == 422

    def test_predict_rechaza_payload_incompleto(self, system_client):
        client, db, model = system_client
        assert client.post("/equipos/EQ-ERR/predict", json={"temperatura": 80.0}).status_code == 422

    def test_sistema_responde_health_ok_cuando_operativo(self, system_client):
        client, db, model = system_client
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
