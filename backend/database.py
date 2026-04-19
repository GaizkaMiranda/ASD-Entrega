"""
database.py
Conexion a MongoDB para el backend de Mantenimiento Predictivo 4.0.
La URI se lee de la variable de entorno MONGO_URI; si no existe, usa localhost.
"""

import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mantenimiento_predictivo"

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[DB_NAME]
