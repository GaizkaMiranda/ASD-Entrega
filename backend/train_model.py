"""
train_model.py
Script para generar datos sinteticos, entrenar un modelo Random Forest
y serializar el artefacto entrenado en disco.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_STATE = 42
MODEL_PATH = "model.joblib"
N_SAMPLES = 2000


def generate_synthetic_data(n_samples: int = N_SAMPLES):
    """
    Genera datos sinteticos de temperatura y vibracion con etiquetas de fallo.
    Clase 1 = fallo, Clase 0 = normal.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Muestras normales: temperatura baja, vibracion baja
    n_normal = int(n_samples * 0.75)
    temperatura_normal = rng.normal(loc=55.0, scale=8.0, size=n_normal)
    vibracion_normal = rng.normal(loc=5.0, scale=1.5, size=n_normal)
    labels_normal = np.zeros(n_normal)

    # Muestras de fallo: temperatura alta, vibracion alta
    n_fallo = n_samples - n_normal
    temperatura_fallo = rng.normal(loc=85.0, scale=10.0, size=n_fallo)
    vibracion_fallo = rng.normal(loc=14.0, scale=2.5, size=n_fallo)
    labels_fallo = np.ones(n_fallo)

    temperatura = np.concatenate([temperatura_normal, temperatura_fallo])
    vibracion = np.concatenate([vibracion_normal, vibracion_fallo])
    labels = np.concatenate([labels_normal, labels_fallo])

    X = np.column_stack([temperatura, vibracion])
    return X, labels


def train_and_save_model(model_path: str = MODEL_PATH):
    """Entrena el modelo y lo guarda en disco."""
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Reporte de clasificacion en conjunto de prueba:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fallo"]))

    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")
    return model


if __name__ == "__main__":
    train_and_save_model()
