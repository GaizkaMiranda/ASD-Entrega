"""
start.py — Arranca el backend y abre el frontend en el navegador.
Uso: python start.py
"""
import subprocess
import threading
import webbrowser
import time
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
FRONTEND = os.path.join(ROOT, "frontend", "index.html")

# Detecta el ejecutable de Python del venv del backend
if sys.platform == "win32":
    PYTHON = os.path.join(BACKEND, "venv", "Scripts", "python.exe")
else:
    PYTHON = os.path.join(BACKEND, "venv", "bin", "python")

if not os.path.exists(PYTHON):
    PYTHON = sys.executable  # fallback al Python actual


def open_browser():
    """Espera a que el servidor arranque y abre el navegador."""
    time.sleep(2)
    webbrowser.open(f"file:///{FRONTEND.replace(os.sep, '/')}")
    print(f"\n  Navegador abierto → {FRONTEND}\n")


threading.Thread(target=open_browser, daemon=True).start()

print("Arrancando backend en http://localhost:8000 ...")
print("Pulsa CTRL+C para detener.\n")

subprocess.run(
    [PYTHON, "-m", "uvicorn", "main:app", "--reload"],
    cwd=BACKEND,
)
