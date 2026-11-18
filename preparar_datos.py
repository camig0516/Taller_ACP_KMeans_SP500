"""Descarga de nuevo los dos archivos crudos que usa el análisis.

Uso:  python preparar_datos.py

El repositorio ya incluye la copia descargada el 17 de noviembre de 2026 en data/,
que es la que produce los resultados del informe. Ejecutar este script reemplaza
esa copia por la versión más reciente (las cifras y los grupos pueden cambiar).
Después de ejecutarlo, actualice FECHA_DESCARGA en analisis.py.
"""

from pathlib import Path
from urllib.request import urlretrieve

FUENTES = {
    "constituents-financials.csv":
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/main/data/constituents-financials.csv",
    "constituents.csv":
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
}

if __name__ == "__main__":
    carpeta = Path(__file__).resolve().parent / "data"
    carpeta.mkdir(exist_ok=True)
    for archivo, url in FUENTES.items():
        urlretrieve(url, carpeta / archivo)
        print("Descargado:", archivo)
