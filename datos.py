"""Lectura de los pasajeros mensuales de avión.

Si hay internet se descarga el archivo original; si no, se usa la copia
guardada en la carpeta datos/.
"""

import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

# La copia local puede estar en "datos" o en "Datos" (Windows no distingue
# mayúsculas, pero así también funciona en otros sistemas).
CARPETA = Path(__file__).parent
RUTAS_LOCALES = [
    CARPETA / "datos" / "airline-passengers.csv",
    CARPETA / "Datos" / "airline-passengers.csv",
]


def descargar_tabla(url):
    """Descarga el CSV y lo convierte en una tabla de pandas."""
    with urllib.request.urlopen(url, timeout=10) as respuesta:
        texto = respuesta.read().decode("utf-8")
    return pd.read_csv(StringIO(texto))


def leer_pasajeros():
    """Devuelve una tabla con dos columnas: Month (fecha) y Passengers (número)."""

    # 1) Intentamos descargar los datos de internet.
    try:
        tabla = descargar_tabla(URL)
        print("Datos descargados de internet.")
    except Exception:
        # 2) Si falla (por ejemplo, no hay internet), usamos el archivo local.
        ruta = next((r for r in RUTAS_LOCALES if r.exists()), None)
        if ruta is None:
            raise FileNotFoundError(
                "No hay internet y tampoco encontré datos/airline-passengers.csv"
            )
        tabla = pd.read_csv(ruta)
        print(f"Datos leídos del archivo local: {ruta}")

    # Nos aseguramos de que las columnas esperadas existan.
    if not {"Month", "Passengers"}.issubset(tabla.columns):
        raise ValueError("El archivo debe tener las columnas 'Month' y 'Passengers'.")

    # Month pasa a ser una fecha real y Passengers un número entero.
    tabla["Month"] = pd.to_datetime(tabla["Month"])
    tabla["Passengers"] = tabla["Passengers"].astype(int)

    return tabla


if __name__ == "__main__":
    df = leer_pasajeros()
    print(df.head())
