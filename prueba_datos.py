"""Prueba rápida de datos.py: imprime un resumen de la serie."""

from datos import leer_pasajeros

df = leer_pasajeros()

meses = len(df)
desde = df["Month"].min()
hasta = df["Month"].max()

fila_min = df.loc[df["Passengers"].idxmin()]
fila_max = df.loc[df["Passengers"].idxmax()]

print()
print(f"Cantidad de meses: {meses}")
print(f"Desde: {desde:%Y-%m}")
print(f"Hasta: {hasta:%Y-%m}")
print(f"Mes con menos pasajeros: {fila_min['Month']:%Y-%m} ({fila_min['Passengers']:,})")
print(f"Mes con más pasajeros:   {fila_max['Month']:%Y-%m} ({fila_max['Passengers']:,})")
