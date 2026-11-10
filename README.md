# ¿Cuántos pasajeros tendremos el próximo año?

App sencilla hecha con Streamlit para analizar y pronosticar los pasajeros
mensuales de avión entre 1949 y 1960, y estimar los 12 meses de 1961.

## Qué muestra la app

- **La historia:** la evolución de los pasajeros mes a mes.
- **La temporada:** el promedio de cada mes del año comparado con un mes normal.
- **El examen:** compara seis formas de pronosticar usando los últimos 24 meses
  como prueba: copiar el año pasado, modelo de temporada (Holt-Winters), SARIMA,
  suavizamiento exponencial, regresión lineal y Prophet.
- **El pronóstico:** permite elegir un modelo y ver los 12 meses de 1961.

## Archivos del proyecto

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La aplicación de Streamlit. |
| `datos.py` | Descarga los datos o usa la copia local. |
| `Datos/airline-passengers.csv` | Copia de respaldo si no hay internet. |
| `requirements.txt` | Paquetes que Streamlit Cloud debe instalar. |
| `prueba_datos.py` | Prueba rápida de la lectura de datos. |
| `iniciar_app.bat` | Abre la app en Windows. |
| `pyproject.toml` | Configuración local del proyecto con `uv`. |

La app intenta descargar los datos desde GitHub. Si no hay internet, busca la
copia local en `Datos/airline-passengers.csv` (el código también acepta la
carpeta en minúscula, `datos/`).

## Probarla en tu computadora

Con `uv`:

```powershell
uv run streamlit run app.py
```

O haciendo doble clic en `iniciar_app.bat` (Windows).

## Subirla a GitHub

1. Crea un repositorio vacío en GitHub (sin README ni licencia).
2. Abre una terminal en esta carpeta y ejecuta, cambiando `TU-USUARIO` y
   `TU-REPO`:

```powershell
git init
git add .
git commit -m "Primera versión de la app de pasajeros"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
git push -u origin main
```

El archivo `.gitignore` ya evita subir `.venv`, archivos temporales y datos que
la app no usa.

## Desplegarla en Streamlit Community Cloud

1. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión con
   tu cuenta de GitHub.
2. Pulsa **Create app** y elige **Yup, I have an app**.
3. Selecciona tu repositorio, la rama `main` y el archivo `app.py`.
4. En **Advanced settings**, deja **Python 3.12** (es la versión que usa el
   proyecto). No hacen falta secretos.
5. Pulsa **Deploy** y espera. La primera vez puede tardar varios minutos porque
   se instalan pandas, statsmodels y Prophet.

Cuando termine, tendrás un enlace parecido a:

```text
https://TU-USUARIO-TU-REPO.streamlit.app
```

## Notas importantes

- No subas la carpeta `.venv`: ocupa mucho y no se necesita en GitHub.
- `requirements.txt` es el archivo de dependencias que usará Streamlit Cloud.
- Prophet es el paquete más pesado. La primera carga entrena varios modelos y
  puede tardar unos segundos; después la app queda más rápida.
- Si la descarga de datos falla, la app usa la copia local incluida en el
  repositorio.
- Si el despliegue falla, revisa los **logs** que aparecen a la derecha de la
  app en Streamlit Cloud.
