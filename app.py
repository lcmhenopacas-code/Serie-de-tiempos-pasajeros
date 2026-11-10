"""App sencilla para pronosticar los pasajeros mensuales de avión."""

import importlib.resources
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datos import leer_pasajeros

# Prophet incluye CmdStan, pero en Windows hay que decirle dónde está su
# librería tbb.dll. Sin esto, Prophet falla al empezar.
for _cmdstan in (
    Path(importlib.resources.files("prophet")) / "stan_model"
).glob("cmdstan-*"):
    _tbb = _cmdstan / "stan" / "lib" / "stan_math" / "lib" / "tbb"
    if _tbb.exists():
        os.environ["PATH"] = str(_tbb) + os.pathsep + os.environ["PATH"]
        break

from prophet import Prophet
from cmdstanpy.utils import get_logger as _get_cmdstanpy_logger

_get_cmdstanpy_logger()
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

st.set_page_config(
    page_title="¿Cuántos pasajeros tendremos el próximo año?",
    page_icon="✈️",
    layout="wide",
)

MESES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def formato_numero(valor):
    """1234.5 -> '1.235' (punto para los miles, como en español)."""
    return f"{valor:,.0f}".replace(",", ".")


def formato_porcentaje(valor):
    """6.39 -> '6,4%' (coma decimal, como en español)."""
    return f"{valor:.1f}%".replace(".", ",")


def fechas_futuras(serie, pasos):
    """Las fechas de los próximos meses, empezando después del último dato."""
    return pd.date_range(
        serie.index[-1] + pd.offsets.MonthBegin(1), periods=pasos, freq="MS"
    )


def pronosticar_temporada(serie, pasos):
    """Holt-Winters con tendencia aditiva y temporada multiplicativa de 12 meses."""
    modelo = ExponentialSmoothing(
        serie,
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    ajuste = modelo.fit()
    pronostico = ajuste.forecast(pasos)
    pronostico.index = fechas_futuras(serie, pasos)
    return pronostico


def pronosticar_suavizamiento(serie, pasos):
    """Suavizamiento exponencial simple: solo mira el nivel más reciente."""
    modelo = SimpleExpSmoothing(serie, initialization_method="estimated")
    ajuste = modelo.fit(optimized=True)
    pronostico = ajuste.forecast(pasos)
    pronostico.index = fechas_futuras(serie, pasos)
    return pronostico


def pronosticar_sarima(serie, pasos):
    """SARIMA: mira el pasado y la temporada de 12 meses."""
    modelo = SARIMAX(
        serie,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    ajuste = modelo.fit(disp=False)
    pronostico = ajuste.forecast(pasos)
    pronostico.index = fechas_futuras(serie, pasos)
    return pronostico


def pronosticar_regresion(serie, pasos):
    """Regresión lineal: una recta de crecimiento más un ajuste por mes."""
    X = pd.DataFrame({"tiempo": np.arange(len(serie)), "mes": serie.index.month})
    X = pd.get_dummies(X, columns=["mes"], drop_first=True).astype(float)
    X = sm.add_constant(X)
    ajuste = sm.OLS(serie.values, X).fit()

    fechas = fechas_futuras(serie, pasos)
    X_futuro = pd.DataFrame(
        {
            "tiempo": np.arange(len(serie), len(serie) + pasos),
            "mes": fechas.month,
        }
    )
    X_futuro = pd.get_dummies(X_futuro, columns=["mes"], drop_first=True).astype(float)
    X_futuro = sm.add_constant(X_futuro)
    X_futuro = X_futuro.reindex(columns=X.columns, fill_value=0.0)

    # np.asarray evita que pandas intente alinear índices distintos.
    return pd.Series(np.asarray(ajuste.predict(X_futuro)), index=fechas)


def pronosticar_prophet(serie, pasos):
    """Prophet: modelo de tendencia y temporada, muy usado en negocios."""
    datos = pd.DataFrame({"ds": serie.index, "y": serie.values})
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    modelo.fit(datos)
    futuro = modelo.make_future_dataframe(periods=pasos, freq="MS")
    prediccion = modelo.predict(futuro)
    pronostico = prediccion.set_index("ds")["yhat"].iloc[-pasos:]
    pronostico.index = fechas_futuras(serie, pasos)
    return pronostico


MODELOS = {
    "Modelo de temporada": pronosticar_temporada,
    "SARIMA": pronosticar_sarima,
    "Suavizamiento exponencial": pronosticar_suavizamiento,
    "Regresión lineal": pronosticar_regresion,
    "Prophet": pronosticar_prophet,
}

ESTILOS = {
    "Copiar el año pasado": dict(color="#E67E22", dash="dash"),
    "Modelo de temporada": dict(color="#27AE60", dash="dash"),
    "SARIMA": dict(color="#8E44AD", dash="dot"),
    "Suavizamiento exponencial": dict(color="#C0392B", dash="dashdot"),
    "Regresión lineal": dict(color="#795548", dash="longdash"),
    "Prophet": dict(color="#16A085", dash="longdashdot"),
}


def error_porcentual(real, pronostico):
    """Error promedio en porcentaje: qué tan lejos quedó el pronóstico."""
    return (abs(real - pronostico) / real).mean() * 100


@st.cache_data(show_spinner="Entrenando los modelos del examen...")
def pronosticos_del_examen(serie_visible):
    """Entrena los cinco modelos con los datos visibles y devuelve 24 meses."""
    return {nombre: funcion(serie_visible, 24) for nombre, funcion in MODELOS.items()}


@st.cache_data(show_spinner="Entrenando el modelo...")
def pronostico_con_modelo(nombre, serie):
    """Entrena el modelo elegido con todos los datos y devuelve 12 meses."""
    return MODELOS[nombre](serie, 12)


@st.cache_data
def cargar_datos():
    return leer_pasajeros()


df = cargar_datos()

# ---------------------------------------------------------------- indicadores
st.title("¿Cuántos pasajeros tendremos el próximo año?")

total_1949 = df.loc[df["Month"].dt.year == 1949, "Passengers"].sum()
total_1960 = df.loc[df["Month"].dt.year == 1960, "Passengers"].sum()
crecimiento = total_1960 / total_1949

col1, col2, col3 = st.columns(3)
col1.metric("Pasajeros en 1949", formato_numero(total_1949))
col2.metric("Pasajeros en 1960", formato_numero(total_1960))
col3.metric(
    "Cuántas veces creció",
    f"{crecimiento:.1f} veces".replace(".", ","),
    delta=f"{(crecimiento - 1) * 100:.0f}% más",
)
st.caption("Suma de pasajeros de cada año (datos mensuales de 1949 a 1960).")

# --------------------------------------------------------------------- tabs
tab_historia, tab_temporada, tab_examen, tab_pronostico = st.tabs(
    ["La historia", "La temporada", "El examen", "El pronóstico"]
)

# ------------------------------------------------------------------ historia
with tab_historia:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Passengers"],
            mode="lines",
            name="Pasajeros",
            line=dict(color="#1F77B4", width=2),
        )
    )
    fig.update_layout(
        title="Pasajeros por mes",
        xaxis_title="Mes",
        yaxis_title="Pasajeros",
        hovermode="x unified",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "La línea muestra cuántos pasajeros viajaron cada mes desde enero de 1949 "
        "hasta diciembre de 1960. Se ve una subida sostenida con el paso de los años "
        "y, dentro de cada año, un patrón que se repite: ciertos meses suben y otros bajan. "
        "Ese patrón repetido es la temporada, y es lo que aprovecha el pronóstico."
    )

# ----------------------------------------------------------------- temporada
with tab_temporada:
    promedio_mes = df.groupby(df["Month"].dt.month)["Passengers"].mean()
    mes_normal = df["Passengers"].mean()

    nombres = [MESES[mes] for mes in promedio_mes.index]
    colores = [
        "#2E86C1" if valor >= mes_normal else "#E67E22"
        for valor in promedio_mes.values
    ]
    etiquetas = [
        f"{(valor / mes_normal - 1) * 100:+.0f}%" for valor in promedio_mes.values
    ]

    fig = go.Figure(
        go.Bar(
            x=nombres,
            y=promedio_mes.values,
            marker_color=colores,
            text=etiquetas,
            textposition="outside",
            hovertemplate="%{x}: %{y:.0f} pasajeros<extra></extra>",
        )
    )
    fig.add_hline(
        y=mes_normal,
        line_dash="dash",
        line_color="#555555",
        annotation_text=f"Mes normal: {mes_normal:.0f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Promedio de pasajeros de cada mes del año",
        xaxis_title="Mes del año",
        yaxis_title="Pasajeros en promedio",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[0, promedio_mes.max() * 1.18]),
    )
    st.plotly_chart(fig, width="stretch")

    mes_alto = promedio_mes.idxmax()
    mes_bajo = promedio_mes.idxmin()
    st.info(
        "Las barras azules están por encima de un mes normal (la línea punteada) "
        "y las naranjas por debajo. "
        f"{MESES[mes_alto].capitalize()} es el mes más fuerte y "
        f"{MESES[mes_bajo]} el más flojo, así que ahí conviene tener más o menos "
        "aviones y personal. Esa diferencia entre meses es la temporada que aprende el modelo."
    )

# -------------------------------------------------------------------- examen
with tab_examen:
    visible = df.iloc[:-24].copy()
    real_oculto = df.iloc[-24:].copy()

    # Forma 1: copiar cada mes del último año visible.
    ultimo_anio = visible["Month"].dt.year.max()
    base = visible.loc[visible["Month"].dt.year == ultimo_anio, ["Month", "Passengers"]].copy()
    base["mes"] = base["Month"].dt.month
    valores_ultimo_anio = base.set_index("mes")["Passengers"]
    pred_copiar = pd.Series(
        [valores_ultimo_anio[mes] for mes in real_oculto["Month"].dt.month],
        index=real_oculto["Month"],
    )

    # Las demás formas se entrenan solo con los datos visibles.
    serie_visible = visible.set_index("Month")["Passengers"].asfreq("MS")
    pronosticos = {"Copiar el año pasado": pred_copiar}
    pronosticos.update(pronosticos_del_examen(serie_visible))

    errores = {
        nombre: error_porcentual(real_oculto["Passengers"].values, pred.values)
        for nombre, pred in pronosticos.items()
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=visible["Month"],
            y=visible["Passengers"],
            mode="lines",
            name="Real (datos visibles)",
            line=dict(color="#1F77B4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=real_oculto["Month"],
            y=real_oculto["Passengers"],
            mode="lines",
            name="Real (los 24 meses tapados)",
            line=dict(color="black", width=3),
        )
    )
    for nombre, pred in pronosticos.items():
        estilo = ESTILOS[nombre]
        fig.add_trace(
            go.Scatter(
                x=pred.index,
                y=pred.values,
                mode="lines",
                name=nombre,
                line=dict(
                    color=estilo["color"],
                    dash=estilo["dash"],
                    width=2,
                ),
            )
        )
    fecha_corte = real_oculto["Month"].iloc[0].strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=fecha_corte,
        x1=fecha_corte,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dot", color="gray", width=1),
    )
    fig.add_annotation(
        x=fecha_corte,
        y=1,
        yref="paper",
        text="Aquí se tapan los datos",
        showarrow=False,
        yanchor="bottom",
    )
    fig.update_layout(
        title="¿Quién adivina mejor los últimos 24 meses?",
        xaxis_title="Mes",
        yaxis_title="Pasajeros",
        hovermode="x unified",
        height=500,
        margin=dict(l=10, r=10, t=50, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            x=0,
            font=dict(size=10),
        ),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "Para poner a prueba las seis formas, escondimos los últimos 24 meses y "
        "les pedimos que los adivinen. La línea negra es lo que realmente pasó, y "
        "mientras más cerca quede un pronóstico de ella, mejor. Los modelos que "
        "tienen en cuenta la temporada y el crecimiento suelen acercarse más."
    )

    filas = sorted(errores.items(), key=lambda fila: fila[1])
    tabla_errores = pd.DataFrame(
        {
            "Forma de pronosticar": [nombre for nombre, _ in filas],
            "Error promedio (%)": [
                formato_porcentaje(valor) for _, valor in filas
            ],
        }
    )
    st.dataframe(tabla_errores, hide_index=True, width="stretch")

    ganador, error_ganador = filas[0]
    segundo, error_segundo = filas[1]
    st.success(
        f"Ganó **{ganador}** con un error promedio de "
        f"{formato_porcentaje(error_ganador)}, el más bajo de los seis. "
        f"El segundo lugar fue para **{segundo}** con "
        f"{formato_porcentaje(error_segundo)}. Cuanto más bajo el error, mejor."
    )

# ---------------------------------------------------------------- pronóstico
with tab_pronostico:
    serie_completa = df.set_index("Month")["Passengers"].asfreq("MS")
    nombre_modelo = st.selectbox(
        "Elige la forma de pronosticar",
        list(MODELOS.keys()),
    )
    pronostico = pronostico_con_modelo(nombre_modelo, serie_completa)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Month"],
            y=df["Passengers"],
            mode="lines",
            name="Pasajeros reales",
            line=dict(color="#1F77B4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pronostico.index,
            y=pronostico.values,
            mode="lines+markers",
            name="Pronóstico 1961",
            line=dict(
                color=ESTILOS[nombre_modelo]["color"],
                dash="dash",
                width=3,
            ),
            marker=dict(size=7),
        )
    )
    fecha_inicio = pronostico.index[0].strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=fecha_inicio,
        x1=fecha_inicio,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dot", color="gray", width=1),
    )
    fig.add_annotation(
        x=fecha_inicio,
        y=1,
        yref="paper",
        text="Empieza 1961",
        showarrow=False,
        yanchor="bottom",
    )
    fig.update_layout(
        title=f"Pasajeros esperados para 1961 con {nombre_modelo}",
        xaxis_title="Mes",
        yaxis_title="Pasajeros",
        hovermode="x unified",
        height=470,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"**{nombre_modelo}** usa toda la historia (1949-1960) y estima los "
        "12 meses de 1961. Los modelos que aprenden la temporada repiten el patrón "
        "de verano alto e invierno bajo; los más simples solo siguen la tendencia "
        "general. Son estimaciones: si el negocio cambia, las cifras reales pueden "
        "ser distintas."
    )

    tabla_pronostico = pd.DataFrame(
        {
            "Mes": [f"{MESES[mes].capitalize()} de 1961" for mes in pronostico.index.month],
            "Pasajeros esperados": [int(round(valor)) for valor in pronostico.values],
        }
    )
    st.dataframe(tabla_pronostico, hide_index=True, width="stretch")
    st.caption(
        f"Total esperado para 1961: {formato_numero(pronostico.sum())} pasajeros."
    )
