"""Funciones del análisis: carga, preparación, ACP y K-Means.

La aplicación (app.py) y el cuaderno (notebooks/analisis_sp500.ipynb) usan
estas mismas funciones, así que las tablas y métricas de los dos coinciden.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

CARPETA_DATOS = Path(__file__).resolve().parent / "data"
ARCHIVO_FINANCIERO = CARPETA_DATOS / "constituents-financials.csv"
ARCHIVO_CONSTITUYENTES = CARPETA_DATOS / "constituents.csv"
FECHA_DESCARGA = "17 de noviembre de 2026"

SEMILLA = 42

# Sector GICS en español. El archivo financiero trae la subindustria;
# el sector se toma del listado de constituyentes (cada subindustria
# pertenece a un único sector).
SECTORES_ES = {
    "Communication Services": "Comunicaciones",
    "Consumer Discretionary": "Consumo discrecional",
    "Consumer Staples": "Consumo básico",
    "Energy": "Energía",
    "Financials": "Financiero",
    "Health Care": "Salud",
    "Industrials": "Industrial",
    "Information Technology": "Tecnología",
    "Materials": "Materiales",
    "Real Estate": "Inmobiliario",
    "Utilities": "Servicios públicos",
}

# Tres subindustrias que ya no aparecen en el listado actual de constituyentes.
SUBINDUSTRIAS_FALTANTES = {
    "Home Furnishings": "Consumer Discretionary",
    "Brewers": "Consumer Staples",
    "Drug Retail": "Consumer Staples",
}

# Acciones de dos clases de la misma empresa: se conserva una sola.
CLASES_DUPLICADAS = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}

VARIABLES = {
    "tamano_log": {
        "nombre": "Tamaño (log10 capitalización)",
        "formula": "log10(capitalización bursátil en USD)",
        "lectura": "Más alto = empresa más grande.",
        "dimension": "Tamaño",
    },
    "rend_utilidades": {
        "nombre": "Rendimiento de utilidades (E/P)",
        "formula": "UPA / precio  (inverso del P/E)",
        "lectura": "Más alto = acción más barata frente a sus utilidades. Negativo = pérdidas.",
        "dimension": "Valoración",
    },
    "libro_mercado": {
        "nombre": "Libro a mercado (B/P)",
        "formula": "1 / (precio / valor en libros)",
        "lectura": "Más alto = acción más barata frente a su patrimonio. Negativo = patrimonio contable negativo.",
        "dimension": "Valoración",
    },
    "ventas_precio": {
        "nombre": "Ventas a precio (S/P)",
        "formula": "1 / (precio / ventas)",
        "lectura": "Más alto = acción más barata frente a sus ventas.",
        "dimension": "Valoración",
    },
    "rend_dividendo": {
        "nombre": "Rendimiento por dividendo",
        "formula": "dividendo anual / precio",
        "lectura": "Más alto = paga más dividendo por cada dólar invertido.",
        "dimension": "Ingreso",
    },
    "margen_neto": {
        "nombre": "Margen neto",
        "formula": "(UPA / precio) × (precio / ventas) = utilidad / ventas",
        "lectura": "Más alto = convierte más ventas en utilidad.",
        "dimension": "Rentabilidad",
    },
    "rango_52s": {
        "nombre": "Amplitud del rango de 52 semanas",
        "formula": "(máximo − mínimo) / máximo",
        "lectura": "Más alto = el precio recorrió más distancia en el año (proxy de volatilidad).",
        "dimension": "Riesgo",
    },
    "posicion_52s": {
        "nombre": "Posición en el rango de 52 semanas",
        "formula": "(precio − mínimo) / (máximo − mínimo)",
        "lectura": "1 = cotiza en su máximo anual; 0 = en su mínimo (proxy de momento).",
        "dimension": "Momento",
    },
}


def nombre(variable):
    return VARIABLES[variable]["nombre"] if variable in VARIABLES else variable


# --------------------------------------------------------------------------
# 1. Carga y unión de fuentes
# --------------------------------------------------------------------------
def cargar_crudo():
    """Lee el archivo financiero y le agrega el sector GICS."""
    financiero = pd.read_csv(ARCHIVO_FINANCIERO)
    constituyentes = pd.read_csv(ARCHIVO_CONSTITUYENTES)

    mapa = (
        constituyentes.drop_duplicates("GICS Sub-Industry")
        .set_index("GICS Sub-Industry")["GICS Sector"]
        .to_dict()
    )
    mapa.update(SUBINDUSTRIAS_FALTANTES)

    crudo = financiero.rename(columns={"Sector": "Subindustria"}).copy()
    crudo["Sector"] = crudo["Subindustria"].map(mapa).map(SECTORES_ES)
    return crudo


# --------------------------------------------------------------------------
# 2. Limpieza con registro de decisiones
# --------------------------------------------------------------------------
def limpiar(crudo):
    """Aplica las exclusiones y devuelve (datos, bitácora de decisiones)."""
    datos = crudo.copy()
    bitacora = []

    sin_datos = datos["Price"].isna()
    bitacora.append({
        "Decisión": "Excluir empresas sin precio ni estados financieros",
        "Motivo": "Yahoo Finance no devolvió datos (empresas adquiridas o retiradas de bolsa, "
                  "o tickers con punto como BRK.B y BF.B que Yahoo escribe con guion).",
        "Empresas": int(sin_datos.sum()),
        "Tickers": ", ".join(datos.loc[sin_datos, "Symbol"]),
    })
    datos = datos[~sin_datos]

    # Precio y capitalización incompatibles con su rango de 52 semanas y su historia:
    # el ticker quedó con datos residuales tras una fusión.
    inconsistente = datos["Symbol"].isin(["PARA"])
    bitacora.append({
        "Decisión": "Excluir registros con datos inconsistentes",
        "Motivo": "PARA reporta precio de 0,10 USD, máximo de 52 semanas de 392 USD y capitalización "
                  "de 207.300 USD: datos residuales del ticker tras la fusión de Paramount.",
        "Empresas": int(inconsistente.sum()),
        "Tickers": ", ".join(datos.loc[inconsistente, "Symbol"]),
    })
    datos = datos[~inconsistente]

    duplicada = datos["Symbol"].isin(CLASES_DUPLICADAS.keys())
    bitacora.append({
        "Decisión": "Conservar una sola clase de acción por empresa",
        "Motivo": "GOOG/GOOGL, FOX/FOXA y NWS/NWSA son la misma empresa: contarlas dos veces "
                  "duplicaría su peso en las distancias. Se conserva la clase A.",
        "Empresas": int(duplicada.sum()),
        "Tickers": ", ".join(datos.loc[duplicada, "Symbol"]),
    })
    datos = datos[~duplicada]

    bitacora.append({
        "Decisión": "No usar EBITDA",
        "Motivo": f"{int(datos['EBITDA'].isna().sum())} empresas, todas del sector financiero, no reportan "
                  "EBITDA (los bancos no lo usan). Imputarlo inventaría un dato; en su lugar se usa "
                  "el margen neto, que está disponible para todas.",
        "Empresas": 0,
        "Tickers": "",
    })
    bitacora.append({
        "Decisión": "Excluir identificadores y texto del modelo",
        "Motivo": "Symbol, Name, SEC Filings, Subindustria y Sector no son medidas cuantitativas. "
                  "El sector se guarda aparte para contrastar los grupos al final.",
        "Empresas": 0,
        "Tickers": "",
    })
    return datos.reset_index(drop=True), pd.DataFrame(bitacora)


# --------------------------------------------------------------------------
# 3. Variables de análisis
# --------------------------------------------------------------------------
def construir_variables(datos):
    """Construye las 8 variables cuantitativas a partir de las columnas de Yahoo."""
    precio = datos["Price"]
    variables = pd.DataFrame(index=datos.index)
    variables["tamano_log"] = np.log10(datos["Market Cap"])
    variables["rend_utilidades"] = datos["Earnings/Share"] / precio
    variables["libro_mercado"] = 1.0 / datos["Price/Book"]
    variables["ventas_precio"] = 1.0 / datos["Price/Sales"]
    variables["rend_dividendo"] = datos["Dividend Yield"]
    variables["margen_neto"] = variables["rend_utilidades"] * datos["Price/Sales"]
    amplitud = datos["52 Week High"] - datos["52 Week Low"]
    variables["rango_52s"] = amplitud / datos["52 Week High"]
    variables["posicion_52s"] = ((precio - datos["52 Week Low"]) / amplitud).clip(0, 1)

    identificacion = datos[["Symbol", "Name", "Sector", "Subindustria"]].rename(
        columns={"Symbol": "Ticker", "Name": "Empresa"}
    )
    return identificacion, variables


def winsorizar(variables, nivel):
    """Recorta cada variable en los percentiles nivel y 1 − nivel."""
    if nivel <= 0:
        return variables.copy(), pd.DataFrame()
    inferior = variables.quantile(nivel)
    superior = variables.quantile(1 - nivel)
    recortadas = variables.clip(lower=inferior, upper=superior, axis=1)
    resumen = pd.DataFrame({
        "Límite inferior": inferior,
        "Límite superior": superior,
        "Valores recortados": (variables.ne(recortadas)).sum(),
    })
    return recortadas, resumen


def estandarizar(variables):
    escalador = StandardScaler()
    z = escalador.fit_transform(variables)
    return pd.DataFrame(z, index=variables.index, columns=variables.columns)


def preparar(nivel_winsor=0.01, seleccion=None):
    """Recorrido completo de preparación. Devuelve un diccionario con cada tramo."""
    crudo = cargar_crudo()
    datos, bitacora = limpiar(crudo)
    identificacion, variables = construir_variables(datos)
    if seleccion is not None:
        variables = variables[list(seleccion)]
    recortadas, resumen_winsor = winsorizar(variables, nivel_winsor)
    z = estandarizar(recortadas)
    return {
        "crudo": crudo,
        "datos": datos,
        "bitacora": bitacora,
        "identificacion": identificacion,
        "variables": variables,
        "recortadas": recortadas,
        "resumen_winsor": resumen_winsor,
        "z": z,
    }


# --------------------------------------------------------------------------
# 4. Análisis de componentes principales
# --------------------------------------------------------------------------
def ajustar_acp(z):
    """ACP con todos los componentes: varianza, valores propios y cargas."""
    acp = PCA(random_state=SEMILLA).fit(z)
    etiquetas = [f"CP{i + 1}" for i in range(acp.n_components_)]
    varianza = pd.DataFrame({
        "Componente": etiquetas,
        "Valor propio": acp.explained_variance_,
        "Varianza explicada": acp.explained_variance_ratio_,
        "Varianza acumulada": np.cumsum(acp.explained_variance_ratio_),
    })
    cargas = pd.DataFrame(acp.components_.T, index=z.columns, columns=etiquetas)
    # Correlación entre cada variable y cada componente (carga × √valor propio).
    correlaciones = cargas * np.sqrt(acp.explained_variance_)
    puntajes = pd.DataFrame(acp.transform(z), index=z.index, columns=etiquetas)
    return {"modelo": acp, "varianza": varianza, "cargas": cargas,
            "correlaciones": correlaciones, "puntajes": puntajes}


def sugerir_componentes(varianza, umbral=0.80):
    """Componentes según Kaiser (valor propio > 1) y según el umbral de varianza acumulada."""
    kaiser = int((varianza["Valor propio"] > 1).sum())
    acumulada = int((varianza["Varianza acumulada"] < umbral).sum() + 1)
    return {"kaiser": max(kaiser, 1), "acumulada": min(acumulada, len(varianza))}


# --------------------------------------------------------------------------
# 5. K-Means
# --------------------------------------------------------------------------
def evaluar_k(matriz, k_min=2, k_max=10, semilla=SEMILLA):
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(matriz)
        filas.append({
            "K": k,
            "Inercia": modelo.inertia_,
            "Silueta": silhouette_score(matriz, modelo.labels_),
            "Grupo más pequeño": int(np.bincount(modelo.labels_).min()),
        })
    return pd.DataFrame(filas)


def ajustar_kmeans(matriz, k, semilla=SEMILLA):
    modelo = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(matriz)
    # Se renumeran los grupos por tamaño (A = el más grande) para que la etiqueta
    # no cambie entre corridas. La letra no implica orden ni jerarquía.
    tamanos = pd.Series(modelo.labels_).value_counts()
    letras = {grupo: chr(65 + i) for i, grupo in enumerate(tamanos.index)}
    etiquetas = pd.Series(modelo.labels_, index=matriz.index).map(letras)
    silueta = pd.Series(silhouette_samples(matriz, modelo.labels_), index=matriz.index)
    return {"modelo": modelo, "grupo": etiquetas.rename("Grupo"), "silueta": silueta.rename("Silueta")}


def estabilidad(matriz, k, semillas=range(10)):
    """Índice de Rand ajustado entre la solución base y otras semillas."""
    from sklearn.metrics import adjusted_rand_score

    base = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit_predict(matriz)
    valores = [
        adjusted_rand_score(base, KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(matriz))
        for s in semillas
    ]
    return float(np.mean(valores)), float(np.min(valores))


def perfiles(variables, grupo):
    """Mediana de cada variable original por grupo, más el total."""
    tabla = variables.groupby(grupo).median().T
    tabla["Total"] = variables.median()
    return tabla


def perfiles_z(z, grupo):
    """Media estandarizada por grupo: cuántas desviaciones se aleja del promedio."""
    return z.groupby(grupo).mean().T


def composicion_sectorial(sector, grupo):
    """Porcentaje de cada sector que cae en cada grupo."""
    return pd.crosstab(sector, grupo, normalize="index")
