"""Estilos de inversión en el S&P 500 · ACP y K-Means.

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analisis as an

st.set_page_config(
    page_title="Estilos de inversión en el S&P 500",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# Identidad visual
# --------------------------------------------------------------------------
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8984"
SUPERFICIE = "#fcfcfb"
REJILLA = "#e6e5e1"
AZUL = "#2a78d6"
AZUL_OSCURO = "#1c5cab"
ROJO = "#e34948"
GRIS_MEDIO = "#f0efec"
# Orden fijo de colores para los grupos; el símbolo del marcador es la segunda
# codificación, para que la identidad del grupo no dependa solo del color.
COLORES_GRUPO = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4",
                 "#008300", "#eda100", "#e34948", "#52514e", "#104281"]
SIMBOLOS_GRUPO = ["circle", "square", "diamond", "triangle-up", "x",
                  "cross", "triangle-down", "star", "pentagon", "hexagon"]
DIVERGENTE = [[0.0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, GRIS_MEDIO],
              [0.75, "#ec8a89"], [1.0, "#b8302f"]]
SECUENCIAL = [[0.0, "#f6f9fe"], [0.35, "#9ec5f4"], [0.7, "#3987e5"], [1.0, "#104281"]]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .etiqueta {color:#52514e; font-size:0.78rem; text-transform:uppercase;
               letter-spacing:0.08em; margin-bottom:0.2rem;}
    .nota {background:#f0efec; border-left:3px solid #1c5cab; padding:0.75rem 1rem;
           border-radius:4px; color:#2b2a28; font-size:0.95rem; margin:0.4rem 0 1rem 0;}
    .aviso {background:#fff6e5; border-left:3px solid #c98500; padding:0.75rem 1rem;
            border-radius:4px; color:#2b2a28; font-size:0.95rem; margin:0.4rem 0 1rem 0;}
    div[data-testid="stMetricValue"] {font-size:1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def nota(texto, clase="nota"):
    st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)


def num(x, dec=2):
    """Número con coma decimal y punto de miles."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return "—" if pd.isna(x) else f"{num(100 * x, dec)} %"


def estilo(fig, alto=420, leyenda=True):
    fig.update_layout(
        height=alto,
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TINTA_2, size=13),
        title_font=dict(color=TINTA, size=16),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=leyenda,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA_2)),
        hoverlabel=dict(bgcolor="white", font_color=TINTA, bordercolor=REJILLA),
        separators=",.",
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor="#cfcec9", linecolor="#cfcec9",
                     title_font=dict(color=TINTA_2))
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor="#cfcec9", linecolor="#cfcec9",
                     title_font=dict(color=TINTA_2))
    return fig


def mapa_calor(tabla, titulo, escala, rango=None, formato=".2f", alto=420, eje_x="", eje_y=""):
    z = tabla.values.astype(float)
    texto = np.vectorize(lambda v: num(v, int(formato[1])) if formato.endswith("f") else pct(v, 0))(z)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(tabla.columns), y=list(tabla.index), colorscale=escala,
        zmin=None if rango is None else rango[0], zmax=None if rango is None else rango[1],
        text=texto, hovertemplate="%{y} · %{x}: %{text}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    # Texto de cada celda con tinta clara u oscura según la intensidad del fondo.
    bajo, alto_r = (np.nanmin(z), np.nanmax(z)) if rango is None else rango
    centro = 0.5 if escala is SECUENCIAL else None
    for i, fila in enumerate(tabla.index):
        for j, col in enumerate(tabla.columns):
            t = (z[i, j] - bajo) / (alto_r - bajo) if alto_r > bajo else 0.5
            oscuro = t > 0.62 if centro is not None else abs(t - 0.5) > 0.36
            fig.add_annotation(x=col, y=fila, text=texto[i, j], showarrow=False,
                               font=dict(size=12, color="white" if oscuro else TINTA))
    fig.update_layout(title=titulo, xaxis_title=eje_x, yaxis_title=eje_y)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return estilo(fig, alto=alto, leyenda=False)


# --------------------------------------------------------------------------
# Parámetros (barra lateral, visibles en todas las páginas)
# --------------------------------------------------------------------------
PAGINAS = [
    "1 · Contexto",
    "2 · Exploración",
    "3 · Preparación",
    "4 · Componentes principales",
    "5 · Segmentación K-Means",
    "6 · Perfil de los grupos",
    "7 · Conclusiones",
]
POR_DEFECTO = {"variables": list(an.VARIABLES), "winsor": 1.0, "componentes": 3, "k": 4, "semilla": 42}

# Los controles de componentes y K viven en páginas distintas: se reasignan en cada
# corrida para que Streamlit no los borre al cambiar de página.
for _clave in ("comp", "k_elegido"):
    if _clave in st.session_state:
        st.session_state[_clave] = st.session_state[_clave]

with st.sidebar:
    st.markdown("### Estilos de inversión en el S&P 500")
    st.caption("ACP y K-Means sobre 482 empresas")
    pagina = st.radio("Sección", PAGINAS, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="etiqueta">Parámetros del análisis</div>', unsafe_allow_html=True)
    seleccion = st.multiselect(
        "Variables que entran al modelo",
        options=list(an.VARIABLES),
        default=POR_DEFECTO["variables"],
        format_func=an.nombre,
        key="vars_sel",
        help="Mínimo cuatro. Todo el análisis se recalcula con las que elijas.",
    )
    winsor = st.slider(
        "Recorte de extremos (% por cola)", 0.0, 5.0, POR_DEFECTO["winsor"], 0.5,
        help="Winsorización: los valores por debajo del percentil p y por encima del 100 − p "
             "se reemplazan por esos percentiles.",
        key="winsor_sel",
    )
    semilla = st.number_input("Semilla de K-Means", 0, 9999, POR_DEFECTO["semilla"], 1, key="semilla_sel")
    if st.button("Volver a los valores por defecto", width="stretch"):
        for clave in ["comp", "k_elegido", "vars_sel", "winsor_sel", "semilla_sel"]:
            st.session_state.pop(clave, None)
        st.rerun()

if len(seleccion) < 4:
    st.warning("Selecciona al menos cuatro variables en la barra lateral para correr el análisis.")
    st.stop()


# --------------------------------------------------------------------------
# Cálculos (en caché: se repiten solo si cambian los parámetros)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def correr_preparacion(variables, nivel):
    return an.preparar(nivel_winsor=nivel / 100, seleccion=variables)


@st.cache_data(show_spinner=False)
def correr_acp(z):
    return an.ajustar_acp(z)


@st.cache_data(show_spinner=False)
def correr_evaluacion(matriz, semilla):
    return an.evaluar_k(matriz, 2, 10, semilla)


@st.cache_data(show_spinner=False)
def correr_kmeans(matriz, k, semilla):
    return an.ajustar_kmeans(matriz, k, semilla)


@st.cache_data(show_spinner=False)
def correr_estabilidad(matriz, k):
    return an.estabilidad(matriz, k)


prep = correr_preparacion(tuple(seleccion), winsor)
acp = correr_acp(prep["z"])
varianza = acp["varianza"]
sugerencia = an.sugerir_componentes(varianza)
n_max = len(varianza)

if "comp" not in st.session_state or st.session_state["comp"] > n_max:
    st.session_state["comp"] = min(sugerencia["kaiser"], n_max) if seleccion != POR_DEFECTO["variables"] \
        else POR_DEFECTO["componentes"]
if "k_elegido" not in st.session_state:
    st.session_state["k_elegido"] = POR_DEFECTO["k"]

n_comp = st.session_state["comp"]
k = st.session_state["k_elegido"]
matriz = acp["puntajes"].iloc[:, :n_comp]
evaluacion = correr_evaluacion(matriz, semilla)
km = correr_kmeans(matriz, k, semilla)
grupo = km["grupo"]
letras = sorted(grupo.unique())
color_grupo = {g: COLORES_GRUPO[i] for i, g in enumerate(letras)}
simbolo_grupo = {g: SIMBOLOS_GRUPO[i] for i, g in enumerate(letras)}
ident = prep["identificacion"]
variables = prep["variables"]
z = prep["z"]
capitalizacion = prep["datos"]["Market Cap"]

es_por_defecto = (
    list(seleccion) == POR_DEFECTO["variables"] and winsor == POR_DEFECTO["winsor"]
    and n_comp == POR_DEFECTO["componentes"] and k == POR_DEFECTO["k"] and semilla == POR_DEFECTO["semilla"]
)


def descriptores(g, n=2):
    """Variables que más separan al grupo del promedio, en palabras."""
    perfil = an.perfiles_z(z, grupo)[g].sort_values()
    altos = [f"{an.nombre(v).split(' (')[0].lower()} alto" for v in perfil.index[::-1][:n] if perfil[v] > 0.3]
    bajos = [f"{an.nombre(v).split(' (')[0].lower()} bajo" for v in perfil.index[:n] if perfil[v] < -0.3]
    return ", ".join(altos + bajos) or "cerca del promedio en todas las variables"


# ==========================================================================
# 1. CONTEXTO
# ==========================================================================
if pagina == PAGINAS[0]:
    st.markdown('<div class="etiqueta">Taller individual · ACP y K-Means</div>', unsafe_allow_html=True)
    st.title("¿Qué estilos de inversión conviven dentro del S&P 500?")
    st.markdown(
        "Un gestor que arma un portafolio de acciones de EE. UU. suele hablar de **estilos**: "
        "empresas de *valor* (baratas frente a sus utilidades o su patrimonio), de *crecimiento* "
        "(caras, pero con márgenes altos), de *dividendo* (pagan mucho y se mueven poco) o "
        "*cíclicas* (con grandes recorridos de precio). Esas etiquetas suelen asignarse con reglas fijas. "
        "Este trabajo pregunta si los datos, por sí solos, reproducen esa clasificación."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas analizadas", f"{len(ident)}")
    c2.metric("Variables cuantitativas", f"{len(seleccion)}")
    c3.metric("Sectores GICS", f"{ident['Sector'].nunique()}")
    c4.metric("Capitalización total", f"{num(capitalizacion.sum() / 1e12, 1)} billones USD")

    st.subheader("Preguntas que orientan el análisis")
    st.markdown(
        "1. **¿Cuántas dimensiones resumen las diferencias entre empresas?** Ocho razones financieras "
        "pueden estar contando unas pocas historias: tamaño, valoración, ingreso, riesgo.\n"
        "2. **¿Existen grupos de empresas con un perfil financiero parecido?** Y si existen, ¿qué los define?\n"
        "3. **¿Esos grupos coinciden con los sectores económicos o los cruzan?** Si un banco y una "
        "empresa de servicios públicos caen juntos, el estilo pesa más que la industria."
    )

    st.subheader("Por qué ACP y K-Means")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Análisis de Componentes Principales.** Las razones de valoración están correlacionadas: "
            "una empresa barata frente a sus ventas suele serlo también frente a su patrimonio. El ACP "
            "combina variables que se mueven juntas en ejes no correlacionados y permite saber cuánta "
            "información se conserva al quedarse con pocos ejes. Además evita que una misma idea "
            "(«barata») pese tres veces en las distancias."
        )
    with col_b:
        st.markdown(
            "**K-Means.** No hay una etiqueta de «estilo» verdadera para cada empresa: es un problema "
            "no supervisado. K-Means agrupa a las empresas que quedan cerca en el espacio de componentes "
            "y devuelve centroides que se pueden leer como perfiles. El sector GICS no entra al modelo; "
            "se usa al final como contraste externo."
        )

    st.subheader("Fuente y cobertura")
    st.markdown(
        f"""
- **Base:** *S&P 500 Companies with Financial Information* — [datahub.io / GitHub](https://github.com/datasets/s-and-p-500-companies-financials),
  la versión mantenida y actualizada del conjunto *S&P 500 Companies with Financial Information* que también se publica en Kaggle.
  Licencia ODC-PDDL (dominio público).
- **Origen de las cifras:** Yahoo Finance (vía `yfinance`), actualizado automáticamente por el repositorio.
  El sector GICS se toma del listado de constituyentes del mismo proyecto (Wikipedia).
- **Corte:** archivo descargado el {an.FECHA_DESCARGA}. Cada fila es una empresa del índice; es una
  fotografía en un momento, no una serie de tiempo.
- **Cobertura:** 503 tickers en el archivo original; 482 empresas después de la limpieza (sección 3).
"""
    )

    st.subheader("Variables del modelo")
    tabla_vars = pd.DataFrame([
        {"Variable": v["nombre"], "Dimensión": v["dimension"], "Cálculo": v["formula"], "Lectura": v["lectura"]}
        for v in an.VARIABLES.values()
    ])
    st.dataframe(tabla_vars, hide_index=True, width="stretch")
    nota(
        "Las razones de valoración se usan invertidas (E/P en lugar de P/E). Así una empresa con "
        "pérdidas da un número negativo y continuo, en lugar de un P/E que se dispara cuando la "
        "utilidad se acerca a cero."
    )

    st.subheader("Limitaciones de la fuente")
    st.markdown(
        "- Es un corte transversal: el rango de 52 semanas y el precio cambian cada día, así que los "
        "grupos describen el momento de la descarga.\n"
        "- Yahoo Finance mezcla cifras de distintos cierres contables (últimos doce meses) y no las audita.\n"
        "- El listado de constituyentes del archivo no estaba al día: 17 tickers ya no tenían datos.\n"
        "- Para bancos y aseguradoras, las «ventas» y el valor en libros no significan lo mismo que para "
        "una empresa industrial; sus razones se comparan con cautela.\n"
        "- Faltan variables útiles para un análisis de estilos (crecimiento de ventas, apalancamiento, beta)."
    )

# ==========================================================================
# 2. EXPLORACIÓN
# ==========================================================================
elif pagina == PAGINAS[1]:
    st.title("Exploración de la base")
    crudo = prep["crudo"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas originales", f"{len(crudo)}")
    c2.metric("Columnas originales", f"{crudo.shape[1]}")
    c3.metric("Celdas faltantes", f"{int(crudo.isna().sum().sum())}")
    c4.metric("Tickers duplicados", f"{int(crudo['Symbol'].duplicated().sum())}")

    st.subheader("Estructura del archivo original")
    estructura = pd.DataFrame({
        "Tipo": crudo.dtypes.astype(str),
        "Faltantes": crudo.isna().sum(),
        "% faltante": (crudo.isna().mean() * 100).round(1),
        "Valores distintos": crudo.nunique(),
    })
    st.dataframe(estructura, width="stretch")
    nota(
        "Diecisiete empresas no tienen ningún dato financiero (fila completa vacía) y 26 más no tienen "
        "EBITDA. Las columnas de texto (ticker, nombre, subindustria, enlace a la SEC) identifican a la "
        "empresa pero no miden nada, así que no entran al modelo."
    )

    with st.expander("Ver las primeras filas del archivo original"):
        st.dataframe(crudo.head(20), width="stretch")

    st.subheader("Estadísticas descriptivas de las columnas numéricas originales")
    st.dataframe(crudo.describe().T.style.format(lambda v: num(v, 2)), width="stretch")
    nota(
        "Las escalas son incomparables: la capitalización llega a 5 billones de dólares y el rendimiento "
        "por dividendo no pasa de 0,08. También hay valores que delatan errores o casos extremos: un "
        "P/E de 4.464, un P/B de −565 (patrimonio contable negativo) y un precio de 0,10 USD.",
        "aviso",
    )

    st.subheader("Distribución de las variables construidas")
    var_hist = st.selectbox("Variable", list(variables.columns), format_func=an.nombre)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(variables, x=var_hist, nbins=50, color_discrete_sequence=[AZUL])
        fig.update_traces(marker_line_color=SUPERFICIE, marker_line_width=1)
        fig.update_layout(title=f"{an.nombre(var_hist)} · antes del recorte",
                          xaxis_title=an.nombre(var_hist), yaxis_title="Empresas")
        st.plotly_chart(estilo(fig, 360, False), width="stretch")
    with col_b:
        fig = px.histogram(prep["recortadas"], x=var_hist, nbins=50, color_discrete_sequence=[AZUL_OSCURO])
        fig.update_traces(marker_line_color=SUPERFICIE, marker_line_width=1)
        fig.update_layout(title=f"{an.nombre(var_hist)} · después del recorte del {num(winsor, 1)} %",
                          xaxis_title=an.nombre(var_hist), yaxis_title="Empresas")
        st.plotly_chart(estilo(fig, 360, False), width="stretch")

    st.subheader("Correlación entre variables")
    corr = prep["recortadas"].corr()
    corr.index = [an.nombre(v) for v in corr.index]
    corr.columns = [an.nombre(v) for v in corr.columns]
    st.plotly_chart(mapa_calor(corr, "Correlación de Pearson (variables recortadas)", DIVERGENTE, (-1, 1), alto=560),
                    width="stretch")
    pares = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
    fuertes = pares.reindex(pares.abs().sort_values(ascending=False).index).head(4)
    nota(
        "Pares con mayor correlación absoluta: " + "; ".join(
            f"{a} – {b} ({num(v, 2)})" for (a, b), v in fuertes.items()) +
        ". Las correlaciones son moderadas: hay redundancia que el ACP puede aprovechar, pero ninguna "
        "variable es copia de otra, así que no cabe esperar que dos componentes lo resuman todo."
    )

    st.subheader("Empresas por sector")
    sectores = ident["Sector"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=sectores.values, y=sectores.index, orientation="h", marker_color=AZUL,
                           hovertemplate="%{y}: %{x} empresas<extra></extra>"))
    fig.update_layout(title="Número de empresas por sector GICS (después de la limpieza)",
                      xaxis_title="Empresas", yaxis_title="")
    st.plotly_chart(estilo(fig, 400, False), width="stretch")

# ==========================================================================
# 3. PREPARACIÓN
# ==========================================================================
elif pagina == PAGINAS[2]:
    st.title("Preparación de los datos")
    st.markdown("Cada decisión se aplica en código (`analisis.py`) y queda registrada en esta bitácora.")

    st.subheader("Bitácora de limpieza")
    st.dataframe(prep["bitacora"], hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas originales", f"{len(prep['crudo'])}")
    c2.metric("Filas excluidas", f"{len(prep['crudo']) - len(prep['datos'])}")
    c3.metric("Empresas en el modelo", f"{len(prep['datos'])}")

    st.subheader("Construcción de variables")
    st.markdown(
        "Las ocho variables se calculan a partir de las columnas de Yahoo. Después de construirlas "
        f"quedan **{int(variables.isna().sum().sum())} valores faltantes**: no hace falta imputar."
    )
    with st.expander("Ver la tabla de variables construidas"):
        st.dataframe(pd.concat([ident, variables], axis=1), width="stretch", hide_index=True)

    st.subheader("Tratamiento de valores extremos")
    if winsor > 0:
        resumen = prep["resumen_winsor"].copy()
        resumen.index = [an.nombre(v) for v in resumen.index]
        st.dataframe(resumen.style.format({"Límite inferior": lambda v: num(v, 3),
                                           "Límite superior": lambda v: num(v, 3)}),
                     width="stretch")
        nota(
            f"Con un recorte del {num(winsor, 1)} % por cola, cada variable pierde sus valores más extremos "
            "(se reemplazan por el percentil límite; la empresa no sale de la base). K-Means usa distancias "
            "y medias: un solo P/E de 4.464 movería un centroide entero. Puedes cambiar el nivel en la barra "
            "lateral y ver cómo cambian el ACP y los grupos."
        )
    else:
        nota("Sin recorte: los valores extremos entran tal cual. Revisa cómo cambian la varianza explicada "
             "y los grupos: suele aparecer un grupo de pocas empresas atípicas.", "aviso")

    st.subheader("Estandarización")
    st.markdown(
        "Cada variable se transforma a $z = (x - \\bar{x}) / s$, con media 0 y desviación 1. Sin este paso "
        "el tamaño (en escala logarítmica, rango ≈ 3) y el dividendo (rango ≈ 0,07) aportarían a la "
        "distancia en proporciones arbitrarias."
    )
    largo = z.melt(var_name="Variable", value_name="z")
    largo["Variable"] = largo["Variable"].map(an.nombre)
    fig = px.box(largo, x="z", y="Variable", color_discrete_sequence=[AZUL], points=False)
    fig.update_layout(title="Variables estandarizadas: todas en la misma escala", xaxis_title="Valor z",
                      yaxis_title="")
    st.plotly_chart(estilo(fig, 420, False), width="stretch")
    resumen_z = z.agg(["mean", "std"]).T
    resumen_z.index = [an.nombre(v) for v in resumen_z.index]
    resumen_z.columns = ["Media", "Desviación estándar"]
    st.dataframe(resumen_z.style.format(lambda v: num(v + 0.0, 2)), width="stretch")
    nota(
        "A diferencia del cuaderno de clase, no se aplica `normalize()` por fila después de estandarizar. "
        "Esa operación lleva a cada empresa a la esfera unitaria y borra la magnitud de sus desviaciones: "
        "una empresa extremadamente barata y otra apenas barata quedarían en el mismo punto."
    )

# ==========================================================================
# 4. ACP
# ==========================================================================
elif pagina == PAGINAS[3]:
    st.title("Análisis de Componentes Principales")

    st.subheader("Varianza explicada")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_bar(x=varianza["Componente"], y=varianza["Varianza explicada"] * 100, name="Individual",
                    marker_color=AZUL, hovertemplate="%{x}: %{y:.1f} %<extra>Individual</extra>")
        fig.add_scatter(x=varianza["Componente"], y=varianza["Varianza acumulada"] * 100, name="Acumulada",
                        mode="lines+markers", line=dict(color=AZUL_OSCURO, width=2), marker=dict(size=8),
                        hovertemplate="%{x}: %{y:.1f} %<extra>Acumulada</extra>")
        fig.add_hline(y=80, line_dash="dot", line_color=TINTA_3,
                      annotation_text="80 %", annotation_position="top left")
        fig.update_layout(title="Varianza explicada por componente (%)", yaxis_title="% de la varianza total",
                          xaxis_title="", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
        st.plotly_chart(estilo(fig, 400), width="stretch")
    with col_b:
        fig = go.Figure(go.Scatter(x=varianza["Componente"], y=varianza["Valor propio"], mode="lines+markers",
                                   line=dict(color=AZUL, width=2), marker=dict(size=9),
                                   hovertemplate="%{x}: valor propio %{y:.2f}<extra></extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color=ROJO, annotation_text="Criterio de Kaiser (λ = 1)",
                      annotation_position="top right")
        fig.update_layout(title="Gráfico de sedimentación", yaxis_title="Valor propio (λ)", xaxis_title="")
        st.plotly_chart(estilo(fig, 400, False), width="stretch")

    tabla_var = varianza.copy()
    st.dataframe(tabla_var.style.format({"Valor propio": lambda v: num(v, 3),
                                         "Varianza explicada": lambda v: pct(v, 1),
                                         "Varianza acumulada": lambda v: pct(v, 1)}),
                 hide_index=True, width="stretch")

    st.subheader("¿Cuántos componentes conservar?")
    c1, c2 = st.columns(2)
    c1.metric("Criterio de Kaiser (λ > 1)", f"{sugerencia['kaiser']} componentes")
    c1.caption(f"Conservan el {pct(varianza['Varianza acumulada'].iloc[sugerencia['kaiser'] - 1])} de la varianza")
    c2.metric("Criterio del 80 % acumulado", f"{sugerencia['acumulada']} componentes")
    c2.caption(f"Conservan el {pct(varianza['Varianza acumulada'].iloc[sugerencia['acumulada'] - 1])} de la varianza")
    st.slider("Componentes que pasan a K-Means", 2, n_max, key="comp",
              help="La segmentación se calcula sobre los puntajes de estos componentes.")
    nota(
        f"Con las ocho variables, solo tres componentes tienen valor propio mayor que 1 y explican el 62,6 %. "
        f"Llegar al 80 % exige cinco. El cuarto componente (λ ≈ 0,92) queda muy cerca del umbral, así que "
        f"la frontera no es tajante. Se conservan <b>tres</b>: cada uno resume más que una variable original, "
        f"tienen lectura financiera clara y, en la sección 5, dan grupos más separados y estables que cinco "
        f"(silueta 0,29 frente a 0,20). La correlación moderada entre las variables explica por qué el ACP no "
        f"comprime la base a dos ejes."
        if es_por_defecto else
        f"Ahora usas {n_comp} componentes, que resumen el "
        f"{pct(varianza['Varianza acumulada'].iloc[n_comp - 1])} de la varianza. Compara con los dos criterios."
    )

    st.subheader("Cargas: qué variable define cada componente")
    st.markdown(
        "Cada celda es la **correlación** entre una variable original y un componente "
        "(carga × √λ). Valores cerca de ±1 indican que esa variable define el eje."
    )
    corr_cp = acp["correlaciones"].iloc[:, :max(n_comp, 3)].copy()
    corr_cp.index = [an.nombre(v) for v in corr_cp.index]
    st.plotly_chart(mapa_calor(corr_cp, "Correlación entre variables y componentes", DIVERGENTE, (-1, 1), alto=460),
                    width="stretch")

    for i in range(n_comp):
        cp = f"CP{i + 1}"
        serie = acp["correlaciones"][cp].sort_values()
        pos = [f"{an.nombre(v)} ({num(serie[v], 2)})" for v in serie.index[::-1][:2] if serie[v] > 0.3]
        neg = [f"{an.nombre(v)} ({num(serie[v], 2)})" for v in serie.index[:2] if serie[v] < -0.3]
        st.markdown(
            f"**{cp}** · {pct(varianza['Varianza explicada'].iloc[i])} de la varianza. "
            f"Polo positivo: {', '.join(pos) or '—'}. Polo negativo: {', '.join(neg) or '—'}."
        )
    if es_por_defecto:
        nota(
            "<b>Lectura.</b> <b>CP1 · valor frente a calidad.</b> En el polo positivo están las empresas "
            "baratas frente a sus ventas y su patrimonio; en el negativo, las grandes y de margen alto. Es "
            "el eje clásico valor–crecimiento. <b>CP2 · riesgo frente a ingreso.</b> Opone la amplitud del "
            "rango anual (volatilidad) al dividendo, al libro a mercado y al rendimiento de utilidades: "
            "acciones que se mueven mucho frente a acciones que pagan. <b>CP3 · momento.</b> Lo define la "
            "posición dentro del rango de 52 semanas, contra el rendimiento de utilidades y la amplitud del rango: separa "
            "a las que cotizan cerca de su máximo de las que vienen cayendo."
        )

    st.subheader("Las empresas en el espacio de componentes")
    col_x, col_y, col_s = st.columns(3)
    opciones_cp = list(varianza["Componente"])
    eje_x = col_x.selectbox("Eje horizontal", opciones_cp, 0)
    eje_y = col_y.selectbox("Eje vertical", opciones_cp, 1)
    sector_dest = col_s.selectbox("Destacar un sector", ["Ninguno"] + sorted(ident["Sector"].unique()))
    puntos = pd.concat([ident, acp["puntajes"]], axis=1)
    resaltado = puntos["Sector"].eq(sector_dest)
    fig = go.Figure()
    fig.add_scatter(x=puntos.loc[~resaltado, eje_x], y=puntos.loc[~resaltado, eje_y], mode="markers",
                    marker=dict(size=8, color="#9ec5f4" if sector_dest == "Ninguno" else "#d6d5d0",
                                line=dict(color=SUPERFICIE, width=1)),
                    name="Resto" if sector_dest != "Ninguno" else "Empresas",
                    customdata=puntos.loc[~resaltado, ["Ticker", "Empresa", "Sector"]],
                    hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    if sector_dest != "Ninguno":
        fig.add_scatter(x=puntos.loc[resaltado, eje_x], y=puntos.loc[resaltado, eje_y], mode="markers",
                        marker=dict(size=10, color=AZUL_OSCURO, line=dict(color=SUPERFICIE, width=1.5)),
                        name=sector_dest, customdata=puntos.loc[resaltado, ["Ticker", "Empresa", "Sector"]],
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    # Vectores de carga (biplot), escalados para que se vean sobre la nube.
    escala = 0.8 * min(puntos[eje_x].abs().max(), puntos[eje_y].abs().max())
    for v in acp["correlaciones"].index:
        dx, dy = acp["correlaciones"].loc[v, eje_x] * escala, acp["correlaciones"].loc[v, eje_y] * escala
        fig.add_annotation(x=dx, y=dy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.5, arrowcolor=TINTA_2, text="")
        fig.add_annotation(x=dx * 1.12, y=dy * 1.12, text=an.nombre(v).split(" (")[0], showarrow=False,
                           font=dict(size=11, color=TINTA), bgcolor="rgba(252,252,251,0.85)")
    i_x, i_y = opciones_cp.index(eje_x), opciones_cp.index(eje_y)
    resumida = varianza["Varianza explicada"].iloc[i_x] + (varianza["Varianza explicada"].iloc[i_y] if i_y != i_x else 0)
    fig.update_layout(title=f"{eje_x} frente a {eje_y} · este plano resume el {pct(resumida)} de la varianza",
                      xaxis_title=f"{eje_x} ({pct(varianza['Varianza explicada'].iloc[i_x])})",
                      yaxis_title=f"{eje_y} ({pct(varianza['Varianza explicada'].iloc[i_y])})",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(estilo(fig, 600), width="stretch")
    nota(
        "Las flechas muestran la correlación de cada variable con los dos ejes (biplot): una empresa "
        "ubicada en la dirección de una flecha tiene un valor alto en esa variable. Pasa el cursor sobre "
        "un punto para ver la empresa. Este plano no muestra toda la información: lo que queda en los demás "
        "componentes no se ve aquí."
    )

# ==========================================================================
# 5. K-MEANS
# ==========================================================================
elif pagina == PAGINAS[4]:
    st.title("Segmentación con K-Means")
    st.markdown(
        f"K-Means corre sobre los puntajes de los **{n_comp} primeros componentes** "
        f"({pct(varianza['Varianza acumulada'].iloc[n_comp - 1])} de la varianza), con 20 inicios aleatorios "
        f"y semilla {semilla}. Los componentes ya están en una escala comparable y no están correlacionados."
    )

    st.subheader("Comparación de valores de K")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Scatter(x=evaluacion["K"], y=evaluacion["Inercia"], mode="lines+markers",
                                   line=dict(color=AZUL, width=2), marker=dict(size=9),
                                   hovertemplate="K = %{x}: inercia %{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Método del codo", xaxis_title="Número de grupos (K)",
                          yaxis_title="Inercia (suma de distancias²)")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(estilo(fig, 380, False), width="stretch")
    with col_b:
        mejor = evaluacion.loc[evaluacion["Silueta"].idxmax()]
        fig = go.Figure(go.Scatter(x=evaluacion["K"], y=evaluacion["Silueta"], mode="lines+markers",
                                   line=dict(color=AZUL_OSCURO, width=2), marker=dict(size=9),
                                   hovertemplate="K = %{x}: silueta %{y:.3f}<extra></extra>"))
        fig.add_annotation(x=mejor["K"], y=mejor["Silueta"], text=f"máximo: K = {int(mejor['K'])}",
                           showarrow=True, arrowhead=2, ay=-35, font=dict(color=TINTA))
        fig.update_layout(title="Coeficiente de silueta promedio", xaxis_title="Número de grupos (K)",
                          yaxis_title="Silueta promedio")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(estilo(fig, 380, False), width="stretch")

    tabla_k = evaluacion.copy()
    tabla_k["Caída de inercia"] = -tabla_k["Inercia"].pct_change()
    st.dataframe(tabla_k.style.format({"Inercia": lambda v: num(v, 1), "Silueta": lambda v: num(v, 3),
                                       "Caída de inercia": lambda v: pct(v, 1)}, na_rep="—"),
                 hide_index=True, width="stretch")

    st.slider("Número de grupos (K)", 2, 10, key="k_elegido")
    estab_media, estab_min = correr_estabilidad(matriz, k)
    c1, c2, c3 = st.columns(3)
    c1.metric("Silueta promedio", num(evaluacion.set_index("K").loc[k, "Silueta"], 3))
    c2.metric("Estabilidad (ARI medio, 10 semillas)", num(estab_media, 2))
    c3.metric("Empresas con silueta negativa", f"{int((km['silueta'] < 0).sum())}")
    if es_por_defecto:
        nota(
            "<b>Elección de K = 4.</b> La silueta alcanza su máximo en K = 4 (0,290), por encima de K = 2 "
            "(0,268) y K = 3 (0,277). En el codo, la inercia cae 22 % al pasar de 2 a 3 grupos y 18 % de 3 a 4; "
            "a partir de 5 cada grupo adicional aporta menos. Además, la solución con K = 4 es estable: al "
            "repetir K-Means con otras semillas, el índice de Rand ajustado promedio es 0,95. Con K = 3 baja "
            "a 0,70 y en alguna semilla cae a 0,24: los grupos cambian según dónde arranque el algoritmo. "
            "Una silueta de 0,29 indica una estructura <b>débil a moderada</b>: hay grupos, pero con "
            "fronteras que se tocan."
        )
    else:
        nota("La silueta mide qué tan cerca está cada empresa de su grupo frente al grupo vecino (de −1 a 1). "
             "El ARI compara la asignación con la de otras semillas: 1 = idéntica.")

    st.subheader("Los grupos en el espacio de componentes")
    col_x, col_y = st.columns(2)
    opciones = list(matriz.columns)
    eje_x = col_x.selectbox("Eje horizontal", opciones, 0, key="km_x")
    eje_y = col_y.selectbox("Eje vertical", opciones, 1, key="km_y")
    puntos = pd.concat([ident, matriz, grupo], axis=1)
    fig = go.Figure()
    for g in letras:
        sub = puntos[puntos["Grupo"] == g]
        fig.add_scatter(x=sub[eje_x], y=sub[eje_y], mode="markers", name=f"Grupo {g} ({len(sub)})",
                        marker=dict(size=9, color=color_grupo[g], symbol=simbolo_grupo[g],
                                    line=dict(color=SUPERFICIE, width=1), opacity=0.85),
                        customdata=sub[["Ticker", "Empresa", "Sector"]],
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}"
                                      f"<extra>Grupo {g}</extra>")
    # Centroides: media de cada grupo en los componentes.
    medias = matriz.groupby(grupo).mean()
    fig.add_scatter(x=medias[eje_x], y=medias[eje_y], mode="markers+text", name="Centroides",
                    marker=dict(size=16, color=TINTA, symbol="x-thin", line=dict(width=3, color=TINTA)),
                    text=[f"  {g}" for g in medias.index], textposition="middle right",
                    textfont=dict(size=14, color=TINTA), hovertemplate="Centroide %{text}<extra></extra>")
    i_x, i_y = opciones.index(eje_x), opciones.index(eje_y)
    resumida = varianza["Varianza explicada"].iloc[i_x] + (varianza["Varianza explicada"].iloc[i_y] if i_y != i_x else 0)
    fig.update_layout(title=f"Grupos sobre {eje_x} y {eje_y} · el plano resume el {pct(resumida)} de la varianza",
                      xaxis_title=eje_x, yaxis_title=eje_y)
    st.plotly_chart(estilo(fig, 580), width="stretch")
    nota("Las letras solo nombran a los grupos (A es el más grande); no indican orden ni calidad. "
         "Los grupos se forman con todos los componentes seleccionados, así que en un plano de dos ejes "
         "pueden verse superpuestos aunque estén separados en una tercera dimensión.")

    st.subheader("Silueta de cada empresa")
    sil = pd.concat([ident, grupo, km["silueta"]], axis=1).sort_values(["Grupo", "Silueta"], ascending=[True, False])
    sil["Orden"] = range(len(sil))
    fig = go.Figure()
    for g in letras:
        sub = sil[sil["Grupo"] == g]
        fig.add_bar(x=sub["Silueta"], y=sub["Orden"], orientation="h", name=f"Grupo {g}",
                    marker=dict(color=color_grupo[g], line_width=0), customdata=sub[["Ticker", "Empresa"]],
                    hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>silueta %{x:.2f}"
                                  f"<extra>Grupo {g}</extra>")
    fig.add_vline(x=km["silueta"].mean(), line_dash="dot", line_color=TINTA_2,
                  annotation_text=f"promedio {num(km['silueta'].mean(), 2)}")
    fig.update_layout(title="Coeficiente de silueta por empresa, ordenado dentro de cada grupo", bargap=0,
                      xaxis_title="Silueta", yaxis=dict(showticklabels=False, title="", autorange="reversed"),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(estilo(fig, 520), width="stretch")
    resumen_sil = pd.DataFrame({
        "Empresas": grupo.value_counts().sort_index(),
        "Silueta media": km["silueta"].groupby(grupo).mean(),
        "Silueta negativa": (km["silueta"] < 0).groupby(grupo).sum(),
    })
    st.dataframe(resumen_sil.style.format({"Silueta media": lambda v: num(v, 3)}), width="stretch")
    dudosas = sil[sil["Silueta"] < 0].sort_values("Silueta")[["Ticker", "Empresa", "Sector", "Grupo", "Silueta"]]
    with st.expander(f"Empresas difíciles de asignar (silueta negativa: {len(dudosas)})"):
        st.dataframe(dudosas.style.format({"Silueta": lambda v: num(v, 3)}), hide_index=True,
                     width="stretch")
        st.caption("Una silueta negativa indica que la empresa está, en promedio, más cerca de otro grupo "
                   "que del suyo: es un caso de frontera.")

# ==========================================================================
# 6. PERFILES
# ==========================================================================
elif pagina == PAGINAS[5]:
    st.title("Perfil de los grupos")
    st.markdown("Los grupos se describen con las **variables originales**, no con los componentes.")

    st.subheader("¿En qué se aleja cada grupo del promedio?")
    pz = an.perfiles_z(z, grupo)
    pz.index = [an.nombre(v) for v in pz.index]
    pz.columns = [f"Grupo {g}" for g in pz.columns]
    st.plotly_chart(mapa_calor(pz, "Media estandarizada de cada variable por grupo (0 = promedio del S&P 500)",
                               DIVERGENTE, (-2, 2), alto=460), width="stretch")
    nota("Azul: el grupo está por debajo del promedio en esa variable; rojo: por encima. Un valor de 1 "
         "equivale a una desviación estándar.")

    st.subheader("Valores típicos en unidades originales (mediana)")
    med = an.perfiles(variables, grupo)
    legible = pd.DataFrame(index=med.columns)
    if "tamano_log" in med.index:
        legible["Capitalización (miles de millones USD)"] = [num(10 ** v / 1e9, 1) for v in med.loc["tamano_log"]]
    if "rend_utilidades" in med.index:
        legible["P/E equivalente"] = [num(1 / v, 1) if v > 0 else "pérdidas" for v in med.loc["rend_utilidades"]]
    for v, etiqueta in [("rend_utilidades", "Rend. de utilidades"), ("libro_mercado", "Libro a mercado"),
                        ("ventas_precio", "Ventas a precio"), ("rend_dividendo", "Rend. por dividendo"),
                        ("margen_neto", "Margen neto"), ("rango_52s", "Amplitud 52 semanas"),
                        ("posicion_52s", "Posición en el rango")]:
        if v in med.index:
            legible[etiqueta] = [pct(x, 1) if v not in ("libro_mercado", "ventas_precio") else num(x, 2)
                                 for x in med.loc[v]]
    legible.insert(0, "Empresas", list(grupo.value_counts().sort_index().values) + [len(grupo)])
    legible.index = [f"Grupo {g}" if g != "Total" else "Todo el índice" for g in legible.index]
    st.dataframe(legible, width="stretch")

    st.subheader("Descripción automática")
    for g in letras:
        n_g = int((grupo == g).sum())
        st.markdown(f"- **Grupo {g}** ({n_g} empresas, {pct(n_g / len(grupo))}): {descriptores(g)}.")

    if es_por_defecto:
        st.subheader("Interpretación (configuración por defecto)")
        st.markdown(
            "- **A · Grandes de calidad y momento (209).** Las empresas más grandes (mediana de 74 mil "
            "millones), caras frente a su patrimonio y sus ventas, con márgenes altos y cotizando cerca de su "
            "máximo anual (posición 79 %). Aquí están Apple, Microsoft, Nvidia, Alphabet, Amazon, Meta y Lilly. "
            "Es el perfil de *crecimiento/calidad* de gran capitalización.\n"
            "- **B · Valor y dividendo defensivo (106).** El dividendo más alto (3,2 %), las más baratas "
            "frente a utilidades y patrimonio, y el rango anual más corto (26 %): se mueven poco. Reúne al "
            "84 % de los servicios públicos, la mitad del sector financiero y del inmobiliario, y telecomunicaciones como "
            "Verizon y AT&T. Es el perfil *valor–ingreso*.\n"
            "- **C · Alta volatilidad, lejos del máximo (105).** El mayor recorrido de precio en el año "
            "(53 %), cotizando en la parte baja de su rango (27 %) y casi sin dividendo. Concentra al 57 % de "
            "la tecnología, sobre todo semiconductores (AMD, Micron, Intel, Applied Materials) e industriales "
            "cíclicas. Es el perfil *cíclico/especulativo* en el momento del corte.\n"
            "- **D · Volumen con margen estrecho (62).** Generan muchas ventas por dólar de capitalización "
            "(ventas a precio 1,5) pero con un margen neto mediano de 1,3 % y un rendimiento de utilidades "
            "bajo. Son distribuidoras farmacéuticas (McKesson, Cencora, Cardinal), aseguradoras de salud "
            "(CVS, Elevance, Cigna, Humana), minoristas y automotrices (GM, Ford): baratas por ventas, no por "
            "utilidades."
        )

    st.subheader("¿Los grupos repiten los sectores?")
    comp = an.composicion_sectorial(ident["Sector"], grupo)
    comp.columns = [f"Grupo {g}" for g in comp.columns]
    st.plotly_chart(mapa_calor(comp, "Porcentaje de las empresas de cada sector que cae en cada grupo",
                               SECUENCIAL, (0, 1), formato="%", alto=480), width="stretch")
    dominante = comp.max(axis=1).sort_values(ascending=False)
    nota(
        f"Solo {int((dominante > 0.6).sum())} de {len(dominante)} sectores tiene más del 60 % de sus empresas "
        f"en un mismo grupo ({', '.join(dominante[dominante > 0.6].index) or 'ninguno'}). El resto se reparte: "
        "los grupos cruzan los sectores. El estilo financiero no es lo mismo que la industria; los sectores "
        "más regulados y estables (servicios públicos) son la excepción."
    )

    st.subheader("Empresas de cada grupo")
    g_sel = st.selectbox("Grupo", letras, format_func=lambda g: f"Grupo {g}")
    tabla = pd.concat([ident, grupo, km["silueta"]], axis=1)
    tabla["Capitalización (miles de millones USD)"] = capitalizacion / 1e9
    tabla = tabla[tabla["Grupo"] == g_sel].sort_values("Capitalización (miles de millones USD)", ascending=False)
    st.dataframe(tabla.drop(columns="Grupo").style.format({"Silueta": lambda v: num(v, 2),
                                                           "Capitalización (miles de millones USD)": lambda v: num(v, 1)}),
                 hide_index=True, width="stretch", height=360)

    st.subheader("Buscar una empresa")
    opciones = (ident["Ticker"] + " · " + ident["Empresa"]).tolist()
    elegida = st.selectbox("Empresa", opciones, index=opciones.index("JPM · JPMorgan Chase")
                           if "JPM · JPMorgan Chase" in opciones else 0)
    i = opciones.index(elegida)
    c1, c2, c3 = st.columns(3)
    c1.metric("Grupo", grupo.iloc[i])
    c2.metric("Sector", ident["Sector"].iloc[i])
    c3.metric("Silueta", num(km["silueta"].iloc[i], 2))
    comparacion = pd.DataFrame({
        "Empresa": variables.iloc[i],
        f"Mediana grupo {grupo.iloc[i]}": variables[grupo == grupo.iloc[i]].median(),
        "Mediana del índice": variables.median(),
    })
    comparacion.index = [an.nombre(v) for v in comparacion.index]
    st.dataframe(comparacion.style.format(lambda v: num(v, 3)), width="stretch")

# ==========================================================================
# 7. CONCLUSIONES
# ==========================================================================
else:
    st.title("Conclusiones")
    if not es_por_defecto:
        nota("Estás viendo una configuración distinta a la del informe. El texto de abajo describe la "
             "configuración por defecto (8 variables, recorte del 1 %, 3 componentes, K = 4, semilla 42); "
             "el resumen de tu corrida aparece primero.", "aviso")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables", f"{len(seleccion)}")
        c2.metric("Varianza conservada", pct(varianza["Varianza acumulada"].iloc[n_comp - 1]))
        c3.metric("K", f"{k}")
        c4.metric("Silueta", num(evaluacion.set_index("K").loc[k, "Silueta"], 3))
        for g in letras:
            st.markdown(f"- **Grupo {g}** ({int((grupo == g).sum())} empresas): {descriptores(g)}.")
        st.divider()

    st.subheader("1. ¿Qué dimensiones resumen a las empresas del S&P 500?")
    st.markdown(
        "Tres componentes con valor propio mayor que 1 conservan el **62,6 %** de la variación de las ocho "
        "variables. Tienen lectura financiera directa: **valor frente a calidad** (CP1, 27,4 %), **riesgo "
        "frente a ingreso** (CP2, 19,4 %) y **momento** (CP3, 15,8 %). Son, en esencia, los factores con los "
        "que la literatura de inversión describe a las acciones, y salieron de los datos sin imponerlos. El "
        "tamaño no forma un eje propio: se une a la rentabilidad en el polo de calidad de CP1."
    )
    st.subheader("2. ¿Hay grupos de empresas con perfil parecido?")
    st.markdown(
        "Sí, pero con fronteras difusas. K = 4 maximiza la silueta (0,29) y es muy estable entre semillas "
        "(ARI 0,95). Los cuatro perfiles son reconocibles para un analista: **grandes de calidad y momento**, "
        "**valor y dividendo defensivo**, **alta volatilidad lejos del máximo** y **volumen con margen "
        "estrecho**. Una silueta de 0,29 dice que las empresas forman un continuo más que islas separadas; "
        "16 empresas tienen silueta negativa y quedan en la frontera, sobre todo entre los grupos C y D, que "
        "son los menos compactos."
    )
    st.subheader("3. ¿Los grupos repiten los sectores?")
    st.markdown(
        "No del todo. Los servicios públicos caen casi completos en el grupo de dividendo (84 %) y la "
        "tecnología se parte entre las grandes de calidad y las volátiles. Pero la mayoría de los sectores "
        "se reparte entre tres o cuatro grupos: hay bancos de valor y bancos de calidad, y empresas de salud "
        "de margen alto (Lilly) junto a aseguradoras de margen estrecho (CVS). Para quien arma un portafolio, "
        "diversificar por sector no garantiza diversificar por estilo."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- **Es una fotografía.** La posición y el rango de 52 semanas cambian a diario: el grupo C "
        "describe a las empresas que venían cayendo en la fecha de descarga, no un rasgo permanente.\n"
        "- **Se pierde información.** Los tres componentes dejan fuera el 37 % de la variación; el cuarto "
        "(λ = 0,92) está cerca del umbral.\n"
        "- **K-Means asume grupos esféricos y de tamaño parecido.** Con una silueta moderada, otras "
        "técnicas (jerárquico, mezclas gaussianas) podrían dar fronteras distintas.\n"
        "- **Las razones no son comparables entre todos los sectores.** Para bancos, el valor en libros y "
        "las ventas tienen otro significado; una versión sin financieras sería un buen contraste.\n"
        "- **Faltan variables** como crecimiento de ventas, apalancamiento o beta, que ayudarían a separar "
        "mejor crecimiento de calidad."
    )
    st.subheader("Uso posible")
    st.markdown(
        "La segmentación sirve como punto de partida para revisar la diversificación de un portafolio por "
        "estilo y no solo por sector, o para encontrar empresas comparables en un ejercicio de valoración por "
        "múltiplos: un comparable debería venir del mismo grupo, no solo de la misma industria."
    )

    st.subheader("Descargar resultados")
    salida = pd.concat([ident, prep["datos"][["Price", "Market Cap"]].rename(
        columns={"Price": "Precio", "Market Cap": "Capitalizacion"}), variables, matriz, grupo, km["silueta"]], axis=1)
    st.download_button("Descargar empresas con su grupo (CSV)", salida.to_csv(index=False).encode("utf-8"),
                       file_name="sp500_grupos.csv", mime="text/csv")
    st.caption(f"{len(salida)} empresas · variables originales, puntajes de componentes, grupo y silueta.")
