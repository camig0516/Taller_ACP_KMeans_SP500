# Estilos de inversión en el S&P 500 · ACP y K-Means

Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali
Autora: Camila Andrea Granja Castro

- **Aplicación:** https://estilos-sp500-acp-kmeans.streamlit.app/
- **Video de sustentación:** `<enlace al video>`

## Contexto y pregunta

Los gestores de portafolios clasifican las acciones por **estilos**: *valor* (baratas frente a sus utilidades o su patrimonio), *crecimiento* (caras, con márgenes altos), *dividendo* (pagan mucho y se mueven poco) o *cíclicas* (con grandes recorridos de precio). Esas etiquetas suelen asignarse con reglas fijas. Este proyecto pregunta si los datos, por sí solos, reproducen esa clasificación:

1. ¿Cuántas dimensiones resumen las diferencias financieras entre las empresas del S&P 500?
2. ¿Existen grupos de empresas con un perfil parecido? ¿Qué los define?
3. ¿Esos grupos coinciden con los sectores económicos o los cruzan?

**Por qué ACP y K-Means.** Las razones de valoración están correlacionadas entre sí (una empresa barata frente a sus ventas suele serlo frente a su patrimonio). El ACP combina las variables que se mueven juntas en ejes no correlacionados, mide cuánta información se conserva y evita que una misma idea pese varias veces en las distancias. No existe una etiqueta de «estilo» verdadera para cada empresa, así que el problema es no supervisado: K-Means agrupa a las empresas cercanas en el espacio de componentes y entrega centroides que se leen como perfiles. El sector GICS no entra al modelo; se usa al final como contraste.

## Datos

| | |
|---|---|
| Base | *S&P 500 Companies with Financial Information* ([datahub.io / GitHub](https://github.com/datasets/s-and-p-500-companies-financials)), versión mantenida del conjunto que también circula en Kaggle |
| Origen de las cifras | Yahoo Finance vía `yfinance`, actualizado automáticamente por el repositorio |
| Sector | Listado de constituyentes del S&P 500 ([GitHub](https://github.com/datasets/s-and-p-500-companies), a partir de Wikipedia) |
| Licencia | ODC-PDDL (dominio público) |
| Corte | Descargado el 17 de noviembre de 2026 (copia en `data/`) |
| Cobertura | 503 tickers originales; 482 empresas después de la limpieza |

**Variables del modelo** (construidas en `analisis.py`):

| Variable | Cálculo | Dimensión |
|---|---|---|
| Tamaño | log10(capitalización) | Tamaño |
| Rendimiento de utilidades (E/P) | UPA / precio | Valoración |
| Libro a mercado (B/P) | 1 / (P/B) | Valoración |
| Ventas a precio (S/P) | 1 / (P/S) | Valoración |
| Rendimiento por dividendo | dividendo / precio | Ingreso |
| Margen neto | (E/P) × (P/S) | Rentabilidad |
| Amplitud del rango de 52 semanas | (máximo − mínimo) / máximo | Riesgo |
| Posición en el rango de 52 semanas | (precio − mínimo) / (máximo − mínimo) | Momento |

Las razones de valoración se usan invertidas: una empresa con pérdidas da un número negativo y continuo, en lugar de un P/E que se dispara cuando la utilidad se acerca a cero.

**Limitaciones de la fuente.** Es una fotografía (el precio y el rango de 52 semanas cambian a diario); Yahoo mezcla cierres contables y no audita las cifras; el listado de constituyentes no estaba al día; las razones de bancos no significan lo mismo que las de una industrial; faltan variables como crecimiento, apalancamiento o beta.

## Metodología

1. **Limpieza** (bitácora completa en la aplicación, sección 3):
   - 17 empresas sin datos en Yahoo (adquiridas, retiradas o con tickers con punto como BRK.B) → excluidas.
   - PARA con datos residuales tras la fusión de Paramount (precio de 0,10 USD) → excluida.
   - GOOG, FOX y NWS duplican a GOOGL, FOXA y NWSA → se conserva una clase por empresa.
   - EBITDA falta en 26 empresas financieras → no se usa; se reemplaza por el margen neto.
   - Identificadores y texto fuera del modelo. Después de construir las variables no quedan faltantes.
2. **Valores extremos:** winsorización al 1 % por cola (ajustable en la app).
3. **Estandarización:** z-score. No se aplica `normalize()` por fila, porque borraría la magnitud de las desviaciones.
4. **ACP:** todos los componentes; se conservan los que cumplen Kaiser (λ > 1) y se contrasta con el 80 % acumulado.
5. **K-Means** sobre los puntajes de los componentes conservados: K de 2 a 10, método del codo, silueta y estabilidad entre semillas (índice de Rand ajustado). 20 inicios aleatorios, semilla 42.
6. **Perfiles** con las variables originales (medianas y medias estandarizadas) y composición sectorial.

## Resultados principales

**ACP.** Tres componentes tienen λ > 1 y conservan el **62,6 %** de la varianza (el 80 % exige cinco; el cuarto tiene λ = 0,92):

| Componente | Varianza | Lectura |
|---|---|---|
| CP1 | 27,4 % | **Valor frente a calidad:** baratas por ventas y patrimonio (+) frente a grandes y de margen alto (−) |
| CP2 | 19,4 % | **Riesgo frente a ingreso:** amplitud del rango anual (+) frente a dividendo y valoración baja (−) |
| CP3 | 15,8 % | **Momento:** cerca del máximo anual (+) frente a rendimiento de utilidades alto y rango amplio (−) |

**K-Means.** K = 4 maximiza la silueta (0,290; K = 3 da 0,277) y es estable (ARI medio 0,95; con K = 3 cae a 0,70). 16 empresas tienen silueta negativa: la estructura es débil a moderada.

| Grupo | Empresas | Perfil | Ejemplos |
|---|---:|---|---|
| A | 209 | Grandes de calidad y momento: capitalización mediana de 74 mil millones, margen alto, cerca del máximo anual | Apple, Microsoft, Nvidia, Alphabet, Lilly |
| B | 106 | Valor y dividendo defensivo: dividendo de 3,2 %, rango anual de 26 % | Bank of America, Verizon, NextEra, AT&T |
| C | 105 | Alta volatilidad, lejos del máximo: rango anual de 53 %, casi sin dividendo | AMD, Micron, Intel, Applied Materials |
| D | 62 | Volumen con margen estrecho: ventas a precio de 1,5, margen neto de 1,3 % | McKesson, CVS, Elevance, GM |

Las letras solo nombran a los grupos (A es el más grande); no indican orden.

**Grupos frente a sectores.** Solo los servicios públicos concentran más del 60 % de sus empresas en un grupo (84 % en B). La tecnología se parte entre A (40 %) y C (57 %); el sector financiero, entre A y B. Diversificar por sector no garantiza diversificar por estilo.

## Estructura del repositorio

```
├── app.py                      Aplicación de Streamlit (7 secciones)
├── analisis.py                 Carga, limpieza, variables, ACP y K-Means (lo usan la app y el cuaderno)
├── preparar_datos.py           Descarga de nuevo los archivos crudos desde GitHub
├── data/
│   ├── constituents-financials.csv   Cifras financieras (Yahoo Finance)
│   └── constituents.csv              Listado y sector GICS
├── notebooks/analisis_sp500.ipynb    El mismo análisis paso a paso, con salidas
├── requirements.txt
└── .streamlit/config.toml      Tema visual
```

## Cómo ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`. Para el cuaderno: `pip install notebook matplotlib` y abrir `notebooks/analisis_sp500.ipynb`.

Los datos de `data/` reproducen exactamente los resultados del informe. Para trabajar con la versión más reciente: `python preparar_datos.py` (las cifras y los grupos cambiarán).

**Despliegue:** en [share.streamlit.io](https://share.streamlit.io) → *Create app* → repositorio público → rama `main` → archivo `app.py`.
