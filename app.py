"""Dashboard interactivo: alerta temprana de bajo puntaje en Saber 11."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from preparar_datos import DEPARTAMENTOS

ROOT = Path(__file__).resolve().parent
RES = pd.read_csv(ROOT / "data" / "resumen_tablero.csv", dtype={"cod_depto": str})
DIST = pd.read_csv(ROOT / "data" / "distribucion_puntaje.csv", dtype={"cod_depto": str})
GEO = json.loads((ROOT / "data" / "colombia_departamentos.geojson").read_text(encoding="utf-8"))
PRED = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv", dtype={"periodo": str, "cod_depto": str})
MODELS = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
IMPORTANCE = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv").head(12)
CORTE_INFO = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))
CORTE = CORTE_INFO["corte_puntaje_global"]
ES_MUESTRA = "muestra" in CORTE_INFO.get("fuente", "muestra")
FUENTE = (f"muestra aleatoria de {CORTE_INFO.get('estudiantes', 0):,} estudiantes" if ES_MUESTRA
          else f"los {CORTE_INFO.get('estudiantes', 0):,} estudiantes del archivo completo").replace(",", ".")

# Colores: una serie = azul; varias series = azul, dorado, verde azulado en ese orden fijo.
AZUL, DORADO, VERDE, GRIS, NAVY = "#1976D2", "#E0A21B", "#26A6A1", "#8A9BAA", "#12304A"
SECUENCIAL = ["#E3F0FB", "#7FB3E6", "#1976D2", "#0D3B66"]
COLOR_NATURALEZA = {"Oficial": AZUL, "No Oficial": DORADO, "Sin dato": GRIS}
COLOR_SEMESTRE = {"2.º semestre": AZUL, "1.er semestre": DORADO}
MIN_ESTUDIANTES = 30  # grupos mas pequenos se ocultan en las barras para no exagerar casos aislados
ANIOS = sorted(int(a) for a in RES.anio.unique())
ORDEN = {
    "Estrato": ["Sin estrato"] + [f"Estrato {i}" for i in range(1, 7)],
    "Educación": ["Ninguno", "Primaria incompleta", "Primaria completa", "Secundaria incompleta", "Secundaria completa",
                  "Técnica incompleta", "Técnica completa", "Profesional incompleta", "Profesional completa", "Postgrado"],
    "Personas por cuarto": ["Hasta 1", "Más de 1 a 2", "Más de 2 a 3", "Más de 3"],
}
NOMBRES = {"departamento": "Departamento", "jornada": "Jornada", "edad": "Edad", "anio": "Año", "cuartos_hogar": "Cuartos del hogar",
           "area": "Zona (urbana/rural)", "educacion_madre": "Educación madre", "personas_por_cuarto": "Personas por cuarto",
           "educacion_padre": "Educación padre", "genero_colegio": "Género del colegio", "calendario": "Calendario",
           "estrato": "Estrato", "naturaleza": "Tipo de colegio", "tiene_computador": "Tiene computador",
           "tiene_internet": "Tiene internet", "genero": "Género", "educacion_max_padres": "Educación máx. padres",
           "bilingue": "Colegio bilingüe", "personas_hogar": "Personas en el hogar", "tiene_automovil": "Tiene automóvil",
           "tiene_lavadora": "Tiene lavadora", "bienes_hogar": "Bienes del hogar (0-4)", "caracter": "Carácter del colegio"}
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Segoe UI, Arial", color="#22313F"), separators=",.",
              margin=dict(l=10, r=10, t=60, b=10), hoverlabel=dict(bgcolor="white"))

app = Dash(__name__, title="Alerta temprana Saber 11", suppress_callback_exceptions=True)
server = app.server


# ---------- utilidades ----------
def num(v):
    return f"{v:,.0f}".replace(",", ".")


def dec(v):
    return f"{v:.2f}".replace(".", ",")


def pct(v, decimales=1):
    return f"{v:.{decimales}%}".replace(".", ",")


def card(title, value_id, note):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(note, className="kpi-note")], className="kpi-card")


def filtrar(df, deptos, nats, areas, cals, anios):
    m = df["anio"].between(anios[0], anios[1])
    for col, vals in [("cod_depto", deptos), ("naturaleza", nats), ("area", areas), ("calendario", cals)]:
        if vals:
            m &= df[col].isin(vals)
    return df[m]


def agrupar(r, cols):
    g = r.groupby(cols, as_index=False)[["estudiantes", "bajo", "suma_puntaje"]].sum()
    g["tasa"] = g.bajo / g.estudiantes
    g["promedio"] = g.suma_puntaje / g.estudiantes
    return g


def cuartiles(tramos, conteos, ancho=5):
    """Cuartiles aproximados (error de alrededor de 1 punto) a partir de cuantos estudiantes hay en cada tramo de puntaje.
    Dentro de cada tramo se supone que los puntajes estan repartidos de forma pareja."""
    orden = np.argsort(tramos)
    t, c = np.asarray(tramos)[orden], np.asarray(conteos, float)[orden]
    n, acum = c.sum(), np.cumsum(c)

    def q(p):
        k = p * n
        i = min(int(np.searchsorted(acum, k)), len(c) - 1)
        previo = acum[i - 1] if i > 0 else 0
        return t[i] + ancho * (k - previo) / c[i]

    q1, med, q3 = q(0.25), q(0.5), q(0.75)
    riq = q3 - q1  # rango intercuartil: el ancho de la caja
    return {"q1": q1, "mediana": med, "q3": q3, "inferior": max(t[0], q1 - 1.5 * riq),
            "superior": min(t[-1] + ancho - 1, q3 + 1.5 * riq), "estudiantes": int(n)}


def vacia(titulo):
    fig = go.Figure()
    fig.add_annotation(text="No hay datos con estos filtros", showarrow=False, font=dict(size=14, color=GRIS))
    fig.update_layout(**LAYOUT, title=titulo, xaxis_visible=False, yaxis_visible=False, height=380)
    return fig


HOVER = "<b>%{x}</b><br>Bajo puntaje: %{y:.1%}<br>Estudiantes: %{customdata[0]:,}<br>Puntaje promedio: %{customdata[1]:.0f}<extra></extra>"


def barras(g, x, titulo, tasa_total, color=None, orden=None, colores=None, texto=True, alto=380, orden_color=None):
    """Barras verticales del % de bajo puntaje, con una linea gris del promedio de la vista."""
    g = g[g.estudiantes >= MIN_ESTUDIANTES]
    if g.empty:
        return vacia(titulo)
    fig = px.bar(g, x=x, y="tasa", color=color, barmode="group", custom_data=["estudiantes", "promedio"],
                 category_orders={**({x: orden} if orden else {}), **({color: orden_color} if orden_color else {})},
                 color_discrete_map=colores, color_discrete_sequence=[AZUL],
                 text=g.tasa.map(lambda v: pct(v, 0)) if texto else None)
    fig.update_traces(hovertemplate=HOVER, marker_line_width=0, textposition="outside", cliponaxis=False)
    fig.add_hline(y=tasa_total, line_dash="dot", line_color=GRIS, annotation_text=f"promedio de la vista {pct(tasa_total, 0)}",
                  annotation_font=dict(color=GRIS, size=11), annotation_position="top right", annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(**LAYOUT, title=titulo, height=alto, yaxis_tickformat=".0%", yaxis_title="% con bajo puntaje",
                      xaxis_title="", legend_title="", bargap=0.5 if g[x].nunique() <= 3 else 0.25, bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


# ---------- estructura ----------
opciones_depto = [{"label": DEPARTAMENTOS.get(c, "Sin dato"), "value": c} for c in sorted(RES.cod_depto.unique(), key=lambda c: DEPARTAMENTOS.get(c, "zz"))]

app.layout = html.Div([
    html.Header([
        html.Div([html.P("PRUEBAS SABER 11 · ICFES · DATOS ABIERTOS", className="eyebrow"),
                  html.H1("Alerta temprana de bajo puntaje en Saber 11"),
                  html.P(f"'Bajo puntaje' = puntaje global menor a {CORTE:.0f}, el 25% más bajo de todos los años juntos "
                         f"(el ICFES reporta el puntaje global en una escala comparable entre años desde 2014-2). "
                         f"Resultados 2014-2 a 2022.", className="subtitle")]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros"),
            html.Label("Departamento del colegio"),
            dcc.Dropdown(opciones_depto, multi=True, id="departamento", placeholder="Todos los departamentos"),
            html.Label("Tipo de colegio"),
            dcc.Dropdown(sorted(RES.naturaleza.unique()), multi=True, id="naturaleza", placeholder="Oficial y no oficial"),
            html.Label("Zona"),
            dcc.Dropdown(sorted(RES.area.unique()), multi=True, id="area", placeholder="Urbana y rural"),
            html.Label("Calendario"),
            dcc.Dropdown(sorted(RES.calendario.unique()), multi=True, id="calendario", placeholder="A, B y otro"),
            html.Label("Años"),
            dcc.RangeSlider(ANIOS[0], ANIOS[-1], 1, value=[ANIOS[0], ANIOS[-1]], marks={a: str(a)[2:] for a in ANIOS}, id="anios"),
            html.P("El examen se aplica dos veces al año: en el 1.er semestre (sobre todo colegios de calendario B) y en el "
                   "2.º semestre (calendario A). El archivo del ICFES no trae el 2.º semestre de 2018, 2020 ni 2021.",
                   className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([card("Estudiantes", "kpi-n", "en la vista filtrada"),
                      card("Con bajo puntaje", "kpi-rate", f"puntaje global < {CORTE:.0f}"),
                      card("Puntaje global promedio", "kpi-avg", "escala de 0 a 500"),
                      card("Corte de bajo puntaje", "kpi-corte", "fijo para todos los años")], className="kpi-grid"),
            dcc.Tabs([
                dcc.Tab(label="Panorama", children=[
                    html.Div([dcc.Graph(id="mapa"), dcc.Graph(id="depto-chart")], className="grid-2"),
                    html.Div([html.Div([dcc.RadioItems([{"label": " % con bajo puntaje", "value": "tasa"},
                                                        {"label": " Puntaje promedio", "value": "promedio"}],
                                                       "tasa", id="metrica-anio", inline=True, className="radio"),
                                        dcc.Graph(id="anio-chart")]),
                              dcc.Graph(id="dist-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="estrato-chart")], className="grid-1"),
                ]),
                dcc.Tab(label="Factores del estudiante y su hogar", children=[
                    html.Div([dcc.Graph(id="educ-chart"), dcc.Graph(id="bienes-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="jornada-chart"), dcc.Graph(id="hacinamiento-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="area-chart"), dcc.Graph(id="genero-chart")], className="grid-2"),
                    html.P(f"Cada barra es el % de estudiantes del grupo con puntaje global menor a {CORTE:.0f}. La línea punteada es el % "
                           f"de toda la vista filtrada. Se ocultan grupos con menos de {MIN_ESTUDIANTES} estudiantes. Son asociaciones, no causas.",
                           className="help note"),
                ]),
                dcc.Tab(label="Modelo", children=[
                    html.Div([
                        html.Div([html.Label("Umbral del puntaje de riesgo"),
                                  dcc.Slider(0.20, 0.80, 0.05, value=0.50, marks={x: f"{x:.1f}" for x in [0.2, 0.4, 0.6, 0.8]}, id="umbral")],
                                 className="umbral-box"),
                        html.P("Desde qué puntaje de riesgo el modelo marca a un estudiante. Más bajo = detecta más estudiantes, "
                               "pero da más falsas alarmas. No reentrena el modelo.", className="help"),
                    ], className="umbral-row"),
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="confusion-chart"), html.Div(id="narrative", className="narrative")], className="grid-2"),
                    html.Div([dcc.Graph(id="model-chart"), dcc.Graph(id="importance-chart")], className="grid-2"),
                ]),
                dcc.Tab(label="Casos de mayor riesgo", children=[
                    html.P("Los 100 estudiantes del conjunto de prueba con mayor puntaje de riesgo, según los filtros. "
                           "El identificador es el consecutivo anónimo del ICFES; no hay nombres ni documentos.", className="help note"),
                    dash_table.DataTable(id="risk-table", page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
                        style_header={"backgroundColor": NAVY, "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[{"if": {"filter_query": "{sobre_umbral} = Sí"}, "backgroundColor": "#FCEFD6"}]),
                ]),
            ]),
            html.Footer(id="footer"),
        ], className="content")
    ], className="shell")
], className="app")


# ---------- callbacks ----------
@app.callback(
    Output("departamento", "value"), Output("naturaleza", "value"), Output("area", "value"), Output("calendario", "value"),
    Output("anios", "value"), Output("umbral", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [ANIOS[0], ANIOS[-1]], 0.50


FILTROS = [Input("departamento", "value"), Input("naturaleza", "value"), Input("area", "value"),
           Input("calendario", "value"), Input("anios", "value")]


@app.callback(
    Output("kpi-n", "children"), Output("kpi-rate", "children"), Output("kpi-avg", "children"), Output("kpi-corte", "children"),
    Output("mapa", "figure"), Output("depto-chart", "figure"), Output("anio-chart", "figure"), Output("dist-chart", "figure"),
    Output("estrato-chart", "figure"), Output("educ-chart", "figure"), Output("bienes-chart", "figure"),
    Output("jornada-chart", "figure"), Output("hacinamiento-chart", "figure"), Output("area-chart", "figure"),
    Output("genero-chart", "figure"), Output("footer", "children"),
    *FILTROS, Input("metrica-anio", "value"))
def actualizar_datos(deptos, nats, areas, cals, anios, metrica):
    r = filtrar(RES, deptos, nats, areas, cals, anios)
    tot = r[r.variable == "Total"]
    n = int(tot.estudiantes.sum())
    footer = (f"Fuente: ICFES, 'Resultados únicos Saber 11' (datos.gov.co), sin registros duplicados. Datos del tablero: {FUENTE}. "
              f"Modelo y casos de riesgo: muestra aleatoria de 60.000 estudiantes.")
    if n == 0:
        figs = [vacia(t) for t in ["Mapa", "Departamentos", "Año", "Distribución", "Estrato", "Educación", "Bienes",
                                   "Jornada", "Personas por cuarto", "Zona", "Género"]]
        return ("0", "--", "--", f"{CORTE:.0f}", *figs, footer)
    tasa_total = tot.bajo.sum() / n
    kpis = (num(n), pct(tasa_total), num(tot.suma_puntaje.sum() / n), f"{CORTE:.0f}")

    # Mapa y ranking por departamento
    dep = agrupar(tot, ["cod_depto"])
    dep["nombre"] = dep.cod_depto.map(DEPARTAMENTOS)
    dep = dep[(dep.estudiantes >= MIN_ESTUDIANTES) & dep.nombre.notna()]  # el mapa usa la misma regla que las barras
    fig_mapa = px.choropleth_map(dep, geojson=GEO, locations="cod_depto", color="tasa", color_continuous_scale=SECUENCIAL,
                                 custom_data=["nombre", "estudiantes", "promedio"], map_style="white-bg",
                                 zoom=4.2, center={"lat": 4.4, "lon": -73.6}, opacity=0.95)
    fig_mapa.update_traces(marker_line_color="white", marker_line_width=1,
                           hovertemplate="<b>%{customdata[0]}</b><br>Bajo puntaje: %{z:.1%}<br>Estudiantes: %{customdata[1]:,}"
                                         "<br>Puntaje promedio: %{customdata[2]:.0f}<extra></extra>")
    fig_mapa.update_layout(**{**LAYOUT, "margin": dict(l=0, r=0, t=60, b=0)}, title="% con bajo puntaje por departamento", height=600,
                           coloraxis_colorbar=dict(title="", tickformat=".0%", thickness=12, len=0.6))

    dep_bar = dep[dep.estudiantes >= MIN_ESTUDIANTES].sort_values("tasa")
    fig_dep = px.bar(dep_bar, x="tasa", y="nombre", orientation="h", custom_data=["estudiantes", "promedio"],
                     text=dep_bar.tasa.map(lambda v: pct(v, 0)), color_discrete_sequence=[AZUL])
    fig_dep.update_traces(hovertemplate="<b>%{y}</b><br>Bajo puntaje: %{x:.1%}<br>Estudiantes: %{customdata[0]:,}"
                                        "<br>Puntaje promedio: %{customdata[1]:.0f}<extra></extra>",
                          textposition="outside", cliponaxis=False, marker_line_width=0)
    fig_dep.add_vline(x=tasa_total, line_dash="dot", line_color=GRIS)
    fig_dep.update_layout(**LAYOUT, title="Ranking de departamentos", height=600, xaxis_tickformat=".0%",
                          xaxis_title="% con bajo puntaje", yaxis_title="", bargap=0.25,
                          yaxis=dict(dtick=1, tickfont=dict(size=11)))

    # Tendencia por ano, una linea por calendario
    an = agrupar(tot, ["anio", "semestre"])
    an = an[an.estudiantes >= MIN_ESTUDIANTES]
    if an.empty:
        fig_anio = vacia("Tendencia por año")
    else:
        # Los anos sin datos quedan como hueco en la linea (no se unen puntos de anos que no son seguidos).
        rejilla = pd.MultiIndex.from_product([range(anios[0], anios[1] + 1), sorted(an.semestre.unique())], names=["anio", "semestre"])
        an = an.set_index(["anio", "semestre"]).reindex(rejilla).reset_index()
        fig_anio = px.line(an, x="anio", y=metrica, color="semestre", markers=True, color_discrete_map=COLOR_SEMESTRE,
                           custom_data=["estudiantes", "tasa", "promedio"],
                           category_orders={"semestre": ["2.º semestre", "1.er semestre"]})
        fig_anio.update_traces(line_width=2, marker_size=9,
                               hovertemplate="<b>%{x}</b><br>Bajo puntaje: %{customdata[1]:.1%}<br>Puntaje promedio: %{customdata[2]:.0f}"
                                             "<br>Estudiantes: %{customdata[0]:,}<extra></extra>")
        fig_anio.update_layout(**LAYOUT, title="Tendencia por año y semestre de aplicación", height=420, legend_title="Aplicación",
                               xaxis=dict(tickmode="array", tickvals=ANIOS, title=""),
                               yaxis=dict(tickformat=".0%" if metrica == "tasa" else "d",
                                          title="% con bajo puntaje" if metrica == "tasa" else "Puntaje global promedio"))

    # Distribucion del puntaje global por tipo de colegio
    dist_f = filtrar(DIST, deptos, nats, areas, cals, anios)
    ds = (dist_f.assign(tramo=dist_f.tramo // 10 * 10)
          .groupby(["naturaleza", "tramo"], as_index=False).estudiantes.sum())
    ds["pct"] = ds.estudiantes / ds.groupby("naturaleza").estudiantes.transform("sum")
    ds["centro"] = ds.tramo + 5
    fig_dist = px.line(ds, x="centro", y="pct", color="naturaleza", color_discrete_map=COLOR_NATURALEZA, line_shape="spline",
                       custom_data=["tramo", "estudiantes"], category_orders={"naturaleza": ["Oficial", "No Oficial"]})
    fig_dist.update_traces(line_width=2, hovertemplate="Puntaje %{customdata[0]}-%{customdata[0]+9}: %{y:.1%} de los estudiantes"
                                                       "<br>(%{customdata[1]:,} estudiantes)<extra></extra>")
    fig_dist.add_vline(x=CORTE, line_dash="dash", line_color=NAVY, annotation_text=f"corte {CORTE:.0f}", annotation_position="top right")
    fig_dist.update_layout(**LAYOUT, title="Distribución del puntaje global", height=420 + 34, xaxis_title="Puntaje global",
                           yaxis_title="% de estudiantes", yaxis_tickformat=".0%", legend_title="Colegio")

    # Cajas (velas) del puntaje global por estrato y tipo de colegio, calculadas desde los conteos por tramo
    fig_est = go.Figure()
    base = dist_f[dist_f.estrato_txt != "Sin dato"].groupby(["naturaleza", "estrato_txt", "tramo"], as_index=False).estudiantes.sum()
    for nat in ["Oficial", "No Oficial"]:
        filas = []
        for est in ORDEN["Estrato"]:
            g = base[(base.naturaleza == nat) & (base.estrato_txt == est)]
            if g.estudiantes.sum() >= MIN_ESTUDIANTES:
                filas.append({"estrato": est, **cuartiles(g.tramo, g.estudiantes)})
        if filas:
            c = pd.DataFrame(filas)
            fig_est.add_trace(go.Box(x=c.estrato, q1=c.q1, median=c.mediana, q3=c.q3, lowerfence=c.inferior,
                                     upperfence=c.superior, name=nat, marker_color=COLOR_NATURALEZA[nat], boxpoints=False,
                                     line_width=1.5))
    if fig_est.data:
        fig_est.add_hline(y=CORTE, line_dash="dash", line_color=NAVY, annotation_text=f"corte {CORTE:.0f}",
                          annotation_position="top left", annotation_font_color=NAVY)
        fig_est.update_layout(**LAYOUT, title="Puntaje global por estrato y tipo de colegio", height=430, boxmode="group",
                              xaxis=dict(categoryorder="array", categoryarray=ORDEN["Estrato"], title=""),
                              yaxis_title="Puntaje global", legend_title="Colegio",
                              legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    else:
        fig_est = vacia("Puntaje global por estrato y tipo de colegio")

    # Factores del hogar
    ed = r[r.variable.isin(["Educación de la madre", "Educación del padre"]) & (r.valor != "Sin dato")]
    ed = agrupar(ed, ["valor", "variable"]).replace({"Educación de la madre": "Madre", "Educación del padre": "Padre"})
    fig_ed = barras(ed, "valor", "Bajo puntaje según educación de la madre y del padre", tasa_total, color="variable",
                    orden=ORDEN["Educación"], colores={"Madre": AZUL, "Padre": DORADO}, texto=False, alto=420, orden_color=["Madre", "Padre"])
    fig_ed.update_xaxes(tickangle=-35)

    bi = agrupar(r[r.variable.isin(["Internet", "Computador", "Automóvil", "Lavadora"]) & r.valor.isin(["Si", "No"])],
                 ["variable", "valor"]).replace({"valor": {"Si": "Lo tiene", "No": "No lo tiene"}})
    fig_bi = barras(bi, "variable", "Bajo puntaje según bienes del hogar", tasa_total, color="valor",
                    orden=["Internet", "Computador", "Automóvil", "Lavadora"], colores={"Lo tiene": AZUL, "No lo tiene": DORADO}, alto=420,
                    orden_color=["Lo tiene", "No lo tiene"])

    jo = agrupar(r[r.variable == "Jornada"], ["valor"]).sort_values("tasa")
    fig_jo = barras(jo, "valor", "Bajo puntaje según jornada del colegio", tasa_total)
    ha = agrupar(r[(r.variable == "Personas por cuarto") & (r.valor != "Sin dato")], ["valor"])
    fig_ha = barras(ha, "valor", "Bajo puntaje según personas por cuarto (hacinamiento)", tasa_total, orden=ORDEN["Personas por cuarto"])
    ar = agrupar(tot, ["area"])
    fig_ar = barras(ar, "area", "Bajo puntaje según zona del colegio", tasa_total)
    ge = agrupar(r[(r.variable == "Género") & (r.valor != "Sin dato")], ["valor"])
    fig_ge = barras(ge, "valor", "Bajo puntaje según género", tasa_total)

    return (*kpis, fig_mapa, fig_dep, fig_anio, fig_dist, fig_est, fig_ed, fig_bi, fig_jo, fig_ha, fig_ar, fig_ge, footer)


# Figuras del modelo que no dependen de los filtros
MODELS["version"] = MODELS.version.replace({"1. Linea base": "1. Línea base", "2. Logistica regularizada": "2. Logística regularizada"})
GANADOR = MODELS.sort_values("roc_auc_validacion_cruzada", ascending=False).version.iloc[0]
mm = MODELS.melt(id_vars="version", value_vars=["roc_auc_validacion_cruzada", "roc_auc_prueba", "recall"], var_name="metrica", value_name="valor")
mm["metrica"] = mm.metrica.map({"roc_auc_validacion_cruzada": "AUC validación cruzada", "roc_auc_prueba": "AUC prueba", "recall": "Detección (recall)"})
FIG_MODELOS = px.bar(mm, x="version", y="valor", color="metrica", barmode="group", color_discrete_sequence=[AZUL, DORADO, VERDE],
                     text=mm.valor.map(lambda v: f"{v:.2f}".replace(".", ",")), category_orders={"version": sorted(MODELS.version)})
FIG_MODELOS.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
FIG_MODELOS.update_layout(**LAYOUT, title=f"Comparación de las 3 versiones (ganador: {GANADOR[3:]})", height=420, yaxis_range=[0, 1],
                          xaxis_title="", yaxis_title="", legend_title="", legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"))
imp = IMPORTANCE.sort_values("importancia").assign(nombre=lambda x: x.variable.map(NOMBRES).fillna(x.variable))
FIG_IMP = px.bar(imp, x="importancia", y="nombre", orientation="h", color_discrete_sequence=[AZUL])
FIG_IMP.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:.4f}<extra></extra>")
FIG_IMP.update_layout(**LAYOUT, title="Qué variables usa más el modelo", height=420, yaxis_title="",
                      xaxis_title="Cuánto empeora el AUC si se desordena la variable")


@app.callback(
    Output("metrics-strip", "children"), Output("confusion-chart", "figure"), Output("model-chart", "figure"),
    Output("importance-chart", "figure"), Output("narrative", "children"),
    Output("risk-table", "data"), Output("risk-table", "columns"),
    *FILTROS, Input("umbral", "value"))
def actualizar_modelo(deptos, nats, areas, cals, anios, umbral):
    p = filtrar(PRED, deptos, nats, areas, cals, anios)
    if len(p) and p.bajo_puntaje.nunique() == 2:
        y = p.bajo_puntaje.to_numpy()
        pred = (p.probabilidad_bajo.to_numpy() >= umbral).astype(int)
        auc = roc_auc_score(y, p.probabilidad_bajo); acc = accuracy_score(y, pred)
        pre = precision_score(y, pred, zero_division=0); rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0); cm = confusion_matrix(y, pred, labels=[0, 1])
        celdas = [("AUC", f"{auc:.2f}".replace(".", ",")), ("Exactitud", pct(acc, 0)), ("Precisión", pct(pre, 0)),
                  ("Detección", pct(rec, 0)), ("F1", f"{f1:.2f}".replace(".", ","))]
        strip = [html.Div([html.Span(k), html.Strong(v)]) for k, v in celdas]
        marcados = int(pred.sum())
        narrative = [
            html.H3("Cómo leer estas cifras"),
            html.P(f"En esta vista hay {num(len(p))} estudiantes del conjunto de prueba (estudiantes que el modelo nunca vio al entrenar). "
                   f"Con un umbral de {dec(umbral)}, el modelo marca a {num(marcados)}."),
            html.Ul([
                html.Li(f"AUC {auc:.2f}".replace(".", ",") + ": si se toma al azar un estudiante que sí sacó bajo puntaje y otro que no, "
                        f"el modelo le da más riesgo al primero el {pct(auc, 0)} de las veces. 0,50 sería lanzar una moneda."),
                html.Li(f"Detección {pct(rec, 0)}: de cada 10 estudiantes que sí sacaron bajo puntaje, el modelo marca a {rec * 10:.0f}."),
                html.Li(f"Precisión {pct(pre, 0)}: de cada 10 estudiantes marcados, {pre * 10:.0f} sí sacaron bajo puntaje; los demás son falsas alarmas."),
                html.Li("F1 resume detección y precisión en un solo número (1 sería perfecto)."),
            ]),
            html.P("El puntaje de riesgo sirve para ordenar a los estudiantes de más a menos riesgo; no es una probabilidad exacta, "
                   "porque el modelo se entrenó dando más peso a los estudiantes con bajo puntaje."),
        ]
    else:
        cm = np.zeros((2, 2), dtype=int)
        strip = [html.Div([html.Span("Sin muestra suficiente"), html.Strong("--")])]
        narrative = [html.H3("Cómo leer estas cifras"), html.P("La selección no tiene estudiantes con y sin bajo puntaje para calcular las métricas.")]

    etiquetas = np.array([["Acierta: no bajo", "Falsa alarma"], ["No detectado", "Acierta: bajo"]])
    fig_cm = go.Figure(go.Heatmap(z=cm, x=["Marca: no bajo", "Marca: bajo"], y=["Real: no bajo", "Real: bajo"],
                                  text=[[f"{etiquetas[i][j]}<br><b>{num(cm[i][j])}</b>" for j in range(2)] for i in range(2)],
                                  texttemplate="%{text}", textfont=dict(color=NAVY, size=14),
                                  colorscale=[[0, "#F2F7FC"], [1, "#8FBCE8"]], showscale=False,
                                  hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
    fig_cm.update_layout(**LAYOUT, title=f"Matriz de confusión (umbral {umbral:.2f})".replace(".", ","), height=420, yaxis_autorange="reversed")

    cols = ["estudiante_id", "departamento", "anio", "calendario", "naturaleza", "area", "jornada", "estrato",
            "punt_global", "probabilidad_bajo", "sobre_umbral", "bajo_real"]
    t = p.sort_values("probabilidad_bajo", ascending=False).head(100).copy()
    t["probabilidad_bajo"] = t.probabilidad_bajo.round(3)
    t["sobre_umbral"] = np.where(t.probabilidad_bajo >= umbral, "Sí", "No")
    t["bajo_real"] = np.where(t.bajo_puntaje == 1, "Sí", "No")
    t["estrato"] = t.estrato.map(lambda v: "Sin dato" if pd.isna(v) else ("Sin estrato" if v == 0 else f"Estrato {int(v)}"))
    nombres = {"estudiante_id": "Id estudiante", "departamento": "Departamento", "anio": "Año", "calendario": "Calendario",
               "naturaleza": "Tipo de colegio", "area": "Zona", "jornada": "Jornada", "estrato": "Estrato",
               "punt_global": "Puntaje global", "probabilidad_bajo": "Puntaje de riesgo", "sobre_umbral": "Marcado por el modelo",
               "bajo_real": "Bajo puntaje real"}
    riesgo = Format(precision=2, scheme=Scheme.fixed).decimal_delimiter(",")
    columnas = [{"name": nombres[c], "id": c, **({"type": "numeric", "format": riesgo} if c == "probabilidad_bajo" else {})} for c in cols]
    return strip, fig_cm, FIG_MODELOS, FIG_IMP, narrative, t[cols].to_dict("records"), columnas


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
