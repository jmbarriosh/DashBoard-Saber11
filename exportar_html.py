"""Genera tablero_saber11.html: el mismo tablero de app.py en UN solo archivo HTML para compartir.

El archivo funciona sin internet y sin Python: lleva adentro los datos resumidos y la libreria de graficos
(plotly.js), y los filtros se calculan en el navegador con JavaScript.

Uso (despues de preparar_datos.py y entrenar_modelo.py):
    python exportar_html.py
"""
from pathlib import Path
import json

import pandas as pd
import plotly

from preparar_datos import DEPARTAMENTOS

ROOT = Path(__file__).resolve().parent
SALIDA = ROOT / "tablero_saber11.html"


def columnar(df, categoricas):
    """Tabla en columnas compactas: los textos repetidos se guardan como numeros + un diccionario."""
    out = {"n": len(df), "dic": {}, "col": {}}
    for c in df.columns:
        if c in categoricas:
            codigos, valores = pd.factorize(df[c].astype(str), sort=True)
            out["dic"][c] = list(valores)
            out["col"][c] = codigos.tolist()
        elif not pd.api.types.is_numeric_dtype(df[c]):
            out["col"][c] = df[c].astype(str).tolist()
        else:
            out["col"][c] = [None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))
                             for v in df[c].tolist()]
    return out


def main():
    res = pd.read_csv(ROOT / "data" / "resumen_tablero.csv", dtype={"cod_depto": str})
    dist = pd.read_csv(ROOT / "data" / "distribucion_puntaje.csv", dtype={"cod_depto": str})
    # El semestre no es un filtro y la distribucion no lo usa: se quita para que el archivo pese menos.
    dist = dist.groupby(["anio", "calendario", "cod_depto", "naturaleza", "area", "estrato_txt", "tramo"],
                        as_index=False).estudiantes.sum()
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv", dtype={"cod_depto": str})
    pred = pred[["estudiante_id", "departamento", "anio", "calendario", "naturaleza", "area", "cod_depto", "jornada",
                 "estrato", "punt_global", "probabilidad_bajo", "bajo_puntaje"]]
    modelos = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
    importancia = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv").head(12)
    corte = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))
    geo = json.loads((ROOT / "data" / "colombia_departamentos.geojson").read_text(encoding="utf-8"))

    datos = {
        "res": columnar(res, {"semestre", "calendario", "cod_depto", "naturaleza", "area", "variable", "valor"}),
        "dist": columnar(dist, {"calendario", "cod_depto", "naturaleza", "area", "estrato_txt"}),
        "pred": columnar(pred, {"departamento", "calendario", "naturaleza", "area", "cod_depto", "jornada"}),
        "modelos": modelos[["version", "roc_auc_validacion_cruzada", "roc_auc_prueba", "recall"]].to_dict("records"),
        "importancia": importancia.to_dict("records"),
        "corte": corte,
        "deptos": DEPARTAMENTOS,
        "geo": geo,
    }
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
    plantilla = (ROOT / "plantillas" / "tablero.html").read_text(encoding="utf-8")
    html = (plantilla.replace("/*__PLOTLY__*/", plotly_js)
            .replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
            .replace("/*__ESTILOS__*/", (ROOT / "assets" / "style.css").read_text(encoding="utf-8")))
    SALIDA.write_text(html, encoding="utf-8")
    print(f"Listo: {SALIDA.name} ({SALIDA.stat().st_size / 1e6:.1f} MB). Abrelo con doble clic; no necesita internet.")


if __name__ == "__main__":
    main()
