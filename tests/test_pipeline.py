from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def cargar():
    data = pd.read_csv(ROOT / "data" / "saber11_preparado.csv", dtype={"periodo": str, "cod_depto": str})
    resumen = pd.read_csv(ROOT / "data" / "resumen_tablero.csv", dtype={"cod_depto": str})
    corte = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))["corte_puntaje_global"]
    return data, resumen, corte


def test_archivos_y_esquema():
    data, _, corte = cargar()
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv")
    results = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
    assert len(data) > 1000
    assert data.estudiante_id.is_unique                     # sin estudiantes repetidos
    assert set(data.bajo_puntaje.unique()) <= {0, 1}
    assert data.periodo.str[:5].ge("20142").all()           # solo periodos con puntaje global
    assert ((data.punt_global < corte) == (data.bajo_puntaje == 1)).all()
    assert 0.15 < data.bajo_puntaje.mean() < 0.35
    assert data.cod_depto.str.len().eq(2).all()             # codigo DANE de 2 digitos para el mapa
    assert pred.probabilidad_bajo.between(0, 1).all()
    assert len(results) == 3
    assert results.roc_auc_prueba.between(0.5, 1).all()


def test_resumen_cuadra_con_los_datos():
    """Cada variable del resumen debe sumar los mismos estudiantes (y, si el tablero usa la muestra,
    exactamente los mismos bajos y puntajes que los datos del modelo)."""
    data, resumen, _ = cargar()
    info = json.loads((ROOT / "outputs" / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))
    sumas = resumen.groupby("variable")[["estudiantes", "bajo", "suma_puntaje"]].sum()
    assert (sumas.estudiantes == info["estudiantes"]).all()
    assert sumas.bajo.nunique() == 1 and sumas.suma_puntaje.nunique() == 1
    if info["fuente"] == "muestra_saber11.csv":
        assert (sumas.estudiantes == len(data)).all()
        assert (sumas.bajo == data.bajo_puntaje.sum()).all()
        assert (sumas.suma_puntaje == data.punt_global.sum()).all()
    else:  # tablero con el archivo completo: el 25% del total debe quedar por debajo del corte
        assert 0.22 < sumas.bajo.iloc[0] / sumas.estudiantes.iloc[0] < 0.28


def test_mapa_tiene_todos_los_departamentos():
    data, _, _ = cargar()
    geo = json.loads((ROOT / "data" / "colombia_departamentos.geojson").read_text(encoding="utf-8"))
    codigos_mapa = {f["id"] for f in geo["features"]}
    assert set(data.cod_depto.unique()) <= codigos_mapa


def test_sin_fuga_de_puntajes():
    """Ningun puntaje del examen puede usarse como pista del modelo."""
    imp = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv")
    assert not imp.variable.str.startswith("punt_").any()


def test_particion_sin_duplicados():
    data, _, _ = cargar()
    pred = pd.read_csv(ROOT / "outputs" / "predicciones_prueba.csv")
    assert pred.estudiante_id.is_unique
    assert abs(len(pred) - 0.25 * len(data)) <= 1
