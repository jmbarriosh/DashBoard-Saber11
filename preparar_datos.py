"""Limpia Saber 11, crea la variable objetivo `bajo_puntaje` y arma las tablas resumen del tablero.

Uso:
    python preparar_datos.py              tablero y modelo con la muestra (rapido, para desarrollar)
    python preparar_datos.py --completo   tablero con TODO el CSV del ICFES; el modelo sigue con la muestra

Entrada : data/muestra_saber11.csv   (sale de sacar_muestra.py) y, con --completo, el CSV original en data/
Salidas : data/saber11_preparado.csv      datos limpios, uno por estudiante (los usa el modelo)
          data/resumen_tablero.csv        conteos por combinacion de filtros (los usa el tablero)
          data/distribucion_puntaje.csv   cuantos estudiantes hay en cada tramo de 5 puntos (por estrato)
          outputs/corte_bajo_puntaje.json el puntaje que separa al 25% mas bajo

Por que tablas resumen: el tablero no necesita a cada estudiante, sino cuantos hay y cuantos sacaron
bajo puntaje en cada combinacion de filtros (ano, semestre, calendario, departamento, tipo de colegio y zona).
Con esos conteos cualquier filtro se resuelve sumando, asi el tablero es rapido aunque los datos
vengan del archivo completo.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "data" / "muestra_saber11.csv"
SALIDA = ROOT / "data" / "saber11_preparado.csv"
RESUMEN = ROOT / "data" / "resumen_tablero.csv"
DISTRIBUCION = ROOT / "data" / "distribucion_puntaje.csv"
CORTE = ROOT / "outputs" / "corte_bajo_puntaje.json"
PERCENTIL_CORTE = 0.25  # el 25% de puntajes mas bajos (de todos los anos juntos) es "bajo puntaje"

# Nombres de departamento por codigo DANE (el mismo codigo que usa el mapa).
DEPARTAMENTOS = {
    "05": "Antioquia", "08": "Atlántico", "11": "Bogotá D.C.", "13": "Bolívar", "15": "Boyacá",
    "17": "Caldas", "18": "Caquetá", "19": "Cauca", "20": "Cesar", "23": "Córdoba",
    "25": "Cundinamarca", "27": "Chocó", "41": "Huila", "44": "La Guajira", "47": "Magdalena",
    "50": "Meta", "52": "Nariño", "54": "Norte de Santander", "63": "Quindío", "66": "Risaralda",
    "68": "Santander", "70": "Sucre", "73": "Tolima", "76": "Valle del Cauca", "81": "Arauca",
    "85": "Casanare", "86": "Putumayo", "88": "San Andrés", "91": "Amazonas", "94": "Guainía",
    "95": "Guaviare", "97": "Vaupés", "99": "Vichada",
}

# Niveles educativos ordenados de menor a mayor (0 = ninguno, 9 = postgrado)
EDUCACION = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
# "No sabe" y "No Aplica" quedan como dato faltante.
EDUCACION_TXT = {0: "Ninguno", 1: "Primaria incompleta", 2: "Primaria completa", 3: "Secundaria incompleta",
                 4: "Secundaria completa", 5: "Técnica incompleta", 6: "Técnica completa",
                 7: "Profesional incompleta", 8: "Profesional completa", 9: "Postgrado"}

# Personas en el hogar: el ICFES cambio de rangos a numeros en algunos periodos.
PERSONAS = {
    "Una": 1, "1 a 2": 1.5, "Dos": 2, "Tres": 3, "3 a 4": 3.5, "Cuatro": 4, "Cinco": 5,
    "5 a 6": 5.5, "Seis": 6, "Siete": 7, "7 a 8": 7.5, "Ocho": 8, "Nueve": 9, "9 o más": 9,
    "Diez": 10, "Once": 11, "Doce o más": 12,
}
CUARTOS = {
    "Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis": 6, "Seis o mas": 6,
    "Siete": 7, "Ocho": 8, "Nueve": 9, "Diez o más": 10,
}
SI_NO = {"Si": "Si", "No": "No", "S": "Si", "N": "No"}
JORNADA = {"MAÑANA": "Mañana", "COMPLETA": "Completa", "TARDE": "Tarde", "UNICA": "Única",
           "SABATINA": "Sabatina", "NOCHE": "Noche"}

# Filtros del tablero: cada fila del resumen es una combinacion de estos valores.
LLAVES = ["anio", "semestre", "calendario", "cod_depto", "naturaleza", "area"]
TRAMO = 5  # ancho en puntos de cada tramo de la distribucion
TAMANO_PEDAZO = 250_000  # filas del CSV completo que se leen en memoria cada vez
COLUMNAS_FUENTE = {
    "PERIODO", "ESTU_CONSECUTIVO", "ESTU_ESTADOINVESTIGACION", "ESTU_FECHANACIMIENTO", "ESTU_GENERO",
    "COLE_COD_DEPTO_UBICACION", "COLE_NATURALEZA", "COLE_AREA_UBICACION", "COLE_JORNADA", "COLE_CALENDARIO",
    "COLE_BILINGUE", "COLE_CARACTER", "COLE_GENERO", "FAMI_ESTRATOVIVIENDA", "FAMI_EDUCACIONMADRE",
    "FAMI_EDUCACIONPADRE", "FAMI_PERSONASHOGAR", "FAMI_CUARTOSHOGAR", "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEAUTOMOVIL", "FAMI_TIENELAVADORA", "PUNT_GLOBAL", "PUNT_MATEMATICAS", "PUNT_LECTURA_CRITICA",
}


def limpiar(raw):
    """Convierte las columnas originales del ICFES en las columnas del proyecto (sin la variable objetivo)."""
    d = raw.copy()
    d["PUNT_GLOBAL"] = pd.to_numeric(d["PUNT_GLOBAL"], errors="coerce")
    d = d[d["PUNT_GLOBAL"].notna() & (d["PUNT_GLOBAL"] > 0) & (pd.to_numeric(d["PERIODO"]) >= 20142)]
    d = d[d["ESTU_ESTADOINVESTIGACION"].eq("PUBLICAR")]
    d = d.drop_duplicates("ESTU_CONSECUTIVO")

    out = pd.DataFrame({
        "estudiante_id": d["ESTU_CONSECUTIVO"],
        "periodo": d["PERIODO"],
        "anio": d["PERIODO"].str[:4].astype(int),
        # Aplicacion del examen: los periodos que terminan en 1 son del primer semestre (la presentan sobre
        # todo colegios de calendario B); los que terminan en 2 o 4, del segundo semestre (calendario A).
        "semestre": np.where(d["PERIODO"].str[-1].eq("1"), "1.er semestre", "2.º semestre"),
        "cod_depto": d["COLE_COD_DEPTO_UBICACION"].str.zfill(2),
        "naturaleza": d["COLE_NATURALEZA"].str.title(),
        "area": d["COLE_AREA_UBICACION"].str.title(),
        "jornada": d["COLE_JORNADA"].map(JORNADA),
        "calendario": d["COLE_CALENDARIO"].replace({"OTRO": "Otro"}),
        "bilingue": d["COLE_BILINGUE"].map(SI_NO),
        "caracter": d["COLE_CARACTER"].str.title(),
        "genero_colegio": d["COLE_GENERO"].str.title(),
        "genero": d["ESTU_GENERO"].map({"F": "Femenino", "M": "Masculino"}),
    })
    out.insert(4, "departamento", out["cod_depto"].map(DEPARTAMENTOS))

    # Edad al presentar la prueba (fuera de 13 a 60 anos se toma como error de digitacion)
    nacimiento = pd.to_datetime(d["ESTU_FECHANACIMIENTO"], format="%d/%m/%Y", errors="coerce")
    edad = out["anio"] - nacimiento.dt.year
    out["edad"] = edad.where(edad.between(13, 60))

    # Estrato: "Sin Estrato" = 0, Estrato 1..6
    out["estrato"] = (d["FAMI_ESTRATOVIVIENDA"].str.extract(r"(\d)")[0].astype(float)
                      .where(d["FAMI_ESTRATOVIVIENDA"].ne("Sin Estrato"), 0))
    out["educacion_madre"] = d["FAMI_EDUCACIONMADRE"].map(EDUCACION)
    out["educacion_padre"] = d["FAMI_EDUCACIONPADRE"].map(EDUCACION)
    out["personas_hogar"] = d["FAMI_PERSONASHOGAR"].map(PERSONAS)
    out["cuartos_hogar"] = d["FAMI_CUARTOSHOGAR"].map(CUARTOS)
    for col, fuente in [("tiene_internet", "FAMI_TIENEINTERNET"), ("tiene_computador", "FAMI_TIENECOMPUTADOR"),
                        ("tiene_automovil", "FAMI_TIENEAUTOMOVIL"), ("tiene_lavadora", "FAMI_TIENELAVADORA")]:
        out[col] = d[fuente].map(SI_NO)

    # Puntajes: se guardan para mostrarlos, pero NO se usan como pistas del modelo.
    out["punt_global"] = d["PUNT_GLOBAL"].astype(int)
    out["punt_matematicas"] = pd.to_numeric(d["PUNT_MATEMATICAS"], errors="coerce")
    out["punt_lectura_critica"] = pd.to_numeric(d["PUNT_LECTURA_CRITICA"], errors="coerce")

    # Los textos faltantes se marcan como "Sin dato" para que se vean en los filtros.
    texto = ["departamento", "naturaleza", "area", "jornada", "calendario", "bilingue", "caracter",
             "genero_colegio", "genero", "tiene_internet", "tiene_computador", "tiene_automovil", "tiene_lavadora"]
    out[texto] = out[texto].fillna("Sin dato")
    out["cod_depto"] = out["cod_depto"].fillna("00")
    return out.reset_index(drop=True)


def etiquetas(df):
    """Versiones en texto de las variables que el tablero muestra por categorias."""
    est = df["estrato"].map(lambda v: "Sin dato" if pd.isna(v) else ("Sin estrato" if v == 0 else f"Estrato {int(v)}"))
    ppc = df["personas_hogar"] / df["cuartos_hogar"]
    hacinamiento = pd.cut(ppc, [0, 1, 2, 3, np.inf], labels=["Hasta 1", "Más de 1 a 2", "Más de 2 a 3", "Más de 3"])
    return {
        "Total": pd.Series("Total", index=df.index),
        "Estrato": est,
        "Educación de la madre": df["educacion_madre"].map(EDUCACION_TXT).fillna("Sin dato"),
        "Educación del padre": df["educacion_padre"].map(EDUCACION_TXT).fillna("Sin dato"),
        "Jornada": df["jornada"],
        "Internet": df["tiene_internet"],
        "Computador": df["tiene_computador"],
        "Automóvil": df["tiene_automovil"],
        "Lavadora": df["tiene_lavadora"],
        "Personas por cuarto": hacinamiento.astype(object).where(hacinamiento.notna(), "Sin dato"),
        "Género": df["genero"],
    }


def resumir(df):
    """Tabla larga: por cada combinacion de filtros y cada categoria, cuantos estudiantes, cuantos con
    bajo puntaje y la suma de puntajes (para sacar promedios despues de filtrar)."""
    partes = []
    for variable, valores in etiquetas(df).items():
        t = (df.assign(valor=valores.values)
             .groupby(LLAVES + ["valor"], as_index=False)
             .agg(estudiantes=("bajo_puntaje", "size"), bajo=("bajo_puntaje", "sum"), suma_puntaje=("punt_global", "sum")))
        t.insert(len(LLAVES), "variable", variable)
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def distribucion(df):
    """Cuantos estudiantes hay en cada tramo de 5 puntos del puntaje global, por combinacion de filtros y estrato.
    Con estos conteos el tablero dibuja la curva de distribucion y calcula los cuartiles del grafico de cajas."""
    tramo = (df["punt_global"] // TRAMO * TRAMO).astype(int)
    return (df.assign(tramo=tramo, estrato_txt=etiquetas(df)["Estrato"].values)
            .groupby(LLAVES + ["estrato_txt", "tramo"], as_index=False).size().rename(columns={"size": "estudiantes"}))


def buscar_csv_completo():
    """El CSV original del ICFES es el archivo de mas de 500 MB dentro de data/."""
    grandes = [p for p in (ROOT / "data").glob("*.csv") if p.stat().st_size > 500_000_000]
    if len(grandes) != 1:
        raise SystemExit("Deja exactamente un CSV grande (el original del ICFES) dentro de data/, o pasa su ruta.")
    return grandes[0]


def leer_por_pedazos(ruta):
    return pd.read_csv(ruta, dtype=str, usecols=lambda c: c in COLUMNAS_FUENTE, chunksize=TAMANO_PEDAZO,
                       encoding="utf-8", low_memory=False)


def id_numerico(serie):
    """Convierte el identificador del ICFES en un numero (hash) para detectar repetidos usando poca memoria."""
    return pd.util.hash_pandas_object(serie, index=False).to_numpy()


def resumen_completo(ruta):
    """Arma el corte y las tablas del tablero con TODO el archivo, leyendolo por pedazos en dos vueltas.
    Vuelta 1: junta el puntaje de cada estudiante unico para calcular el corte del 25%.
    Vuelta 2: limpia cada pedazo, descarta estudiantes ya vistos, marca bajo puntaje y suma los conteos."""
    print(f"Vuelta 1 de 2: calculando el corte con {ruta.name} ...")
    ids, puntajes, leidas = [], [], 0
    for pedazo in leer_por_pedazos(ruta):
        leidas += len(pedazo)
        pg = pd.to_numeric(pedazo["PUNT_GLOBAL"], errors="coerce")
        ok = (pg > 0) & (pd.to_numeric(pedazo["PERIODO"], errors="coerce") >= 20142) & pedazo["ESTU_ESTADOINVESTIGACION"].eq("PUBLICAR")
        ids.append(id_numerico(pedazo.loc[ok, "ESTU_CONSECUTIVO"]))
        puntajes.append(pg[ok].to_numpy())
        print(f"  {leidas:,} filas leidas", end="\r", flush=True)
    ids, puntajes = np.concatenate(ids), np.concatenate(puntajes)
    _, primero = np.unique(ids, return_index=True)          # una sola vez cada estudiante
    corte = float(pd.Series(puntajes[primero]).quantile(PERCENTIL_CORTE))
    unicos = len(primero)
    print(f"\n  {unicos:,} estudiantes unicos | corte: puntaje global menor a {corte:.0f}")

    print("Vuelta 2 de 2: armando las tablas del tablero ...")
    vistos = np.array([], dtype=np.uint64)
    partes_res, partes_dist, total, leidas = [], [], 0, 0
    for pedazo in leer_por_pedazos(ruta):
        leidas += len(pedazo)
        limpio = limpiar(pedazo)
        h = id_numerico(limpio["estudiante_id"])
        nuevos = ~np.isin(h, vistos)
        limpio, vistos = limpio[nuevos], np.concatenate([vistos, h[nuevos]])
        limpio["bajo_puntaje"] = (limpio["punt_global"] < corte).astype(int)
        total += len(limpio)
        partes_res.append(resumir(limpio))
        partes_dist.append(distribucion(limpio))
        print(f"  {leidas:,} filas leidas | {total:,} estudiantes", end="\r", flush=True)
    print()
    claves_res = LLAVES + ["variable", "valor"]
    resumen = pd.concat(partes_res).groupby(claves_res, as_index=False)[["estudiantes", "bajo", "suma_puntaje"]].sum()
    dist = pd.concat(partes_dist).groupby(LLAVES + ["estrato_txt", "tramo"], as_index=False).estudiantes.sum()
    if total != unicos:
        raise SystemExit(f"Error de consistencia: vuelta 1 conto {unicos:,} estudiantes y vuelta 2 {total:,}.")
    return corte, total, resumen, dist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--completo", nargs="?", const="auto", default=None,
                        help="arma el tablero con el CSV completo del ICFES (opcional: su ruta)")
    args = parser.parse_args()

    raw = pd.read_csv(ENTRADA, dtype=str)
    out = limpiar(raw)

    if args.completo:
        ruta = buscar_csv_completo() if args.completo == "auto" else Path(args.completo)
        corte, estudiantes, resumen, dist = resumen_completo(ruta)
        fuente = ruta.name
    else:
        # Variable objetivo: un solo corte para todos los anos, porque el ICFES reporta el puntaje global
        # en una escala historica comparable entre aplicaciones desde 2014-2.
        corte = float(out["punt_global"].quantile(PERCENTIL_CORTE))
        estudiantes, fuente = len(out), ENTRADA.name

    # La muestra (que usa el modelo) siempre se marca con el mismo corte que el tablero.
    out["bajo_puntaje"] = (out["punt_global"] < corte).astype(int)
    if not args.completo:
        resumen, dist = resumir(out), distribucion(out)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    CORTE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALIDA, index=False, encoding="utf-8")
    resumen.to_csv(RESUMEN, index=False, encoding="utf-8")
    dist.to_csv(DISTRIBUCION, index=False, encoding="utf-8")
    CORTE.write_text(json.dumps({"percentil": PERCENTIL_CORTE, "corte_puntaje_global": corte,
                                 "fuente": fuente, "estudiantes": int(estudiantes)}, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    print(f"Muestra para el modelo: {len(out):,} estudiantes | {out['bajo_puntaje'].mean():.1%} con bajo puntaje")
    print(f"Tablero: {estudiantes:,} estudiantes ({fuente})")
    print(f"Corte de bajo puntaje: puntaje global menor a {corte:.0f}")
    print(f"Tabla resumen: {len(resumen):,} filas ({RESUMEN.stat().st_size / 1e6:.1f} MB) | "
          f"distribucion: {len(dist):,} filas ({DISTRIBUCION.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
