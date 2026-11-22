"""Saca una muestra aleatoria de Saber 11 a partir del CSV completo del ICFES (~3,6 GB).

Que hace distinto a la version anterior:
  1. Lee solo las columnas que usa el proyecto (mas rapido y la muestra pesa mucho menos).
  2. Filtra MIENTRAS lee: solo periodos con puntaje global (2014-2 en adelante), puntaje mayor a 0
     y resultados en estado "PUBLICAR". Asi las 60.000 filas de la muestra son todas utiles.
  3. Quita los duplicados del archivo oficial. Se verifico que el archivo trae copias exactas
     (los periodos 20194 y 20224 vienen dos veces y 20181 tiene un 39% repetido).
  4. Hace un sorteo exacto en una sola pasada: a cada ESTUDIANTE le asigna un numero al azar que
     depende de su identificador (ESTU_CONSECUTIVO), asi sus copias reciben el mismo numero y no
     tiene mas oportunidades de salir que los demas. Al final se queda con los 60.000 numeros mas
     pequenos. Equivale a sacar 60.000 papelitos de una bolsa con un papelito por estudiante.
  5. Guarda cuantas filas y cuantos estudiantes unicos hay por periodo en el archivo completo y en
     la muestra, para comprobar que la muestra representa bien al archivo.

Uso (con el entorno .venv activado, desde la carpeta del proyecto):
    python sacar_muestra.py                      # busca el CSV grande dentro de data/
    python sacar_muestra.py "C:\\ruta\\archivo.csv"
    python sacar_muestra.py --filas 80000        # otro tamano de muestra

Salidas (en data/):
    muestra_saber11.csv          la muestra
    conteo_periodos.csv          filas por periodo: archivo completo vs muestra
    muestra_info.json            resumen de como se saco la muestra
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

FILAS_OBJETIVO = 60_000   # tamano final de la muestra
TAMANO_PEDAZO = 250_000   # filas que se leen en memoria cada vez
SEMILLA = 42              # para que la muestra salga siempre igual
CLAVE_SORTEO = f"semilla_{SEMILLA:08d}"  # 16 caracteres, la pide la funcion de hash
PERIODO_MINIMO = 20142    # desde 2014-2 existe el puntaje global

ID = "ESTU_CONSECUTIVO"
COLUMNAS = [
    "PERIODO", ID, "ESTU_ESTADOINVESTIGACION", "ESTU_FECHANACIMIENTO", "ESTU_GENERO",
    "COLE_COD_DEPTO_UBICACION", "COLE_DEPTO_UBICACION", "COLE_MCPIO_UBICACION", "COLE_NATURALEZA", "COLE_AREA_UBICACION",
    "COLE_JORNADA", "COLE_CALENDARIO", "COLE_BILINGUE", "COLE_CARACTER", "COLE_GENERO",
    "FAMI_ESTRATOVIVIENDA", "FAMI_EDUCACIONMADRE", "FAMI_EDUCACIONPADRE", "FAMI_PERSONASHOGAR",
    "FAMI_CUARTOSHOGAR", "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR", "FAMI_TIENEAUTOMOVIL",
    "FAMI_TIENELAVADORA",
    "PUNT_GLOBAL", "PUNT_MATEMATICAS", "PUNT_LECTURA_CRITICA", "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_C_NATURALES", "PUNT_INGLES",
]
OBLIGATORIAS = ["PERIODO", ID, "PUNT_GLOBAL"]


def buscar_csv():
    candidatos = [p for p in DATA.glob("*.csv") if p.stat().st_size > 500_000_000]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise SystemExit("No encontre el CSV grande dentro de data/. Pasa la ruta: python sacar_muestra.py \"C:\\ruta\\archivo.csv\"")
    raise SystemExit("Hay varios CSV grandes en data/. Indica cual: " + ", ".join(p.name for p in candidatos))


def detectar_formato(ruta):
    """Mira el inicio del archivo para saber la codificacion (tildes) y el separador."""
    with open(ruta, "rb") as f:
        inicio = f.read(20_000_000)
    inicio = inicio[: inicio.rfind(b"\n")]  # no cortar un caracter a la mitad
    if inicio.startswith(b"\xef\xbb\xbf"):
        codificacion = "utf-8-sig"
    else:
        try:
            inicio.decode("utf-8")
            codificacion = "utf-8"
        except UnicodeDecodeError:
            codificacion = "latin-1"
    encabezado = inicio.split(b"\n", 1)[0].decode(codificacion, errors="replace")
    separador = ";" if encabezado.count(";") > encabezado.count(",") else ","
    columnas = [c.strip().strip('"').lstrip("\ufeff") for c in encabezado.strip().split(separador)]
    return codificacion, separador, columnas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ruta", nargs="?", help="CSV completo de Resultados unicos Saber 11")
    parser.add_argument("--filas", type=int, default=FILAS_OBJETIVO)
    args = parser.parse_args()

    ruta = Path(args.ruta.strip().strip('"').strip("'")) if args.ruta else buscar_csv()
    if not ruta.exists():
        raise SystemExit(f"No encuentro el archivo: {ruta}")

    codificacion, separador, encabezado = detectar_formato(ruta)
    faltan = [c for c in OBLIGATORIAS if c not in encabezado]
    if faltan:
        raise SystemExit(f"Al archivo le faltan columnas obligatorias: {faltan}")
    usar = [c for c in COLUMNAS if c in encabezado]
    no_estan = [c for c in COLUMNAS if c not in encabezado]
    print(f"Archivo: {ruta.name} ({ruta.stat().st_size / 1e9:.2f} GB) | codificacion {codificacion} | separador '{separador}'")
    if no_estan:
        print(f"Aviso: estas columnas no vienen en el archivo y se omiten: {no_estan}")

    guardadas = None
    total_por_periodo, validas_por_periodo = {}, {}
    ids_validos, periodos_validos = [], []
    leidas = validas = 0
    t0 = time.time()

    lector = pd.read_csv(ruta, dtype=str, usecols=usar, sep=separador, encoding=codificacion,
                         chunksize=TAMANO_PEDAZO, low_memory=False)
    for pedazo in lector:
        leidas += len(pedazo)
        for k, v in pedazo["PERIODO"].value_counts().items():
            total_por_periodo[k] = total_por_periodo.get(k, 0) + int(v)

        periodo = pd.to_numeric(pedazo["PERIODO"], errors="coerce")
        puntaje = pd.to_numeric(pedazo["PUNT_GLOBAL"], errors="coerce")
        ok = (periodo >= PERIODO_MINIMO) & (puntaje > 0)
        if "ESTU_ESTADOINVESTIGACION" in pedazo:
            ok &= pedazo["ESTU_ESTADOINVESTIGACION"].str.strip().eq("PUBLICAR")
        buenas = pedazo[ok].copy()
        validas += len(buenas)
        for k, v in buenas["PERIODO"].value_counts().items():
            validas_por_periodo[k] = validas_por_periodo.get(k, 0) + int(v)

        # Sorteo: cada estudiante recibe un numero al azar entre 0 y 1 que depende solo de su
        # identificador. Las copias de un mismo estudiante reciben el mismo numero.
        h = pd.util.hash_pandas_object(buenas[ID], index=False, hash_key=CLAVE_SORTEO).to_numpy()
        buenas["_azar"] = h / 2.0**64
        ids_validos.append(h)
        periodos_validos.append(buenas["PERIODO"].to_numpy())
        juntas = buenas if guardadas is None else pd.concat([guardadas, buenas], ignore_index=True)
        juntas = juntas.sort_values("_azar").drop_duplicates(ID, keep="first")
        guardadas = juntas.head(args.filas)

        mins = (time.time() - t0) / 60
        print(f"  {leidas:>12,} filas leidas | {validas:>12,} validas | {mins:5.1f} min", flush=True)

    if guardadas is None or guardadas.empty:
        raise SystemExit("No quedo ninguna fila valida. Revisa el archivo.")

    muestra = guardadas.sort_values(["PERIODO", "_azar"]).drop(columns="_azar").reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(DATA / "muestra_saber11.csv", index=False, encoding="utf-8")

    unicos = (pd.DataFrame({"periodo": np.concatenate(periodos_validos), "id": np.concatenate(ids_validos)})
              .drop_duplicates().groupby("periodo").size())
    conteo = pd.DataFrame({"filas_archivo": pd.Series(total_por_periodo),
                           "filas_validas": pd.Series(validas_por_periodo),
                           "estudiantes_unicos": unicos}).fillna(0).astype(int)
    conteo.index.name = "periodo"
    conteo["filas_muestra"] = muestra["PERIODO"].value_counts().reindex(conteo.index).fillna(0).astype(int)
    conteo["pct_unicos"] = (conteo.estudiantes_unicos / conteo.estudiantes_unicos.sum() * 100).round(2)
    conteo["pct_muestra"] = (conteo.filas_muestra / conteo.filas_muestra.sum() * 100).round(2)
    conteo = conteo.sort_index()
    conteo.to_csv(DATA / "conteo_periodos.csv", encoding="utf-8")

    info = {
        "archivo_origen": ruta.name,
        "fecha_muestra": datetime.now().isoformat(timespec="seconds"),
        "filas_leidas": leidas,
        "filas_validas": validas,
        "estudiantes_unicos_validos": int(conteo.estudiantes_unicos.sum()),
        "filas_muestra": len(muestra),
        "fraccion_muestreada": round(len(muestra) / int(conteo.estudiantes_unicos.sum()), 5),
        "semilla": SEMILLA,
        "filtros": f"PERIODO >= {PERIODO_MINIMO}, PUNT_GLOBAL > 0, ESTU_ESTADOINVESTIGACION = PUBLICAR, sin duplicados de {ID}",
        "metodo": "Muestreo aleatorio simple sin reemplazo sobre estudiantes unicos (numero al azar por identificador, se conservan los mas bajos)",
        "columnas": list(muestra.columns),
    }
    (DATA / "muestra_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nListo en {(time.time() - t0) / 60:.1f} min.")
    print(f"Filas leidas: {leidas:,} | validas: {validas:,} | estudiantes unicos validos: {int(conteo.estudiantes_unicos.sum()):,} | muestra: {len(muestra):,}")
    print(conteo.to_string())


if __name__ == "__main__":
    main()
