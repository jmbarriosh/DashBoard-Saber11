# Dashboard de alerta temprana de bajo puntaje en Saber 11 (Dash)

Adaptacion del proyecto base "Dashboard de desercion academica" a datos reales del ICFES:
[Resultados unicos Saber 11](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe/about_data) (datos.gov.co).

**Pregunta:** con datos del estudiante, su hogar y su colegio (sin usar sus puntajes), ¿se puede estimar la
probabilidad de que quede en el 25% de puntajes globales mas bajos?

## Estructura

```text
app.py                             Dashboard Dash (4 pestañas: panorama, factores del hogar, modelo, casos de riesgo)
sacar_muestra.py                   Lee el CSV completo (~3,6 GB), quita duplicados y saca una muestra aleatoria de estudiantes
preparar_datos.py                  Limpieza, variable objetivo bajo_puntaje y tablas resumen del tablero
entrenar_modelo.py                 Entrenamiento, ajuste y exportacion de resultados (3 iteraciones)
exportar_html.py                   Genera tablero_saber11.html: el mismo tablero en un solo archivo para compartir
plantillas/tablero.html            Plantilla (HTML + JavaScript) que usa exportar_html.py
data/muestra_saber11.csv           Muestra aleatoria de 60.000 estudiantes unicos
data/conteo_periodos.csv           Filas y estudiantes unicos por periodo: archivo completo vs muestra
data/muestra_info.json             Como se saco la muestra (filtros, semilla, totales)
data/saber11_preparado.csv         Datos limpios usados por el modelo
data/resumen_tablero.csv           Conteos por combinacion de filtros (los usa el tablero)
data/distribucion_puntaje.csv      Estudiantes por tramo de 10 puntos (grafico de distribucion)
data/colombia_departamentos.geojson  Mapa de departamentos (codigo DANE)
outputs/                           Corte, metricas, predicciones e importancia
models/                            Modelo serializado
assets/style.css                   Estilos del dashboard
tests/test_pipeline.py             Pruebas de integridad
```

## Decisiones principales

- Solo se usan los periodos **2014-2 a 2022**: antes de 2014-2 el examen tenia otra estructura y no existe el puntaje global.
- Se excluyen registros con puntaje 0 o con estado distinto a "PUBLICAR".
- **Duplicados:** el archivo oficial trae copias exactas de filas (los periodos 20194 y 20224 vienen dos veces y 20181
  tiene un 39% repetido). Se verifico que ningun estudiante aparece con datos distintos, asi que se quitan antes de muestrear.
- **Muestra:** muestreo aleatorio simple de 60.000 estudiantes unicos (cada estudiante tiene la misma probabilidad de salir).
- **Variable objetivo:** `bajo_puntaje = 1` si el puntaje global es menor al percentil 25 de todos los anos juntos.
  Se usa un solo corte porque el ICFES reporta el puntaje global en una escala historica comparable entre aplicaciones.
- **Semestre de aplicacion:** los periodos que terminan en 1 son del primer semestre (sobre todo calendario B) y los que
  terminan en 2 o 4, del segundo (calendario A). El archivo no trae el segundo semestre de 2018, 2020 ni 2021.
- **Sin fuga de informacion:** los puntajes por materia no se usan como variables del modelo, porque el puntaje global se calcula a partir de ellos.
- Tres iteraciones: linea base (3 variables), regresion logistica regularizada y bosque aleatorio. Se elige la de mayor
  ROC-AUC en validacion cruzada (solo entrenamiento); el conjunto de prueba solo se usa para medir al ganador.

## Ejecucion rapida

Requiere Python 3.11 a 3.13 (con 3.15 algunas librerias aun no traen instaladores listos).

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Abra `http://127.0.0.1:8050`. Los datos preparados y los resultados ya vienen incluidos.
Tambien puede abrir el dashboard con doble clic en `ejecutar_dashboard.bat` (despues de crear `.venv`).

Para regenerar todo (con el CSV completo del ICFES dentro de `data/`):

```powershell
python sacar_muestra.py                # 1-2 minutos: muestra de 60.000 estudiantes para el modelo
python preparar_datos.py --completo    # 3-5 minutos: tablero con los 3,4 millones de estudiantes
python entrenar_modelo.py              # 2-4 minutos
python exportar_html.py                # tablero_saber11.html para compartir
```

Sin `--completo`, el tablero se arma solo con la muestra (mas rapido, util para hacer cambios).

## Version HTML para compartir

`tablero_saber11.html` es el mismo tablero en un solo archivo: se abre con doble clic y funciona sin internet
ni Python (lleva adentro los datos resumidos y la libreria de graficos; los filtros se calculan en el navegador).
Se verifico que da exactamente los mismos numeros que `app.py` (totales, departamentos, jornada, matriz de
confusion y AUC) en varias combinaciones de filtros.

## Pruebas

```bash
pytest -q
```

## Publicacion en GitHub y Binder

1. Cree un repositorio publico y cargue el contenido de esta carpeta (sin `.venv`).
2. Reemplace `USUARIO` y `REPOSITORIO` en `ENLACES.txt`.
3. Abra el enlace Binder. El archivo `binder/start` inicia Dash y lo expone mediante `jupyter-server-proxy`.

## Creditos del mapa

`data/colombia_departamentos.geojson` se convirtio desde el TopoJSON de departamentos del repositorio
[jacasta2/colombian_map](https://github.com/jacasta2/colombian_map) (licencia MIT), elaborado a partir del
Marco Geoestadistico Nacional del DANE.

## Nota etica

Los datos son publicos y anonimos (sin nombres ni documentos). El modelo muestra asociaciones, no causas:
variables como estrato, departamento o jornada reflejan desigualdades del contexto y no deben usarse para
etiquetar ni excluir estudiantes. Es un ejercicio academico, no una herramienta de decision.
