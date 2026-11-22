# Guia de entrega y correspondencia con la actividad

## Entregable 1. Informe del proceso

- PENDIENTE (lo redacta el equipo): informe del proceso con problema, decisiones, datos, procesamiento, modelos, regularizacion, hiperparametros, metricas, conclusiones, limitaciones y pruebas.
- Los numeros para el informe estan en `outputs/resumen_modelo.json` y `outputs/comparacion_modelos.csv`.

## Entregable 2. Carpeta del dashboard

- `app.py`: tablero Dash con graficos, tablas, filtros y callbacks.
- `assets/style.css`: apariencia y diseno adaptable.
- `data/`: muestra Saber 11, datos preparados y diccionario.
- `outputs/`: comparacion de modelos, predicciones de prueba, importancia y resumen.
- `models/`: modelo seleccionado.
- `sacar_muestra.py`, `preparar_datos.py` y `entrenar_modelo.py`: reproduccion de datos y analisis.
- `requirements.txt`, `environment.yml` y `Procfile`: configuracion.
- `tests/`: pruebas automatizadas.

## Entregable 3. Archivo con enlaces

- `ENLACES.txt`: contiene las plantillas de GitHub y Binder.

Los enlaces quedan deliberadamente como plantilla hasta que el autor publique la carpeta en un repositorio de su cuenta. Tras reemplazar `USUARIO` y `REPOSITORIO`, el enlace de Binder abre el dashboard mediante el proxy del puerto 8050.

## Inicio rapido

1. Cree y active un entorno virtual.
2. Ejecute `pip install -r requirements.txt`.
3. Ejecute `python app.py`.
4. Abra `http://127.0.0.1:8050`.

Los datos y resultados ya estan incluidos. Solo ejecute `preparar_datos.py` y `entrenar_modelo.py` si desea regenerarlos.
