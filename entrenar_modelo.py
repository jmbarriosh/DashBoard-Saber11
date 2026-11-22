"""Entrena, compara y documenta tres iteraciones del modelo de bajo puntaje en Saber 11.

Pregunta: con datos del estudiante, su hogar y su colegio (sin usar sus puntajes),
¿se puede estimar la probabilidad de que quede en el 25% de puntajes globales mas bajos?
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "saber11_preparado.csv"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
TARGET = "bajo_puntaje"
ID = "estudiante_id"

CATEGORICAS = [
    "departamento", "naturaleza", "area", "jornada", "calendario", "bilingue", "caracter",
    "genero_colegio", "genero", "tiene_internet", "tiene_computador", "tiene_automovil", "tiene_lavadora",
]
NUMERICAS = ["anio", "edad", "estrato", "educacion_madre", "educacion_padre", "personas_hogar", "cuartos_hogar"]
RAW_FEATURES = NUMERICAS + CATEGORICAS
ENGINEERED = ["bienes_hogar", "personas_por_cuarto", "educacion_max_padres"]
BASE_FEATURES = ["estrato", "educacion_madre", "naturaleza"]
# Columnas que se guardan en las predicciones para el dashboard (incluye puntajes solo para mostrar).
COLUMNAS_TABLA = [ID, TARGET, "punt_global", "periodo", "cod_depto"] + RAW_FEATURES


def agregar_caracteristicas(df):
    x = df.copy()
    bienes = ["tiene_internet", "tiene_computador", "tiene_automovil", "tiene_lavadora"]
    x["bienes_hogar"] = (x[bienes] == "Si").sum(axis=1)                      # 0 a 4 bienes en casa
    x["personas_por_cuarto"] = x["personas_hogar"] / x["cuartos_hogar"]     # hacinamiento
    x["educacion_max_padres"] = x[["educacion_madre", "educacion_padre"]].max(axis=1)
    return x


def _preprocesador(features, escalar):
    cat = [c for c in features if c in CATEGORICAS]
    num = [c for c in features if c not in cat]
    pasos_num = [("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if escalar else [])
    return ColumnTransformer([
        ("num", Pipeline(pasos_num), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Sin dato")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def pipeline_logistica(features):
    return Pipeline([("prep", _preprocesador(features, True)), ("model", LogisticRegression(max_iter=3000, random_state=42))])


def pipeline_bosque(features):
    return Pipeline([("prep", _preprocesador(features, False)), ("model", RandomForestClassifier(random_state=42, n_jobs=-1))])


def metricas(nombre, modelo, X_train, X_test, y_train, y_test, variables, regularizacion, params, auc_cv):
    p_train = modelo.predict_proba(X_train)[:, 1]
    p_test = modelo.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.50).astype(int)
    return {
        "version": nombre,
        "variables": variables,
        "regularizacion": regularizacion,
        "hiperparametros": json.dumps(params, ensure_ascii=False, default=str),
        "roc_auc_validacion_cruzada": auc_cv,
        "roc_auc_entrenamiento": roc_auc_score(y_train, p_train),
        "roc_auc_prueba": roc_auc_score(y_test, p_test),
        "brecha_auc": roc_auc_score(y_train, p_train) - roc_auc_score(y_test, p_test),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }, p_test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = agregar_caracteristicas(pd.read_csv(DATA, dtype={"periodo": str, "cod_depto": str}))
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=df[TARGET])
    train, test = df.iloc[idx_train].copy(), df.iloc[idx_test].copy()
    y_train, y_test = train[TARGET], test[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados, predicciones = [], {}
    full_features = RAW_FEATURES + ENGINEERED
    n_vars = f"{len(RAW_FEATURES)} originales + {len(ENGINEERED)} derivadas"

    print("1/3 Linea base...")
    base = pipeline_logistica(BASE_FEATURES).set_params(model__class_weight="balanced")
    auc_cv_base = cross_val_score(base, train[BASE_FEATURES], y_train, scoring="roc_auc", cv=cv, n_jobs=-1).mean()
    base.fit(train[BASE_FEATURES], y_train)
    row, p = metricas("1. Línea base", base, train[BASE_FEATURES], test[BASE_FEATURES], y_train, y_test,
                      "3 variables (estrato, educ. madre, naturaleza)", "L2, C=1, clases balanceadas", {"C": 1.0, "class_weight": "balanced"},
                      auc_cv_base)
    resultados.append(row); predicciones[row["version"]] = p

    print("2/3 Logistica regularizada (busqueda de hiperparametros)...")
    log_search = GridSearchCV(
        pipeline_logistica(full_features),
        {"model__C": [0.05, 0.2, 1.0, 5.0], "model__penalty": ["l1", "l2"],
         "model__solver": ["liblinear"], "model__class_weight": [None, "balanced"]},
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    log_search.fit(train[full_features], y_train)
    log_model = log_search.best_estimator_
    bp = log_search.best_params_
    row, p = metricas("2. Logística regularizada", log_model, train[full_features], test[full_features], y_train, y_test,
                      n_vars, f"{bp['model__penalty'].upper()}, C={bp['model__C']}", bp, log_search.best_score_)
    resultados.append(row); predicciones[row["version"]] = p

    print("3/3 Bosque aleatorio (busqueda de hiperparametros, puede tardar unos minutos)...")
    rf_search = GridSearchCV(
        pipeline_bosque(full_features),
        {"model__n_estimators": [150], "model__max_depth": [8, 14],
         "model__min_samples_leaf": [20], "model__max_features": ["sqrt"],
         "model__class_weight": ["balanced"]},
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    rf_search.fit(train[full_features], y_train)
    rf_model = rf_search.best_estimator_
    row, p = metricas("3. Bosque ajustado", rf_model, train[full_features], test[full_features], y_train, y_test,
                      n_vars, "Control por profundidad y hoja minima", rf_search.best_params_, rf_search.best_score_)
    resultados.append(row); predicciones[row["version"]] = p

    # El ganador se elige con la validacion cruzada (solo datos de entrenamiento). El conjunto de prueba
    # se reserva para medir al ganador, asi su nota no queda inflada por haberlo usado para escoger.
    resultados_df = pd.DataFrame(resultados).sort_values("roc_auc_validacion_cruzada", ascending=False).reset_index(drop=True)
    ganador = resultados_df.iloc[0]["version"]
    modelos = {"1. Línea base": base, "2. Logística regularizada": log_model, "3. Bosque ajustado": rf_model}
    best_model = modelos[ganador]
    best_features = BASE_FEATURES if ganador == "1. Línea base" else full_features

    pred_test = test[COLUMNAS_TABLA].copy()
    pred_test["probabilidad_bajo"] = predicciones[ganador]
    pred_test["prediccion_050"] = (pred_test["probabilidad_bajo"] >= 0.50).astype(int)
    pred_test["nivel_riesgo"] = pd.cut(pred_test["probabilidad_bajo"], [-0.01, 0.30, 0.60, 1.0], labels=["Bajo", "Medio", "Alto"])

    print("Calculando importancia de variables...")
    perm = permutation_importance(best_model, test[best_features], y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"variable": best_features, "importancia": perm.importances_mean}).sort_values("importancia", ascending=False)

    resultados_df.to_csv(OUT / "comparacion_modelos.csv", index=False)
    pred_test.to_csv(OUT / "predicciones_prueba.csv", index=False)
    importance.to_csv(OUT / "importancia_variables.csv", index=False)
    joblib.dump({"modelo": best_model, "variables": best_features, "ganador": ganador}, MODEL_DIR / "modelo_bajo_puntaje.joblib")
    corte = json.loads((OUT / "corte_bajo_puntaje.json").read_text(encoding="utf-8"))
    resumen = {
        "modelo_seleccionado": ganador,
        "registros_totales": int(len(df)),
        "registros_prueba": int(len(test)),
        "tasa_bajo_puntaje": float(df[TARGET].mean()),
        "corte_puntaje_global": corte["corte_puntaje_global"],
        "auc_validacion_cruzada_ganador": float(resultados_df.iloc[0]["roc_auc_validacion_cruzada"]),
        "auc_prueba_ganador": float(resultados_df.iloc[0]["roc_auc_prueba"]),
        "semilla": 42,
        "criterio_seleccion": "Mayor ROC-AUC promedio en validacion cruzada de 5 particiones (solo entrenamiento); la prueba solo se usa para medir",
    }
    (OUT / "resumen_modelo.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(resultados_df[["version", "roc_auc_validacion_cruzada", "roc_auc_entrenamiento", "roc_auc_prueba", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
