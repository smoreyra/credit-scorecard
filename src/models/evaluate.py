"""
=========================================
MODEL EVALUATION METRICS - QUICK GUIDE
=========================================

--- CLASIFICACIÓN (Binary Classification) ---

Accuracy:
    Proporción de predicciones correctas.
    Uso: útil cuando las clases están balanceadas.
    ⚠️ Puede ser engañoso con datos desbalanceados.

Precision:
    TP / (TP + FP)
    Qué porcentaje de los positivos predichos son correctos.
    Uso: cuando el costo de falsos positivos es alto.

Recall (Sensitivity):
    TP / (TP + FN)
    Qué proporción de positivos reales detecta el modelo.
    Uso: cuando perder positivos es costoso.

Specificity:
    TN / (TN + FP)
    Qué proporción de negativos detecta correctamente.

F1 Score:
    Media armónica entre precision y recall.
    Uso: balance entre FP y FN (muy usado en datasets desbalanceados).

--- PROBABILÍSTICAS / RANKING ---

ROC-AUC:
    Mide capacidad del modelo para rankear positivos > negativos.
    Uso: evaluación general de discriminación.
    Valor:
        0.5 = azar
        1 = perfecto

PR-AUC:
    Área bajo la curva Precision-Recall.
    Uso: mejor que ROC-AUC cuando hay clases desbalanceadas.

Log Loss:
    Penaliza probabilidades incorrectas (especialmente seguras).
    Uso: optimización de modelos probabilísticos.

Brier Score:
    Error cuadrático medio de probabilidades.
    Uso: mide calidad de probabilidades (calibración + error).
    Valor:
        0 = perfecto

--- THRESHOLD-DEPENDENT ---

Confusion Matrix:
    TP, FP, TN, FN
    Uso: análisis detallado de errores.

F1 (with threshold):
    F1 calculado usando un threshold específico.
    Uso: elegir punto de decisión del modelo.

--- REGRESIÓN ---

MAE:
    Error absoluto promedio.
    Uso: robusto a outliers.

MSE:
    Error cuadrático promedio.
    Uso: penaliza errores grandes.

RMSE:
    Raíz de MSE (misma escala que el target).
    Uso: interpretación más intuitiva.

R²:
    Proporción de varianza explicada por el modelo.
    Valor:
        1 = perfecto
        0 = no mejora sobre baseline

--- CALIBRACIÓN ---

Calibration (deciles):
    Compara probabilidad predicha vs frecuencia real.
    Uso: verificar si las probabilidades son confiables.
    Ejemplo:
        Si el modelo predice 0.8 → debería ocurrir ~80% de las veces.

=========================================
RECOMMENDATIONS
=========================================

- Nunca usar una sola métrica.
- Para clasificación:
    -> ROC-AUC / PR-AUC + F1
- Para probabilidades:
    -> Brier + Log Loss + Calibration
- Para regresión:
    -> RMSE + MAE + R²

"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    # Clasificación
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    
    # Regresión
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Verificaciones
def _has_proba(model):
    """ 
    Verifica si el modelo tiene método predict_proba
    """

    return hasattr(model, "predict_proba")


def _safe_divide(a, b):
    """ 
    Evita división por cero 
    """

    return a / b if b != 0 else 0


# Clasificación
def evaluate_classification(model, X, y, average="weighted", threshold=0.5):
    """
    Evaluación general para modelos de clasificación (binaria y multiclase).

    Parámetros
    ----------
    model : object
        Modelo entrenado con método `predict` y opcionalmente `predict_proba`.

    X : array-like (n_samples, n_features)
        Features de entrada.

    y : array-like (n_samples,)
        Labels verdaderos.

    average : str, opcional (default="weighted")
        Tipo de promedio para métricas multiclase:
        - "binary" (solo binaria)
        - "micro"
        - "macro"
        - "weighted"

    threshold : float, opcional (default=0.5)
        Umbral para clasificación binaria (solo usado si hay probabilidades y problema binario).

    Retorna
    -------
    dict
        Diccionario con métricas de evaluación. Puede contener:

        Métricas básicas:
        - accuracy : float
            Proporción de predicciones correctas.
            Rango: [0, 1]
            Ideal: cercano a 1
            Nota: puede ser engañosa en datasets desbalanceados.

        - f1 : float
            Media armónica entre precision y recall (equilibrio entre ambos).
            Rango: [0, 1]
            Ideal: cercano a 1

        - precision : float
            De todos los positivos predichos, cuántos son correctos.
            Rango: [0, 1]
            Ideal: alto (especialmente importante cuando los falsos positivos son costosos).

        - recall : float
            De todos los positivos reales, cuántos detecta el modelo.
            Rango: [0, 1]
            Ideal: alto (importante cuando no querés perder casos positivos).

        Matriz de confusión:
        - confusion_matrix : array (n_classes x n_classes)
            Tabla que muestra conteos de aciertos y errores por clase.

        Métricas adicionales para clasificación binaria:
        - tp : int
            Verdaderos positivos (predijo 1 y era 1).
            Ideal: alto

        - fp : int
            Falsos positivos (predijo 1 pero era 0).
            Ideal: bajo

        - tn : int
            Verdaderos negativos (predijo 0 y era 0).
            Ideal: alto

        - fn : int
            Falsos negativos (predijo 0 pero era 1).
            Ideal: bajo

        - specificity : float
            Capacidad de identificar correctamente la clase negativa.
            Fórmula: TN / (TN + FP)
            Rango: [0, 1]
            Ideal: cercano a 1

        Métricas probabilísticas (si el modelo soporta `predict_proba`):
        - roc_auc : float
            Área bajo la curva ROC (capacidad de separar clases).
            Rango: [0, 1]
            Ideal: > 0.8 bueno, > 0.9 excelente, 0.5 = azar

        - pr_auc : float
            Área bajo la curva Precision-Recall.
            Rango: [0, 1]
            Ideal: cercano a 1 (más útil en datasets desbalanceados)

        - log_loss : float
            Penaliza probabilidades incorrectas (qué tan bien calibradas están).
            Rango: [0, +∞)
            Ideal: cercano a 0

        Solo para clasificación binaria:
        - brier_score : float
            Error cuadrático entre probabilidades predichas y valores reales.
            Rango: [0, 1]
            Ideal: cercano a 0

        - f1_threshold : float
            F1 score usando el threshold definido manualmente.
            Rango: [0, 1]
            Ideal: cercano a 1 (útil para ajustar decisiones según el problema)

        Notas
        -----
        - Algunas métricas pueden no estar presentes según:
          - tipo de modelo
          - disponibilidad de probabilidades
          - número de clases
        - Los valores "ideales" dependen del contexto del problema.
    """

    result = {}

    # predict: devuelve la respuesta final (0 o 1)
    # usa internamente threshold = 0.5
    y_pred = model.predict(X)

    # Número de clases
    classes = np.unique(y)
    n_classes = len(classes)

    # Métricas básicas
    result["accuracy"] = accuracy_score(y, y_pred)
    result["f1"] = f1_score(y, y_pred, average=average)
    result["precision"] = precision_score(y, y_pred, average=average, zero_division=0)
    result["recall"] = recall_score(y, y_pred, average=average, zero_division=0)

    # Confusion matrix (general)
    cm = confusion_matrix(y, y_pred)
    result["confusion_matrix"] = cm

    # Si es binaria, agregamos tp/tn/etc
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "specificity": _safe_divide(tn, tn + fp)
        })

    # Probabilidades
    if _has_proba(model):
        # predict_proba: devuelve qué tan seguro está el modelo,
        # qué tan probable es cada clase para cada fila
        y_proba = model.predict_proba(X)

        # ROC AUC. Se usa try/except porque pueden romperse fácilmente.
        # Ej: una sola clase, formato incorrecto de probabilidades, problemas de multiclase. 
        try:
            if n_classes == 2:
                result["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
            else:
                result["roc_auc"] = roc_auc_score(y, y_proba, multi_class="ovr", average=average)
        except Exception:
            pass

        # PR AUC
        try:
            if n_classes == 2:
                result["pr_auc"] = average_precision_score(y, y_proba[:, 1])
            else:
                result["pr_auc"] = average_precision_score(y, y_proba, average=average)
        except Exception:
            pass

        # Log loss (compatible multiclase)
        result["log_loss"] = log_loss(y, y_proba)

        # Brier score → solo bien definida para binaria
        if n_classes == 2:
            # Significa dame la probabilidad de que sea clase 1
            result["brier_score"] = brier_score_loss(y, y_proba[:, 1])

            # Threshold custom: recalcula métricas dependientes de threshold con el nuevo umbral
            # Es como un médico: Probabilidad de enfermedad: 0.7. Vos decidís: “si es mayor a 0.6 → doy tratamiento”
            y_pred_thr = (y_proba[:, 1] >= threshold).astype(int)
            result["f1_threshold"] = f1_score(y, y_pred_thr)

    return result


# Regresión
def evaluate_regression(model, X, y):
    """
    Evaluación general para modelos de regresión.

    Parámetros
    ----------
    model : object
        Modelo de regresión entrenado con método `predict`.

    X : array-like (n_samples, n_features)
        Features de entrada.

    y : array-like (n_samples,)
        Valores reales (target continuo).

    Retorna
    -------
    dict
        Diccionario con métricas de evaluación:

        Métricas principales:
        - mae : float
            Error absoluto medio (promedio del error absoluto).
            Rango: [0, +∞)
            Ideal: cercano a 0
            Interpretación: cuánto se equivoca el modelo en promedio.

        - mse : float
            Error cuadrático medio.
            Rango: [0, +∞)
            Ideal: cercano a 0
            Interpretación: penaliza más los errores grandes.

        - rmse : float
            Raíz del error cuadrático medio.
            Rango: [0, +∞)
            Ideal: cercano a 0
            Interpretación: error típico en las mismas unidades del target. 

        - r2 : float
            Coeficiente de determinación.
            Rango: (-∞, 1]
            Ideal: cercano a 1
            Interpretación:
                - 1 → predicción perfecta
                - 0 → igual que predecir la media
                - < 0 → peor que un modelo trivial

        Notas
        -----
        - Todas las métricas se calculan comparando valores reales vs predichos.
        - MAE es más robusto a outliers que MSE/RMSE.
        - RMSE es más interpretable porque está en la misma escala que `y`.
    """

    result = {}

    # Predicciones del modelo
    y_pred = model.predict(X)

    # Métricas base
    result["mae"] = mean_absolute_error(y, y_pred)

    mse = mean_squared_error(y, y_pred)
    result["mse"] = mse

    # RMSE: qué tan grande es el error del modelo, en promedio. Pero expresado en la misma unidad que tus datos
    result["rmse"] = np.sqrt(mse)

    result["r2"] = r2_score(y, y_pred)

    return result


# Calibración
def calibration_table(model, X, y, bins=10):
    """
    Genera una tabla de calibración para modelos de clasificación binaria.

    La función agrupa las predicciones en intervalos (bins) según su
    probabilidad y compara la probabilidad promedio predicha con la tasa
    real de eventos en cada grupo.

    Parámetros
    ----------
    model : object
        Modelo entrenado que implementa `predict_proba`.

    X : array-like (n_samples, n_features)
        Features de entrada.

    y : array-like (n_samples,)
        Labels verdaderos (0 o 1).

    bins : int, opcional (default=10)
        Cantidad de bins para agrupar las probabilidades.
        Por default se usan 10 (deciles).

    Retorna
    -------
    pandas.DataFrame o None
        Tabla de calibración con las siguientes columnas:

        - bin : categoría
            Intervalo de probabilidades.

        - avg_pred : float
            Probabilidad promedio predicha dentro del bin.
            Rango: [0, 1]

        - actual : float
            Tasa real de la clase positiva dentro del bin.
            Rango: [0, 1]

        - count : int
            Cantidad de observaciones en el bin.

        Interpretación:
        - avg_pred ≈ actual → ✅ modelo bien calibrado
        - avg_pred > actual → ❌ modelo sobreestima (overconfident)
        - avg_pred < actual → ❌ modelo subestima (underconfident)

        Si el modelo no tiene `predict_proba`, devuelve None.

    Notas
    -----
    - Solo aplica a clasificación binaria.
    - Usa `qcut`, por lo que cada bin tiene aproximadamente la misma cantidad de muestras.
    - Es útil para evaluar la confiabilidad de las probabilidades, no solo la precisión del modelo.
    """

    if not _has_proba(model):
        return None

    y_proba = model.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "y": y,
        "proba": y_proba
    })

    # La calibración responde: “¿cuando el modelo dice X%, realmente ocurre X%?”
    # Un modelo puede ser preciso… pero no confiable en sus probabilidades. La calibración mide esa confianza.
    df["bin"] = pd.qcut(df["proba"], q=bins, duplicates="drop")

    calibration = df.groupby("bin").agg(
        avg_pred=("proba", "mean"),
        actual=("y", "mean"),
        count=("y", "size")
    ).reset_index()

    return calibration


# Detección automática de tipo de problema 
# (clasificación binaria, multiclase o regresión)
def detect_problem_type(y):
    """
    Detecta tipo de problema automáticamente
    """

    unique = len(np.unique(y))

    if set(np.unique(y)).issubset({0, 1}):
        return "binary"
    elif len(np.unique(y)) < 20:
        return "multiclass"
    else:
        return "regression"



# Llama la función de la métrica de acuerdo al tipo de modelo
def evaluate_model(model, X, y, problem_type=None):
    """
    Evaluador universal
    """

    if problem_type is None:
        problem_type = detect_problem_type(y)

    if problem_type in ["binary", "multiclass"]:
        return evaluate_classification(model, X, y)
    elif problem_type == "regression":
        return evaluate_regression(model, X, y)
    else:
        raise NotImplementedError(f"{problem_type} no soportado aún")


# # Multi models
# def evaluate_models(models, X, y):
#     """
#     Evalúa múltiples modelos de forma homogénea
#     """

#     results = []

#     for name, model in models.items():
#         metrics = evaluate_model(model, X, y)

#         results.append({
#             "model": name,
#             **metrics
#         })

#     return pd.DataFrame(results)


# # Display
# def summarize_results(df):
#     """
#     Resumen ordenado automático
#     """

#     cols_priority = [
#         "model",
#         "roc_auc",
#         "pr_auc",
#         "f1",
#         "accuracy",
#         "r2",
#         "rmse"
#     ]

#     cols = [c for c in cols_priority if c in df.columns]

#     return df[cols].sort_values(by=cols[1], ascending=False)