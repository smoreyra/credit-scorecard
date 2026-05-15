from sklearn.model_selection import cross_val_score
import numpy as np


def train_model(model, X_train, y_train, fit_params=None):
    """
    Entrena un modelo de forma genérica.

    Parameters
    ----------
    model : estimator (sklearn-like)
    X_train : array-like
    y_train : array-like
    fit_params : dict, optional
        Parámetros extra para .fit() (ej: early stopping)

    Returns
    -------
    model : trained model
    """

    if fit_params is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, **fit_params)

    return model


def cross_validate_model(model, X_train, y_train, cv=5, scoring="roc_auc"):
    """
    Hace cross-validation para evaluar estabilidad del modelo.

    Returns
    -------
    dict con media y std del score
    """

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return {
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores),
        "cv_scores": scores.tolist()
    }


def train_with_cv(model, X_train, y_train, cv=5, scoring="roc_auc"):
    """
    Entrena modelo + calcula CV en un solo paso.
    Muy útil para benchmarking.
    """

    cv_results = cross_validate_model(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    trained_model = train_model(model, X_train, y_train)

    return trained_model, cv_results


def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val):
    """
    Entrenamiento con early stopping (para boosting models).
    Compatible con LightGBM / XGBoost / CatBoost.
    """

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model