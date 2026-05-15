import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame):
    """
    Preprocesamiento simple para modelo de churn.
    """

    df = df.copy()

    # Limpieza
    df = df.replace({"None": np.nan})

    # Fechas a datetime
    # TODO: poner vector de fechas
    date_cols = [
        "fecha_snapshot"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Feature Engineering
    # TODO: poner funciones útiles
    # TODO: poner flags útiles


    # Tratamiento de categóricas

    # NPS categórico
    df["desc_nps_score"] = df["desc_nps_score"].fillna("unknown")
    df = pd.get_dummies(df, columns=["desc_nps_score"], drop_first=True)

    # Limpieza final

    drop_cols = [
        "id_cliente",
        "fecha_snapshot",
        "antiguedad_continua",
        "antiguedad_absoluta",
        "fec_nps_score",
        "cod_ramo_cliente",
        "nps_score"  # opcional si es muy sparse
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill NA restantes
    df = df.fillna(0)

    # Split X / y

    # TODO: target col
    target_col = "flg_gestion_mala"  

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica de datos.
    """

    df = df.drop_duplicates()

    df = df.fillna(0)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera features relevantes para churn.
    """

    # ✅ Ejemplo básico (adaptar a tu negocio)

    if "last_activity_days" in df.columns:
        df["is_inactive"] = (df["last_activity_days"] > 30).astype(int)

    if "usage_30d" in df.columns and "usage_7d" in df.columns:
        df["usage_drop"] = df["usage_30d"] - df["usage_7d"]

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding de variables categóricas.
    """

    categorical_cols = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def scale_features(X: pd.DataFrame):
    """
    Escalado de variables numéricas.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def split_features_target(df: pd.DataFrame, target_col="target"):
    """
    Separa features y target.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def preprocess_data(df: pd.DataFrame, scale=True):
    """
    Pipeline completo de preprocessing.
    """

    # ✅ 1. Limpieza
    df = clean_data(df)

    # ✅ 2. Features
    df = feature_engineering(df)

    # ✅ 3. Encoding
    df = encode_features(df)

    # ✅ 4. Split X / y
    X, y = split_features_target(df)

    # ✅ 5. Escalado (opcional)
    if scale:
        X, scaler = scale_features(X)
        return X, y

    return X, y


def create_time_features(df, date_col):

    df = df.copy()

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df[date_col].dt.dayofweek

    return df


def create_lag_features(df, target_col, lags=[1, 7, 14]):

    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df