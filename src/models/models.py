# Imports clásicos
import numpy as np
import pandas as pd

# Estadísticos
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet (puede variar el import según versión)
from prophet import Prophet

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Supervisados
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    IsolationForest
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

# No supervisado
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Boosting modernos
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.feature_extraction.text import TfidfVectorizer


# Supervisado
# Clasificación
def get_classification_models(random_state: int = 42):
    """
    Devuelve un diccionario de modelos de clasificación listos para entrenar.
    Modelos: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, LightGBM, XGBoost, CatBoost
    """

    models = {}

    # Básicos
    models["logistic"] = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state
    )

    models["decision_tree"] = DecisionTreeClassifier(
        max_depth=5,
        random_state=random_state
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )

    models["knn"] = KNeighborsClassifier(
        n_neighbors=5
    )

    models["svm"] = SVC(
        probability=True
    )

    # Boosting
    models["gradient_boosting"] = GradientBoostingClassifier(

    )
    
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=200
    )
    
    models["adaboost"] = AdaBoostClassifier(

    )

    # Boosting moderno
    models["lightgbm"] = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state
    )

    models["xgboost"] = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=None,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state
    )

    models["catboost"] = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=random_state
    )

    # Naive Bayes (clave NLP)
    models["naive_bayes"] = GaussianNB(

    )

    # Red neuronal clásica
    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500
    )

    return models


# Regresión
def get_regression_models(random_state: int = 42):
    """
    Devuelve un diccionario de modelos de regresión listos para entrenar.
    Modelos: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, KNN, SVR, LightGBM, XGBoost, CatBoost
    """

    models = {}

    # Lineales
    models["linear"] = LinearRegression()

    models["ridge"] = Ridge(alpha=1.0)

    models["lasso"] = Lasso(alpha=0.1)

    # Árboles
    models["decision_tree"] = DecisionTreeRegressor(
        max_depth=5,
        random_state=random_state
    )

    models["random_forest"] = RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=random_state
    )

    # Otros
    models["knn"] = KNeighborsRegressor(
        n_neighbors=5
    )

    models["svr"] = SVR()

    # Boosting
    models["gradient_boosting"] = GradientBoostingRegressor(

    )
    
    models["extra_trees"] = ExtraTreesRegressor(
        n_estimators=200
    )
    
    models["adaboost"] = AdaBoostRegressor(

    )

    # Red neuronal
    models["mlp"] = MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=500
    )

    # Boosting modernos
    models["lightgbm"] = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state
    )

    models["xgboost"] = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state
    )

    models["catboost"] = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        verbose=0,
        random_state=random_state
    )

    return models

# No Supervisado
def get_unsupervised_models():

    models = {}

    # Clustering
    models["kmeans"] = KMeans(n_clusters=5)
    models["dbscan"] = DBSCAN(eps=0.5)
    models["hierarchical"] = AgglomerativeClustering(n_clusters=5)
    models["gmm"] = GaussianMixture(n_components=5)

    # Reducción de dimensionalidad
    models["pca"] = PCA(n_components=2)
    models["tsne"] = TSNE(n_components=2)

    return models


# Forecasting (Series temporales)
# Nota: muchos libs de forecasting no siguen sklearn puro
def get_forecasting_models(random_state=42):

    models = {}

    # ARIMA
    models["arima"] = ARIMA

    # SARIMA
    models["sarima"] = SARIMAX

    # Prophet
    models["prophet"] = Prophet

    # ML (regresión con lags)
    models["random_forest_ts"] = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    models["xgboost_ts"] = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state
    )

    # LSTM (factory function)
    # ojo: función, no clase directa
    models["lstm"] = build_lstm_model  

    return models


# LSTM Factory
def build_lstm_model(input_shape):
    """
    Devuelve un modelo LSTM (sin entrenar)
    """

    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# Anomaly Detection
def get_anomaly_detection_models():

    models = {}

    models["isolation_forest"] = IsolationForest(contamination=0.05)
    models["one_class_svm"] = OneClassSVM(nu=0.05)

    return models


# NLP
def get_nlp_pipeline_model():

    """
    Pipeline básico NLP:
    TF-IDF + Logistic Regression
    """

    vectorizer = TfidfVectorizer(max_features=5000)

    model = LogisticRegression(max_iter=1000)

    return vectorizer, model


# Deep Learning
def build_dense_nn(input_dim):
    """
    Red neuronal fully-connected
    """

    model = Sequential()

    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # cambiar según problema

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model