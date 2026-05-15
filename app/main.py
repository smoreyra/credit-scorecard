import os
from datetime import datetime
import pandas as pd

import gcp_utils as gcp
import ingest 
import preprocess
import models
import train
import evaluate

from sklearn.model_selection import train_test_split


def main():

    print("\n Inicio de pipeline\n")

    env = os.getenv("ENV", "dev")
    cfg = gcp.read_config(env=env)

    print("Config:", cfg, "\n")

    print("Carga de datos \n")
    df = ingest.load_dataset_from_config(cfg)

    print("Preprocesamiento de datos \n")
    X, y = preprocess(df)

    print("Separo data en entrenamiento y testeo \n")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Carga de modelos \n")
    models = models()
    trained_models = {}

    print("Entrenamiento de modelos \n")
    for name, model in models.items():
        print(f"Training: {name}")

        trained_model = train(model, X_train, y_train)
        trained_models[name] = trained_model

    print("Evaluación de modelos \n")
    results = evaluate(trained_models, X_test, y_test)

    print(results)

    best_model = max(results, key=lambda x: x["roc_auc"])

    print("Mejor modelo:")
    print(best_model)

    # Predicción
    def predict_model(model, X_test):
        return model.predict(X_test)


if __name__ == "__main__":
    main()