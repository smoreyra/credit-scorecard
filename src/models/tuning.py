from sklearn.model_selection import GridSearchCV

def get_param_grid(model_name):
    grids = {
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        },
        "SVC": {
            "model__C": [0.1, 1, 10]
        }
    }

    return grids.get(model_name, {})

def tune_model(pipeline, model_name, X_train, y_train):

    param_grid = get_param_grid(model_name)

    if not param_grid:
        print("⚠️ No tuning para este modelo")
        return pipeline

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_