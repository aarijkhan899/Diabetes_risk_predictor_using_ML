        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def _tune_xgb(X_res, y_res) -> XGBClassifier:
    """Grid-search XGBoost with 5-fold stratified CV (same strategy as train_model.py)."""
    log.info("    Grid-searching XGBoost hyperparameters (5-fold CV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [3, 4, 5],
        "learning_rate":   [0.05, 0.1],
        "subsample":       [0.8, 1.0],
        "min_child_weight":[3, 5],
    }
    gs = GridSearchCV(
        XGBClassifier(
            colsample_bytree=0.8, gamma=0.1,
            reg_alpha=0.5, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        param_grid, cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_res, y_res)
    log.info(f"    Best CV AUC={gs.best_score_:.4f}  params={gs.best_params_}")
    return gs.best_estimator_, gs.best_params_


def train_all_models(X_res, y_res, X_test, y_test) -> dict:
    """
    Train all 5 dissertation classifiers on the SMOTE-balanced set.
    XGBoost is tuned via GridSearchCV to maximise AUC.
    Evaluate on the held-out 20 % test split.
    """
    log.info("    Training Logistic Regression ...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_res, y_res)

    log.info("    Training Random Forest ...")
