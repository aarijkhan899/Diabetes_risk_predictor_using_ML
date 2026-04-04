    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_res, y_res)

    log.info("    Training XGBoost (with hyperparameter tuning) ...")
    xgb, xgb_params = _tune_xgb(X_res, y_res)

    log.info("    Training LightGBM ...")
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_samples=10, reg_alpha=0.5, random_state=42, verbose=-1
    )
    lgbm.fit(X_res, y_res)

    log.info("    Training SVM ...")
    svm = SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=42)
    svm.fit(X_res, y_res)

    candidates = {
        "Logistic Regression": lr,
        "Random Forest":        rf,
        "XGBoost":              xgb,
        "LightGBM":             lgbm,
        "SVM":                  svm,
    }

    results = {}
    for name, clf in candidates.items():
        y_train_pred  = clf.predict(X_res)
        y_test_pred   = clf.predict(X_test)
        y_test_proba  = clf.predict_proba(X_test)[:, 1]

        results[name] = {
            "model":          clf,
            "train_accuracy": accuracy_score(y_res, y_train_pred),
            "test_accuracy":  accuracy_score(y_test, y_test_pred),
            "test_auc":       roc_auc_score(y_test, y_test_proba),
            "test_f1":        f1_score(y_test, y_test_pred, average="weighted"),
            "test_recall":    recall_score(y_test, y_test_pred, pos_label=1),
            "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
        }
        log.info(
