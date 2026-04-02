    for name, estimator, param_grid in build_candidates():
        log.info(f"Training {name} ...")
        gs = GridSearchCV(
            estimator, param_grid, cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True, verbose=0
        )
        gs.fit(X, y)
        best = gs.best_estimator_

        # Evaluate on full resampled set (CV already guards against overfit)
        y_pred  = best.predict(X)
        y_proba = best.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_proba)
        f1  = f1_score(y, y_pred, average="macro")
        rec = recall_score(y, y_pred, pos_label=1)
        acc = accuracy_score(y, y_pred)
        pre = precision_score(y, y_pred, average="macro")

        log.info(
            f"  {name}: AUC={auc:.4f} F1={f1:.4f} Recall={rec:.4f} "
            f"Acc={acc:.4f} | best_params={gs.best_params_}"
        )
        results.append({
            "name": name, "model": best, "auc": auc,
            "f1": f1, "recall": rec, "accuracy": acc, "precision": pre,
            "best_params": gs.best_params_
        })

    results.sort(key=lambda r: r["auc"], reverse=True)
    return results
