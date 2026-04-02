    meta_path   = os.path.join(MODELS_DIR, "model_meta.json")

    joblib.dump(best["model"], model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "model_name":    best["name"],
        "auc":           round(best["auc"], 4),
        "f1":            round(best["f1"], 4),
        "recall":        round(best["recall"], 4),
        "accuracy":      round(best["accuracy"], 4),
        "precision":     round(best["precision"], 4),
        "features":      FEATURES,
        "best_params":   best.get("best_params", {}),
        "fallback_used": best.get("fallback_used", False),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info(f"Saved model  -> {model_path}")
    log.info(f"Saved scaler -> {scaler_path}")
    log.info(f"Saved meta   -> {meta_path}")

    # Print full classification report on raw (unsmoted) test split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    y_pred_te = best["model"].predict(X_te)
    log.info("Classification report on held-out 20%% test set:\n" +
             classification_report(y_te, y_pred_te,
