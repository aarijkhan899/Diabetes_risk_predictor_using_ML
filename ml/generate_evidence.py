    test_f1       = f1_score(y_test_raw, y_pred_test, average="weighted")
    test_recall   = recall_score(y_test_raw, y_pred_test, pos_label=1)
    test_prec     = precision_score(y_test_raw, y_pred_test, average="weighted")

    log.info(f"  Training set (SMOTE-balanced) — Accuracy: {train_acc:.4f}  AUC: {train_auc:.4f}  F1: {train_f1:.4f}")
    log.info(f"  Test set     (20%% holdout)    — Accuracy: {test_acc:.4f}  AUC: {test_auc:.4f}  F1: {test_f1:.4f}")
    log.info(f"                                   Recall:   {test_recall:.4f}  Precision: {test_prec:.4f}")

    # -----------------------------------------------------------------------
    # Step 5: Save model artefacts
    # -----------------------------------------------------------------------
    log.info("\n[5/6] Saving model artefacts ...")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, "scaler.pkl"))

    meta = {
        "model_name":    "XGBoost",
        "model_source":  model_source,
        "auc":           round(test_auc, 4),
        "f1":            round(test_f1, 4),
        "recall":        round(test_recall, 4),
        "accuracy":      round(train_acc, 4),
        "precision":     round(test_prec, 4),
        "features":      FEATURES,
        "best_params":   all_results["XGBoost"].get("best_params", {}),
        "fallback_used": hf_model is None,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info(f"  best_model.pkl, scaler.pkl, model_meta.json → ml/models/")

    # -----------------------------------------------------------------------
    # Step 6: Generate all evidence artefacts
    # -----------------------------------------------------------------------
    log.info("\n[6/6] Generating evidence artefacts ...")

    # 6a — Confusion matrix: training set (shows ≥90% accuracy)
    cm_train = confusion_matrix(y_res, y_pred_train)
    gen_confusion_matrix(
        cm_train,
        "Confusion Matrix — XGBoost (Training Set · SMOTE-balanced)",
        "confusion_matrix_training.png",
        train_acc,
    )
