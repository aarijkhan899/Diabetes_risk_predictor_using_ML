
    # 6b — Confusion matrix: test set (honest held-out evaluation)
    cm_test = confusion_matrix(y_test_raw, y_pred_test)
    gen_confusion_matrix(
        cm_test,
        "Confusion Matrix — XGBoost (Test Set · 20 % Holdout)",
        "confusion_matrix_test.png",
        test_acc,
    )

    # 6c — ROC curve
    roc_auc_final = gen_roc_curve(best_model, X_test_raw, y_test_raw, "roc_curve.png")

    # 6d — Precision-Recall curve
    gen_precision_recall_curve(best_model, X_test_raw, y_test_raw, "precision_recall_curve.png")

    # 6e — Feature importance
    gen_feature_importance(best_model, "feature_importance.png")

    # 6f — Multi-model comparison
    gen_model_comparison(all_results, "model_comparison.png")

    # 6g — Cross-validation bar chart
    cv_scores = gen_cv_bar(X_res, y_res, best_model, "cross_validation_scores.png")

    # 6h — Classification report (text)
    report = classification_report(
        y_test_raw, y_pred_test, target_names=LABEL_NAMES
    )
    report_path = os.path.join(EVIDENCE_DIR, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write("=" * 65 + "\n")
        fh.write("DIABETES RISK PREDICTOR — CLASSIFICATION REPORT\n")
        fh.write("Dissertation: Md Aariz | MSc Big Data Technologies | UEL\n")
        fh.write("=" * 65 + "\n\n")
        fh.write("Model  : XGBoost (optimised, n_estimators=300, max_depth=6)\n")
        fh.write("Source : " + model_source + "\n")
        fh.write("Data   : Pima Indians Diabetes Dataset (UCI ML Repo, id=34)\n\n")
        fh.write("─" * 65 + "\n")
        fh.write("TEST SET CLASSIFICATION REPORT (20 % stratified holdout)\n")
        fh.write("─" * 65 + "\n")
        fh.write(report)
        fh.write("\n" + "─" * 65 + "\n")
        fh.write("TRAINING SET METRICS (SMOTE-balanced, N=" + str(len(y_res)) + ")\n")
