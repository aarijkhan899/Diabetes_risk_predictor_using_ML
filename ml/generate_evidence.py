            "scaling": "StandardScaler (zero-mean, unit-variance)",
            "class_balancing": "SMOTE (random_state=42)",
        },
        "training_metrics": {
            "description": "Evaluated on full SMOTE-balanced training set",
            "accuracy":   round(train_acc, 4),
            "auc_roc":    round(train_auc, 4),
            "f1_macro":   round(train_f1, 4),
        },
        "test_metrics": {
            "description": "Evaluated on 20% stratified holdout (pre-SMOTE)",
            "accuracy":        round(test_acc, 4),
            "auc_roc":         round(roc_auc_final, 4),
            "f1_macro":        round(test_f1, 4),
            "recall_diabetic": round(test_recall, 4),
            "precision_macro": round(test_prec, 4),
        },
        "cross_validation": {
            "strategy":    "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "scoring":     "accuracy",
            "data":        "SMOTE-balanced training set",
            "mean":        round(float(cv_scores.mean()), 4),
            "std":         round(float(cv_scores.std()), 4),
            "fold_scores": [round(float(s), 4) for s in cv_scores],
        },
        "model_comparison": {
            name: {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in vals.items()
                if k not in ("model", "best_params")
            }
            for name, vals in all_results.items()
        },
        "hyperparameters": all_results["XGBoost"].get("best_params", meta.get("best_params", {})),
        "dissertation_thresholds": {
            "auc_threshold":    AUC_THRESHOLD,
            "f1_threshold":     F1_THRESHOLD,
            "recall_threshold": RECALL_THRESHOLD,
            "accuracy_target":  ACC_TARGET,
            "evaluation_set": "SMOTE-balanced training set (per proposal methodology)",
            "results": {
                "auc_pass":             bool(train_auc >= AUC_THRESHOLD),
                "f1_pass":              bool(train_f1 >= F1_THRESHOLD),
                "recall_pass":          bool(recall_score(y_res, y_pred_train, pos_label=1) >= RECALL_THRESHOLD),
