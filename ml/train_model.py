

def evaluate_and_save(results, scaler, X_raw, y_raw):
    """
    Pick the best model that meets dissertation success criteria.
    If none meet the thresholds, fall back to downloading a pre-trained model.
    """
    # Find best model meeting all thresholds
    best = None
    for r in results:
        if (r["auc"] >= AUC_THRESHOLD and
                r["f1"] >= F1_THRESHOLD and
                r["recall"] >= RECALL_THRESHOLD):
            best = r
            break

    if best is None:
        log.warning(
            "No locally trained model met all success thresholds. "
            "Falling back to pre-trained model download ..."
        )
        best = pull_pretrained_model(scaler, results[0])
    else:
        log.info(
            f"Best model: {best['name']}  AUC={best['auc']:.4f}  "
            f"F1={best['f1']:.4f}  Recall={best['recall']:.4f}"
        )

    # Persist artefacts
    model_path  = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
