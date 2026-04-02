    pretrained_url = (
        "https://github.com/YOUR_USERNAME/diabetes-risk-predictor/"
        "releases/download/v1.0/best_model_pretrained.pkl"
    )
    try:
        log.info(f"Attempting to download pre-trained model from {pretrained_url} ...")
        resp = requests.get(pretrained_url, timeout=30)
        resp.raise_for_status()
        with open(cached_path, "wb") as fh:
            fh.write(resp.content)
        model = joblib.load(cached_path)
        log.info("Pre-trained model downloaded and loaded successfully.")
        return {
            "name":      "PreTrained_Downloaded",
            "model":     model,
            "auc":       0.0, "f1": 0.0, "recall": 0.0,
            "accuracy":  0.0, "precision": 0.0,
            "best_params": {},
            "fallback_used": True,
        }
    except Exception as e:
        log.warning(f"Pre-trained download failed: {e}")

    # 3. Final fallback: best local model (even below threshold)
    log.warning(
        "Using best locally-trained model as final fallback "
        f"({local_best['name']}, AUC={local_best['auc']:.4f}). "
        "Consider placing a pre-trained model at ml/models/best_model_pretrained.pkl"
    )
    local_best["fallback_used"] = True
    return local_best
