                                   target_names=["Non-Diabetic", "Diabetic"]))
    return best


def pull_pretrained_model(scaler, local_best):
    """
    Fallback: attempt to load a pre-trained model artefact.

    Strategy (in order):
      1. If a cached best_model_pretrained.pkl exists locally, load it.
      2. Try to download from a known public URL (GitHub Releases / Hugging Face).
      3. If both fail, log a warning and return the best locally-trained model
         (even if it misses thresholds) so the app remains functional.
    """
    cached_path = os.path.join(MODELS_DIR, "best_model_pretrained.pkl")

    # 1. Local cache
    if os.path.exists(cached_path):
        log.info(f"Loading cached pre-trained model from {cached_path}")
        model = joblib.load(cached_path)
        return {
            "name":      "PreTrained_Cached",
            "model":     model,
            "auc":       0.0, "f1": 0.0, "recall": 0.0,
            "accuracy":  0.0, "precision": 0.0,
            "best_params": {},
            "fallback_used": True,
        }

    # 2. Remote download
    # Replace this URL with a real hosted model if available
