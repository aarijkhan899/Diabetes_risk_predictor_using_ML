ACC_TARGET       = 0.90   # ≥90% training accuracy target


# ===========================================================================
# STEP 0 — HuggingFace pre-trained model attempt
# ===========================================================================
def try_download_hf_model():
    """
    Try to pull a pre-trained sklearn-compatible diabetes model from HF Hub.
    Returns a loaded model on success, None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download, list_models
        log.info("  Connecting to HuggingFace Hub (token-authenticated)...")

        # Known HF repos that host a Pima diabetes sklearn pickle
        candidates = [
            ("Ilyabarigou/pima-diabetes-xgboost",     "xgb_model.pkl"),
            ("scikit-learn/diabetes-prediction",        "model.pkl"),
            ("Falah/Ghamizi-diabetes-prediction",       "model.pkl"),
        ]

        for repo_id, filename in candidates:
            try:
                log.info(f"  Trying {repo_id}/{filename} ...")
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=HF_TOKEN,
                    cache_dir=os.path.join(MODELS_DIR, ".hf_cache"),
                )
                model = joblib.load(path)
                log.info(f"  Pre-trained model loaded from HuggingFace: {repo_id}")
                return model, repo_id
            except Exception:
                continue

        log.warning("  No compatible HuggingFace model found; falling back to local training.")
        return None, None

    except ImportError:
        log.warning("  huggingface_hub not installed; skipping HF download.")
        return None, None
    except Exception as e:
