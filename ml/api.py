    except FileNotFoundError as e:
        log.error(
            f"Model artefacts not found ({e}). "
            "Run `python ml/train_model.py` first."
        )


load_artefacts()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "model_name":   meta.get("model_name", "none"),
        "auc":          meta.get("auc", 0),
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify(meta)


@app.route("/predict", methods=["POST"])
