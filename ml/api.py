def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    data = request.get_json(force=True)

    # Validate input
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        raw = np.array([[float(data[f]) for f in FEATURES]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # Scale
    X_scaled = scaler.transform(raw)

    # Predict
    proba      = model.predict_proba(X_scaled)[0]
    pred       = int(np.argmax(proba))
    confidence = float(proba[pred])

    # SHAP values
    try:
