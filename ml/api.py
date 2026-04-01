            "clinical markers. Continue routine preventive health monitoring."
        )

    return jsonify({
        "prediction":   pred,
        "label":        label,
        "confidence":   round(confidence * 100, 2),
        "probabilities": {
            "non_diabetic": round(float(proba[0]) * 100, 2),
            "diabetic":     round(float(proba[1]) * 100, 2),
        },
        "shap_values":  shap_dict,
        "guidance":     guidance,
        "model_used":   meta.get("model_name", "unknown"),
        "disclaimer": (
            "This tool is a decision-support aid only. "
            "It is not a diagnostic instrument. "
            "Clinician oversight is mandatory before any patient-facing action. "
            "Model trained on Pima Indian female population only."
        )
    })


if __name__ == "__main__":
    port = int(os.environ.get("ML_API_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
