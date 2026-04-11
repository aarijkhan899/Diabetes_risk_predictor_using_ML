"""
Flask REST API for diabetes risk prediction (loads joblib models from ml/models/).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import shap
from flask import Flask, jsonify, request
from flask_cors import CORS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

app = Flask(__name__)
CORS(app)

clf = None
scaler = None
meta: dict = {}


def _load_json_meta() -> dict:
    path = MODELS_DIR / "model_meta.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_model_path() -> Path | None:
    for name in ("best_model.pkl", "best_model_pretrained.pkl"):
        p = MODELS_DIR / name
        if p.is_file():
            return p
    return None


def load_artifacts() -> None:
    global clf, scaler, meta
    meta = _load_json_meta()
    model_path = _find_model_path()
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not model_path or not scaler_path.is_file():
        log.error("Missing model artefacts (need scaler.pkl and best_model.pkl or best_model_pretrained.pkl).")
        clf, scaler = None, None
        return
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    log.info("Loaded classifier from %s", model_path)


def _guidance(pred: int, confidence: float) -> str:
    if pred == 1:
        if confidence >= 70:
            return (
                "Elevated predicted risk for diabetes. This is not a diagnosis. "
                "Discuss results with a clinician; consider fasting glucose or HbA1c testing."
            )
        return (
            "Borderline elevated risk. Continue routine preventive health monitoring "
            "and share these inputs with a clinician if symptoms or risk factors apply."
        )
    if confidence >= 70:
        return (
            "Predicted lower risk in this model's output distribution. "
            "Maintain healthy lifestyle; screening intervals should follow clinical guidelines."
        )
    return (
        "Predicted lower risk with moderate model confidence. "
        "Continue routine preventive health monitoring."
    )


def _shap_for_row(x_row: np.ndarray) -> dict[str, float]:
    if clf is None:
        return {}
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(x_row)
        if isinstance(sv, list):
            # binary: use positive class
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv).ravel()
    except Exception:
        try:
            explainer = shap.Explainer(clf.predict_proba, x_row)
            exp = explainer(x_row)
            sv = np.asarray(exp.values).reshape(-1)
            if sv.size > len(FEATURES):
                sv = sv[-len(FEATURES) :]
        except Exception as ex:
            log.warning("SHAP skipped: %s", ex)
            return {}
    names = meta.get("features") if isinstance(meta.get("features"), list) else FEATURES
    if len(sv) != len(names):
        names = FEATURES[: len(sv)]
    return {str(names[i]): float(sv[i]) for i in range(len(sv))}


load_artifacts()


@app.route("/health")
def health():
    payload = {
        "status": "ok" if clf is not None else "no_model",
        "model_name": meta.get("model_name", "unknown"),
        "auc": meta.get("auc"),
    }
    if clf is None:
        return jsonify(payload), 503
    return jsonify(payload), 200


@app.route("/model_info")
def model_info():
    if not meta:
        return jsonify({"error": "model_meta.json not found"}), 404
    return jsonify(meta)


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(force=True, silent=True) or {}
    row = []
    for f in FEATURES:
        if f not in body:
            return jsonify({"error": f"Missing key: {f}"}), 400
        try:
            row.append(float(body[f]))
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid numeric value for {f}"}), 400

    x = np.asarray(row, dtype=np.float64).reshape(1, -1)
    x_s = scaler.transform(x)
    proba = clf.predict_proba(x_s)[0]
    pred = int(np.argmax(proba))
    confidence = float(proba[pred])
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    shap_dict = _shap_for_row(x_s)

    return jsonify(
        {
            "prediction": pred,
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "non_diabetic": round(float(proba[0]) * 100, 2),
                "diabetic": round(float(proba[1]) * 100, 2),
            },
            "shap_values": shap_dict,
            "guidance": _guidance(pred, round(confidence * 100, 2)),
            "model_used": meta.get("model_name", "unknown"),
            "auc": meta.get("auc"),
            "disclaimer": (
                "This tool is a decision-support aid only. "
                "It is not a diagnostic instrument. "
                "Clinician oversight is mandatory before any patient-facing action. "
                "Model trained on Pima Indian female population only."
            ),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("ML_API_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
