FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# Load artefacts at startup
model, scaler, meta, explainer = None, None, {}, None


def load_artefacts():
    global model, scaler, meta, explainer
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(META_PATH) as fh:
            meta = json.load(fh)
        # Build SHAP explainer (TreeExplainer for tree models, KernelExplainer otherwise)
        model_name = meta.get("model_name", "")
        if model_name in ("RandomForest", "XGBoost", "LightGBM"):
            explainer = shap.TreeExplainer(model)
        else:
            # Use a small background sample for speed
            bg = np.zeros((50, len(FEATURES)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
        log.info(f"Loaded model: {model_name}  AUC={meta.get('auc')}")
