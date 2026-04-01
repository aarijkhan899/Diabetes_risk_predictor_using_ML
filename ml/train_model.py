from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -- Success thresholds (from dissertation proposal) --------------------------
AUC_THRESHOLD    = 0.85
F1_THRESHOLD     = 0.80
RECALL_THRESHOLD = 0.78
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -- Column names for the raw CSV fallback ------------------------------------
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
FEATURES = COLUMNS[:-1]
TARGET   = "Outcome"
# Columns where 0 is biologically invalid -> replace with NaN then impute
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset() -> pd.DataFrame:
    """Load Pima Indians Diabetes Dataset via ucimlrepo or CSV fallback."""
    log.info("Attempting to load dataset via ucimlrepo ...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=34)
        X = dataset.data.features
        y = dataset.data.targets
        # targets may be Series or DataFrame
