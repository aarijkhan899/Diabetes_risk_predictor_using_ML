
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
EVIDENCE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "evidence")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
FEATURES            = COLUMNS[:-1]
TARGET              = "Outcome"
ZERO_INVALID_COLS   = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# HuggingFace token (set HUGGINGFACE_TOKEN in the environment; never commit secrets)
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Dissertation success thresholds
AUC_THRESHOLD    = 0.85
F1_THRESHOLD     = 0.80
RECALL_THRESHOLD = 0.78
