            "RandomForest",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5]},
        ),
        (
            "XGBoost",
            XGBClassifier(eval_metric="logloss", random_state=42,
                          use_label_encoder=False, verbosity=0),
            {"n_estimators": [100, 200], "max_depth": [3, 6],
             "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
        ),
        (
            "LightGBM",
            LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [-1, 6],
             "learning_rate": [0.05, 0.1]},
        ),
        (
            "SVM",
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
        ),
    ]


def train_all(X, y):
    """Grid-search all candidates with 5-fold CV; return sorted results list."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

