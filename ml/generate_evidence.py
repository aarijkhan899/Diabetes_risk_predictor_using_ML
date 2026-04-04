    df = df.copy()

    # Replace biologically invalid zeros
    df[ZERO_INVALID_COLS] = df[ZERO_INVALID_COLS].replace(0, np.nan)
    for col in ZERO_INVALID_COLS:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        log.info(f"    Imputed {col} zeros → median = {median:.2f}")

    X_raw = df[FEATURES].values
    y_raw = df[TARGET].values

    # --- SPLIT FIRST (no leakage) ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    log.info(f"  Train/test split: {X_train_raw.shape} / {X_test_raw.shape}")

    # --- Scale on TRAIN only ---
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc  = scaler.transform(X_test_raw)

    # --- SMOTE on TRAIN only ---
    smote          = SMOTE(random_state=42)
    X_res, y_res   = smote.fit_resample(X_train_sc, y_train)
    log.info(f"  After SMOTE (train only): {np.bincount(y_res.astype(int))} (balanced)")

    return X_res, y_res, X_test_sc, y_test, scaler


# ===========================================================================
# STEP 3 — Local model training
# ===========================================================================
def _xgb_model():
    """XGBoost with regularisation to prevent overfitting on small dataset."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
