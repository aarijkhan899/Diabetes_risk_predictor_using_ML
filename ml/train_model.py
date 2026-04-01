    for col in ZERO_INVALID_COLS:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        log.info(f"  Imputed {col} zeros with median={median:.2f}")

    log.info(f"Class distribution before SMOTE: {df[TARGET].value_counts().to_dict()}")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    log.info(f"After SMOTE: {np.bincount(y_res.astype(int))}")

    return X_res, y_res, scaler


def build_candidates():
    """Return list of (name, estimator, param_grid) tuples."""
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"]},
        ),
        (
