

def main():
    log.info("=== Diabetes Risk Predictor - Model Training ===")

    # 1. Load data
    df = load_dataset()
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Class distribution: {df[TARGET].value_counts().to_dict()}")

    # 2. Preprocess
    X_res, y_res, scaler = preprocess(df)

    # Keep raw scaled (pre-SMOTE) for final held-out evaluation
    df_clean = df[FEATURES].replace(0, np.nan)
    for col in ZERO_INVALID_COLS:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    X_raw_scaled = scaler.transform(df_clean.values)
    y_raw = df[TARGET].values

    # 3. Train all models
    results = train_all(X_res, y_res)

    # 4. Evaluate, select best, save
    evaluate_and_save(results, scaler, X_raw_scaled, y_raw)

    log.info("=== Training complete. Models saved to ml/models/ ===")


if __name__ == "__main__":
    main()
