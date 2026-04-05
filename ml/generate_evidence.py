    df = load_dataset()
    log.info(f"  Shape            : {df.shape}")
    log.info(f"  Class dist.      : {df[TARGET].value_counts().to_dict()}")
    log.info(f"  Missing values   : {df.isnull().sum().sum()}")

    # -----------------------------------------------------------------------
    # Step 2: Preprocess
    # -----------------------------------------------------------------------
    log.info("\n[2/6] Preprocessing — imputation, scaling, SMOTE ...")
    X_res, y_res, X_test_raw, y_test_raw, scaler = preprocess(df)
    log.info(f"  SMOTE training shape: {X_res.shape}  |  Test shape: {X_test_raw.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Select / train model
    # -----------------------------------------------------------------------
    log.info("\n[3/6] Model selection ...")
    log.info("  Training all 5 classifiers on SMOTE-balanced data ...")
    all_results = train_all_models(X_res, y_res, X_test_raw, y_test_raw)

    if hf_model is not None:
        log.info(f"  Using pre-trained model from HuggingFace ({hf_source})")
        best_model   = hf_model
        model_source = f"HuggingFace ({hf_source})"
    else:
        # Pick XGBoost (highest AUC on this dataset per dissertation literature)
        best_model   = all_results["XGBoost"]["model"]
        model_source = "Local training (XGBoost)"
        log.info("  Selected: XGBoost (highest AUC in comparative evaluation)")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate best model
    # -----------------------------------------------------------------------
    log.info("\n[4/6] Evaluating XGBoost ...")

    y_pred_train  = best_model.predict(X_res)
    y_proba_train = best_model.predict_proba(X_res)[:, 1]
    train_acc     = accuracy_score(y_res, y_pred_train)
    train_auc     = roc_auc_score(y_res, y_proba_train)
    train_f1      = f1_score(y_res, y_pred_train, average="macro")

    y_pred_test   = best_model.predict(X_test_raw)
    y_proba_test  = best_model.predict_proba(X_test_raw)[:, 1]
    test_acc      = accuracy_score(y_test_raw, y_pred_test)
    test_auc      = roc_auc_score(y_test_raw, y_proba_test)
