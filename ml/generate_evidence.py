        log.warning(f"  HuggingFace download error: {e}")
        return None, None


# ===========================================================================
# STEP 1 — Dataset loading
# ===========================================================================
def load_dataset() -> pd.DataFrame:
    log.info("  Loading Pima Indians Diabetes Dataset via ucimlrepo ...")
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=34)
        X  = ds.data.features
        y  = ds.data.targets
        if hasattr(y, "iloc"):
            y = y.iloc[:, 0] if y.ndim > 1 else y
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.columns = COLUMNS
        log.info(f"  Loaded via ucimlrepo: {df.shape}")
        return df
    except Exception as e:
        log.warning(f"  ucimlrepo failed ({e}); trying CSV fallback ...")

    csv_url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/pima-indians-diabetes.csv"
    )
    df = pd.read_csv(csv_url, header=None, names=COLUMNS)
    log.info(f"  Loaded from CSV fallback: {df.shape}")
    return df


# ===========================================================================
# STEP 2 — Preprocessing
# ===========================================================================
def preprocess(df: pd.DataFrame):
    """
    Leak-free preprocessing pipeline:
      1. Impute biologically invalid zeros with column medians
      2. Stratified 80/20 train-test split  (split BEFORE scaling/SMOTE)
      3. Fit StandardScaler on train only; transform both splits
      4. Apply SMOTE to training split only
    Returns (X_train_res, y_train_res, X_test, y_test, scaler)
    """
