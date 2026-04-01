        if hasattr(y, "iloc"):
            y = y.iloc[:, 0] if y.ndim > 1 else y
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.columns = COLUMNS
        log.info(f"Loaded via ucimlrepo: {df.shape}")
        return df
    except Exception as e:
        log.warning(f"ucimlrepo failed ({e}); falling back to direct CSV download ...")

    csv_url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/"
        "master/pima-indians-diabetes.csv"
    )
    try:
        df = pd.read_csv(csv_url, header=None, names=COLUMNS)
        log.info(f"Loaded from CSV URL: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Cannot load dataset. ucimlrepo and CSV fallback both failed: {e}"
        )


def preprocess(df: pd.DataFrame):
    """Clean, impute, scale and SMOTE-balance the dataset."""
    df = df.copy()

    # Replace biologically invalid zeros with NaN
    df[ZERO_INVALID_COLS] = df[ZERO_INVALID_COLS].replace(0, np.nan)

    # Median imputation per column
