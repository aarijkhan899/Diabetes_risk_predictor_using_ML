    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_title(f"{title}\nAccuracy: {accuracy:.2%}", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, filename)


def gen_roc_curve(model, X_test, y_test, filename: str) -> float:
    y_proba          = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _      = roc_curve(y_test, y_proba)
    roc_auc_val      = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2.5,
            label=f"XGBoost  (AUC = {roc_auc_val:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title(
        "ROC Curve — XGBoost\nPima Indians Diabetes Dataset  (20 % test holdout)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, filename)
    return roc_auc_val


def gen_precision_recall_curve(model, X_test, y_test, filename: str):
    y_proba          = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap               = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recall, precision, color="#D32F2F", where="post", lw=2.5,
            label=f"XGBoost  (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.08, color="#D32F2F", step="post")
    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="grey", lw=1.2,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
