    metric_labels = ["Train Accuracy", "Test Accuracy", "AUC-ROC", "F1 (macro)"]
    bar_colors    = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

    x     = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, bar_colors)):
        vals  = [comparison[n][key] for n in names]
        rects = ax.bar(x + i * width, vals, width,
                       label=label, color=color, alpha=0.85, edgecolor="white")
        for rect in rects:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.004,
                f"{rect.get_height():.2f}",
                ha="center", va="bottom", fontsize=8.5, rotation=90,
            )

    ax.axhline(0.90, color="red", linestyle="--", lw=1.5, alpha=0.6, label="90 % target")
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(
        "Multi-Model Comparison — Pima Indians Diabetes Dataset\n"
        "All classifiers trained on SMOTE-balanced set; evaluated on 20 % holdout",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=11, rotation=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, filename)


def gen_cv_bar(X_res, y_res, best_model, filename: str):
    """5-fold stratified CV accuracy bars using the tuned XGBoost."""
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores   = cross_val_score(best_model, X_res, y_res, cv=cv, scoring="accuracy")
    mean_s   = scores.mean()
    std_s    = scores.std()

    fold_labels = [f"Fold {i+1}" for i in range(5)]
    colors = ["#1565C0" if s == scores.max() else "#42A5F5" for s in scores]
