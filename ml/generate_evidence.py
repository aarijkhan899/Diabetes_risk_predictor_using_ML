    ax.set_ylim([0.0, 1.05]); ax.set_xlim([0.0, 1.0])
    ax.set_title(
        "Precision-Recall Curve — XGBoost\nPima Indians Diabetes Dataset  (20 % test holdout)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, filename)


def gen_feature_importance(model, filename: str):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    colors      = plt.cm.Blues(np.linspace(0.4, 0.9, len(FEATURES)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(FEATURES)),
        importances[indices],
        color=colors, edgecolor="white", linewidth=0.8,
    )
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in indices], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Feature Importance (XGBoost gain)", fontsize=13)
    ax.set_title(
        "Feature Importance — XGBoost\nPima Indians Diabetes Dataset",
        fontsize=13, fontweight="bold",
    )
    for bar, imp in zip(bars, importances[indices]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{imp:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(importances) * 1.18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, filename)


def gen_model_comparison(comparison: dict, filename: str):
    names         = list(comparison.keys())
    metric_keys   = ["train_accuracy", "test_accuracy", "test_auc", "test_f1"]
