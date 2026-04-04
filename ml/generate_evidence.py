            f"      {name:<22} Train={results[name]['train_accuracy']:.4f}  "
            f"Test={results[name]['test_accuracy']:.4f}  "
            f"AUC={results[name]['test_auc']:.4f}  "
            f"F1={results[name]['test_f1']:.4f}"
        )

    # Store best XGBoost params for metadata
    results["XGBoost"]["best_params"] = xgb_params
    return results


# ===========================================================================
# STEP 4 — Evidence artifact generation
# ===========================================================================
LABEL_NAMES = ["Non-Diabetic", "Diabetic"]
BLUE_PALETTE = ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#90CAF9"]


def _save(fig, filename: str):
    path = os.path.join(EVIDENCE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → evidence/{filename}")


def gen_confusion_matrix(cm, title: str, filename: str, accuracy: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(LABEL_NAMES))
    ax.set_xticks(ticks); ax.set_xticklabels(LABEL_NAMES, fontsize=12)
    ax.set_yticks(ticks); ax.set_yticklabels(LABEL_NAMES, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label", fontsize=13)
