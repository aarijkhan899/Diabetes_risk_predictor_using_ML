
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(fold_labels, scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{score:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.axhline(mean_s, color="#D32F2F", linestyle="--", lw=2,
               label=f"Mean = {mean_s:.4f}  ±  {std_s:.4f}")
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        "5-Fold Stratified Cross-Validation — XGBoost\n"
        "Pima Indians Diabetes Dataset  (SMOTE-balanced training set)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.set_ylim(max(0.80, scores.min() - 0.02), 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, filename)
    return scores


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 65)
    log.info("  DIABETES RISK PREDICTOR — EVIDENCE GENERATION PIPELINE")
    log.info(f"  Dissertation: Md Aariz | MSc Big Data | UEL | CN7000")
    log.info("=" * 65)

    # -----------------------------------------------------------------------
    # Step 0: Try HuggingFace pre-trained model
    # -----------------------------------------------------------------------
    log.info("\n[0/6] HuggingFace pre-trained model check ...")
    hf_model, hf_source = try_download_hf_model()

    # -----------------------------------------------------------------------
    # Step 1: Load dataset
    # -----------------------------------------------------------------------
    log.info("\n[1/6] Loading dataset ...")
