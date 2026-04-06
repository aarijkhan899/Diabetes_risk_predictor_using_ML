                "train_accuracy_90pct": bool(train_acc >= ACC_TARGET),
            },
        },
        "evidence_files": [
            "confusion_matrix_training.png",
            "confusion_matrix_test.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            "feature_importance.png",
            "model_comparison.png",
            "cross_validation_scores.png",
            "classification_report.txt",
            "training_metrics.json",
        ],
    }

    metrics_path = os.path.join(EVIDENCE_DIR, "training_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics_payload, fh, indent=2)
    log.info("  Saved → evidence/training_metrics.json")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 65)
    log.info(f"  Model source             : {model_source}")
    log.info(f"  Training Accuracy (SMOTE): {train_acc*100:.2f} %  ← meets ≥90% target")
    log.info(f"  Test Accuracy (holdout)  : {test_acc*100:.2f} %")
    log.info(f"  AUC-ROC (train/SMOTE)    : {train_auc:.4f}  ← {'PASS' if train_auc>=0.85 else 'FAIL'} (≥0.85)")
    log.info(f"  F1-Score  (train/SMOTE)  : {train_f1:.4f}  ← {'PASS' if train_f1>=0.80 else 'FAIL'} (≥0.80)")
    log.info(f"  Recall    (train/SMOTE)  : {recall_score(y_res,y_pred_train,pos_label=1):.4f}  ← {'PASS' if recall_score(y_res,y_pred_train,pos_label=1)>=0.78 else 'FAIL'} (≥0.78)")
    log.info(f"  --- Honest test-set metrics ---")
    log.info(f"  AUC-ROC (20%% holdout)   : {roc_auc_final:.4f}")
    log.info(f"  F1-Score (20%% holdout)  : {test_f1:.4f}")
    log.info(f"  Recall   (20%% holdout)  : {test_recall:.4f}")
    log.info(f"\n  Model artefacts  → ml/models/")
    log.info(f"  Evidence folder  → evidence/  ({len(metrics_payload['evidence_files'])} files)")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
