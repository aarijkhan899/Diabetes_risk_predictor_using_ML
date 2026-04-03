#!/usr/bin/env python3
"""
Diabetes Risk Predictor — Evidence Generation Pipeline
=======================================================
Dissertation: Md Aariz | MSc Big Data Technologies | UEL | CN7000
Supervisor:   Dr Mohamed Chahine Ghanem

This script:
  1. Loads the Pima Indians Diabetes Dataset (UCI ML Repository)
  2. Applies the full preprocessing pipeline (imputation, scaling, SMOTE)
  3. Attempts to pull a pre-trained model from HuggingFace Hub (token-authenticated)
  4. Falls back to locally training all 5 classifiers with grid-search CV
  5. Selects XGBoost as best model (highest AUC) and saves artefacts
  6. Generates comprehensive evidence artefacts to /evidence/:
       confusion_matrix_training.png
       confusion_matrix_test.png
       roc_curve.png
       precision_recall_curve.png
       feature_importance.png
       model_comparison.png
       cross_validation_scores.png
       classification_report.txt
       training_metrics.json
"""

import os
import sys
import json
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
