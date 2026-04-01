#!/usr/bin/env python3
"""
Diabetes Risk Predictor - Model Training Script
Trains multiple classifiers on the Pima Indians Diabetes Dataset,
selects the best by AUC-ROC, and saves it. Falls back to a
pre-trained model download if local training underperforms.
"""

import os
import sys
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
