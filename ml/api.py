#!/usr/bin/env python3
"""
Diabetes Risk Predictor — Flask REST API
Serves predictions and SHAP explanations.
"""

import os
import json
import logging
import numpy as np
import joblib
import shap

from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH   = os.path.join(MODELS_DIR, "model_meta.json")

