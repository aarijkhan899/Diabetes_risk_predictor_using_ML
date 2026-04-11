# DiabetesRisk AI — Dissertation Research Project

A full-stack diabetes risk prediction system combining:
- **ML layer**: Python (scikit-learn, XGBoost, LightGBM, SHAP) + Flask REST API (`ml/requirements.txt`)
- **Web layer**: Ruby on Rails 7 frontend

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ruby | 3.3.x (see `diabetes_app/Gemfile`) |
| Bundler | 2.x |
| Docker + Docker Compose | 24+ (optional) |

---

## 2. Training the Model

```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

The script will:
1. Download the Pima Indians Diabetes Dataset (via `ucimlrepo`, with CSV fallback).
2. Impute invalid zeros, apply `StandardScaler`, and SMOTE-balance the classes.
3. Grid-search five algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM).
4. Save the best model (by AUC-ROC) to `ml/models/`:
   - `best_model.pkl`
   - `scaler.pkl`
   - `model_meta.json`

Expected performance targets (dissertation success criteria):
- AUC-ROC ≥ 0.85
- F1 ≥ 0.80
- Recall ≥ 0.78

---

## 3. Starting the Flask API

```bash
cd ml
python api.py
```

The API runs on **http://localhost:5001** by default.

### Key endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check — returns model name and AUC |
| GET | `/model_info` | Full model metadata |
| POST | `/predict` | Accepts JSON body, returns prediction + SHAP values |

### Example predict request

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

---

## 4. Setting up and Starting the Rails App

```bash
cd diabetes_app
bundle install
bundle exec rails server
```

The web app runs on **http://localhost:3000**.

The app expects the Flask API at `http://localhost:5001` by default.
Override with the `ML_API_URL` environment variable:

```bash
ML_API_URL=http://my-api-host:5001 bundle exec rails server
```

---

## 5. Docker Compose (local build)

Ship `ml/models/` in the repo (e.g. `best_model.pkl`, `scaler.pkl`, `model_meta.json`) so the API image can start without a training step. If those files are missing, train first:

```bash
cd ml && pip install -r requirements.txt && python train_model.py && cd ..
```

Build and run both services (Compose v2):

```bash
docker compose up --build
```

- Flask API: http://localhost:5001
- Rails app: http://localhost:3000

The Rails image runs `rails db:prepare` on startup via `diabetes_app/docker-entrypoint.sh` so a **named volume** on `./db` (see `docker-compose.yml`) always gets migrations and an empty SQLite database when the container starts. The API image copies `ml/` including `ml/models/` into the image context at build time.

Override the production secret in real deployments:

```bash
SECRET_KEY_BASE=$(openssl rand -hex 64) docker compose up --build
```

### Pre-built images (Docker Hub)

Images are tagged `aariz-ml-api` and `aariz-rails-app` under your Docker Hub user (example: `isammalik`). Set **`DOCKERHUB_USER`** so the compose file can resolve image names (there is no hard-coded namespace in `docker-compose.hub.yml`).

Push multi-arch (`linux/amd64`, `linux/arm64`) images after logging in to Docker Hub:

```bash
export DOCKERHUB_USER=isammalik   # optional if detectable from `~/.docker/config.json`
./bin/push-dockerhub.sh
```

The script reminds you to set **Repository visibility → Public** for each new repo so anonymous `docker pull` works.

Pull and run published images only:

```bash
export DOCKERHUB_USER=isammalik
docker compose -f docker-compose.hub.yml pull && docker compose -f docker-compose.hub.yml up
```

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ML_API_URL` | `http://localhost:5001` | Base URL of the Flask prediction API |
| `ML_API_PORT` | `5001` | Port the Flask API listens on |
| `RAILS_ENV` | `development` (local Rails) / `production` (Dockerfile) | Rails environment |
| `DOCKERHUB_USER` | _(required for hub compose)_ | Docker Hub namespace for `docker-compose.hub.yml` image references |
| `SECRET_KEY_BASE` | _(set in compose for Docker)_ | Rails secret; generate a strong value for production |
| `PRETRAINED_MODEL_URL` | _(empty)_ | Optional URL to download `joblib` model during training fallback |

---

## 7. Fallback Model Behaviour

If none of the locally trained models meet the success thresholds, the training script
attempts the following fallback chain in order:

1. **Local cache** — loads `ml/models/best_model_pretrained.pkl` if it exists.
2. **Remote download** — if `PRETRAINED_MODEL_URL` is set, `train_model.py` attempts to download a `joblib` model before falling back to the best grid-search model.
3. **Best local model** — if both above fail, the highest-AUC locally trained model is
   used regardless of whether it meets thresholds. A warning is logged.

To supply your own pre-trained model, place a `joblib`-serialised sklearn-compatible
classifier at `ml/models/best_model_pretrained.pkl` before running the training script.

---

## 8. Dataset Reference

> Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).
> *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*.
> Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.

UCI ML Repository: https://archive.ics.uci.edu/dataset/34/diabetes

---

## Disclaimer

This project is for academic research purposes only. It has not been clinically validated
and must not be used to make medical decisions. The model was trained on data from
adult females of Pima Indian heritage aged ≥ 21. Clinician oversight is mandatory
before any patient-facing use.
