# Mental Health Tracker

A machine-learning-powered wellness check-in. Fill in a few details about your age, digital habits, and lifestyle, and a trained model returns a predicted mental health score out of 10 — served through a FastAPI backend and a single-page frontend, deployed as one service on Render.

**Live demo:** https://mental-health-tracker-1-hj76.onrender.com/

> Built for informational purposes only — this is not a clinical assessment. If you're struggling, please talk to someone you trust.

---

## How it works

1. **Model training** (`Health_Score.ipynb`) — cleans and explores a student social-media/wellness dataset, builds a `ColumnTransformer` + `Pipeline` (imputing, scaling, ordinal/one-hot encoding), trains and tunes a Random Forest via `RandomizedSearchCV`, and saves the full pipeline with `joblib`.
2. **API** (`main.py`) — loads the saved pipeline once at startup, validates incoming requests with Pydantic, and exposes a `/predict` endpoint that returns a score from 0–10.
3. **Frontend** (`static/`) — a single HTML page with vanilla JS that collects form input, posts it to `/predict`, and renders the result on a live gauge.
4. **Deployment** — one FastAPI app serves both the API and the static frontend from the same origin, so there's no separate frontend host and no CORS configuration needed.

## Features

- End-to-end ML pipeline: cleaning → feature engineering → encoding → tuned Random Forest, all wrapped in one `Pipeline` object
- FastAPI backend with Pydantic request/response validation (bad input returns a structured `422`, not a crash)
- Single-page frontend with client-side validation, loading/error states, and a live score meter
- Same-origin deployment — one Render service, no CORS headaches

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | pandas, scikit-learn, joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML, CSS, vanilla JavaScript |
| Hosting | Render |

## Project structure

```
.
├── Health_Score.ipynb        # data cleaning, EDA, model training & tuning
├── main.py                   # FastAPI app + /predict endpoint
├── mental_health_model.pkl   # trained pipeline (preprocessing + model)
├── requirements.txt
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## API

### `POST /predict`

**Request body**

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 60,
  "study_hours": 3.0,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 6.5,
  "stress_level": "Medium"
}
```

**Response**

```json
{
  "predicted_mental_health_score": 6.42
}
```

Invalid input (out-of-range values, missing fields, unrecognized categories) returns a `422` with field-level error details.

## Running locally

```bash
git clone https://github.com/<your-username>/mental-health-tracker.git
cd mental-health-tracker
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser — the frontend and API are served from the same address.

## Deployment (Render)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Render sets `$PORT` automatically at runtime; the app must bind to it for the platform's proxy to route traffic correctly.

## Notes on the model

`scikit-learn`, `pandas`, and `numpy` versions are pinned in `requirements.txt` to match the versions used when the pipeline was trained and pickled. If you retrain the model, keep the pinned versions in sync with whatever environment produced `mental_health_model.pkl` — an unpickling error will occur otherwise.

## Disclaimer

This tool estimates a wellness signal from self-reported habits using a machine learning model trained on a specific dataset. It is not a diagnostic or clinical tool, and its predictions should not be treated as medical or psychological advice.
