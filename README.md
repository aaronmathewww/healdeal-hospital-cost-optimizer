# HealDeal

**An AI-powered hospital cost comparison and recommendation system.**

HealDeal demonstrates how a user can filter a generated hospital-like dataset and compare records using a configurable Cost-Effectiveness Score (CES). It includes a Flask REST API, a browser-based single-page interface, an optional Streamlit interface, text sentiment analysis, geocoding, file-based demo accounts, and an experimental cost-regression model.

> [!IMPORTANT]
> **Synthetic data only.** The bundled dataset is generated for software demonstration. Provider names, prices, ratings, review text, review counts, bed counts, accreditation flags, insurance tags, and coordinates are synthetic. They do not represent real hospitals, patient experiences, official accreditation, insurer acceptance, or quotations. Do not use HealDeal for medical, financial, insurance, or provider-selection decisions.

## What is implemented

- **Flask API:** search, metadata, record detail, authentication, favorites, search history, analytics, and chatbot endpoints.
- **Browser UI:** search and filters, map demonstration, record detail, comparison, analytics, profile, and chatbot views.
- **CES ranking:** normalized weighted score over synthetic cost, sample rating, simulated distance, and generated review-text sentiment. Lower scores rank earlier within the current result set; this is not clinical quality scoring.
- **Sentiment analysis:** VADER when installed, with a small rule-based fallback. Input text is generated sample review text, not patient feedback.
- **Geocoding:** local city-centre lookup first, then optional OpenStreetMap Nominatim lookup for other place names. Coordinates associated with the generated records are simulated offsets around city centres.
- **Demo authentication:** password hashes and JSON-file persistence for local demonstrations. Bearer tokens are stored in process memory and expire after eight hours.
- **Experimental cost regression:** Random Forest pipeline with one-hot categorical preprocessing. It is trained and evaluated only against the synthetic dataset. Search uses sample CSV costs by default; the experimental model is not enabled by default.
- **Optional Streamlit interface:** `app.py` is an alternate interface that uses the same demo data and core modules.

## Architecture

```text
Browser (frontend/index.html)
        │  same-origin /api/* requests
        ▼
Flask REST API (backend/app.py) ──────── Optional Gemini / Hugging Face APIs
        │
        ├── ml/ces_engine.py       weighted CES + Haversine distance
        ├── ml/cost_model.py       experimental Random Forest regression
        ├── ml/sentiment.py        VADER / rule-based sentiment
        ├── utils/auth.py          local JSON demo accounts
        ├── utils/geocoder.py      city lookup + Nominatim + local cache
        └── data/hospitals.csv     generated synthetic records

Optional alternate UI: Streamlit (app.py)
```

The Flask interface is the primary application. `app.py` is a separate Streamlit prototype, not the Flask entry point.

## Quick start

Python 3.11 or newer is recommended. The project has been checked with Python 3.13; if you use a newer Python version, install dependencies in a fresh virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/run_demo.py
```

This single command regenerates the deterministic synthetic dataset, trains and evaluates the experimental cost model, runs the automated test suite, and starts the Flask application. Open <http://127.0.0.1:5000>. Press `Ctrl+C` in the terminal to stop it.

To prepare and test without starting the server, use `python scripts/run_demo.py --no-server`. To skip tests during a later quick launch, use `python scripts/run_demo.py --skip-tests`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python scripts/run_demo.py
```

This single command regenerates the deterministic synthetic dataset, trains and evaluates the experimental cost model, runs the automated test suite, and starts the Flask application. Open <http://127.0.0.1:5000>. Press `Ctrl+C` in the terminal to stop it.

To prepare and test without starting the server, use `python scripts/run_demo.py --no-server`. To skip tests during a later quick launch, use `python scripts/run_demo.py --skip-tests`.

The local `.env.example` works for development with optional chatbot keys left blank. A temporary secret key is generated when `SECRET_KEY` is empty. For a deployed environment, set `FLASK_ENV=production` and provide a unique, strong `SECRET_KEY`; the application refuses to start in production without one.

### Optional Streamlit UI

```bash
streamlit run app.py
```

The Streamlit UI is an alternate interface and should be treated with the same synthetic-data limitations.

## Dataset and model

The repository includes 612 reproducible synthetic records across 20 city labels and 20 specialties. Regenerate them with:

```bash
python data/generate_dataset.py
```

Train the experimental cost model with:

```bash
python ml/cost_model.py
```

The current saved model was trained on 612 synthetic rows. A reproducible 80/20 holdout split produced:

| Metric | Result |
|---|---:|
| Holdout MAE | ₹17,846.43 |
| Holdout RMSE | ₹29,962.32 |
| Holdout R² | 0.7818 |
| 5-fold shuffled CV R² | 0.7799 ± 0.0414 |

These metrics describe performance against generated sample values only. They are **not** evidence of real-world hospital price accuracy, generalization to actual providers, or suitability for patient decisions. Full metrics and caveat are in `models/cost_model_metrics.json`.

## Configuration

Copy `.env.example` to `.env`. Optional chatbot configuration:

- `GEMINI_API_KEY` / `GEMINI_MODEL`
- `HF_API_KEY` / `HF_MODEL` (fallback)
- `CHAT_TIMEOUT_SECONDS`

API keys must remain server-side. Do not put them in frontend JavaScript or commit `.env`. If no key is configured, HealBot uses a limited offline informational response. The chatbot is not a medical professional and must not be used for diagnosis or treatment decisions.

`CORS_ORIGINS` should contain only trusted frontend origins. The default allows the local app origins only.

## API overview

All endpoints are served under `/api`.

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| GET | `/api/health` | Service and dataset status | No |
| GET | `/api/hospitals/meta` | Cities, specialties, tiers and sample insurance tags | No |
| POST | `/api/hospitals/search` | Filter, rank and paginate demo records | No |
| GET | `/api/hospitals/<hospital_id>` | Demo record detail and generated text snippets | No |
| POST | `/api/auth/register` | Create local demo account | No |
| POST | `/api/auth/login` | Obtain in-memory bearer token | No |
| GET | `/api/auth/profile` | Current profile | Yes |
| POST | `/api/auth/logout` | Invalidate current token | Yes |
| GET | `/api/user/favorites` | List favorites | Yes |
| POST | `/api/user/favorites` | Save a known demo record | Yes |
| DELETE | `/api/user/favorites/<hospital_id>` | Remove a favorite | Yes |
| GET | `/api/user/history` | Recent search history | Yes |
| GET | `/api/analytics/overview` | Aggregate demo dataset statistics | No |
| POST | `/api/chat` | Optional informational chatbot | No |

Search request example:

```json
{
  "location": "Bengaluru",
  "specialty": "Any",
  "tier": "Any",
  "insurance": "Any",
  "max_km": 50,
  "page": 1,
  "per_page": 12,
  "use_ml": false,
  "weights": {
    "cost": 0.4,
    "rating": 0.3,
    "distance": 0.2,
    "sentiment": 0.1
  }
}
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers the core ranking functions, generated-data integrity, model output constraints, and Flask API routes. External geocoding and LLM calls are not required by the tests.

## Important limitations and production gaps

- The dataset is synthetic; there is no live hospital, treatment-price, review, accreditation, or insurer integration.
- The generated coordinates are illustrative, not facility addresses. The map is a visualization of synthetic points.
- The optional regression model is a learning demonstration, not a validated healthcare pricing model.
- The CES is a configurable ranking heuristic, not a clinical quality or safety measure.
- Local account storage is a JSON file and is not designed for concurrent multi-user production deployment. Tokens are in-memory and are lost on restart.
- No account verification, password reset, persistent session store, rate limiting, audit logging, or relational database is implemented.
- The optional chatbot is informational only and is not a substitute for a doctor, hospital billing office, or insurer.
- External CDN assets and Nominatim access require an internet connection.

For details about the generated dataset, see [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md). Security considerations and current limitations are listed in [`docs/SECURITY_NOTES.md`](docs/SECURITY_NOTES.md).

## Licensing

No open-source license is currently declared. Do not assume the project or third-party assets may be reused or redistributed under a particular license. Review the licenses and terms for all dependencies and external services before publication or commercial use.
