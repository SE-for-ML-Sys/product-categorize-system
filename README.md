# Product Categorization System — Monorepo

This repository consolidates all components of the Product Categorization System into a single unified codebase.

## Repository Structure

```
product-categorize-system/
├── app/                    # Main Application
│   ├── backend/            # FastAPI (prediction API, drift orchestrator, SQLite ORM)
│   └── frontend/           # Next.js UI (product upload, prediction results, history)
├── monitoring/             # Monitoring Dashboard
│   ├── app/                # FastAPI monitoring API
│   ├── src/monitoring/     # Monitoring logic (store, orchestrator, quality)
│   └── frontend/           # Vite / React dashboard (charts, review queue, alerts)
├── ml-training/            # ML Training Pipeline (notebook, training scripts)
├── docker-compose.yml      # Run all services with one command
└── README.md
```

## Components

### 🖥️ `app/` — Main Application
The core product categorisation app.
- **Backend** (port 8000): REST API for image prediction, drift orchestrator, SQLite persistence
- **Frontend** (port 3000): Next.js UI for submitting product images and viewing results

### 📊 `monitoring/` — Monitoring Dashboard
A dedicated monitoring service that visualises model performance and data drift.
- **Backend** (port 8001): FastAPI API (KPIs, drift trends, review queue, alerts)
- **Frontend** (port 5173): Vite / React dashboard

### 🤖 `ml-training/` — ML Training Pipeline
Data preprocessing, feature engineering, model training scripts and a Jupyter notebook.  
No server — see [`ml-training/README.md`](./ml-training/README.md) for instructions.

---

## Shared Database

Both the `app-backend` and `monitoring-backend` services **share a single SQLite file**
(`product_categorization.db`) via the `db_data` Docker volume.  All four tables
(`prediction_events`, `human_feedback`, `drift_events`, `alerts`) live in this one file
so the monitoring dashboard always reflects real prediction data.

---

## 🚀 Quick Start — Docker (Recommended)

> **Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose plugin)

```bash
# 1. Clone the repo
git clone https://github.com/SE-for-ML-Sys/product-categorize-system.git
cd product-categorize-system

# 2. Build and start all four services
docker compose up --build

# 3. (First-time or after schema changes) the DB is created automatically on
#    startup — no extra step needed.
```

> **Tip:** add `-d` to run in the background: `docker compose up --build -d`

### Service URLs

| Service | URL |
|---|---|
| App Backend (FastAPI) | <http://localhost:8000> |
| App Frontend (Next.js) | <http://localhost:3000> |
| Monitoring Backend (FastAPI) | <http://localhost:8001> |
| Monitoring Frontend (Vite) | <http://localhost:5173> |
| DB Web Preview | <http://localhost:8001/monitoring/db-web> |
| API Docs (app) | <http://localhost:8000/docs> |
| API Docs (monitoring) | <http://localhost:8001/docs> |

### Useful Docker Commands

```bash
# View running containers and their status
docker compose ps

# Tail logs for a specific service
docker compose logs -f app-backend

# Stop all services
docker compose down

# Stop and remove the shared database volume (resets all data)
docker compose down -v
```

---

## 🛠 Manual Setup (without Docker)

Run the services in separate terminals. A shared database path can be configured
via environment variables (see notes below).

### Step 1 — App Backend

```bash
cd app/backend

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: point to a custom DB path (defaults to /data/product_categorization.db)
export DATABASE_URL="sqlite:///./product_categorization.db"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API available at <http://localhost:8000> · Docs at <http://localhost:8000/docs>

### Step 2 — App Frontend

```bash
cd app/frontend

cp .env.example .env             # edit NEXT_PUBLIC_API_URL if needed
npm install
npm run dev
```

Frontend at <http://localhost:3000>

### Step 3 — Monitoring Backend

```bash
cd monitoring

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Point to the SAME database file used by the app backend
export DB_PATH="$(pwd)/../app/backend/product_categorization.db"

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Monitoring API at <http://localhost:8001> · DB preview at <http://localhost:8001/monitoring/db-web>

### Step 4 — Monitoring Frontend

```bash
cd monitoring/frontend

# Set the API URL to the monitoring backend
echo "VITE_API_URL=http://localhost:8001" > .env

npm install
npm run dev
```

Dashboard at <http://localhost:5173>

### Step 5 — Seed Sample Data (optional)

You can seed the database with mock data to test the monitoring dashboard.

**Option A: Using Docker Compose (Recommended)**
*(Ensure `docker compose up -d` is running)*
```bash
docker compose exec monitoring-backend python scripts/seed_mock_data.py --reset
```

**Option B: Manual Setup (Local)**
```bash
cd monitoring
source venv/bin/activate
# Specify the path to the app's database and use python3
DB_PATH=../app/backend/product_categorization.db python3 scripts/seed_mock_data.py --reset
```

---

## API Reference

### App Backend (port 8000)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload a JPG/PNG image for classification |
| `GET` | `/history` | Prediction history (`?limit=20&offset=0`) |
| `GET` | `/healthz` | Health check |
| `POST` | `/feedback` | Submit human feedback for low-confidence predictions |

### Monitoring Backend (port 8001)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/monitoring/kpi` | Today's KPIs (requests, low-confidence rate, drift status) |
| `GET` | `/monitoring/confidence-trend` | Avg confidence over last 24 h |
| `GET` | `/monitoring/drift-trend` | Drift scores over time |
| `GET` | `/monitoring/class-ratio` | Class distribution over time |
| `GET` | `/monitoring/review-queue` | Predictions needing human labels |
| `POST` | `/monitoring/review-queue/{id}/label` | Submit a label |
| `GET` | `/monitoring/alerts` | Active and resolved alerts |
| `POST` | `/monitoring/alerts/{id}/resolve` | Resolve an alert |
| `GET` | `/monitoring/db-web` | Browser-based DB viewer |

---

## Database Schema

All four tables live in a single SQLite file shared by both backends:

| Table | Purpose |
|---|---|
| `prediction_events` | Every prediction request (class, confidence, quality metrics) |
| `human_feedback` | Human-provided true labels for low-confidence predictions |
| `drift_events` | Results of drift detection runs |
| `alerts` | System alerts for administrators |

### Reset the Database (Development)

```bash
# Docker
docker compose down -v            # removes the db_data volume
docker compose up --build

# Manual
rm -f app/backend/product_categorization.db
# Restart the app backend — tables are recreated automatically
```

---

## Repository Origins

| Component | Original Repository |
|---|---|
| `app/` | [Janeninie/product-categorization-system-app](https://github.com/Janeninie/product-categorization-system-app) |
| `monitoring/` | [kcopyk/Monitoring_Dashboard](https://github.com/kcopyk/Monitoring_Dashboard) |
| `ml-training/` | [View-MG/product-categorization-system](https://github.com/View-MG/product-categorization-system) |
