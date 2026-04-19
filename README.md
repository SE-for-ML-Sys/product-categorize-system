# Product Categorization System — Monorepo

This repository consolidates all components of the Product Categorization System into a single unified codebase.

## Repository Structure

```
product-categorize-system/
├── monitoring/        # Monitoring Dashboard (FastAPI API + Vite/React UI)
├── ml-training/       # ML Training Pipeline (Jupyter notebook, training scripts, feature engineering)
└── app/               # Main Application (FastAPI backend + Next.js frontend)
```

## Components

### 🖥️ `app/` — Main Application
The core product categorization app with a FastAPI backend and Next.js frontend.
- Backend: REST API for product prediction, drift monitoring orchestrator, SQLite persistence
- Frontend: Next.js UI for submitting products and viewing categorization results

See [`app/README.md`](./app/README.md) for setup and usage instructions.

---

### 📊 `monitoring/` — Monitoring Dashboard
A dedicated monitoring service with a FastAPI API and a Vite/React dashboard for visualizing model performance and data drift metrics.

See [`monitoring/README.md`](./monitoring/README.md) for setup and usage instructions.

---

### 🤖 `ml-training/` — ML Training Pipeline
The machine learning training pipeline, including data preprocessing, feature engineering, model training scripts, and a Jupyter notebook.

See [`ml-training/README.md`](./ml-training/README.md) for setup and usage instructions.

---

## Getting Started

Each component has its own `requirements.txt` (Python) or `package.json` (Node.js). Refer to the README in each subdirectory for specific setup instructions.

## Repository Origins

| Component | Original Repository |
|---|---|
| `app/` | [Janeninie/product-categorization-system-app](https://github.com/Janeninie/product-categorization-system-app) |
| `monitoring/` | [kcopyk/Monitoring_Dashboard](https://github.com/kcopyk/Monitoring_Dashboard) |
| `ml-training/` | [View-MG/product-categorization-system](https://github.com/View-MG/product-categorization-system) |
