# 🧬 Contact Tracing System — Machine Learning Project

## Project Overview

A full-stack **Contact Tracing System** powered by **Machine Learning** using Python Flask.  
The system tracks individuals, logs physical contact events, and uses a **Random Forest Classifier + Graph Propagation** pipeline to predict each person's infection risk.

---

## 🏗️ Project Structure

```
contact_tracing/
├── app.py                 # Flask backend (REST API)
├── ml_model.py            # ML engine (RandomForest + Graph BFS)
├── requirements.txt       # Python dependencies
├── models/
│   ├── rf_model.pkl       # Saved RandomForest model (auto-generated)
│   └── scaler.pkl         # Saved StandardScaler (auto-generated)
└── templates/
    └── index.html         # Full-stack frontend (D3.js dashboard)
```

---

## 🤖 Machine Learning Architecture

### 1. Algorithm: Random Forest Classifier
- **Library**: scikit-learn `RandomForestClassifier`
- **Trees**: 150 estimators
- **Max depth**: 10
- **Class weights**: Balanced (handles imbalanced classes)
- **Output**: 3 classes — Low / Medium / High risk

### 2. Feature Engineering (12 Features per Person)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `num_contacts` | Total distinct individuals met |
| 2 | `num_infected_contacts` | Count of contacts with infected status |
| 3 | `total_exposure_minutes` | Cumulative minutes of physical proximity |
| 4 | `avg_duration_minutes` | Average contact duration |
| 5 | `max_duration_minutes` | Longest single contact event |
| 6 | `recent_contacts_24h` | Contacts in last 24 hours |
| 7 | `recent_contacts_48h` | Contacts in last 48 hours |
| 8 | `infected_exposure_minutes` | Minutes spent near infected persons |
| 9 | `contact_degree` | Node degree in contact graph |
| 10 | `time_since_last_contact_h` | Hours since most recent contact |
| 11 | `proportion_infected` | Fraction of contacts who are infected |
| 12 | `high_risk_location_count` | Contacts in high-risk venues |

### 3. Graph-Based Risk Propagation (NetworkX BFS)

- Builds an undirected weighted graph of all contact events
- **Edge weights** = `time_decay × duration_factor`
  - Time decay: exponential with 48-hour half-life
  - Duration factor: minutes / 60, capped at 1.0
- **BFS propagation**: Starts from infected nodes, risk decays per hop (factor 0.55)
- **Max depth**: 3 hops
- **Centrality**: Degree centrality identifies super-spreaders

### 4. Score Blending

```
final_risk_score = 0.6 × RF_probability × (class/2) + 0.4 × graph_risk
```

### 5. Training Data

- **2,000 synthetic samples** generated with realistic distributions
- 80/20 train/test split with stratification
- Features generated per risk class (Low / Medium / High) with class-specific distributions

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/people` | List all people |
| POST | `/api/people` | Add a person |
| PUT | `/api/people/<id>` | Update person (status etc.) |
| DELETE | `/api/people/<id>` | Remove a person |
| GET | `/api/contacts` | List all contact events |
| POST | `/api/contacts` | Log a contact event |
| GET | `/api/predict` | Run full ML prediction pipeline |
| POST | `/api/train` | Retrain the Random Forest model |
| GET | `/api/stats` | Aggregate statistics |
| POST | `/api/simulate` | Generate random contact events |
| POST | `/api/reset` | Restore demo data |

---

## 🚀 Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask app
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

The ML model is automatically trained on first startup and saved to `models/`.

---

## 📊 Dashboard Features

| Tab | Features |
|-----|----------|
| **Dashboard** | Stats overview, infection rate meter, top-risk table |
| **People** | Add/edit/delete individuals, status tracking |
| **Contacts** | Log contact events, location-based risk |
| **ML Predictions** | RF + graph risk scores, probability breakdown |
| **Contact Graph** | D3.js force-directed network visualization |
| **Model Info** | Feature importances, training metrics, pipeline diagram |

---

## 🔬 ML Pipeline Flow

```
Raw Data (people + contacts)
        ↓
Feature Engineering (12 features/person)
        ↓
StandardScaler (normalization)
        ↓
RandomForest (150 trees) → class + probabilities
        ↓
NetworkX Graph → BFS risk propagation
        ↓
Score Blending (60% RF + 40% Graph)
        ↓
Risk Label: LOW / MEDIUM / HIGH
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.0 |
| ML | scikit-learn, NumPy, pandas |
| Graph | NetworkX |
| Serialization | joblib |
| Frontend | Vanilla JS, D3.js v7 |
| Fonts | Space Mono, DM Sans |

---

## 📈 Example Model Output

```json
{
  "id": "P002",
  "name": "Bob",
  "status": "healthy",
  "risk_label": "High",
  "risk_score": 0.743,
  "ml_proba": [0.08, 0.21, 0.71],
  "graph_risk": 0.612,
  "centrality": 0.0182,
  "hops_to_infected": 1
}
```
