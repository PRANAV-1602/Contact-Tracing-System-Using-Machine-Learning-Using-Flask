"""
ml_model.py — Core Machine Learning Engine for Contact Tracing

Models Used:
  1. RandomForestClassifier  — predicts infection risk (High/Medium/Low)
  2. Graph-based BFS propagation — spreads risk scores through contact network
  3. Time-decay weighting — recent contacts carry higher risk
  4. Degree centrality  — super-spreader detection
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict, deque
from datetime import datetime, timedelta
import math
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(person_id, people, contacts):
    """
    Build a feature vector for a given person:

    Feature                     | Description
    ─────────────────────────────────────────────────────
    num_contacts                | Total distinct people met
    num_infected_contacts       | How many contacts are infected
    total_exposure_minutes      | Cumulative exposure time (all contacts)
    avg_duration_minutes        | Average contact duration
    max_duration_minutes        | Longest single contact
    recent_contacts_24h         | Contacts in last 24 hours
    recent_contacts_48h         | Contacts in last 48 hours
    infected_exposure_minutes   | Minutes spent near infected people
    contact_degree              | Degree in contact graph (centrality proxy)
    time_since_last_contact_h   | Hours since most recent contact
    proportion_infected         | infected_contacts / total_contacts
    high_risk_location_count    | Contacts in high-risk locations
    """
    now = datetime.utcnow()#universal codinated time
    person_contacts = [c for c in contacts
                       if c['person_a'] == person_id or c['person_b'] == person_id]

    if not person_contacts:
        return [0] * 12

    neighbors = set()
    infected_neighbors = set()
    total_minutes = 0
    infected_minutes = 0
    durations = []
    recent_24 = 0
    recent_48 = 0
    high_risk_locs = 0
    timestamps = []

    for c in person_contacts:
        other_id = c['person_b'] if c['person_a'] == person_id else c['person_a']
        dur = c.get('duration_minutes', 15)
        ts = datetime.fromisoformat(c['timestamp'])
        hours_ago = (now - ts).total_seconds() / 3600

        neighbors.add(other_id)
        total_minutes += dur
        durations.append(dur)
        timestamps.append(ts)

        if hours_ago <= 24:
            recent_24 += 1
        if hours_ago <= 48:
            recent_48 += 1

        if c.get('location_risk', 'low') == 'high':
            high_risk_locs += 1

        other = people.get(other_id, {})
        if other.get('status') == 'infected':
            infected_neighbors.add(other_id)
            infected_minutes += dur

    last_contact_h = (now - max(timestamps)).total_seconds() / 3600 if timestamps else 999
    proportion_infected = len(infected_neighbors) / len(neighbors) if neighbors else 0

    features = [
        len(neighbors),                         # num_contacts
        len(infected_neighbors),                # num_infected_contacts
        total_minutes,                          # total_exposure_minutes
        np.mean(durations) if durations else 0, # avg_duration_minutes
        max(durations) if durations else 0,     # max_duration_minutes
        recent_24,                              # recent_contacts_24h
        recent_48,                              # recent_contacts_48h
        infected_minutes,                       # infected_exposure_minutes
        len(neighbors),                         # contact_degree (proxy)
        last_contact_h,                         # time_since_last_contact_h
        proportion_infected,                    # proportion_infected
        high_risk_locs,                         # high_risk_location_count
    ]
    return features


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_data(n_samples=2000):
    """
    Generates labelled synthetic training data with realistic distributions.
    Labels: 0 = Low risk, 1 = Medium risk, 2 = High risk
    """
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_samples):
        # Randomly assign ground truth risk class
        risk_class = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        if risk_class == 0:   # Low
            num_contacts              = np.random.randint(0, 5)
            num_infected_contacts     = np.random.randint(0, 1)
            total_exposure_minutes    = np.random.randint(0, 30)
            avg_duration              = np.random.uniform(5, 20)
            max_duration              = np.random.uniform(10, 30)
            recent_24                 = np.random.randint(0, 2)
            recent_48                 = np.random.randint(0, 3)
            infected_minutes          = np.random.randint(0, 10)
            degree                    = np.random.randint(0, 5)
            last_contact_h            = np.random.uniform(48, 200)
            proportion_infected       = np.random.uniform(0, 0.1)
            high_risk_locs            = 0

        elif risk_class == 1:  # Medium
            num_contacts              = np.random.randint(3, 15)
            num_infected_contacts     = np.random.randint(1, 3)
            total_exposure_minutes    = np.random.randint(30, 120)
            avg_duration              = np.random.uniform(15, 45)
            max_duration              = np.random.uniform(30, 90)
            recent_24                 = np.random.randint(1, 5)
            recent_48                 = np.random.randint(2, 8)
            infected_minutes          = np.random.randint(10, 60)
            degree                    = np.random.randint(3, 12)
            last_contact_h            = np.random.uniform(12, 48)
            proportion_infected       = np.random.uniform(0.1, 0.4)
            high_risk_locs            = np.random.randint(0, 2)

        else:                  # High
            num_contacts              = np.random.randint(10, 40)
            num_infected_contacts     = np.random.randint(3, 10)
            total_exposure_minutes    = np.random.randint(120, 600)
            avg_duration              = np.random.uniform(30, 120)
            max_duration              = np.random.uniform(60, 240)
            recent_24                 = np.random.randint(5, 15)
            recent_48                 = np.random.randint(8, 25)
            infected_minutes          = np.random.randint(60, 300)
            degree                    = np.random.randint(10, 40)
            last_contact_h            = np.random.uniform(0, 12)
            proportion_infected       = np.random.uniform(0.4, 1.0)
            high_risk_locs            = np.random.randint(1, 5)

        X.append([
            num_contacts, num_infected_contacts, total_exposure_minutes,
            avg_duration, max_duration, recent_24, recent_48,
            infected_minutes, degree, last_contact_h,
            proportion_infected, high_risk_locs
        ])
        y.append(risk_class)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """Train the Random Forest model and save to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X, y = generate_training_data(2000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Low', 'Medium', 'High'],
                                   output_dict=True)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {
        "accuracy": round(acc * 100, 2),
        "report": report,
        "feature_importances": clf.feature_importances_.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH-BASED RISK PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

class ContactGraph:
    """
    Builds a NetworkX graph from contacts and runs:
      - BFS risk propagation with exponential time-decay
      - Degree centrality (super-spreader detection)
      - Shortest path from each person to nearest infected
    """
    HALF_LIFE_HOURS  = 48     # time decay
    MAX_DEPTH        = 3      # BFS hops
    HOP_DECAY        = 0.55   # per-hop decay factor

    def __init__(self, people, contacts):
        self.people   = people
        self.contacts = contacts
        self.G        = self._build_graph()

    def _build_graph(self):
        G = nx.Graph()
        now = datetime.utcnow()
        for pid, p in self.people.items():
            G.add_node(pid, **p)
        for c in self.contacts:
            a, b = c['person_a'], c['person_b']
            hours_ago = (now - datetime.fromisoformat(c['timestamp'])).total_seconds() / 3600
            t_decay   = math.exp(-0.693 * hours_ago / self.HALF_LIFE_HOURS)
            dur_weight = min(c.get('duration_minutes', 15) / 60.0, 1.0)
            weight    = t_decay * dur_weight
            if G.has_edge(a, b):
                G[a][b]['weight'] = max(G[a][b]['weight'], weight)
            else:
                G.add_edge(a, b, weight=weight, **c)
        return G

    def propagate_risk(self):
        """BFS from infected nodes, propagating decayed risk scores."""
        risk = defaultdict(float)
        for pid, p in self.people.items():
            if p['status'] == 'infected':
                risk[pid] = 1.0

        queue = deque([(pid, 0) for pid in risk])
        visited = {pid: 0 for pid in risk}

        while queue:
            curr, depth = queue.popleft()
            if depth >= self.MAX_DEPTH:
                continue
            for nbr in self.G.neighbors(curr):
                edge_w = self.G[curr][nbr].get('weight', 0.1)
                contribution = risk[curr] * edge_w * (self.HOP_DECAY ** (depth + 1))
                if contribution > risk[nbr]:
                    risk[nbr] = contribution
                if visited.get(nbr, self.MAX_DEPTH + 1) > depth + 1:
                    visited[nbr] = depth + 1
                    queue.append((nbr, depth + 1))

        # Infected always stay at 1.0
        for pid, p in self.people.items():
            if p['status'] == 'infected':
                risk[pid] = 1.0

        return dict(risk)

    def centrality(self):
        if len(self.G.nodes) == 0:
            return {}
        return nx.degree_centrality(self.G)

    def shortest_path_to_infected(self, person_id):
        infected_ids = [pid for pid, p in self.people.items()
                        if p['status'] == 'infected']
        min_dist = None
        for inf_id in infected_ids:
            if nx.has_path(self.G, person_id, inf_id):
                d = nx.shortest_path_length(self.G, person_id, inf_id)
                if min_dist is None or d < min_dist:
                    min_dist = d
        return min_dist  # None means no path

    def get_graph_data(self):
        """Serialise graph for D3.js visualisation."""
        nodes = []
        for pid, data in self.G.nodes(data=True):
            nodes.append({
                "id": pid,
                "name": data.get("name", pid),
                "status": data.get("status", "healthy"),
                "age": data.get("age", 0),
            })
        edges = []
        for a, b, data in self.G.edges(data=True):
            edges.append({
                "source": a,
                "target": b,
                "weight": round(data.get("weight", 0), 3),
            })
        return {"nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICT
# ─────────────────────────────────────────────────────────────────────────────

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

def predict_all_risks(people, contacts):
    """
    Run the full ML + graph pipeline and return enriched person records.
    """
    if not people:
        return {}

    clf, scaler = load_model()
    cg = ContactGraph(people, contacts)
    graph_risk  = cg.propagate_risk()
    centrality  = cg.centrality()

    results = {}
    for pid, p in people.items():
        if p['status'] == 'infected':
            results[pid] = {
                **p,
                "risk_label": "Infected",
                "risk_score": 1.0,
                "risk_color": "#dc2626",
                "ml_class": 2,
                "graph_risk": 1.0,
                "centrality": round(centrality.get(pid, 0), 4),
                "hops_to_infected": 0,
            }
            continue

        features = extract_features(pid, people, contacts)
        X_scaled = scaler.transform([features])
        ml_class = int(clf.predict(X_scaled)[0])
        ml_proba = clf.predict_proba(X_scaled)[0]

        g_risk = graph_risk.get(pid, 0.0)
        # Blend: 60% ML probability score + 40% graph propagation
        blended = 0.6 * float(ml_proba[ml_class]) * (ml_class / 2.0) + 0.4 * g_risk
        blended = min(blended, 0.99)

        hops = cg.shortest_path_to_infected(pid)

        results[pid] = {
            **p,
            "risk_label": RISK_LABELS[ml_class],
            "risk_score": round(blended, 3),
            "risk_color": RISK_COLORS[ml_class],
            "ml_class": ml_class,
            "ml_proba": [round(x, 3) for x in ml_proba.tolist()],
            "graph_risk": round(g_risk, 3),
            "centrality": round(centrality.get(pid, 0), 4),
            "hops_to_infected": hops,
        }

    return results, cg.get_graph_data()
