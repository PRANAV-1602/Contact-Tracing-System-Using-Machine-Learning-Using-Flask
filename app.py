"""
app.py — Flask Backend for Contact Tracing System
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import random, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from ml_model import predict_all_risks, train_model

app = Flask(__name__)

# ─── In-Memory Store ─────────────────────────────────────────────────────────
people   = {}
contacts = []
next_id  = [1]

LOCATIONS = ["Hospital","Market","School","Office","Restaurant",
             "Gym","Airport","Mall","Park","University"]
LOCATION_RISK = {
    "Hospital":"high","Airport":"high","Mall":"high",
    "Market":"medium","Restaurant":"medium","Gym":"medium",
    "School":"low","Office":"low","Park":"low","University":"low"
}

# ─── Seed Demo Data ──────────────────────────────────────────────────────────
def seed_demo_data():
    global people, contacts
    people = {}; contacts = []; next_id[0] = 1

    demo_people = [
        ("Alice",28,"F","infected"), ("Bob",34,"M","healthy"),
        ("Carol",22,"F","healthy"),  ("David",45,"M","quarantined"),
        ("Eve",31,"F","healthy"),    ("Frank",55,"M","infected"),
        ("Grace",27,"F","healthy"),  ("Hank",38,"M","healthy"),
        ("Iris",24,"F","healthy"),   ("Jack",41,"M","healthy"),
        ("Karen",36,"F","recovered"),("Liam",29,"M","healthy"),
    ]
    for name,age,gender,status in demo_people:
        pid = f"P{next_id[0]:03d}"; next_id[0] += 1
        people[pid] = {"id":pid,"name":name,"age":age,
                       "gender":gender,"status":status,
                       "created_at":datetime.utcnow().isoformat()}

    ids = list(people.keys())
    pairs = [
        (ids[0],ids[1],45,2,"Hospital","high"),
        (ids[0],ids[2],30,5,"Market","medium"),
        (ids[1],ids[2],60,10,"Office","low"),
        (ids[1],ids[3],20,25,"Restaurant","medium"),
        (ids[2],ids[4],15,48,"Park","low"),
        (ids[3],ids[4],90,3,"Hospital","high"),
        (ids[4],ids[5],30,1,"Airport","high"),
        (ids[5],ids[6],45,6,"Mall","high"),
        (ids[5],ids[7],60,8,"Gym","medium"),
        (ids[6],ids[8],25,12,"School","low"),
        (ids[7],ids[8],40,18,"Office","low"),
        (ids[8],ids[9],35,36,"University","low"),
        (ids[9],ids[10],50,72,"Market","medium"),
        (ids[10],ids[11],20,96,"Park","low"),
        (ids[0],ids[5],80,4,"Hospital","high"),
    ]
    for a,b,dur,hrs,loc,lrisk in pairs:
        ts = (datetime.utcnow()-timedelta(hours=hrs)).isoformat()
        contacts.append({"person_a":a,"person_b":b,"duration_minutes":dur,
                         "timestamp":ts,"location":loc,"location_risk":lrisk})

seed_demo_data()

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/people", methods=["GET"])
def get_people():
    return jsonify(list(people.values()))

@app.route("/api/people", methods=["POST"])
def add_person():
    data = request.json
    pid = f"P{next_id[0]:03d}"; next_id[0] += 1
    person = {"id":pid,"name":data.get("name","Unknown"),
              "age":int(data.get("age",0)),"gender":data.get("gender","Unknown"),
              "status":data.get("status","healthy"),
              "created_at":datetime.utcnow().isoformat()}
    people[pid] = person
    return jsonify(person), 201

@app.route("/api/people/<pid>", methods=["PUT"])
def update_person(pid):
    if pid not in people: return jsonify({"error":"Not found"}), 404
    data = request.json
    people[pid].update({k:v for k,v in data.items() if k!="id"})
    return jsonify(people[pid])

@app.route("/api/people/<pid>", methods=["DELETE"])
def delete_person(pid):
    if pid not in people: return jsonify({"error":"Not found"}), 404
    del people[pid]
    global contacts
    contacts = [c for c in contacts if c['person_a']!=pid and c['person_b']!=pid]
    return jsonify({"deleted":pid})

@app.route("/api/contacts", methods=["GET"])
def get_contacts():
    enriched = []
    for c in contacts:
        ec = dict(c)
        ec['person_a_name'] = people.get(c['person_a'],{}).get('name',c['person_a'])
        ec['person_b_name'] = people.get(c['person_b'],{}).get('name',c['person_b'])
        enriched.append(ec)
    return jsonify(enriched)

@app.route("/api/contacts", methods=["POST"])
def add_contact():
    data = request.json
    a,b = data.get("person_a"), data.get("person_b")
    if not a or not b: return jsonify({"error":"Both persons required"}), 400
    if a not in people or b not in people: return jsonify({"error":"Person not found"}), 404
    loc = data.get("location","Unknown")
    contact = {"person_a":a,"person_b":b,
               "duration_minutes":int(data.get("duration_minutes",15)),
               "timestamp":data.get("timestamp",datetime.utcnow().isoformat()),
               "location":loc,"location_risk":LOCATION_RISK.get(loc,"low")}
    contacts.append(contact)
    return jsonify(contact), 201

@app.route("/api/predict", methods=["GET"])
def predict():
    if not people: return jsonify({"error":"No people"}), 400
    results, graph_data = predict_all_risks(people, contacts)
    return jsonify({"predictions":results,"graph":graph_data})

@app.route("/api/train", methods=["POST"])
def retrain():
    stats = train_model()
    return jsonify(stats)

@app.route("/api/stats", methods=["GET"])
def stats():
    sc = {}
    for p in people.values():
        sc[p['status']] = sc.get(p['status'],0)+1
    total = len(people)
    return jsonify({
        "total_people":total,
        "infected":sc.get('infected',0),
        "recovered":sc.get('recovered',0),
        "healthy":sc.get('healthy',0),
        "quarantined":sc.get('quarantined',0),
        "total_contacts":len(contacts),
        "infection_rate":round(sc.get('infected',0)/total*100,1) if total else 0
    })

@app.route("/api/reset", methods=["POST"])
def reset():
    seed_demo_data()
    return jsonify({"message":"Demo data restored"})

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json or {}
    n = min(int(data.get("n",10)),50)
    ids = list(people.keys())
    if len(ids)<2: return jsonify({"error":"Need 2+ people"}), 400
    for _ in range(n):
        a,b = random.sample(ids,2)
        loc = random.choice(LOCATIONS)
        ts = (datetime.utcnow()-timedelta(hours=random.uniform(0,120))).isoformat()
        contacts.append({"person_a":a,"person_b":b,
                         "duration_minutes":random.randint(5,120),
                         "timestamp":ts,"location":loc,
                         "location_risk":LOCATION_RISK.get(loc,"low")})
    return jsonify({"added":n,"total_contacts":len(contacts)})

if __name__ == "__main__":
    print("Training ML model on startup...")
    train_model()
    print("Model ready. Starting Flask...")
    app.run(debug=True, port=5000)
