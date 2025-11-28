from flask import Flask, request, jsonify, send_file, render_template
import csv
import uuid
import threading
import json
from datetime import datetime
from pathlib import Path
import tempfile
import glob

app = Flask(__name__)

# ---- Demo catalogue for picking objects ----
PRODUCTS = [
    {"product_name": "green_cube", "product_id": 0},
    {"product_name": "red_cube", "product_id": 1},
]

# ---- Action catalogue for manipulator ----
# You can extend targets or constraints here.
ACTION_CATALOG = [
    {"action_type": "pick", "label": "Взять кубик", "needs_target": False},
    {"action_type": "place", "label": "Положить", "needs_target": True},
    {"action_type": "throw", "label": "Бросить", "needs_target": True},
    {"action_type": "move_to_point", "label": "Перейти в точку", "needs_target": True},
    {"action_type": "move_to_cube", "label": "Подъехать к кубику", "needs_target": False},
]

cart = []
action_queue = []
queue_lock = threading.Lock()
QUEUE_LOG = Path(__file__).parent / "pending_actions.json"


def load_queue():
    if QUEUE_LOG.exists():
        try:
            data = json.loads(QUEUE_LOG.read_text())
            return data.get("queue", [])
        except Exception:
            return []
    return []


def persist_queue():
    data = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "queue": action_queue,
    }
    with QUEUE_LOG.open("w") as f:
        json.dump(data, f, indent=2)


@app.route('/')
def home():
    return render_template('index.html')


# -------- Catalogues --------
@app.route('/products', methods=['GET'])
def get_products():
    return jsonify(PRODUCTS)


@app.route('/actions/catalog', methods=['GET'])
def get_actions_catalog():
    return jsonify(ACTION_CATALOG)


# -------- Cart workflow (kept for CSV order) --------
@app.route('/cart/add', methods=['POST'])
def add_to_cart():
    data = request.json
    product_id = data.get('product_id')
    product = next((p for p in PRODUCTS if p['product_id'] == product_id), None)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    cart.append(product)
    return jsonify({"message": "Product added"}), 201


@app.route('/order', methods=['POST'])
def order():
    if not cart:
        return jsonify({"error": "Cart is empty"}), 400

    filename = f"order_{uuid.uuid4()}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['product_name', 'product_id'])
        for item in cart:
            writer.writerow([item['product_name'], item['product_id']])

    cart.clear()
    return send_file(filename, as_attachment=True)


# -------- Manipulator actions API --------
def _validate_action(payload):
    action_type = payload.get("action_type")
    target = payload.get("target")
    catalog_item = next((a for a in ACTION_CATALOG if a["action_type"] == action_type), None)
    if not catalog_item:
        return False, "Unsupported action_type"
    if catalog_item["needs_target"]:
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            return False, "target must be [x, y, z] for this action"
        try:
            target = [float(x) for x in target]
        except (TypeError, ValueError):
            return False, "target must contain numeric values"
    return True, None


@app.route('/actions', methods=['POST'])
def enqueue_action():
    payload = request.get_json(force=True)
    ok, msg = _validate_action(payload)
    if not ok:
        return jsonify({"error": msg}), 400
    action = {
        "action_type": payload["action_type"],
        "target": payload.get("target"),
        "requested_at": datetime.utcnow().isoformat() + "Z",
        "status": "queued",
    }
    with queue_lock:
        action_queue.append(action)
        persist_queue()
    return jsonify({"message": "queued", "queue_length": len(action_queue)})


@app.route('/actions/queue', methods=['GET'])
def list_queue():
    with queue_lock:
        return jsonify(action_queue)


@app.route('/actions/next', methods=['POST'])
def pop_next_action():
    with queue_lock:
        if not action_queue:
            return jsonify({"message": "empty"}), 204
        action = action_queue.pop(0)
        action["status"] = "dispatched"
        persist_queue()
    return jsonify(action)


@app.route('/actions/clear', methods=['POST'])
def clear_queue():
    with queue_lock:
        action_queue.clear()
        persist_queue()
    return jsonify({"message": "cleared"})


# -------- Scene export (placements -> CSV) --------
@app.route('/scene/export', methods=['POST'])
def export_scene():
    payload = request.get_json(force=True)
    placements = payload.get("placements", [])
    if not isinstance(placements, list) or not placements:
        return jsonify({"error": "placements must be a non-empty list"}), 400

    # validate and map product ids to names
    product_map = {p["product_id"]: p["product_name"] for p in PRODUCTS}
    rows = []
    for p in placements:
        pid = p.get("product_id")
        pos = p.get("position")
        if pid not in product_map:
            return jsonify({"error": f"unknown product_id {pid}"}), 400
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            return jsonify({"error": f"position must be [x,y,z] for product_id {pid}"}), 400
        try:
            pos = [float(x) for x in pos]
        except (TypeError, ValueError):
            return jsonify({"error": f"position must be numeric for product_id {pid}"}), 400
        rows.append((product_map[pid], pid, *pos))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="scene_", dir=".")
    with open(tmp.name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["product_name", "product_id", "x", "y", "z"])
        writer.writerows(rows)

    return send_file(tmp.name, as_attachment=True, download_name="scene_placements.csv")


@app.route('/media/latest', methods=['GET'])
def latest_media():
    static_dir = Path(__file__).parent / "static"
    candidates = sorted(static_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return jsonify({"message": "no media"}), 204
    latest_path = candidates[0]
    return jsonify({
        "url": f"/static/{latest_path.name}",
        "updated_at": latest_path.stat().st_mtime,
    })


if __name__ == '__main__':
    # Load queue from disk once on startup
    with queue_lock:
        action_queue.extend(load_queue())
    app.run(debug=False, use_reloader=False)
