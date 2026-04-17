"""
app.py — Neuromorphic AI Learning Platform Backend
===================================================
Run:  python app.py
Then open:  http://localhost:5000
"""

import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("model", "neural_classifier.joblib")
METADATA_PATH = os.path.join("model", "class_info.joblib")

try:
    model      = joblib.load(MODEL_PATH)
    class_info = joblib.load(METADATA_PATH)
    print("✅  Model loaded successfully.")
except Exception as e:
    model      = None
    class_info = {}
    print(f"⚠️  Model not found: {e}")
    print("    → Run: python model/train_model.py")

# ── Helper ────────────────────────────────────────────────────────────────────
DARK_BG = "#0D1117"
GRID_C  = "#21262D"

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

# ── Chart Generators ──────────────────────────────────────────────────────────
COLORS = {
    "EXCITATORY":  "#f97316",   # orange
    "INHIBITORY":  "#ef4444",   # red
    "OSCILLATORY": "#664e86",   # neural blue
    "SPARSE":      "#3D0357",   # yellow
}

def generate_spike_train(label):
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    n_neurons = 20
    t_max     = 200
    col       = COLORS.get(label, "#60A5FA")

    for n in range(n_neurons):
        if label == "EXCITATORY":
            spikes = np.where(np.random.rand(t_max) < 0.25)[0]
        elif label == "INHIBITORY":
            spikes = np.where(np.random.rand(t_max) < 0.05)[0]
        elif label == "OSCILLATORY":
            period = np.random.uniform(18, 22)
            base   = np.arange(np.random.randint(0, int(period)), t_max, period).astype(int)
            spikes = np.clip(base + np.random.randint(-2, 3, size=len(base)), 0, t_max - 1)
        else:
            spikes = np.random.choice(t_max, size=np.random.randint(1, 4), replace=False)

        ax.vlines(spikes, n + 0.1, n + 0.9, color=col, linewidth=1.2, alpha=0.85)

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, n_neurons)
    ax.set_xlabel("Time (ms)", color="#8B949E", fontsize=10)
    ax.set_ylabel("Neuron #",  color="#8B949E", fontsize=10)
    ax.set_title(f"Spike Train — {label}", color="#E6EDF3", fontsize=13, pad=12)
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color(GRID_C)
    ax.grid(axis="x", color=GRID_C, linestyle="--", linewidth=0.5)
    return fig_to_base64(fig)


def generate_membrane_potential(label):
    np.random.seed(7)
    t   = np.linspace(0, 100, 1000)
    dt  = t[1] - t[0]
    col = COLORS.get(label, "#60A5FA")
    V   = np.full(len(t), -70.0)
    thr = -55.0

    if label == "EXCITATORY":
        for i in range(1, len(t)):
            V[i] = V[i-1] + 0.18 * dt + 0.3 * np.random.randn()
            if V[i] >= thr:
                V[i] = 40.0
                if i + 10 < len(t):
                    V[i+1:i+10] = np.linspace(40, -75, 9)
    elif label == "INHIBITORY":
        for i in range(1, len(t)):
            V[i] = V[i-1] - 0.05 * (V[i-1] + 70) * dt + 0.15 * np.random.randn()
        V = np.clip(V, -90, -55)
    elif label == "OSCILLATORY":
        V = -70 + 12 * np.sin(2 * np.pi * 0.08 * t) + 1.5 * np.random.randn(len(t))
    else:
        for i in range(1, len(t)):
            V[i] = V[i-1] - 0.03 * (V[i-1] + 70) * dt + 0.1 * np.random.randn()
        for i in np.random.choice(len(t) - 15, size=3, replace=False):
            V[i] = 40.0
            V[i+1:i+10] = np.linspace(40, -75, 9)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.plot(t, V, color=col, linewidth=1.4, alpha=0.9)
    ax.axhline(thr, color="#F43F5E", linewidth=0.8, linestyle="--", label=f"Threshold ({thr} mV)")
    ax.axhline(-70, color="#6B7280", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Time (ms)", color="#8B949E", fontsize=10)
    ax.set_ylabel("Membrane Potential (mV)", color="#8B949E", fontsize=10)
    ax.set_title(f"Membrane Potential — {label}", color="#E6EDF3", fontsize=13, pad=12)
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color(GRID_C)
    ax.grid(color=GRID_C, linestyle="--", linewidth=0.5)
    ax.legend(facecolor=DARK_BG, labelcolor="#8B949E", fontsize=8)
    return fig_to_base64(fig)


def generate_network_activity():
    labels_all = ["EXCITATORY", "INHIBITORY", "OSCILLATORY", "SPARSE"]
    values     = [0.82, 0.54, 0.71, 0.38]
    colors_all = [COLORS[l] for l in labels_all]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bars = ax.bar(labels_all, values, color=colors_all, width=0.5, edgecolor=DARK_BG)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", color="#E6EDF3", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Relative Activation", color="#8B949E", fontsize=10)
    ax.set_title("Network Signal Type Overview", color="#E6EDF3", fontsize=13, pad=12)
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color(GRID_C)
    ax.grid(axis="y", color=GRID_C, linestyle="--", linewidth=0.5)
    return fig_to_base64(fig)


def generate_confidence_chart(proba_dict):
    labels_all = list(proba_dict.keys())
    values     = list(proba_dict.values())
    colors_all = [COLORS.get(l, "#60A5FA") for l in labels_all]
    order      = np.argsort(values)
    labels_all = [labels_all[i] for i in order]
    values     = [values[i]     for i in order]
    colors_all = [colors_all[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bars = ax.barh(labels_all, values, color=colors_all, height=0.5, edgecolor=DARK_BG)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", color="#E6EDF3", fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Confidence", color="#8B949E", fontsize=10)
    ax.set_title("Classifier Confidence Scores", color="#E6EDF3", fontsize=13, pad=12)
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color(GRID_C)
    ax.grid(axis="x", color=GRID_C, linestyle="--", linewidth=0.5)
    return fig_to_base64(fig)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"status": "online", "model_loaded": model is not None})

@app.route("/api/classify", methods=["POST"])
def classify():
    if model is None:
        return jsonify({"error": "Model not loaded. Run: python model/train_model.py"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please provide some text input."}), 400

    label       = model.predict([text])[0]
    proba       = model.predict_proba([text])[0]
    class_names = model.classes_
    proba_dict  = {c: round(float(p), 4) for c, p in zip(class_names, proba)}
    confidence  = round(float(proba.max()) * 100, 1)
    info        = class_info.get(label, {})

    return jsonify({
        "label":         label,
        "confidence":    confidence,
        "emoji":         info.get("emoji", "🧠"),
        "color":         info.get("color", "#60A5FA"),
        "description":   info.get("description", ""),
        "analogy":       info.get("analogy", ""),
        "probabilities": proba_dict,
        "charts": {
            "spike_train": generate_spike_train(label),
            "membrane":    generate_membrane_potential(label),
            "confidence":  generate_confidence_chart(proba_dict),
        },
    })

@app.route("/api/visualize", methods=["GET"])
def visualize():
    return jsonify({"chart": generate_network_activity()})

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)