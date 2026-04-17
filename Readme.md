# Neuronix — Neuromorphic Computing Learning Platform

A full-stack web application that combines machine learning with an interactive educational platform on neuromorphic computing. Built with Flask, scikit-learn, and vanilla HTML/CSS/JS.

---

## What It Does

Neuronix lets users explore brain-inspired computing through four sections:

- **Learn** — Visual explanations of SNNs, LIF neurons, STDP, and real neuromorphic hardware
- **Resources** — Curated PDFs and external links for deeper study
- **Demo** — A live ML classifier that detects neural signal patterns from plain text input
- **Visualize** — Real-time matplotlib charts showing network activity

---

## ML Classifier

The core model classifies text into four neural signal categories:

| Label | Description |
|---|---|
| `EXCITATORY` | Dense, rapid burst activity |
| `INHIBITORY` | Suppressed, GABA-driven silencing |
| `OSCILLATORY` | Rhythmic, synchronized wave patterns |
| `SPARSE` | Minimal activation, maximum information |

**Pipeline:** TF-IDF (1–3 ngrams) → Logistic Regression  
**Training data:** 100 labeled examples (25 per class)  
**Evaluation:** ~90%+ test accuracy with 5-fold cross-validation

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn, joblib, numpy |
| Visualization | Matplotlib (server-side, base64 encoded) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Fonts | Playfair Display, Syne, JetBrains Mono |

---

## Project Structure

```
neuronix/
│
├── app.py                  # Flask backend, API routes, chart generators
├── templates/
│   └── index.html          # Single-page frontend (all CSS + JS inline)
├── model/
│   ├── train_model.py      # Training script
│   ├── neural_classifier.joblib
│   └── class_info.joblib
└── static/
    ├── intro_snn.pdf
    ├── stdp_learning.pdf
    ├── Survey_on_Hardware.pdf
    └── my_conference_paper.pdf
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install flask flask-cors scikit-learn numpy matplotlib joblib
```

### 2. Train the model (run once)

```bash
python model/train_model.py
```

### 3. Start the server

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve frontend |
| `GET` | `/api/status` | Check server and model status |
| `POST` | `/api/classify` | Classify text → label, confidence, charts |
| `GET` | `/api/visualize` | Generate network activity bar chart |

### POST `/api/classify`

**Request body:**
```json
{ "text": "GABA release suppresses neural firing significantly" }
```

**Response:**
```json
{
  "label": "INHIBITORY",
  "confidence": 94.2,
  "emoji": "🛑",
  "description": "...",
  "analogy": "...",
  "probabilities": { "EXCITATORY": 0.02, "INHIBITORY": 0.94, ... },
  "charts": {
    "spike_train": "<base64>",
    "membrane": "<base64>",
    "confidence": "<base64>"
  }
}
```

---

## Charts Generated

For each classification, three matplotlib charts are returned as base64 PNGs:

- **Spike Train** — Raster plot of 20 simulated neurons over 200ms
- **Membrane Potential** — LIF-style voltage trace with threshold line
- **Confidence Chart** — Horizontal bar chart of classifier probabilities

---

## Inspiration

This project was inspired by a conference paper on neuromorphic edge computing. The goal was to make a niche but important field accessible — not just as an academic read, but as something you can interact with.

---

## Skills Demonstrated

- End-to-end ML pipeline (data, training, evaluation, deployment)
- Flask REST API design
- Server-side data visualization with Matplotlib
- Frontend UI design without any CSS framework
- Connecting backend inference to a live interactive interface

---

## Author

**Hoor-Ain Rajput**  
GitHub: [hoorainrajput](https://github.com/hoorainrajput)  
LinkedIn: [hoor-ain5](https://linkedin.com/in/hoor-ain5)  
Instagram: [@astral.syntax](https://instagram.com/astral.syntax)

---

*Built as a portfolio project — 2026*