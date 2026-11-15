# PlantX Detector – Plant Disease Detector

AI-powered plant leaf disease classifier with high-speed ONNX inference, a Flask backend, and a modern leaf-themed web dashboard. Deployed and hosted serverlessly on **Vercel**.

🌐 **Live Production App:** [https://plantx-vercel.vercel.app](https://plantx-vercel.vercel.app)

---

## Performance & Metrics

Trained on 1,826 authentic field leaf images from agricultural benchmarks across 4 key cotton classes:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| **Bacterial Blight** | 99% | 96% | 0.97 |
| **Cotton Leaf Curl Virus (CLCuV)** | 99% | 98% | 0.98 |
| **Fusarium Wilt** | 97% | 100% | 0.99 |
| **Healthy Leaf** | 95% | 98% | 0.97 |
| **Overall Validation Accuracy** | — | — | **97.53%** |

* **Model format:** ONNX (`models/plantx_model.onnx`, 4.12 MB)
* **Inference latency:** < 50ms on CPU
* **Serverless footprint:** Fully compliant with Vercel's 250 MB bundle limit

---

## Project Structure

```
plantx/
├── api/
│   └── index.py                             # Vercel Serverless Function entry point
├── app.py                                   # Flask backend & REST API
├── train_deep.py                            # End-to-end deep training pipeline
├── vercel.json                              # Vercel serverless routing configuration
├── requirements.txt                         # Lightweight production dependencies
├── models/
│   ├── plantx_model.onnx                    # Trained deep vision model (4.12 MB)
│   ├── plantx_classes.json                  # Class mappings
│   └── metrics.json                         # Validation report & metrics
├── templates/
│   └── index.html                           # Responsive diagnostic UI
└── static/
    ├── css/style.css
    ├── js/app.js                            # UI state & API interactions
    └── img/*.svg                            # Vector plant icons
```

---

## Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web server
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Re-training the Model

To train the deep vision model on fresh data:
```bash
python train_deep.py
```
This downloads benchmark agricultural datasets, trains with data augmentation, evaluates validation metrics, and automatically exports `models/plantx_model.onnx` and `models/metrics.json`.

---

## Vercel Deployment

Deploy directly using the Vercel CLI:
```bash
vercel deploy --prod
```
