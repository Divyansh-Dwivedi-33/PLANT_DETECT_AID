# PlantX Detector – Plant Disease Detector

Swin-Transformer leaf-disease classifier with a Flask backend and a green, leaf-themed web UI.

## Project layout
```
plantx/
├── app.py                                   # Flask backend (API + page)
├── plantx_model.py                          # Swin model (shared by notebook and backend)
├── PlantX_Detector_Swin_Transformer.ipynb   # training notebook
├── requirements.txt
├── models/                                  # put trained weights + class list here
├── templates/index.html
└── static/{css,js,img}
```

## Setup (Python 3.8 or 3.9 - required by TensorFlow 2.7.0)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Train
Arrange the dataset as `dataset/train/<class_name>/*.jpg`, open the notebook, set `DATA_ROOT`, and run all cells.
It writes `models/plantx_swin.weights.h5` and `models/plantx_classes.json`.

## 2. Run the app
```bash
python app.py
```
Open http://127.0.0.1:5000

## API
| Route | Method | Description |
|---|---|---|
| `/api/health` | GET | model status + class list |
| `/api/predict` | POST (`multipart/form-data`, field `image`) | label, confidence, per-class probabilities, symptoms, actions |

If the weights file is missing, the page still loads, the status pill shows "Model offline", and `/api/predict` returns a clear 503 message.
