"""
PlantX Detector - Plant Disease Detector (Flask backend)

Run:
    python app.py
Then open http://127.0.0.1:5000

Expected files (produced by the training notebook):
    models/plantx_swin.weights.h5   - trained weights
    models/plantx_classes.json      - ordered list of class names
"""
import json
import logging
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import plantx_model as pm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "plantx_swin.weights.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "plantx_classes.json")

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_UPLOAD_MB = 8
LOW_CONFIDENCE = 0.60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("plantx")

# --------------------------------------------------------------------------
# Disease knowledge base (cotton)
# --------------------------------------------------------------------------
DISEASE_INFO = {
    "healthy": {
        "title": "Healthy Leaf",
        "status": "healthy",
        "summary": "No visible signs of disease. The leaf looks vigorous and normal.",
        "symptoms": ["Uniform green colour", "No spots, curling or wilting"],
        "actions": [
            "Keep up regular watering and balanced fertilisation",
            "Continue weekly scouting for early symptoms",
            "Maintain good field hygiene and crop rotation",
        ],
    },
    "bacterial_blight": {
        "title": "Bacterial Blight",
        "status": "disease",
        "summary": "A bacterial infection (Xanthomonas) that damages leaves, stems and bolls.",
        "symptoms": [
            "Angular, water-soaked spots bounded by leaf veins",
            "Spots turn brown/black and leaves may tear",
            "Dark lesions on stems and bolls in severe cases",
        ],
        "actions": [
            "Use certified, disease-free or acid-delinted seed",
            "Remove and destroy infected plant debris",
            "Apply copper-based bactericide as advised locally",
            "Avoid overhead irrigation and working in wet fields",
        ],
    },
    "curl_virus": {
        "title": "Cotton Leaf Curl Virus",
        "status": "disease",
        "summary": "A whitefly-transmitted virus that deforms leaves and stunts the plant.",
        "symptoms": [
            "Upward or downward curling of leaf edges",
            "Thickened, dark green veins and small leaf-like outgrowths",
            "Stunted growth and reduced boll formation",
        ],
        "actions": [
            "Control whiteflies (neem oil, yellow sticky traps, recommended insecticides)",
            "Rogue out and destroy infected plants early",
            "Plant tolerant varieties and remove weed hosts",
            "Avoid staggered sowing near older infected crops",
        ],
    },
    "fussarium_wilt": {
        "title": "Fusarium Wilt",
        "status": "disease",
        "summary": "A soil-borne fungus (Fusarium oxysporum) that blocks water transport.",
        "symptoms": [
            "Yellowing and drooping of lower leaves",
            "Browning of vascular tissue inside the stem",
            "Whole-plant wilting and death",
        ],
        "actions": [
            "Rotate with non-host crops for 3+ years",
            "Plant wilt-resistant cultivars",
            "Improve drainage and avoid root injury",
            "Use soil solarisation or recommended bio-control agents (Trichoderma)",
        ],
    },
}

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
model = None
class_names = list(pm.DEFAULT_CLASSES)
model_error = None


def load_model():
    """Load class list + weights once at start-up."""
    global model, class_names, model_error
    try:
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r", encoding="utf-8") as fh:
                class_names = json.load(fh)
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                "Trained weights not found at 'models/plantx_swin.weights.h5'. "
                "Run the training notebook first and copy the file there."
            )
        net = pm.build_plantx_model(len(class_names))
        net.load_weights(WEIGHTS_PATH)
        # warm-up call so the first real request is fast
        net.predict(np.zeros((1,) + pm.INPUT_SHAPE, dtype="float32"), verbose=0)
        model = net
        model_error = None
        log.info("Model loaded (%d classes: %s)", len(class_names), class_names)
    except Exception as exc:  # noqa: BLE001
        model = None
        model_error = str(exc)
        log.error("Model could not be loaded: %s", exc)


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", diseases=DISEASE_INFO)


@app.route("/api/health")
def health():
    return jsonify(
        model_loaded=model is not None,
        classes=class_names,
        error=model_error,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error=model_error or "Model is not loaded."), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose a leaf image to analyse."), 400
    if not allowed(file.filename):
        return jsonify(error="Unsupported file type. Use JPG, PNG, BMP or WEBP."), 400

    raw = np.frombuffer(file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="That file could not be read as an image."), 400

    batch = np.expand_dims(pm.prepare_image(bgr), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    top = int(np.argmax(probs))
    label = class_names[top]
    confidence = float(probs[top])
    info = DISEASE_INFO.get(label, {
        "title": label.replace("_", " ").title(),
        "status": "disease",
        "summary": "", "symptoms": [], "actions": [],
    })

    return jsonify(
        label=label,
        title=info["title"],
        status=info["status"],
        confidence=round(confidence * 100, 2),
        low_confidence=confidence < LOW_CONFIDENCE,
        summary=info["summary"],
        symptoms=info["symptoms"],
        actions=info["actions"],
        probabilities=[
            {
                "label": name,
                "title": DISEASE_INFO.get(name, {}).get("title", name),
                "value": round(float(p) * 100, 2),
            }
            for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1])
        ],
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is too large (max {MAX_UPLOAD_MB} MB)."), 413


load_model()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
