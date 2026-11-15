import os
import io
import json
import logging
import re
from flask import Flask, request, jsonify, render_template

# --------------------------------------------------------------------------
# Optional backends
# --------------------------------------------------------------------------
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np

# --------------------------------------------------------------------------
# Paths & Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_PATH = os.path.join(MODELS_DIR, "plantx_model.onnx")
KERAS_WEIGHTS_PATH = os.path.join(MODELS_DIR, "swin_transformer.weights.h5")
CLASSES_PATH = os.path.join(MODELS_DIR, "plantx_classes.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_UPLOAD_MB = 8
LOW_CONFIDENCE = 0.55

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("plantx")

# --------------------------------------------------------------------------
# Complete Disease Knowledge Base (9 Comprehensive Agricultural Classes)
# --------------------------------------------------------------------------
DISEASE_INFO = {
    "healthy": {
        "title": "Healthy Leaf",
        "status": "healthy",
        "category": "Normal Foliage",
        "severity": "None · Healthy Crop",
        "summary": "Normal vigor, good chlorophyll density, and fully intact cuticular cellular architecture.",
        "symptoms": [
            "Uniform vibrant green coloration",
            "No necrotic spots, veinal chlorosis, or curling",
            "Smooth blade margins with normal turgor pressure"
        ],
        "actions": [
            "Maintain balanced N-P-K fertilization and regular irrigation cycles",
            "Continue weekly field scouting for early pest or blight signs",
            "Practice proactive crop rotation and good soil aeration"
        ],
        "chemical_rx": "No synthetic chemical fungicides or pesticides required. Avoid unnecessary prophylactic sprays to preserve beneficial predatory insects.",
        "organic_rx": "Foliar spray compost tea, seaweed kelp extract @ 2 mL/L, or fermented cow dung slurry every 21 days to sustain immune vigor.",
        "cultural_rx": "Maintain 90 cm x 60 cm row spacing for optimal air circulation, scout fields twice weekly, and keep field borders weed-free.",
        "irrigation_rx": "Adopt alternate-furrow or drip irrigation. Irrigate every 10–14 days based on soil moisture tensiometer reading."
    },
    "bacterial_blight": {
        "title": "Bacterial Blight (Angular Leaf Spot)",
        "status": "disease",
        "category": "Bacterial Infection",
        "severity": "High Risk · Destructive Pathogen",
        "summary": "Caused by Xanthomonas citri pv. malvacearum, producing vein-delimited angular necrotic lesions and blackarm stem cankers.",
        "symptoms": [
            "Angular, water-soaked lesions bounded by veins",
            "Lesions rapidly turning brown, black, and shedding tissue",
            "Stem cankers (blackarm) and boll rot in advanced infection"
        ],
        "actions": [
            "Foliar spray Copper Oxychloride (50 WP) @ 2.5 g/L mixed with Streptocycline @ 0.1 g/L",
            "Plant certified acid-delinted and disease-resistant seeds",
            "Eradicate and safely dispose of infected crop residues",
            "Avoid overhead irrigation to minimize bacterial splash dispersal"
        ],
        "chemical_rx": "Spray Copper Oxychloride 50 WP @ 2.5–3.0 g/L combined with Streptocycline (plant antibiotic) @ 0.1 g/L (1 g in 10 L water). Repeat at 12-day intervals if rainfall persists.",
        "organic_rx": "Foliar spray 5% Neem Seed Kernel Extract (NSKE) or Pseudomonas fluorescens @ 10 g/L early morning. Apply approved copper hydroxide (Bordeaux 1%).",
        "cultural_rx": "Use certified acid-delinted seeds. Rogue out and safely burn heavily blighted plants. Deep plough post-harvest to bury bacterial inoculum.",
        "irrigation_rx": "Strictly avoid overhead sprinkler irrigation to prevent bacterial splash dispersal. Ensure rapid surface drainage after heavy rainfall."
    },
    "curl_virus": {
        "title": "Cotton Leaf Curl Begomovirus (CLCuV)",
        "status": "disease",
        "category": "Viral Pathology",
        "severity": "Critical Risk · Vector-Borne Virus",
        "summary": "A destructive whitefly-transmitted begomovirus leading to architectural leaf deformation, vein enations, and severe stunting.",
        "symptoms": [
            "Upward or downward cupping and curling of leaf margins",
            "Vein thickening, enations (leaf-like outgrowths) on abaxial leaf surface",
            "Stunted plant growth and significant drop in boll production"
        ],
        "actions": [
            "Control whitefly vectors using Diafenthiuron 50% WP (1.2 g/L) or Neem Azadirachtin (1500 ppm)",
            "Install 10-12 yellow sticky traps per acre along field perimeter",
            "Rogue out and destroy viral host weeds and early-infected plants",
            "Cultivate virus-tolerant and early-maturing hybrid cultivars"
        ],
        "chemical_rx": "Vector control: Spray Flonicamid 50 WG @ 0.3 g/L or Diafenthiuron 50 WP @ 1.2 g/L or Pyriproxyfen 10 EC @ 2 mL/L targeting underside of leaves.",
        "organic_rx": "Spray cold-pressed pure Neem Oil (1500 ppm Azadirachtin) @ 5 mL/L with soap solution. Install 15-20 yellow sticky traps/acre along borders.",
        "cultural_rx": "Rogue out early-infected plants within first 45 days after sowing. Eradicate alternate host weeds (Abutilon indicum, Parthenium hysterophorus).",
        "irrigation_rx": "Maintain consistent soil moisture; drought-stressed plants exhibit aggravated viral stunting and attract higher whitefly foraging."
    },
    "fussarium_wilt": {
        "title": "Fusarium Vascular Wilt",
        "status": "disease",
        "category": "Fungal Vascular Disease",
        "severity": "Critical Risk · Soil Vascular Fungus",
        "summary": "Soil-borne pathogen (Fusarium oxysporum f. sp. vasinfectum) clogging water and nutrient xylem vessels, causing irreversible collapse.",
        "symptoms": [
            "Interveinal yellowing, chlorosis, and progressive wilting of lower foliage",
            "Dark brown to black vascular ring discoloration inside split stem",
            "Permanent wilting and sudden collapse under daytime heat"
        ],
        "actions": [
            "Drench root zone with Carbendazim (1 g/L) or Copper Hydroxide",
            "Enrich soil with Trichoderma harzianum bio-agent mixed with farmyard manure",
            "Enforce a minimum 3-year crop rotation with non-host crops (cereals/sorghum)",
            "Improve field drainage and avoid root injuries during inter-cultivation"
        ],
        "chemical_rx": "Drench root zone with Carbendazim 50 WP @ 1.0 g/L or Copper Hydroxide 53.8 DF @ 2.0 g/L around affected plant basins.",
        "organic_rx": "Soil application of Trichoderma viride or T. harzianum (2 kg/acre) pre-incubated in 100 kg enriched farmyard manure. Drench with fermented jeevamrut.",
        "cultural_rx": "Enforce strict 3-year crop rotation with cereals (sorghum, pearl millet, maize). Deep summer ploughing and solarization of infected field blocks.",
        "irrigation_rx": "Avoid standing water in field furrows; waterlogged conditions accelerate root asphyxiation and rapid fungal xylem penetration."
    },
    "verticillium_wilt": {
        "title": "Verticillium Wilt",
        "status": "disease",
        "category": "Fungal Vascular Disease",
        "severity": "High Risk · Soil Vascular Fungus",
        "summary": "Soil-borne fungus (Verticillium dahliae) causing characteristic 'tiger-stripe' foliar chlorosis, internal vascular staining, and defoliation.",
        "symptoms": [
            "Mottled interveinal yellowing and browning resembling tiger stripes",
            "Olive-brown vascular staining visible in longitudinal stem cuts",
            "Premature defoliation advancing from lower to upper canopy"
        ],
        "actions": [
            "Avoid over-irrigation during cool spells which favor fungal growth",
            "Apply balanced Potassium (muriate of potash) to strengthen vascular cell walls",
            "Practice deep summer ploughing and solarization of infested fields",
            "Rotate with resistant crops like maize, sorghum, or mustard"
        ],
        "chemical_rx": "Soil drenching with Thiophanate-Methyl 70 WP @ 1.0 g/L or Benomyl 50 WP @ 1.0 g/L around root zones showing early chlorotic striping.",
        "organic_rx": "Practice bio-fumigation by incorporating chopped mustard or brassica crop residues into soil 4 weeks prior to cotton planting.",
        "cultural_rx": "Rotate with corn, sorghum, or alfalfa. Avoid moving contaminated tractor tillage equipment from infested to clean fields.",
        "irrigation_rx": "Avoid irrigation when ambient soil temperatures drop below 24°C, as cool damp conditions dramatically accelerate Verticillium proliferation."
    },
    "alternaria_leaf_spot": {
        "title": "Alternaria Leaf Spot",
        "status": "disease",
        "category": "Foliar Fungal Infection",
        "severity": "Moderate Risk · Foliar Fungal",
        "summary": "Fungal pathogen (Alternaria macrospora / alternata) producing circular target-board concentric spots on foliage with shot-hole centers.",
        "symptoms": [
            "Brown concentric rings creating a distinct target-board pattern",
            "Necrotic centres cracking and dropping out ('shot-hole' effect)",
            "Premature leaf drop during warm, humid rainfall periods"
        ],
        "actions": [
            "Spray Mancozeb 75 WP @ 2.5 g/L or Azoxystrobin @ 1 mL/L upon first sighting",
            "Collect and burn fallen infected leaf litter to reduce spore inoculum",
            "Ensure adequate crop spacing to maintain air flow within the canopy"
        ],
        "chemical_rx": "Foliar spray Mancozeb 75 WP @ 2.5 g/L or Azoxystrobin 23 SC @ 1.0 mL/L or Propiconazole 25 EC @ 1.0 mL/L at first appearance of target spots.",
        "organic_rx": "Spray 1% Bordeaux Mixture or Bacillus subtilis bio-fungicide @ 5 g/L. Spray in early mornings to allow optimal bio-film formation.",
        "cultural_rx": "Maintain proper crop spacing to promote air circulation. Collect and bury infected fallen leaf litter to eliminate spore banks.",
        "irrigation_rx": "Irrigate during early morning hours so foliage dries quickly before nightfall. Avoid overhead sprinkler systems."
    },
    "leaf_reddening": {
        "title": "Leaf Reddening / Anthocyanosis",
        "status": "stress",
        "category": "Physiological & Nutrient Stress",
        "severity": "Moderate Risk · Abiotic Nutrient Stress",
        "summary": "Abiotic disorder triggered by sudden cold night temperatures, magnesium deficiency, or waterlogged root hypoxia, inducing anthocyanin pigments.",
        "symptoms": [
            "Reddish-purple anthocyanin discoloration starting on leaf margins and spreading between veins",
            "Foliage becomes brittle and leathery",
            "Reduced boll weight, lower lint quality, and premature senescence"
        ],
        "actions": [
            "Foliar spray Magnesium Sulfate (MgSO4) @ 10 g/L combined with Urea @ 10 g/L",
            "Improve field drainage immediately to eliminate root zone hypoxia",
            "Apply foliar micronutrient cocktail during peak squaring and boll formation"
        ],
        "chemical_rx": "Foliar spray 1% Magnesium Sulfate (MgSO4 @ 10 g/L) + 1% Urea (10 g/L) or 2% 19:19:19 water-soluble NPK. Repeat after 10 days.",
        "organic_rx": "Apply liquid humic acid (12%) @ 3 mL/L + fermented seaweed kelp extract to stimulate root nutrient transport and chlorophyll synthesis.",
        "cultural_rx": "Test soil pH (acidic soils induce Magnesium fixation). Apply dolomitic limestone if soil pH is below 6.0.",
        "irrigation_rx": "Aerate waterlogged soil immediately. Cold, water-saturated root zones inhibit magnesium cation uptake."
    },
    "jassid_hopper_damage": {
        "title": "Leaf Hopper / Jassid Pest Damage (Hopperburn)",
        "status": "disease",
        "category": "Insect Pest Infestation",
        "severity": "Moderate to High · Sucking Insect Pest",
        "summary": "Sucking insect pest (Amrasca biguttula biguttula) sucking plant sap and injecting toxic salivary secretions, causing leaf margin hopperburn.",
        "symptoms": [
            "Marginal yellowing followed by triangular browning and scorched edges (hopperburn)",
            "Downward curling and crinkling of leaf blades",
            "Stunted terminal growth with leathery leaves"
        ],
        "actions": [
            "Foliar spray Flonicamid 50 WG @ 0.3 g/L or Imidacloprid 17.8 SL @ 0.3 mL/L",
            "Conserve natural predators including chrysopids, ladybird beetles, and spiders",
            "Install yellow sticky traps (15-20/acre) to monitor and suppress adult hoppers"
        ],
        "chemical_rx": "Spray Flonicamid 50 WG @ 0.3 g/L or Thiamethoxam 25 WG @ 0.25 g/L or Imidacloprid 17.8 SL @ 0.3 mL/L targeting lower canopy leaf surfaces.",
        "organic_rx": "Spray 5% NSKE (Neem Seed Kernel Extract) or Verticillium lecanii entomopathogenic bio-fungus @ 5 g/L. Conserve green lacewings and coccinellids.",
        "cultural_rx": "Sow hairy-leaf tolerant cotton hybrids (pubescent leaf trichomes deter hopper nymph feeding and egg oviposition).",
        "irrigation_rx": "Avoid excessive nitrogen application which produces tender succulent vegetative shoots highly vulnerable to hopper infestation."
    },
    "herbicide_damage": {
        "title": "Herbicide Injury / Chemical Phytotoxicity",
        "status": "stress",
        "category": "Chemical Phytotoxicity",
        "severity": "Variable Risk · Auxin Chemical Phytotoxicity",
        "summary": "Non-parasitic injury caused by synthetic auxin or non-selective herbicide spray drift (e.g. 2,4-D, Dicamba, Glyphosate), causing leaf strappiness and epinasty.",
        "symptoms": [
            "Fan-shaped, strapped, or shoestring leaf distortion with parallel veins",
            "Epinastic bending, twisted petioles, and stem curvature",
            "Chlorotic bleached spots without bacterial ooze or fungal mycelia"
        ],
        "actions": [
            "Provide immediate irrigation to dilute active herbicide residues in the rhizosphere",
            "Apply foliar biostimulants (seaweed extract or humic acid @ 2 mL/L) to aid physiological recovery",
            "Use spray hoods, proper nozzle calibration, and avoid spraying during windy periods (> 5 km/h)"
        ],
        "chemical_rx": "Foliar spray Potassium Nitrate (KNO3 @ 10 g/L) + Gibberellic Acid (GA3 @ 10 ppm) to stimulate terminal elongation and counteract auxin twisting.",
        "organic_rx": "Apply liquid humic acid @ 5 mL/L + amino acid biostimulants (2 mL/L) to stimulate endogenous cytochrome P450 detoxifying enzymes.",
        "cultural_rx": "Use dedicated herbicide sprayers with drift guard nozzles. Never spray non-selective herbicides when wind speeds exceed 5 km/h.",
        "irrigation_rx": "Provide an immediate deep flush irrigation to leach soil-mobile herbicide chemistries below the cotton root absorption zone."
    },
}

# --------------------------------------------------------------------------
# Model state & loader
# --------------------------------------------------------------------------
backend_type = None
model = None
onnx_input_name = None
class_names = list(DISEASE_INFO.keys())
model_error = None
model_metrics = None


def load_model():
    """Load model once at start-up (prefers ONNX, falls back to Keras)."""
    global backend_type, model, onnx_input_name, class_names, model_error, model_metrics

    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as fh:
                class_names = json.load(fh)
        except Exception as e:
            log.warning("Could not read classes JSON: %s", e)

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as fh:
                model_metrics = json.load(fh)
        except Exception:
            pass

    if os.path.exists(ONNX_PATH) and HAS_ORT:
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])
            model = sess
            onnx_input_name = sess.get_inputs()[0].name
            backend_type = "onnx"
            model_error = None
            log.info("Loaded ONNX deep model (%d classes: %s)", len(class_names), class_names)
            return
        except Exception as exc:
            log.warning("Could not load ONNX model: %s", exc)

    if os.path.exists(KERAS_WEIGHTS_PATH):
        try:
            import plantx_model as pm
            net = pm.build_plantx_model(len(class_names))
            net.load_weights(KERAS_WEIGHTS_PATH)
            model = net
            backend_type = "keras"
            model_error = None
            log.info("Loaded Keras Swin model (%d classes: %s)", len(class_names), class_names)
            return
        except Exception as exc:
            model = None
            backend_type = None
            model_error = str(exc)
            log.error("Failed to load Keras model: %s", exc)
            return

    model = None
    backend_type = None
    model_error = "Model weights not found. Please ensure plantx_model.onnx is present in models/."
    log.warning(model_error)


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(image_bytes):
    """Normalize leaf image to 224x224 RGB float32 for model inference."""
    if HAS_PIL:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    elif HAS_CV2:
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        arr = rgb.astype(np.float32) / 255.0
    else:
        raise RuntimeError("Neither Pillow nor OpenCV is installed.")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    ch_first = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(ch_first, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# AI Agronomist Consultation Engine ("Ask In Chat")
# --------------------------------------------------------------------------
def generate_agronomist_reply(message, context=None):
    msg = message.lower().strip()
    detected_label = context.get("label") if context else None
    detected_title = context.get("title") if context else None
    confidence = context.get("confidence") if context else None

    # Retrieve info if context label is recognized
    info = DISEASE_INFO.get(detected_label) if detected_label else None

    # 1. Greetings
    if re.search(r"\b(hi|hello|hey|namaste|greetings)\b", msg) and len(msg.split()) <= 3:
        if detected_label and detected_label != "healthy":
            return (
                f"Hello! I am your **PlantX AI Agronomist**. I see your recent leaf scan identified "
                f"**{detected_title}** ({confidence}% confidence). You can ask me about organic remedies, "
                f"chemical dosages, irrigation adjustments, or how to stop it from spreading."
            )
        return (
            "Hello! I am your **PlantX AI Agronomist**. I can provide immediate agronomic advice on crop diseases, "
            "chemical fungicides/pesticide dosages, organic bio-management, or irrigation schedules. "
            "How can I help your field today?"
        )

    # 2. Organic / Neem / Bio-control queries
    if any(k in msg for k in ["organic", "neem", "biological", "bio", "home remedy", "natural"]):
        if detected_label and info:
            organic_advice = info.get("organic_rx", "")
            return (
                f"**Organic & Biological Management for {detected_title}:**\n\n"
                f"{organic_advice}\n\n"
                f"**Key Cultural Actions:**\n"
                + "\n".join(f"- {act}" for act in info.get("actions", [])[:2])
            )
        return (
            "**General Organic Crop Health Practice:**\n"
            "1. Apply vermicompost or composted cow manure (2-3 tons/acre) enriched with *Trichoderma viride*.\n"
            "2. Foliar spray 5% Neem Seed Kernel Extract (NSKE) or 1500 ppm Azadirachtin @ 5 mL/L.\n"
            "3. Install yellow and blue sticky traps (10-15 traps/acre) to control sucking vector pests.\n"
            "4. Spray *Pseudomonas fluorescens* (10 g/L) to boost natural systemic resistance."
        )

    # 3. Chemical / Dosage / Fungicide / Pesticide queries
    if any(k in msg for k in ["chemical", "dose", "dosage", "spray", "fungicide", "pesticide", "medicine", "cure"]):
        if detected_label and info:
            chem_rx = info.get("chemical_rx", "")
            return (
                f"**Recommended Chemical Prescription for {detected_title}:**\n\n"
                f"{chem_rx}\n\n"
                f"⚠️ *Safety Advisory:* Wear protective equipment (gloves, mask) and maintain a minimum 14-day pre-harvest interval (PHI)."
            )
        return (
            "**Agricultural Spray Guidance:**\n"
            "For fungal infections: Mancozeb 75 WP @ 2.5 g/L or Azoxystrobin @ 1 mL/L.\n"
            "For bacterial blights: Copper Oxychloride 50 WP @ 2.5 g/L + Streptocycline @ 0.1 g/L.\n"
            "For sucking pests: Flonicamid 50 WG @ 0.3 g/L or Diafenthiuron 50 WP @ 1.2 g/L.\n"
            "Please specify your crop disease or scan a leaf above for exact dosage calibration."
        )

    # 4. Spreading / Contagion / Danger queries
    if any(k in msg for k in ["spread", "contagious", "danger", "risk", "neighbor", "infect"]):
        if detected_label == "curl_virus":
            return (
                "**High Vector Contagion Alert:** Cotton Leaf Curl Virus is transmitted rapidly from plant to plant by **whiteflies (*Bemisia tabaci*)**. "
                "Immediate vector control is critical. Install yellow sticky traps along the boundary and spray Flonicamid 50 WG or Diafenthiuron. "
                "Rogue out and burn infected plants within 45 days of sowing."
            )
        elif detected_label in ["fussarium_wilt", "verticillium_wilt"]:
            return (
                f"**Soil-Borne Vascular Risk:** {detected_title} spreads through soil water, infected root contact, and contaminated tillage tools. "
                "Isolate the affected field zone, stop surface furrow runoff to neighboring plots, and avoid moving tractors through wet soil."
            )
        elif detected_label == "bacterial_blight":
            return (
                "**Rain-Splash Spreading Risk:** Bacterial Blight spreads swiftly via wind-driven rain droplets, dew, and sprinkler irrigation. "
                "Avoid overhead irrigation, do not work in wet fields, and apply a prophylactic copper oxychloride barrier spray on neighboring rows."
            )
        elif detected_label == "herbicide_damage":
            return (
                "**Non-Contagious Condition:** Herbicide drift is abiotic (chemical injury). It does **not** spread biologically to unaffected plants. "
                "Flush the root zone with fresh water and apply foliar humic acid/biostimulants to promote vegetative recovery."
            )
        elif detected_label == "healthy":
            return "Your scanned leaf shows healthy characteristics with intact cellular structure. No infection risk is present."
        return (
            "Crop diseases spread primarily through wind-borne spores, splashing rain, contaminated soil, or insect vectors (whiteflies, aphids, thrips). "
            "Maintain wide plant spacing, sanitize pruning tools with 70% alcohol, and scout downwind rows."
        )

    # 5. Irrigation / Water queries
    if any(k in msg for k in ["water", "irrigation", "soil", "drip", "flood", "rain"]):
        if detected_label and info:
            return (
                f"**Irrigation & Soil Protocol for {detected_title}:**\n\n"
                f"{info.get('irrigation_rx', 'Maintain well-drained soil moisture.')}\n\n"
                f"- **Cultural step:** {info.get('cultural_rx', 'Avoid water stagnation.')}"
            )
        return (
            "**Cotton Irrigation Management:**\n"
            "- Cotton requires 650–800 mm of water throughout its growing cycle.\n"
            "- Critical stages: Flowering (squaring) and boll development.\n"
            "- Avoid waterlogging at all times: stagnant water for >24 hours triggers root hypoxia, leaf reddening, and flower drop."
        )

    # 6. Specific disease direct questions
    for key, d_info in DISEASE_INFO.items():
        if key in msg or d_info["title"].lower() in msg:
            return (
                f"**Field Profile: {d_info['title']} ({d_info['category']})**\n\n"
                f"**Summary:** {d_info['summary']}\n\n"
                f"**Symptoms:**\n"
                + "\n".join(f"- {s}" for s in d_info["symptoms"]) + "\n\n"
                f"**Prescription:**\n"
                f"- *Chemical:* {d_info['chemical_rx']}\n"
                f"- *Organic:* {d_info['organic_rx']}"
            )

    # 7. Fallback contextual guidance
    if detected_label and info:
        return (
            f"Regarding **{detected_title}** ({info['category']}):\n\n"
            f"- **Summary:** {info['summary']}\n"
            f"- **Chemical Rx:** {info['chemical_rx']}\n"
            f"- **Organic Bio-Control:** {info['organic_rx']}\n"
            f"- **Irrigation:** {info['irrigation_rx']}\n\n"
            "Feel free to ask for specific dilution rates, sprayer calibration, or tank mixing compatibility."
        )

    return (
        "I am ready to help. You can ask me:\n"
        "• *'What is the chemical spray dosage for Bacterial Blight?'*\n"
        "• *'How do I make organic neem extract (NSKE 5%)?'*\n"
        "• *'How does Leaf Curl Begomovirus spread?'*\n"
        "• *'What causes leaf reddening in cotton?'*\n"
        "Or upload a leaf image above to get a customized agronomist prescription!"
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        diseases=DISEASE_INFO,
        metrics=model_metrics,
        backend=backend_type or "offline"
    )


@app.route("/api/health")
def health():
    return jsonify(
        model_loaded=model is not None,
        backend=backend_type,
        classes=class_names,
        total_classes=len(class_names),
        metrics=model_metrics,
        error=model_error,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error=model_error or "Model is currently offline."), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose a leaf image to analyse."), 400
    if not allowed(file.filename):
        return jsonify(error="Unsupported file format. Please upload JPG, PNG, BMP, or WEBP."), 400

    try:
        content = file.read()
        batch = preprocess_image(content)

        if backend_type == "onnx":
            outputs = model.run(None, {onnx_input_name: batch})
            probs = outputs[0][0]
        elif backend_type == "keras":
            import plantx_model as pm
            if HAS_CV2:
                raw = np.frombuffer(content, dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                k_batch = np.expand_dims(pm.prepare_image(bgr), axis=0)
                probs = model.predict(k_batch, verbose=0)[0]
            else:
                raise RuntimeError("Keras backend requires OpenCV.")
        else:
            return jsonify(error="Unsupported model backend."), 500

    except Exception as exc:
        log.exception("Inference error: %s", exc)
        return jsonify(error=f"Image processing error: {exc}"), 400

    top_idx = int(np.argmax(probs))
    label = class_names[top_idx]
    confidence = float(probs[top_idx])

    info = DISEASE_INFO.get(label, {
        "title": label.replace("_", " ").title(),
        "status": "disease" if label != "healthy" else "healthy",
        "category": "Field Condition",
        "severity": "Moderate",
        "summary": "Cotton leaf condition detected.",
        "symptoms": [],
        "actions": [],
        "chemical_rx": "Consult your local agricultural extension service.",
        "organic_rx": "Spray 5% Neem Seed Kernel Extract (NSKE).",
        "cultural_rx": "Practice crop rotation and field sanitation.",
        "irrigation_rx": "Maintain appropriate soil moisture."
    })

    probabilities = [
        {
            "label": name,
            "title": DISEASE_INFO.get(name, {}).get("title", name.replace("_", " ").title()),
            "value": round(float(p) * 100, 2),
        }
        for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1])
    ]

    return jsonify(
        label=label,
        title=info["title"],
        status=info["status"],
        category=info.get("category", "Field Condition"),
        severity=info.get("severity", "Moderate"),
        confidence=round(confidence * 100, 2),
        low_confidence=confidence < LOW_CONFIDENCE,
        summary=info["summary"],
        symptoms=info["symptoms"],
        actions=info["actions"],
        chemical_rx=info.get("chemical_rx", ""),
        organic_rx=info.get("organic_rx", ""),
        cultural_rx=info.get("cultural_rx", ""),
        irrigation_rx=info.get("irrigation_rx", ""),
        probabilities=probabilities,
        backend=backend_type,
        can_chat=True
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """AI Agronomist consultation chat endpoint."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify(error="Message cannot be empty."), 400

    context = data.get("context", {})
    reply = generate_agronomist_reply(message, context=context)
    return jsonify(reply=reply)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image exceeds maximum allowed size ({MAX_UPLOAD_MB} MB)."), 413


load_model()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
