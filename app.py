from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io, base64, os
import cv2

# ============================================================
#                   FLASK CONFIG
# ============================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ============================================================
#                   LOAD MODEL (SAFE)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "veg_fruit_classify.keras")

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Model load failed: {e}")

IMG_HEIGHT = 180
IMG_WIDTH = 180

# ============================================================
#                MODEL LABELS
# ============================================================
data_cat = [
 'apple','avocado','banana','barbados cherry','beetroot','bell pepper',
 'berries','blackberry','brocolli','cabbage','cantaloupe','capsicum',
 'carrot','cauliflower','cherry','chilli pepper','corn','courgette',
 'cucumber','dates','dragon fruit','eggplant','fig','garlic','ginger',
 'grapes','jalepeno','kiwi','lemon','lettuce','lychee','mango','nectarine',
 'olive','onion','orange','paprika','passion','pawpaw','peach','pear','peas',
 'pepino','pineapple','plum','pomegranate','potato','pumpkin','raddish',
 'soy beans','spinach','strawberry','sugar apple','sweetcorn','sweetpotato',
 'tangarine','tomato','turnip','watermelon'
]

# ============================================================
#           IMAGE PREPROCESSING
# ============================================================
def preprocess_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = ImageOps.autocontrast(img)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# ============================================================
#     IMAGE QUALITY + SPOILAGE ANALYSIS
# ============================================================
def analyze_quality_and_spoilage(pil_image):
    np_img = np.array(pil_image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    sharpness = min(100.0, (cv2.Laplacian(gray, cv2.CV_64F).var() / 300) * 100)
    brightness = (gray.mean() / 255) * 100
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).mean() * 100

    quality = max(0, min(100,
        0.6 * sharpness +
        0.2 * (100 - abs(brightness - 55)) +
        0.2 * (100 - abs(edge_density - 40))
    ))

    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    s = hsv[:,:,1]

    spoilage = ((v < 60).mean() * 0.6 + ((v < 110) & (s < 50)).mean() * 0.4) * 100

    status = "Fresh" if spoilage < 15 else "Slightly Aged" if spoilage < 35 else "Possibly Spoiled"

    return {
        "quality": round(quality, 1),
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "edge_density": round(edge_density, 1),
        "spoilage_score": round(spoilage, 1),
        "status": status
    }

# ============================================================
#                    ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    if not files:
        return "No images uploaded", 400

    results = []

    for file in files:
        try:
            img_bytes = file.read()
            if not img_bytes:
                raise ValueError("Empty file")

            arr, pil_img = preprocess_image_bytes(img_bytes)

            preds = model(arr, training=False)
            probs = tf.nn.softmax(preds[0]).numpy()

            top3_idx = probs.argsort()[-3:][::-1]
            top3 = [(data_cat[i], round(float(probs[i]*100), 2)) for i in top3_idx]

            quality = analyze_quality_and_spoilage(pil_img)

            results.append({
                "prediction": top3[0][0],
                "confidence": top3[0][1],
                "top3": top3,
                "image_b64": base64.b64encode(img_bytes).decode(),
                "quality": quality
            })

        except Exception as e:
            results.append({"error": str(e)})

    app.latest_results = results
    return render_template("result.html", results=results)

@app.route("/download_csv")
def download_csv():
    results = getattr(app, "latest_results", [])
    if not results:
        return "No results", 404

    def generate():
        yield "prediction,confidence,top3\n"
        for r in results:
            if "prediction" in r:
                t = ";".join([f"{a}:{b}" for a,b in r["top3"]])
                yield f"{r['prediction']},{r['confidence']},{t}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )

# ============================================================
#                    RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
