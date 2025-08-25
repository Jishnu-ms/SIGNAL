from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests
from collections import defaultdict

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# HuggingFace pipeline
pipe = pipeline("image-classification", model="wambugu71/crop_leaf_diseases_vit")

# Inference knobs
TOP_K = 3
CONF_THRESHOLD = 0.60
USE_TTA = True


# -----------------------------
# Helpers
# -----------------------------
def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def run_pipe(img: Image.Image, top_k: int = 5):
    img = _to_rgb(img)
    preds = pipe(img, top_k=top_k)
    return list(preds)

def merge_and_normalize(score_maps):
    accum = defaultdict(float)
    for preds in score_maps:
        for p in preds:
            accum[p['label']] += float(p['score'])

    total = sum(accum.values()) or 1.0
    merged = [{"label": k, "score": v / total} for k, v in accum.items()]
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged

def predict_with_tta(img: Image.Image, want_top_k: int = TOP_K, do_tta: bool = USE_TTA):
    if not do_tta:
        preds = run_pipe(img, top_k=max(5, want_top_k))
        return sorted(preds, key=lambda x: x["score"], reverse=True)[:want_top_k]

    views = []
    views.append(img)
    views.append(ImageOps.mirror(img))
    views.append(img.rotate(10, resample=Image.BICUBIC))
    views.append(img.rotate(-10, resample=Image.BICUBIC))

    per_view_preds = [run_pipe(v, top_k=max(5, want_top_k)) for v in views]
    merged = merge_and_normalize(per_view_preds)
    return merged[:want_top_k]


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop-disease')
def crop_disease():
    return render_template('crop_disease.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template("error.html", message="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template("error.html", message="No selected file")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run predictions
    image = Image.open(file_path)
    results = predict_with_tta(image, want_top_k=TOP_K, do_tta=USE_TTA)

    warning = None
    if not results or results[0]['score'] < CONF_THRESHOLD:
        warning = "⚠️ Low confidence prediction. Try another photo (clear, centered leaf)."

    return render_template("result.html", results=results, image_path=file_path, warning=warning)

@app.route('/search_disease')
def search_disease():
    query = request.args.get('query', '')

    try:
        sources = [
            f"https://www.planetnatural.com/pest-problem-solver/plant-disease/{quote_plus(query)}",
            f"https://extension.umn.edu/search?search={quote_plus(query)}",
            f"https://www.gardeningknowhow.com/?s={quote_plus(query)}"
        ]

        description = ""
        solutions = []
        source_list = []

        for source_url in sources[:2]:
            response = requests.get(source_url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                paragraphs = soup.find_all('p')[:3]
                description += " ".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])

                lists = soup.find_all(['ul', 'ol'])
                for lst in lists:
                    items = lst.find_all('li')
                    solutions.extend([item.text.strip() for item in items if len(item.text.strip()) > 20])

                source_list.append({
                    "url": source_url,
                    "title": soup.title.string if soup.title else "Reference"
                })

        description = (description[:500] + "...") if len(description) > 500 else description
        solutions = list(dict.fromkeys(solutions))[:5]

        return jsonify({
            "description": description or "No detailed description available.",
            "solutions": solutions or ["No specific solutions found. Please consult with a local agricultural expert."],
            "sources": source_list
        })

    except Exception as e:
        print(f"Error fetching disease information: {str(e)}")
        return jsonify({
            "description": "Error fetching information. Please try again later.",
            "solutions": ["Unable to retrieve solutions at this time."],
            "sources": []
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
