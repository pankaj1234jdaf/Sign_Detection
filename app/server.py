import io
import json
import os
import logging
import warnings
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pandas as pd
import mediapipe as mp
try:
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', category=UserWarning)
    from tensorflow import keras
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    keras = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
DEFAULT_MODEL = os.path.join(BASE_DIR, "model.keras")
H5_FALLBACK = os.path.join(BASE_DIR, "model_v2.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

app = FastAPI(title="ISL Sign Detection API")

# Add CORS middleware for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if web directory exists
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="static")


@app.get("/")
async def root_index():
    """Serve the main web interface"""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ISL Sign Detector API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "message": "ISL Sign Detector is running"}


def load_labels(path: str) -> List[str]:
    """Load label mapping from JSON file"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            classes = data.get("classes")
            if isinstance(classes, list) and classes:
                return classes
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading labels from {path}: {e}")
    
    # Fallback to default labels
    import string as _string
    return [str(i) for i in range(1, 10)] + list(_string.ascii_uppercase)


def load_model(model_path: str) -> keras.Model:
    """Load the trained model"""
    path_to_use = model_path if os.path.exists(model_path) else H5_FALLBACK
    if not os.path.exists(path_to_use):
        raise FileNotFoundError(f"Model file not found at {path_to_use}. Train the model first.")
    
    try:
        return keras.models.load_model(path_to_use)
    except Exception as e:
        print(f"Error loading model from {path_to_use}: {e}")
        raise


# Initialize global variables
LABELS: List[str] = []
MODEL: keras.Model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and labels on startup"""
    global LABELS, MODEL
    try:
        LABELS = load_labels(LABELS_PATH)
        if keras is not None:
            MODEL = load_model(DEFAULT_MODEL)
            logger.info(f"✅ Model loaded successfully with {len(LABELS)} labels")
        else:
            logger.warning("⚠️ TensorFlow not available, model loading skipped")
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not load model: {e}")
        logger.warning("⚠️ The API will work but predictions will fail until model is available")


def extract_landmarks(img_np: np.ndarray, hand_landmarks) -> List[List[int]]:
    """Extract hand landmarks from MediaPipe results"""
    img_height, img_width = img_np.shape[0], img_np.shape[1]
    points = []
    for landmark in hand_landmarks.landmark:
        x = min(int(landmark.x * img_width), img_width - 1)
        y = min(int(landmark.y * img_height), img_height - 1)
        points.append([x, y])
    return points


def preprocess_landmarks(landmarks: List[List[int]]) -> List[float]:
    """Preprocess landmarks for model input"""
    if not landmarks:
        return [0.0] * 42  # Return zeros for 21 landmarks * 2 coordinates
    
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    rel = [[x - base_x, y - base_y] for x, y in landmarks]
    flat = [v for pair in rel for v in pair]
    max_val = max(map(abs, flat)) if flat else 1.0
    return [v / max_val for v in flat] if max_val != 0 else flat


def predict_image(pil_img: Image.Image) -> np.ndarray:
    """Run prediction on a single image"""
    if MODEL is None:
        logger.error("Model not loaded, cannot predict")
        return None
    
    # Convert PIL image to numpy array
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Convert RGB to BGR for MediaPipe
        img_np = img_np[:, :, ::-1]
    
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as hands:
        results = hands.process(img_np)
        if not results.multi_hand_landmarks:
            logger.warning("No hand landmarks detected")
            return None
            
        hand_landmarks = results.multi_hand_landmarks[0]
        points = extract_landmarks(img_np, hand_landmarks)
        features = preprocess_landmarks(points)
        
        # Ensure we have the right number of features
        expected_features = 42  # 21 landmarks * 2 coordinates
        if len(features) != expected_features:
            logger.error(f"Expected {expected_features} features, got {len(features)}")
            return None
        
        df = pd.DataFrame([features])
        try:
            probs = MODEL.predict(df, verbose=0)[0]
            logger.info(f"Prediction successful, max confidence: {np.max(probs):.3f}")
            return probs
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Predict sign language from uploaded image"""
    logger.info(f"Received prediction request for file: {image.filename}")
    
    try:
        content = await image.read()
        base_img = Image.open(io.BytesIO(content)).convert("RGB")
        logger.info(f"Image loaded successfully, size: {base_img.size}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if MODEL is None:
        logger.error("Model not available for prediction")
        return JSONResponse({"label": None, "confidence": 0.0, "message": "Model not loaded"})

    # Test-time augmentation: slight resizes around the base image
    scales = [1.0, 0.9, 1.1]
    probs_accum = None
    valid = 0

    for s in scales:
        if s == 1.0:
            img_s = base_img
        else:
            w, h = base_img.size
            nw, nh = int(w * s), int(h * s)
            resized = base_img.resize((nw, nh))
            # center-crop or pad to original size
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            left = max(0, (w - nw) // 2)
            top = max(0, (h - nh) // 2)
            canvas.paste(resized, (left, top))
            img_s = canvas
        
        probs = predict_image(img_s)
        if probs is not None:
            probs_accum = probs if probs_accum is None else (probs_accum + probs)
            valid += 1
            logger.info(f"Valid prediction for scale {s}")

    if not valid:
        logger.warning("No valid predictions from any scale")
        return JSONResponse({"label": None, "confidence": 0.0, "message": "No hand detected"})

    probs_avg = probs_accum / valid
    idx = int(np.argmax(probs_avg))
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    conf = float(probs_avg[idx])
    
    logger.info(f"Final prediction: {label} (confidence: {conf:.3f})")
    return {"label": label, "confidence": conf}


@app.get("/web/")
async def web_index():
    """Serve the web interface"""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Web UI not found. Build web assets in /web"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
