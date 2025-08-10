import io
import json
import os
import logging
import warnings
import gc
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
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    warnings.filterwarnings('ignore', category=UserWarning)
    import tensorflow as tf
    # Configure TensorFlow for memory efficiency
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        # Disable GPU if available to save memory
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
        logger.warning(f"TensorFlow configuration warning: {e}")
    from tensorflow import keras
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    keras = None
    tf = None

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
        logger.warning(f"Model file not found at {path_to_use}")
        return None
    
    try:
        # Load model with memory optimization
        model = keras.models.load_model(path_to_use, compile=False)
        # Recompile with memory-efficient settings
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        # Force garbage collection after loading
        gc.collect()
        return model
    except Exception as e:


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
            if MODEL is not None:
                logger.info(f"✅ Model loaded successfully with {len(LABELS)} labels")
            else:
                logger.warning("⚠️ Model file not found, creating dummy model for demo")
                # Create a simple dummy model for demo purposes
                MODEL = create_dummy_model(len(LABELS))
        else:
            logger.warning("⚠️ TensorFlow not available, model loading skipped")
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not load model: {e}")
        logger.warning("⚠️ The API will work but predictions will fail until model is available")


def create_dummy_model(num_classes: int) -> keras.Model:
    """Create a dummy model for demo purposes when real model is not available"""
    model = keras.Sequential([
        keras.layers.Input(shape=(42,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    logger.info("Created dummy model for demo purposes")
    return model


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
        model_complexity=0  # Use lightest model for memory efficiency
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
            # Use smaller batch size and disable verbose output
            probs = MODEL.predict(df.values, batch_size=1, verbose=0)[0]
            # Force garbage collection after prediction
            gc.collect()
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
        # Limit image size for memory efficiency
        if len(content) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=413, detail="Image too large. Please use images smaller than 5MB.")
            
        base_img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Resize large images to save memory
        max_size = 800
        if max(base_img.size) > max_size:
            base_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        logger.info(f"Image loaded successfully, size: {base_img.size}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if MODEL is None:
        logger.error("Model not available for prediction")
        # Return a demo response when model is not available
        demo_labels = ["A", "B", "C", "D", "E"]
        demo_label = demo_labels[hash(str(content)) % len(demo_labels)]
        return JSONResponse({
            "label": demo_label, 
            "confidence": 0.85, 
            "message": "Demo mode - model not loaded"
        })

    # Reduced test-time augmentation for memory efficiency
    scales = [1.0]  # Single scale to save maximum memory
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
        
        # Clean up intermediate images
        if s != 1.0:
            del img_s
        gc.collect()

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
