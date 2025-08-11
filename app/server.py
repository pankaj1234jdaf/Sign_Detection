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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pandas as pd
import mediapipe as mp

# Suppress warnings and configure for minimal memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    # Aggressive memory optimization
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_memory_growth = True
    from tensorflow import keras
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    keras = None
    tf = None

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced logging
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
MODEL_PATH = os.path.join(BASE_DIR, "model_v2.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

app = FastAPI(title="ISL Sign Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
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
        except Exception as e:
            logger.warning(f"Error loading labels: {e}")
    
    # Fallback to default ISL labels
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9']

def create_minimal_model(num_classes: int) -> keras.Model:
    """Create a minimal model for demo purposes"""
    if keras is None:
        return None
        
    model = keras.Sequential([
        keras.layers.Input(shape=(42,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
        run_eagerly=False
    )
    return model

def load_model(model_path: str, num_classes: int) -> keras.Model:
    """Load model or create minimal one"""
    if keras is None:
        return None
        
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            with tf.device('/CPU:0'):
                model = keras.models.load_model(model_path, compile=False)
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"],
                run_eagerly=False
            )
            gc.collect()
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    
    # Create minimal model for demo
    logger.info("Creating minimal demo model")
    return create_minimal_model(num_classes)

# Global variables
LABELS: List[str] = []
MODEL: keras.Model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and labels on startup"""
    global LABELS, MODEL
    try:
        LABELS = load_labels(LABELS_PATH)
        if keras is not None:
            MODEL = load_model(MODEL_PATH, len(LABELS))
            if MODEL is not None:
                logger.info(f"✅ Model ready with {len(LABELS)} labels")
            else:
                logger.warning("⚠️ Model not available")
        else:
            logger.warning("⚠️ TensorFlow not available")
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")

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
    if not landmarks or len(landmarks) != 21:
        return [0.0] * 42
    
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    rel = [[x - base_x, y - base_y] for x, y in landmarks]
    flat = [v for pair in rel for v in pair]
    
    # Normalize
    max_val = max(map(abs, flat)) if flat else 1.0
    if max_val == 0:
        return flat
    return [v / max_val for v in flat]

def predict_image(pil_img: Image.Image) -> np.ndarray:
    """Run prediction on a single image"""
    if MODEL is None:
        return None
    
    # Convert and resize image for memory efficiency
    img_np = np.array(pil_img.convert('RGB'))
    
    # Limit image size
    max_size = 640
    if max(img_np.shape[:2]) > max_size:
        h, w = img_np.shape[:2]
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_np = np.array(pil_resized)
    
    # Convert RGB to BGR for MediaPipe
    img_np = img_np[:, :, ::-1]
    
    # Use minimal MediaPipe configuration
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0
    ) as hands:
        results = hands.process(img_np)
        if not results.multi_hand_landmarks:
            return None
            
        hand_landmarks = results.multi_hand_landmarks[0]
        points = extract_landmarks(img_np, hand_landmarks)
        features = preprocess_landmarks(points)
        
        if len(features) != 42:
            return None
        
        try:
            # Single prediction with minimal batch size
            input_data = np.array([features], dtype=np.float32)
            probs = MODEL.predict(input_data, batch_size=1, verbose=0)[0]
            
            # Force garbage collection
            del input_data
            gc.collect()
            
            return probs
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Predict sign language from uploaded image"""
    try:
        content = await image.read()
        
        # Strict size limit for free tier
        if len(content) > 2 * 1024 * 1024:  # 2MB limit
            raise HTTPException(status_code=413, detail="Image too large. Please use images smaller than 2MB.")
            
        base_img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Aggressive resizing for memory efficiency
        max_size = 512
        if max(base_img.size) > max_size:
            base_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if MODEL is None:
        # Demo response when model not available
        import hashlib
        hash_val = int(hashlib.md5(content[:50]).hexdigest(), 16)
        demo_label = LABELS[hash_val % len(LABELS)] if LABELS else "A"
        return {
            "label": demo_label, 
            "confidence": 0.75 + (hash_val % 20) / 100,
            "message": "Demo mode - using simulated prediction"
        }

    # Single prediction (no test-time augmentation for memory efficiency)
    probs = predict_image(base_img)
    
    # Force cleanup
    del base_img
    gc.collect()
    
    if probs is None:
        return {"label": None, "confidence": 0.0, "message": "No hand detected"}

    idx = int(np.argmax(probs))
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    conf = float(probs[idx])
    
    return {"label": label, "confidence": conf}

@app.get("/web/")
async def web_index():
    """Serve the web interface"""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Web UI not found"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)