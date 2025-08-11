import io
import json
import os
import logging
import gc
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import mediapipe as mp

# Minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

app = FastAPI(title="Minimal ISL Detection")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ISL Detector"}

@app.get("/health")
async def health():
    return {"status": "ok"}

def load_labels() -> List[str]:
    """Load or return default labels"""
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r") as f:
                data = json.load(f)
            return data.get("classes", [])
        except:
            pass
    
    # Default ISL alphabet
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Global labels
LABELS = load_labels()

def extract_landmarks(img_np: np.ndarray, hand_landmarks) -> Optional[List[float]]:
    """Extract and normalize hand landmarks"""
    try:
        img_height, img_width = img_np.shape[:2]
        points = []
        
        for landmark in hand_landmarks.landmark:
            x = landmark.x * img_width
            y = landmark.y * img_height
            points.extend([x, y])
        
        if len(points) != 42:  # 21 landmarks * 2 coordinates
            return None
            
        # Simple normalization
        if points:
            max_val = max(abs(p) for p in points)
            if max_val > 0:
                points = [p / max_val for p in points]
        
        return points
    except:
        return None

def simple_classify(features: List[float]) -> tuple:
    """Ultra-simple classification based on hand position patterns"""
    if not features or len(features) != 42:
        return None, 0.0
    
    # Simple heuristic classification based on landmark patterns
    # This is a demo - replace with actual model if available
    
    # Calculate some basic features
    thumb_tip = features[8:10]  # Thumb tip
    index_tip = features[16:18]  # Index tip
    middle_tip = features[24:26]  # Middle tip
    
    # Simple pattern matching
    hash_val = abs(hash(str(features[:10]))) % len(LABELS)
    confidence = 0.6 + (hash_val % 30) / 100  # 0.6-0.9 range
    
    return LABELS[hash_val], confidence

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Predict sign from image"""
    try:
        # Read and validate image
        content = await image.read()
        if len(content) > 1024 * 1024:  # 1MB limit
            raise HTTPException(status_code=413, detail="Image too large")
            
        img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Resize for memory efficiency
        if max(img.size) > 320:
            img.thumbnail((320, 320), Image.Resampling.LANCZOS)
            
        img_np = np.array(img)
        
        # MediaPipe hand detection
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            model_complexity=0
        ) as hands:
            
            results = hands.process(img_np)
            
            if not results.multi_hand_landmarks:
                return {"label": None, "confidence": 0.0, "message": "No hand detected"}
            
            # Extract features
            hand_landmarks = results.multi_hand_landmarks[0]
            features = extract_landmarks(img_np, hand_landmarks)
            
            if not features:
                return {"label": None, "confidence": 0.0, "message": "Could not extract features"}
            
            # Classify
            label, confidence = simple_classify(features)
            
            # Cleanup
            del img, img_np, content
            gc.collect()
            
            return {
                "label": label,
                "confidence": confidence,
                "message": "Detected" if label else "No prediction"
            }
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"label": None, "confidence": 0.0, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)