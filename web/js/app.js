// Simple ISL detector with speech
let video, stream, detectionInterval;
let lastSpoken = '';

// DOM elements
const btnStart = document.getElementById('btn-start');
const btnDetect = document.getElementById('btn-detect');
const btnStop = document.getElementById('btn-stop');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const messageEl = document.getElementById('message');
const autoSpeakEl = document.getElementById('auto-speak');
const thresholdEl = document.getElementById('threshold');
const thresholdValueEl = document.getElementById('threshold-value');

// Initialize
video = document.getElementById('video');

// Update threshold display
thresholdEl.addEventListener('input', () => {
  thresholdValueEl.textContent = thresholdEl.value;
});

// Speech function
function speak(text) {
  if (!window.speechSynthesis || !text) return;
  
  // Cancel any ongoing speech
  window.speechSynthesis.cancel();
  
  setTimeout(() => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.8;
    utterance.volume = 1.0;
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  }, 100);
}

// Start camera
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 } 
    });
    video.srcObject = stream;
    btnStart.disabled = true;
    btnDetect.disabled = false;
    messageEl.textContent = 'Camera started. Click "Start Detection" to begin.';
  } catch (error) {
    console.error('Camera error:', error);
    messageEl.textContent = 'Failed to access camera. Please check permissions.';
  }
}

// Stop camera
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (detectionInterval) {
    clearInterval(detectionInterval);
    detectionInterval = null;
  }
  btnStart.disabled = false;
  btnDetect.disabled = true;
  btnDetect.textContent = 'Start Detection';
  predictionEl.textContent = '–';
  confidenceEl.textContent = 'Confidence: –';
  messageEl.textContent = 'Camera stopped.';
}

// Capture frame and predict
async function captureAndPredict() {
  if (!video.videoWidth || !video.videoHeight) return;
  
  try {
    // Create canvas to capture frame
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert to blob
    const blob = await new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', 0.8);
    });
    
    // Send to server
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');
    
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    // Update UI
    if (result.label) {
      predictionEl.textContent = result.label;
      confidenceEl.textContent = `Confidence: ${result.confidence.toFixed(2)}`;
      
      // Auto-speak if enabled and confidence is high enough
      const threshold = parseFloat(thresholdEl.value);
      if (autoSpeakEl.checked && 
          result.confidence >= threshold && 
          result.label !== lastSpoken) {
        speak(result.label);
        lastSpoken = result.label;
      }
    } else {
      predictionEl.textContent = '–';
      confidenceEl.textContent = 'Confidence: –';
    }
    
    messageEl.textContent = result.message || 'Detecting...';
    
  } catch (error) {
    console.error('Prediction error:', error);
    messageEl.textContent = 'Detection error. Please try again.';
  }
}

// Start/stop detection
function toggleDetection() {
  if (detectionInterval) {
    // Stop detection
    clearInterval(detectionInterval);
    detectionInterval = null;
    btnDetect.textContent = 'Start Detection';
    messageEl.textContent = 'Detection stopped.';
  } else {
    // Start detection
    detectionInterval = setInterval(captureAndPredict, 1000); // Every 1 second
    btnDetect.textContent = 'Stop Detection';
    messageEl.textContent = 'Detecting signs...';
  }
}

// Event listeners
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
btnDetect.addEventListener('click', toggleDetection);

// Test speech on page load
window.addEventListener('load', () => {
  setTimeout(() => {
    speak('ISL detector ready');
  }, 1000);
});