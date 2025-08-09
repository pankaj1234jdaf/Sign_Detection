# Indian Sign Language to Speech/Text Converter

A real-time Indian Sign Language detection system using deep learning, with a modern web interface for accessibility.

## 🚀 Features

- **Real-time Detection**: Live webcam-based sign language recognition
- **High Accuracy**: Enhanced with test-time augmentation and improved MediaPipe detection
- **Professional UI**: Modern black, white, and grey theme with smooth animations
- **Sentence Building**: Auto-add labels to build complete sentences
- **Text-to-Speech**: Automatic speech synthesis for detected signs
- **Upload Support**: Image upload for offline detection
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Indian-Sign-Language-to-Speech-Text-Converter-using-Deep-Learning-main
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv311
   source .venv311/bin/activate  # On Windows: .venv311\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if needed)
   ```bash
   python train_model.py
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

## 🌐 Deployment on Render

### Option 1: Using Render Dashboard

1. **Fork/Clone** this repository to your GitHub account
2. **Sign up** for [Render](https://render.com) (free tier available)
3. **Create New Web Service**:
   - Connect your GitHub repository
   - Choose **Python** environment
   - Set **Build Command**: `pip install -r requirements.txt`
   - Set **Start Command**: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
   - Choose **Free** plan
4. **Deploy** and wait for build to complete
5. **Access** your app at the provided URL

### Option 2: Using render.yaml (Recommended)

1. **Push** your code to GitHub
2. **Connect** repository to Render
3. **Render will automatically** detect the `render.yaml` file
4. **Deploy** with one click

## 📁 Project Structure

```
├── app/
│   └── server.py              # FastAPI backend server
├── web/
│   ├── index.html             # Main web interface
│   ├── js/
│   │   └── app.js             # Frontend JavaScript
│   ├── terms.html             # Terms of service
│   └── usage.html             # Usage instructions
├── train_model.py             # Model training script
├── detect.py                  # Standalone detection script
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── Procfile                   # Process file for deployment
├── runtime.txt                # Python version specification
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHON_VERSION`: Python version (default: 3.11.0)

### Model Files

Ensure these files are present in the root directory:
- `model.keras` or `model_v2.h5` - Trained model
- `labels.json` - Label mapping

## 🎮 Usage

### Web Interface

1. **Start Camera**: Click "Start Camera" and allow camera access
2. **Start AI**: Click "Start AI" for continuous detection
3. **Adjust Settings**: Use confidence threshold and auto-features
4. **Build Sentences**: Use auto-add or manual buttons
5. **Speak**: Click "Speak" to hear the sentence

### API Endpoints

- `GET /` - Main web interface
- `GET /health` - Health check
- `POST /predict` - Predict sign from image
- `GET /docs` - API documentation

## 🛠️ Development

### Training Model

```bash
python train_model.py --csv hand_keypoints.csv --model_out model.keras
```

### Standalone Detection

```bash
python detect.py --model model.keras --labels labels.json
```

## 🔍 Troubleshooting

### Common Issues

1. **Model not found**: Ensure `model.keras` or `model_v2.h5` exists
2. **Camera not working**: Check browser permissions
3. **Dependencies error**: Use Python 3.11 and install from `requirements.txt`

### Render Deployment Issues

1. **Build fails**: Check `requirements.txt` for compatibility
2. **App not starting**: Verify `Procfile` and start command
3. **Model loading**: Ensure model files are in repository

## 📄 License

This project is built for accessibility and educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub

---

**Built with ❤️ for accessibility**

