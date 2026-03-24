# 😊 Facial Expression Detector

Real-time facial expression detection using your **webcam**, powered by [DeepFace](https://github.com/serengil/deepface) and OpenCV.

> Detects 7 emotions: **Happy · Sad · Angry · Surprise · Fear · Disgust · Neutral**

---

## 📸 Preview

The app opens your webcam and:
- Draws a **color-coded rounded bounding box** around your face
- Shows a **floating emotion tag** with emoji above the box
- Renders a **live emotion confidence bar chart** in the side panel
- Displays real-time **FPS counter**

---

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/MxthThunder/facial-expression-detector.git
cd facial-expression-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run will auto-download the DeepFace model weights (~100 MB). Internet required.

---

## ▶️ Run

```bash
python detector.py
```

Press **`Q`** to quit.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `deepface` | Emotion analysis AI model |
| `opencv-python` | Webcam capture & drawing |
| `tensorflow` | DeepFace backend |
| `numpy` | Array operations |

---

## ⚙️ How It Works

1. **Webcam feed** is captured and mirrored horizontally
2. Every ~180 ms, **DeepFace** analyzes the frame using the `opencv` face detector
3. Returns a dictionary of **emotion scores** (0–100%) and the **dominant emotion**
4. OpenCV draws bounding box, tag, and confidence bars overlaid on the live feed

---

## 🧠 Supported Emotions

| Emotion | Emoji | Box Color |
|---|---|---|
| Happy | 😊 | Green |
| Sad | 😢 | Blue-Orange |
| Angry | 😠 | Red |
| Surprise | 😲 | Cyan |
| Fear | 😨 | Purple |
| Disgust | 🤢 | Dark Green |
| Neutral | 😐 | Grey |

---

## 📁 Project Structure

```
facial-expression-detector/
├── detector.py        # Main application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🐛 Troubleshooting

**Camera not opening?**
- Make sure no other app is using your webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `detector.py`

**Slow performance?**
- Increase `analysis_interval` in `detector.py` (e.g. `0.35`)
- Use a lighter backend: change `detector_backend="opencv"` to `"haarcascade"`

**TensorFlow issues on Mac M1/M2?**
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
