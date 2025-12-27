<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=200&section=header&text=Real-Time%20Face%20Detection&fontSize=50&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=Ronaldo%20Recognition%20with%20Web%20Interface&descAlignY=55" width="100%"/>
</div>

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
  ![ResNet](https://img.shields.io/badge/ResNet-DC143C?style=for-the-badge)
  ![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge)
  
</div>

<h3 align="center">🎯 Real-time Face Recognition System with Web Dashboard</h3>

<p align="center">
  A comprehensive real-time face detection and classification system that identifies Cristiano Ronaldo using ResNet architecture, MediaPipe face detection, and displays results on an interactive web interface powered by Flask.
</p>

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=flat-square)](/)
[![Real-time](https://img.shields.io/badge/Real--time-Detection-orange?style=flat-square)](/)
[![Web Interface](https://img.shields.io/badge/Web-Interface-blue?style=flat-square)](/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Web Interface](#-web-interface)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements an advanced **real-time face recognition system** that combines computer vision and deep learning to detect and classify faces, specifically identifying **Cristiano Ronaldo**. The system features:

- **Real-time Processing**: Webcam-based live face detection
- **Deep Learning**: ResNet architecture with transfer learning
- **Face Detection**: MediaPipe for accurate face localization
- **Web Dashboard**: Flask-powered interface for live results
- **Interactive UI**: Dynamic table with images, scores, and timestamps

### 🌟 Key Highlights

- ✅ **Dual Architecture**: MediaPipe + ResNet for robust detection
- ✅ **Web-Based**: Beautiful Flask interface with real-time updates
- ✅ **RESTful API**: Clean API for data transmission
- ✅ **Visual Feedback**: Images and color-coded results
- ✅ **Production Ready**: Scalable Flask application

---

## ✨ Features

<table>
  <tr>
    <td width="50%" valign="top">
      
### 🎥 Real-time Detection
- Live webcam face detection
- MediaPipe face localization
- Instant classification results
- Multi-face support

    </td>
    <td width="50%" valign="top">
      
### 🌐 Web Dashboard
- Flask-based web interface
- Real-time data updates
- Interactive results table
- Responsive design

    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      
### 🧠 Deep Learning
- ResNet transfer learning
- Pre-trained model support
- High accuracy classification
- Confidence score display

    </td>
    <td width="50%" valign="top">
      
### 🎨 Visual Interface
- Color-coded results
- Representative images
- Timestamp tracking
- Professional UI/UX

    </td>
  </tr>
</table>

---

## 🏗️ System Architecture

```
┌─────────────┐
│   Webcam    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   MediaPipe     │  Face Detection
│ Face Detection  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  ResNet Model   │  Classification
│ (Transfer Learn)│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Flask API     │  Data Processing
│ (POST /api/...)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Web Dashboard  │  Visualization
│   (HTML/CSS/JS) │
└─────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework
- **Flask**: Web framework and API
- **MediaPipe**: Face detection library
- **OpenCV**: Computer vision operations

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling and animations
- **JavaScript**: Dynamic interactions
- **Bootstrap** (optional): Responsive design

### Machine Learning
- **ResNet**: Classification model
- **Transfer Learning**: Pre-trained weights
- **PIL**: Image processing

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8 or higher
Webcam (built-in or external)
pip (Python package manager)
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ai-rezzak/real-time-ronaldo-and_person-detection-with-html.git
cd real-time-ronaldo-and_person-detection-with-html
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements include:**
- torch
- torchvision
- flask
- mediapipe
- opencv-python
- pillow
- numpy

### Step 3: Download Pre-trained Model

Place the pre-trained model file in the project root:
```
real-time-ronaldo-and_person-detection-with-html/
├── person_ronaldo_model.pth  ← Place model here
└── ...
```

### Step 4: Verify Webcam

Ensure your webcam is connected and working:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam Error')"
```

---

## 💻 Usage

### Quick Start

#### 1️⃣ Start the Flask Web Server

Open a terminal and run:

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Running on http://localhost:5000
```

#### 2️⃣ Start the Webcam Classifier

Open a **second terminal** and run:

```bash
python main.py
```

This will:
- Open your webcam
- Start detecting faces
- Send results to Flask API

#### 3️⃣ View the Web Dashboard

Open your browser and navigate to:
```
http://localhost:5000/veri-tablosu
```

**You'll see:**
- Live detection results
- Classification scores
- Timestamps
- Representative images

---

## 📁 Project Structure

```
real-time-ronaldo-and_person-detection-with-html/
│
├── 📄 app.py                           # Flask web application
├── 📄 main.py                          # Webcam detection script
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # Project documentation
│
├── 📁 models/
│   └── person_ronaldo_model.pth       # Pre-trained ResNet model
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── style.css                  # Custom styles
│   ├── 📁 js/
│   │   └── script.js                  # JavaScript functionality
│   └── 📁 images/
│       ├── ron.png                    # Ronaldo image
│       └── person.png                 # Person image
│
└── 📁 templates/
    └── veri_tablosu.html              # Web dashboard template
```

### File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| **app.py** | Flask web server | API endpoints, data handling, rendering |
| **main.py** | Face detection & classification | Webcam capture, MediaPipe detection, model inference |
| **veri_tablosu.html** | Web dashboard | Display results, dynamic table |
| **style.css** | Styling | UI design, animations |
| **script.js** | Frontend logic | AJAX requests, DOM manipulation |

---

## 🔌 API Documentation

### POST `/api/save-data`

Receives detection data from the webcam classifier.

**Request Body:**
```json
{
  "class": "ronaldo",
  "score": "0.95",
  "time": "14:02:30 / 15-10-2024"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Data saved successfully"
}
```

**Example using cURL:**
```bash
curl -X POST http://localhost:5000/api/save-data \
  -H "Content-Type: application/json" \
  -d '{"class":"ronaldo","score":"0.95","time":"14:02:30 / 15-10-2024"}'
```

### GET `/veri-tablosu`

Displays the web dashboard with detection results.

**Response:** HTML page with live results

---

## 🌐 Web Interface

### Features

<table>
  <tr>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/fluency/96/000000/table.png" width="50px"/><br>
      <b>Results Table</b>
      <p><sub>Dynamic data display</sub></p>
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/fluency/96/000000/image.png" width="50px"/><br>
      <b>Visual Feedback</b>
      <p><sub>Representative images</sub></p>
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/fluency/96/000000/clock.png" width="50px"/><br>
      <b>Timestamps</b>
      <p><sub>Detection time tracking</sub></p>
    </td>
    <td align="center" width="25%">
      <img src="https://img.icons8.com/fluency/96/000000/graph.png" width="50px"/><br>
      <b>Confidence Scores</b>
      <p><sub>Classification accuracy</sub></p>
    </td>
  </tr>
</table>

### Table Columns

| Column | Description | Example |
|--------|-------------|---------|
| **Class** | Detected class | Ronaldo / Person |
| **Score** | Confidence score | 0.95 (95%) |
| **Time** | Detection timestamp | 14:02:30 / 15-10-2024 |
| **Image** | Visual representation | ![Ronaldo](static/images/ron.png) |

### Color Coding

- 🟢 **Green Row**: Ronaldo detected
- 🔵 **Blue Row**: Person detected
- 🔴 **Red Text**: Low confidence (<0.7)
- 🟢 **Green Text**: High confidence (>0.9)

---

## ⚙️ How It Works

### Step-by-Step Process

#### 1. Webcam Capture (`main.py`)
```python
# Capture frame from webcam
ret, frame = cap.read()
```

#### 2. Face Detection (MediaPipe)
```python
# Detect faces in the frame
results = face_detection.process(rgb_frame)
```

#### 3. Face Extraction
```python
# Crop detected face region
face_crop = frame[y1:y2, x1:x2]
```

#### 4. Preprocessing
```python
# Resize and normalize for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

#### 5. Classification (ResNet)
```python
# Predict class
with torch.no_grad():
    output = model(face_tensor)
    prediction = torch.argmax(output, dim=1)
```

#### 6. Send to API
```python
# POST detection results to Flask
requests.post('http://localhost:5000/api/save-data', json=data)
```

#### 7. Display on Web
```javascript
// Update table dynamically
fetch('/api/get-latest-data')
    .then(response => response.json())
    .then(data => updateTable(data));
```

---

## 🔧 Configuration

### Webcam Settings (`main.py`)

```python
# Change camera index (0=built-in, 1=external)
cap = cv2.VideoCapture(0)

# Adjust resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Model Settings

```python
# Change confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Adjust class names
CLASSES = ['person', 'ronaldo']
```

### Flask Settings (`app.py`)

```python
# Change port
app.run(host='0.0.0.0', port=5000)

# Enable debug mode
app.run(debug=True)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Webcam Not Opening**
```bash
# Try different camera index
python main.py --camera 1
```

**2. Model File Not Found**
```bash
# Ensure model is in correct location
ls person_ronaldo_model.pth
```

**3. Flask Port Already in Use**
```bash
# Change port in app.py
app.run(port=5001)
```

**4. API Connection Error**
```python
# Check if Flask server is running
curl http://localhost:5000/api/save-data
```

**5. Low Detection Accuracy**
```python
# Adjust confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Lower threshold
```

---

## 🎨 Customization

### Adding Custom Images

1. Add images to `static/images/`:
```
static/images/
├── ron.png          # Ronaldo image
├── person.png       # Generic person
└── custom.png       # Your custom image
```

2. Update image paths in template:
```html
<img src="{{ url_for('static', filename='images/custom.png') }}" />
```

### Changing UI Colors

Edit `static/css/style.css`:
```css
/* Ronaldo row color */
.ronaldo-row {
    background-color: #28a745;  /* Green */
}

/* Person row color */
.person-row {
    background-color: #007bff;  /* Blue */
}
```

---

## 📊 Performance Optimization

### Speed Improvements

```python
# 1. Use GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Reduce frame processing
if frame_count % 2 == 0:  # Process every 2nd frame
    detect_and_classify(frame)

# 3. Lower webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. 🔀 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Contact

**Abdurrezzak ŞIK**

[![Email](https://img.shields.io/badge/Email-rezzak.eng%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:rezzak.eng@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdurrezzak-%C5%9F%C4%B1k-64b919233/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ai-rezzak)

---

## 🌟 Show Your Support

Give a ⭐ if this project helped you build real-time face recognition systems!

<div align="center">
  
### 📚 Related Projects

[![ResNet Person Detection](https://img.shields.io/badge/ResNet_Detection-DC143C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/person_ronaldo_with_ResNet)
[![VGGNet Detection](https://img.shields.io/badge/VGGNet_Detection-4B0082?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/person_ronaldo_with_VGGnet)
[![Dental X-Ray Detection](https://img.shields.io/badge/Dental_X--Ray-00FFFF?style=for-the-badge&logo=github&logoColor=black)](https://github.com/Ai-rezzak/dental-xray-yolov8-detection)

</div>

---

## 📚 References

- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24&height=120&section=footer" width="100%"/>
  
  <br>
  
  <sub>Made with ❤️ by Abdurrezzak ŞIK</sub>
  
  <br><br>
  
  <sub>"Real-time face recognition with the power of AI" 🎯</sub>
  
</div>
