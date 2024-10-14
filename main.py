# KÜTÜPHANELER
import torch
import cv2
import os  
from torchvision import transforms
import mediapipe as mp
from PIL import Image
from datetime import datetime
import requests

# Cihaz ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VERİ ÖN İŞLEME
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MODEL YÜKLEME
model_path = 'models/person_ronaldo_model.pth'
model = torch.load(model_path, map_location=device)
model.eval()  # Modeli değerlendirme moduna al

# Kullanım örneği
detected_classes = set()  # Daha önce tespit edilen sınıfları saklamak için

def send_data_to_app(cls_name, score):
    url = "http://localhost:5000/api/save-data"  # Flask uygulamanın URL'si
    time_data = datetime.now().strftime("%S:%M:%H / %d-%m-%Y")
    
    data = {
        "class": cls_name.upper(),
        "score": f"{score:.2f}",
        "time": time_data
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Data sent successfully!")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# ÇİZİM
def draw(frame, bboxC, cls_name, score):
    h, w, _ = frame.shape
    x_min = int(bboxC.xmin * w)
    y_min = int(bboxC.ymin * h)
    box_width = int(bboxC.width * w)
    box_height = int(bboxC.height * h)

    cv2.putText(frame, f"{cls_name}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)

# FRAME GÖSTERME
def göster(frame):
    cv2.imshow("Frame", frame)
    return cv2.waitKey(1) & 0xFF == ord("q")

# MEDİPİPE'de ALGILANAN YÜZ CNN MODELİ TESPİT İŞLEMİ YAPILACAK
def cnn(frame_copy):
    frame_copy = cv2.resize(frame_copy, (224, 224))  # Boyutlandırma
    frame_copy = Image.fromarray(frame_copy)  # OpenCV'den PIL'e dönüştürme
    frame_copy = transform(frame_copy).unsqueeze(0).to(device)  # Tensor haline getirme

    class_names = ['person', 'ronaldo']  # Sınıflar

    with torch.no_grad():
        output = model(frame_copy)  # Modelden çıktı al
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)  # En yüksek olasılıklı sınıfı al
        cls_name = class_names[predicted_class]
        
        score = probabilities[0][predicted_class.item()].item()
        return cls_name, score  # Model çıktısını döndür

# Kamera ayarları
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Kamera açılamadı!")

# MEDİPİPE İLE YÜZ TESPİTİ
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    frame = cv2.flip(frame, 1)  
    results = mp_face_detection.process(frame)  # Yüz tespiti yap

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_region = frame[int(bboxC.ymin * frame.shape[0]):int((bboxC.ymin + bboxC.height) * frame.shape[0]),
                                int(bboxC.xmin * frame.shape[1]):int((bboxC.xmin + bboxC.width) * frame.shape[1])]
            
            cls_name, score = cnn(face_region)  # CNN modelini çalıştır
            
            if cls_name not in detected_classes:
                send_data_to_app(cls_name, score)  # Veriyi gönder
                detected_classes.add(cls_name)  # Yeni sınıfı tespit edilen sınıflar set'ine ekle
            
            draw(frame, bboxC, cls_name, score)  # Çizim fonksiyonunu çağır

    if göster(frame):
        break

cap.release()
cv2.destroyAllWindows()
