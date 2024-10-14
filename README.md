# Ronaldo ve Kişi Tespit Projesi

Bu proje, önceden eğitilmiş bir ResNet modelini (transfer öğrenimi ile) kullanarak **Cristiano Ronaldo** ve normal kişileri ayırt etmek amacıyla tasarlanmıştır. Gerçek zamanlı olarak webcam'den yakalanan yüzler, bir CNN modeline gönderilerek "ronaldo" ya da "kişi" olarak sınıflandırılır. Sonuçlar (sınıf, skor ve tespit zamanı) bir web tablosunda görüntülenir.

## İçindekiler

- [Proje Açıklaması](#proje-açıklaması)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum](#kurulum)
- [Nasıl Çalışır?](#nasıl-çalışır)
- [Projeyi Çalıştırma](#projeyi-çalıştırma)
- [API Uç Noktaları](#api-uç-noktaları)
- [Web Arayüzü](#web-arayüzü)
- [Resim Ekleme Özelliği](#resim-ekleme-özelliği)

## Proje Açıklaması

Bu proje, webcam'den alınan gerçek zamanlı görüntüyü kullanarak **MediaPipe** ile yüz tespiti yapar ve önceden eğitilmiş bir CNN modeli ile sınıflandırma gerçekleştirir. Sınıflandırma sonuçları Flask tabanlı bir web arayüzüne gönderilir ve burada sınıf, güven skoru ve tespit zamanı görüntülenir.

## Kullanılan Teknolojiler

- Python
- PyTorch
- MediaPipe (yüz tespiti için)
- OpenCV (video yakalama ve gösterim için)
- Flask (web arayüzü için)
- HTML/CSS (frontend için)
- JavaScript (dinamik etkileşim için)
- PIL (görüntü işleme için)

## Kurulum

1. Projeyi yerel makinenize klonlayın.
   ```bash
   git clone https://github.com/Ai-rezzak/real-time-ronaldo-and_person-detection-with-html.git
   cd real-time-ronaldo-and_person-detection-with-html
   ```

2. Gerekli Python kütüphanelerini yükleyin.
   ```bash
   pip install -r requirements.txt
   ```

3. Önceden eğitilmiş modeli (`person_ronaldo_model.pth`) indirip ana dizine yerleştirin.

4. Webcam'inizin çalıştığından emin olun.

## Nasıl Çalışır?

1. **Yüz Tespiti:** Webcam'den alınan görüntü MediaPipe kullanılarak gerçek zamanlı olarak işlenir ve yüzler tespit edilir.
2. **Sınıflandırma:** Tespit edilen yüzler, **Cristiano Ronaldo** ile diğer kişileri ayırt eden önceden eğitilmiş CNN modeline gönderilir.
3. **Veri Gönderme:** Sınıflandırma sonucu (sınıf, skor ve zaman) Flask tabanlı arka uç sunucusuna gönderilir.
4. **Web Tablosu:** Sonuçlar web arayüzünde farklı renkler ve resimlerle gösterilir.

## Projeyi Çalıştırma

### 1. Webcam Sınıflandırıcısını Başlat

Gerçek zamanlı yüz sınıflandırması yapmak için `main.py` dosyasını çalıştırın.

```bash
python main.py
```

### 2. Flask Uygulamasını Başlat

Web arayüzünü çalıştırmak için Flask sunucusunu başlatın.

```bash
python app.py
```

### 3. Web Arayüzünü Görüntüleyin

Tarayıcınızdan `http://localhost:5000/veri-tablosu` adresine giderek canlı tespit sonuçlarını görüntüleyin.

## API Uç Noktaları

- **POST /api/save-data:** `main.py` dosyasından sınıf, skor ve tespit zamanını Flask sunucusuna göndermek için kullanılır.
  - Veri formatı:
    ```json
    {
      "class": "ronaldo",
      "score": "0.95",
      "time": "14:02:30 / 15-10-2024"
    }
    ```

## Web Arayüzü

Web arayüzü, her bir tespit için aşağıdaki bilgileri gösteren bir tablo içerir:

- **Sınıf**: Tespit edilen sınıf (ronaldo veya kişi).
- **Skor**: Sınıflandırmanın güven skoru.
- **Zaman**: Tespitin yapıldığı tarih ve saat.
- **Resim**: Tespit edilen sınıfın temsilci bir resmi.

## Resim Ekleme Özelliği

Her bir sınıf için (Ronaldo ve diğer kişi) belirli resimler kullanılarak web arayüzünde daha iyi bir görsel temsil sağlanır. Bu resimler:

- **Ronaldo:** Ronaldo'nun bir resmi (`ronaldo_image.png`).
- **Kişi:** Genel bir kişi resmi (`person_image.png`).

Bu resimler, sınıflandırma sonucuna göre ekranda değişir. Eğer bir yüz **Cristiano Ronaldo** olarak sınıflandırılırsa, Ronaldo'nun resmi gösterilir. Eğer yüz bir **kişi** olarak sınıflandırılırsa, kişiyi temsil eden genel bir resim gösterilir.

Resim dosyalarını `static/images` klasörüne yerleştirin ve bu dosyaların adlarının doğru olduğundan emin olun:

- `ronaldo_image.png`
- `person_image.png`

---
