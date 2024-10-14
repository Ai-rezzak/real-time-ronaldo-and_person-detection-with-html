from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Bellekte veri saklamak için liste
detections = []

# Her sınıf için renk ve resim dosyaları
renkler = {
    'person': '#00fffb',  # Perdon için yeşil
    'ronaldo': '#ff0880'  # Ronaldo için cyan
}

resimler = {
    'person': 'images/person.png',   # Happy sınıfı için resim
    'ronaldo': 'images/ron.png',       # Angry sınıfı için resim
    # Diğer sınıflar için de resim ekleyebilirsiniz
}

@app.route('/api/save-data', methods=['POST'])
def save_data():
    data = request.get_json()
    cls_name = data.get('class').lower()  # Küçük harfe dönüştür
    score = data.get('score')
    time_data = data.get('time')

    # Veriyi listeye ekle
    detections.append({'class': cls_name, 'score': score, 'time': time_data})
    
    print(f"Class: {cls_name}, Score: {score}, Time: {time_data}")
    
    return jsonify({"message": "Data received successfully!"}), 200

@app.route('/veri-tablosu')
def data_table():
    return render_template('table.html', veriler=detections, renkler=renkler, resimler=resimler)

@app.route('/')
def index():
    return "Hoş geldiniz! Bu uygulama çalışıyor."


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

