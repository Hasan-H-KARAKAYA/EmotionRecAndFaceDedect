# Hasan Hüseyin KARAKAYA tarafından yazıldı.
# ATP 9/A 2512
# pip install -r requirements.txt

import cv2
import numpy as np
from keras.models import model_from_json

# Duygu etiketleri ve sayısal karşılıkları
emotion_dict = {0: "Kizgin", 1: "Nefret", 2: "Korkmus",
                3: "Mutlu", 4: "Dogal", 5: "Uzgun", 6: "Sasgin"}


# Duygu modelinin JSON dosyasını oku ve modeli yükle
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights("model/emotion_model.h5")
print("Model diskten yüklendi..")

# Kamera kaynağını başlat
# Ben telefonumun kamerasını kullandığım için 1 şeklinde kullandım.
# USB bir kamerada çalıştırmak için cap = cv2.VideoCapture(0) olarak kodu düzenleyin.
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    # Yüz tespit sınıflayıcısını yükle
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    # Kareyi gri tonlamaya dönüştür
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı kare üzerinde yüzleri tespit et
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

# Tespit edilen yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Duygu tahmini yap
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),  # Dikdörtgenin üzerine duyguyu yazdır
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# Sonuçları ekranda göster
    cv2.imshow('Duygu Tespiti', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
