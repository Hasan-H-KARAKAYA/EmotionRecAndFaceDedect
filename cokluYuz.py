#OpenCV: bilgisayarla görü, makine öğrenimi, görüntü işleme, video analizi gibi uygulamalar için kullanılan devasa bir açık kaynak kodlu kütüphanedir. 
import cv2

# Cascade Classifier'ı yükle
#dijital bir resim ya da video karesi içerisinde bulunan belirli bir nesnenin tespit edilmesi amacıyla kullanılmaktadır.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




# Webcam başlatma
#Ben telefonumun kamerasını kullandığım için 1 şeklinde kullandım.
# USB bir kamerada çalıştırmak için cap = cv2.VideoCapture(0) olarak kodu düzenleyin.
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlamalı olarak görüntüyü yükle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti yap
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Tespit edilen yüzlerin sayısını yazdır
    print("Yüz Sayısı:", len(faces))

    # Yüzleri dikdörtgen içine al
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow('Webcam Yüz Tespiti', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
