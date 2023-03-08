import cv2
import numpy as np
import tensorflow as tf
import pyautogui

# TensorFlow Object Detection API'ı kullanarak önceden eğitilmiş bir modeli yükleme
model = tf.saved_model.load('C:/Users/s/Desktop/yolooo/yolov5')

# Ekran görüntüsü almak için pencere boyutlarını ayarlama
screen_size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, screen_size)

while True:
    # Ekran görüntüsü al
    img = pyautogui.screenshot()

    # Ekran görüntüsünü numpy dizisine dönüştür
    frame = np.array(img)

    # Görüntüyü boyutlandırma
    frame = cv2.resize(frame, (800, 600))

    # Görüntüyü TensorFlow Object Detection API'ı kullanarak işleme sokma
    output_dict = model(np.expand_dims(frame, axis=0))

    # Sonuçları görüntü üzerine işaretleyerek ekranda gösterme
    for i, score in enumerate(output_dict['detection_scores'][0]):
        if score > 0.5:
            box = output_dict['detection_boxes'][0][i]
            y1, x1, y2, x2 = box
            x1, y1, x2, y2 = int(x1*800), int(y1*600), int(x2*800), int(y2*600)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # İşaretlenmiş ekran görüntüsünü kaydetme ve ekranda gösterme
    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow("Screen Capture", frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) == ord("q"):
        break

# Kayıt işlemini durdurma ve pencereleri kapatma
out.release()
cv2.destroyAllWindows()
