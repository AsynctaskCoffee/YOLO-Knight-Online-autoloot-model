## Kedikodu

```python
import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO

# YOLO modelini 'best.pt' dosyasından yükle
model = YOLO('best.pt')

def get_center_half_window_image():
    # Pencerenin ekran görüntüsünü al ve numpy array'ine dönüştür
    bbox = (0, 0, 1920, 1080)  # Tüm ekranı yakalar
    img = np.array(ImageGrab.grab(bbox=bbox))
    return img

def main():
    while True:
        img = get_center_half_window_image()
        if img is not None:
            img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = model(img_cv2, stream=True)
            
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        print(f"Koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow("YOLO Detection", img_cv2)
            
            # 'q' tuşuna basıldığında döngüyü kır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

Şimdi bu kodun her satırını detaylı bir şekilde açıklayalım:

```python
import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO
```
Bu satırlar gerekli kütüphaneleri içe aktarıyor:
- `cv2`: OpenCV kütüphanesi, görüntü işleme için kullanılır.
- `numpy`: Numpy kütüphanesi, sayısal işlemler ve array manipülasyonları için kullanılır.
- `ImageGrab`: PIL kütüphanesinden, ekran görüntüsü almak için kullanılır.
- `YOLO`: YOLO (You Only Look Once) modelini içeren ultralytics kütüphanesi.

```python
# YOLO modelini 'best.pt' dosyasından yükle
model = YOLO('best.pt')
```
Bu satır, YOLO modelini 'best.pt' dosyasından yükler.

```python
def get_center_half_window_image():
    # Pencerenin ekran görüntüsünü al ve numpy array'ine dönüştür
    bbox = (0, 0, 1920, 1080)  # Tüm ekranı yakalar
    img = np.array(ImageGrab.grab(bbox=bbox))
    return img
```
Bu fonksiyon, ekranın belirtilen `bbox` (bounding box) alanından bir ekran görüntüsü alır ve bunu numpy array'ine dönüştürür. Burada `bbox` tüm ekranı kapsar (1920x1080 çözünürlük).

```python
def main():
```
Ana fonksiyonun başlangıcı.

```python
    while True:
```
Sonsuz bir döngü oluşturur. Bu döngü, görüntülerin sürekli olarak işlenmesini sağlar.

```python
        img = get_center_half_window_image()
```
Ekrandan bir görüntü alır.

```python
        if img is not None:
```
Eğer bir görüntü alındıysa (boş değilse) işlemeye devam eder.

```python
            img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```
Görüntüyü RGB renk uzayından BGR renk uzayına dönüştürür. OpenCV, görüntüleri BGR formatında işler.

```python
            results = model(img_cv2, stream=True)
```
YOLO modelini kullanarak görüntüdeki nesneleri tespit eder ve sonuçları `results` değişkenine atar.

```python
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        print(f"Koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
```
Tespit edilen her bir nesne için:
- Eğer nesne kutuları varsa (`r.boxes`):
  - Her bir kutu için koordinatları `x1, y1, x2, y2` olarak alır.
  - Koordinatları konsola yazdırır.
  - Kutunun etrafına yeşil bir dikdörtgen çizer (`cv2.rectangle`).

```python
            # Görüntüyü göster
            cv2.imshow("YOLO Detection", img_cv2)
```
Güncellenmiş görüntüyü bir pencerede gösterir.

```python
            # 'q' tuşuna basıldığında döngüyü kır
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
```
Eğer kullanıcı 'q' tuşuna basarsa, döngü kırılır ve program sonlanır.

```python
    cv2.destroyAllWindows()
```
Tüm OpenCV pencerelerini kapatır.


## BULUNAN KORDİNATLARA TIKLAMA VE TOPLAMA İŞLEMLERİ SERVERDAN SERVERA DEĞİŞEBİLİR. YUKARIDAKİ KOD SADECE OYUNDA KUTUNUN KORDİNATLARINI DÖNER :) İYİ ÇALIŞMALAR :rocket:
