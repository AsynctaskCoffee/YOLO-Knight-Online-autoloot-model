## YOLO Knight Online AutoLoot (Tkinter UI)

Bu repo artık tek dosyalık, çalıştırılabilir bir **Python masaüstü aracı** içerir:

- Ekranın belirlenen bölgesini tarar
- `best.pt` (veya seçtiğiniz başka `.pt`) ile tespit yapar
- Bulduğu nesnenin merkezine **sol / sağ / iki tık** (left/right/both) atar
- Gerçek zamanlı log ve önizleme penceresi sunar

> ⚠️ Uyarı: Oyun otomasyonu, bazı sunucularda kurallara aykırı olabilir. Sorumluluk kullanıcıya aittir.

---

## Kurulum

Python 3.10+ önerilir.

```bash
pip install ultralytics opencv-python pillow pyautogui numpy
```

---

## Çalıştırma

Proje klasöründe:

```bash
python autoloot_ui.py
```

Açılan arayüzde:

1. Model dosyasını seçin (`best.pt`)
2. Confidence, tarama aralığı ve ekran bölgesini ayarlayın
3. Tıklama modunu seçin (`left`, `right`, `both`)
4. `Başlat` düğmesine basın

---

## Özellikler

- **Model seçimi:** UI'dan `.pt` dosyası seçme
- **Confidence slider:** Tespit eşiğini canlı ayarlama
- **Bölge tarama:** `x1, y1, x2, y2` ile performans odaklı tarama
- **Click offset:** Merkez noktasına ofset ekleyerek daha doğru toplama
- **Left/Right/Both click:** Oyun davranışına göre tıklama stratejisi
- **Önizleme penceresi:** Tespit kutularını cv2 penceresinde görme
- **Log paneli:** Her adımı zaman damgasıyla takip etme

---

## Notlar / Sorun Giderme

- `pyautogui` bazı sistemlerde güvenlik izinleri ister (özellikle macOS).
- Oyun tam ekran çalışırken ekran yakalama API davranışı değişebilir.
- Çok sık tarama CPU/GPU kullanımını artırır; `Tarama Aralığı` değerini yükseltebilirsiniz.
- Hedef kutu yanlış tıklanıyorsa `Offset X/Y` ile ince ayar yapın.

---

## Dosyalar

- `autoloot_ui.py`: Tüm uygulama (UI + YOLO inference + tıklama akışı)
- `best.pt`: Örnek model dosyası
