[![Videoyu İzle: Knight Online AutoLoot Demo](https://img.youtube.com/vi/KATgKl8bgaQ/hqdefault.jpg)](https://youtu.be/KATgKl8bgaQ?si=Yh4QmymAisHCRc9C)

## YOLO Knight Online AutoLoot (Tkinter UI)

Bu projeyi sıfırdan, sade bir dille anlatalım.

Bu araç şunu yapar:
1. Ekrandaki görüntüyü alır.
2. Bu görüntüde modelin öğretilmiş olduğu nesneleri arar.
3. Bulduğu nesnenin koordinatını çıkarır.
4. Seçtiğin moda göre oraya sol/sağ tık atar.

Kısaca: **“Ekrana bak + hedefi bul + tıkla”**.

---

## YOLO nedir? (Uzun ve net anlatım)

YOLO = **You Only Look Once**.

Bu bir nesne tespit yaklaşımıdır. Yani sadece “resimde bir şey var mı?” demez; aynı zamanda **resimde nerede** olduğunu da söyler.

Örnek:
- Sınıflandırma modeli: “Bu fotoğrafta kedi var.”
- YOLO gibi tespit modeli: “Bu fotoğrafta kedi var ve şu kutunun içinde (x1,y1,x2,y2).”

Bu projede model (`best.pt`) ekrandan alınan görüntüde loot/hedef olarak eğitilmiş nesneleri yakalar.

YOLO neden tercih edilir?
- Hızlıdır (gerçek zamanlıya uygundur).
- Tek geçişte (single-pass) tespit yapar.
- Ekran akışında tekrar tekrar çalıştırması kolaydır.

Bu repo’daki akışta YOLO’nun görevi şudur:
- Görüntüde kutuları üretmek.
- Her kutu için bir güven skoru vermek (`confidence`).
- Eşik üstündeki kutulardan en mantıklısını seçmek (kodda en yüksek skorlu olan seçiliyor).

---

## Bu proje teknik olarak nasıl çalışıyor?

`autoloot_ui.py` dosyası tek başına bütün sistemi yönetir.

### 1) Arayüz (Tkinter)
Arayüzden şunları ayarlarsın:
- Model dosyası (`.pt`)
- Confidence (tespit eşiği)
- Tarama aralığı (kaç saniyede bir tarasın)
- Ekran bölgesi (`x1, y1, x2, y2`)
- Tıklama modu (`left`, `right`, `both`)
- Offset (`X/Y`) ve ikinci tık gecikmesi
- Önizleme penceresi açık/kapalı

### 2) Ekran görüntüsü alma
Kod belirlediğin ekran alanını alır (`ImageGrab`).

### 3) YOLO ile tespit
Alınan görüntü YOLO modeline verilir.
- Kutu koordinatları çıkar.
- Her kutunun confidence değeri alınır.
- Eşik altı elenir.
- En yüksek skorlu kutu seçilir.

### 4) Tıklama
Seçilen kutunun merkez noktası alınır.
- `left`: sol tık
- `right`: sağ tık
- `both`: önce sol sonra (kısa gecikmeyle) sağ tık

Offset varsa bu merkeze eklenir.

### 5) Döngü
Bu işlem, verdiğin tarama aralığına göre sürekli tekrar eder.

---

## Kodun kısa sade açıklaması (dosya içinde ne var?)

- `DetectionConfig`: Bütün ayarları tek yerde tutan yapı.
- `AutoLootApp`: Uygulamanın ana sınıfı.
  - `_build_ui`: Ekrandaki buton/alanları oluşturur.
  - `_sync_config_from_ui`: Girilen değerleri kontrol eder, hatalıysa kullanıcıya söyler.
  - `_load_model`: YOLO modelini yükler.
  - `_capture_region`: Ekran alanını görüntü olarak alır.
  - `_find_best_box_center`: Kutular içinde en iyi adayı bulur.
  - `_perform_click`: Seçilen noktaya tıklar.
  - `_run_loop`: Tüm bu adımları thread içinde sürekli döndürür.

Yani kod da sade olarak şu mantıkta:
**Ayarları al → modeli yükle → ekranı tara → hedefi bul → tıkla → bekle → tekrar et.**

---

## Kurulum

Python 3.10+ önerilir.

```bash
pip install ultralytics opencv-python pillow pyautogui numpy
```

> Not: `pyautogui` bazı sistemlerde güvenlik izni ister.

---

## Çalıştırma

```bash
python autoloot_ui.py
```

Açılan arayüzde adım adım:
1. `best.pt` (veya başka `.pt`) seç.
2. Confidence değerini ayarla (örn. 0.45–0.65 arası başlayabilirsin).
3. Ekran bölgesini gir (mümkünse tüm ekran yerine sadece oyun alanı).
4. Tıklama modunu seç.
5. Gerekirse offset ver.
6. `Başlat` de.

---

## Ayarları nasıl düşünmelisin? (Pratik rehber)

### Confidence
- Çok düşükse: yanlış hedeflere tıklayabilir.
- Çok yüksekse: gerçek hedefleri kaçırabilir.
- Genelde orta seviye başlayıp testle ayarlamak en doğrusu.

### Tarama Aralığı
- Çok düşük (çok sık): daha hızlı tepki ama daha fazla sistem yükü.
- Daha yüksek: daha az yük, daha yavaş tepki.

### Bölge (x1,y1,x2,y2)
- Ne kadar küçük ve doğru alan seçersen performans o kadar iyi olur.
- Tüm ekran taramak en ağır seçenektir.

### Offset
- Model kutunun merkezini verir ama oyunda gerçek tıklanacak nokta merkezden biraz kayık olabilir.
- Offset ile bu farkı düzeltirsin.

---

## Sık sorunlar

### “Hiç tıklamıyor”
- Model gerçekten doğru nesneleri görüyor mu? (Önizleme aç.)
- Confidence çok yüksek olabilir.
- Ekran bölgesi yanlış olabilir.

### “Yanlış yere tıklıyor”
- Offset ayarla.
- Model eğitim kalitesini gözden geçir.
- `both` yerine tek modla test et.

### “Donuyor / yavaş”
- Bölgeyi küçült.
- Tarama aralığını artır.
- Önizlemeyi kapat.

---

## Güvenlik ve sorumluluk notu

- Bu bir otomasyon aracıdır.
- Oyuna/sunucuya göre kullanım kuralları farklıdır.
- Hesap riski doğurabilecek ortamlarda kullanmadan önce kuralları kontrol et.
- Sorumluluk kullanıcıdadır.

---

## Dosyalar

- `autoloot_ui.py`: Uygulamanın tamamı (UI + YOLO inference + tıklama)
- `best.pt`: Örnek model dosyası
