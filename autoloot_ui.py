"""YOLO tabanlı otomatik toplama aracı (Tkinter arayüzlü).

Özellikler:
- Ekran görüntüsünü alıp YOLO ile nesne tespiti
- Eşik skoru / tarama gecikmesi ayarı
- Sol/Sağ/Tekrar tık kombinasyonu
- Algılanan kutuları önizleme penceresinde gösterme
- Koordinat ofsetleri ve ekran bölgesi tanımlama

Not: Oyun otomasyonları sunucu kurallarını ihlal edebilir.
Kullanmadan önce ilgili oyunun kullanım şartlarını kontrol edin.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


@dataclass
class DetectionConfig:
    model_path: str = "best.pt"
    confidence: float = 0.5
    interval_sec: float = 0.15
    region_x1: int = 0
    region_y1: int = 0
    region_x2: int = 1920
    region_y2: int = 1080
    click_mode: str = "left"  # left | right | both
    second_click_delay: float = 0.04
    click_offset_x: int = 0
    click_offset_y: int = 0
    preview: bool = True


class AutoLootApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO AutoLoot - Tkinter UI")
        self.root.geometry("860x620")

        self.config = DetectionConfig()
        self.model: Optional[YOLO] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.running = False

        self._build_ui()
        self._set_defaults()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="YOLO Knight Online AutoLoot", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        # Model & çalışma ayarları
        cfg_frame = ttk.LabelFrame(main, text="Model ve Çalışma Ayarları", padding=10)
        cfg_frame.pack(fill="x", pady=6)

        self.model_var = tk.StringVar()
        self.conf_var = tk.DoubleVar()
        self.interval_var = tk.DoubleVar()
        self.preview_var = tk.BooleanVar()

        ttk.Label(cfg_frame, text="Model (.pt)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(cfg_frame, textvariable=self.model_var, width=60).grid(row=0, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(cfg_frame, text="Seç", command=self._choose_model).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(cfg_frame, text="Confidence").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Scale(cfg_frame, from_=0.05, to=0.99, variable=self.conf_var, orient="horizontal").grid(
            row=1, column=1, sticky="we", padx=4, pady=4
        )
        self.conf_label = ttk.Label(cfg_frame, text="0.50")
        self.conf_label.grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(cfg_frame, text="Tarama Aralığı (sn)").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(cfg_frame, textvariable=self.interval_var, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(cfg_frame, text="Önizleme penceresi (cv2)", variable=self.preview_var).grid(
            row=2, column=2, sticky="w", padx=4, pady=4
        )

        cfg_frame.columnconfigure(1, weight=1)

        # Bölge ayarları
        region_frame = ttk.LabelFrame(main, text="Ekran Bölgesi", padding=10)
        region_frame.pack(fill="x", pady=6)

        self.x1_var = tk.IntVar()
        self.y1_var = tk.IntVar()
        self.x2_var = tk.IntVar()
        self.y2_var = tk.IntVar()

        ttk.Label(region_frame, text="x1").grid(row=0, column=0, padx=4, pady=4)
        ttk.Entry(region_frame, textvariable=self.x1_var, width=8).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(region_frame, text="y1").grid(row=0, column=2, padx=4, pady=4)
        ttk.Entry(region_frame, textvariable=self.y1_var, width=8).grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(region_frame, text="x2").grid(row=0, column=4, padx=4, pady=4)
        ttk.Entry(region_frame, textvariable=self.x2_var, width=8).grid(row=0, column=5, padx=4, pady=4)
        ttk.Label(region_frame, text="y2").grid(row=0, column=6, padx=4, pady=4)
        ttk.Entry(region_frame, textvariable=self.y2_var, width=8).grid(row=0, column=7, padx=4, pady=4)

        ttk.Button(region_frame, text="Tam Ekran Değerlerini Doldur", command=self._fill_screen_size).grid(
            row=0, column=8, padx=8, pady=4
        )

        # Tıklama ayarları
        click_frame = ttk.LabelFrame(main, text="Tıklama / Toplama Ayarları", padding=10)
        click_frame.pack(fill="x", pady=6)

        self.click_mode_var = tk.StringVar()
        self.second_delay_var = tk.DoubleVar()
        self.off_x_var = tk.IntVar()
        self.off_y_var = tk.IntVar()

        ttk.Label(click_frame, text="Tıklama Modu").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Combobox(
            click_frame,
            textvariable=self.click_mode_var,
            values=["left", "right", "both"],
            width=10,
            state="readonly",
        ).grid(row=0, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(click_frame, text="2. Tık Gecikmesi (both)").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(click_frame, textvariable=self.second_delay_var, width=10).grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(click_frame, text="Offset X").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(click_frame, textvariable=self.off_x_var, width=10).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(click_frame, text="Offset Y").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(click_frame, textvariable=self.off_y_var, width=10).grid(row=1, column=3, padx=4, pady=4, sticky="w")

        # Kontrol butonları
        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(controls, text="Başlat", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(controls, text="Durdur", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(controls, text="Kaydet (Ayarları Uygula)", command=self._sync_config_from_ui).pack(side="left", padx=4)

        # Log alanı
        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True, pady=8)
        self.log_text = tk.Text(log_frame, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        self.conf_var.trace_add("write", self._on_conf_change)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_defaults(self) -> None:
        self.model_var.set(self.config.model_path)
        self.conf_var.set(self.config.confidence)
        self.interval_var.set(self.config.interval_sec)
        self.preview_var.set(self.config.preview)

        self.x1_var.set(self.config.region_x1)
        self.y1_var.set(self.config.region_y1)
        self.x2_var.set(self.config.region_x2)
        self.y2_var.set(self.config.region_y2)

        self.click_mode_var.set(self.config.click_mode)
        self.second_delay_var.set(self.config.second_click_delay)
        self.off_x_var.set(self.config.click_offset_x)
        self.off_y_var.set(self.config.click_offset_y)

        self._on_conf_change()

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(title="YOLO model seç", filetypes=[("PyTorch", "*.pt")])
        if path:
            self.model_var.set(path)

    def _on_conf_change(self, *_args) -> None:
        self.conf_label.configure(text=f"{self.conf_var.get():.2f}")

    def _fill_screen_size(self) -> None:
        w, h = pyautogui.size()
        self.x1_var.set(0)
        self.y1_var.set(0)
        self.x2_var.set(int(w))
        self.y2_var.set(int(h))
        self._log(f"Ekran boyutu algılandı: {w}x{h}")

    def _sync_config_from_ui(self) -> bool:
        try:
            model_path = self.model_var.get().strip()
            if not model_path:
                raise ValueError("Model yolu boş olamaz.")
            if not Path(model_path).exists():
                raise ValueError(f"Model dosyası bulunamadı: {model_path}")

            x1, y1, x2, y2 = self.x1_var.get(), self.y1_var.get(), self.x2_var.get(), self.y2_var.get()
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Bölge değeri hatalı: x2>x1 ve y2>y1 olmalı.")

            interval = float(self.interval_var.get())
            second_delay = float(self.second_delay_var.get())
            conf = float(self.conf_var.get())
            if interval < 0.01:
                raise ValueError("Tarama aralığı en az 0.01 sn olmalı.")
            if not 0.01 <= conf <= 0.99:
                raise ValueError("Confidence 0.01 ile 0.99 arasında olmalı.")

            self.config = DetectionConfig(
                model_path=model_path,
                confidence=conf,
                interval_sec=interval,
                region_x1=int(x1),
                region_y1=int(y1),
                region_x2=int(x2),
                region_y2=int(y2),
                click_mode=self.click_mode_var.get(),
                second_click_delay=max(0.0, second_delay),
                click_offset_x=int(self.off_x_var.get()),
                click_offset_y=int(self.off_y_var.get()),
                preview=bool(self.preview_var.get()),
            )
            self._log("Ayarlar güncellendi.")
            return True
        except Exception as exc:  # GUI tarafı için kullanıcı dostu hata
            messagebox.showerror("Ayar Hatası", str(exc))
            self._log(f"Ayar hatası: {exc}")
            return False

    def _load_model(self) -> bool:
        try:
            if self.model is None or Path(self.config.model_path).name != Path(getattr(self.model, "ckpt_path", "")).name:
                self._log(f"Model yükleniyor: {self.config.model_path}")
                self.model = YOLO(self.config.model_path)
            return True
        except Exception as exc:
            messagebox.showerror("Model Yükleme Hatası", str(exc))
            self._log(f"Model yüklenemedi: {exc}")
            return False

    def start(self) -> None:
        if self.running:
            return
        if not self._sync_config_from_ui():
            return
        if not self._load_model():
            return

        self.stop_event.clear()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()
        self._log("Tarama başlatıldı.")

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._log("Durdurma sinyali gönderildi.")

    def _capture_region(self) -> np.ndarray:
        bbox = (
            self.config.region_x1,
            self.config.region_y1,
            self.config.region_x2,
            self.config.region_y2,
        )
        return np.array(ImageGrab.grab(bbox=bbox))

    def _find_best_box_center(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, float]]:
        assert self.model is not None

        best: Optional[Tuple[int, int, float]] = None
        results = self.model(frame_bgr, verbose=False)
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                if score < self.config.confidence:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if best is None or score > best[2]:
                    best = (cx, cy, score)

                if self.config.preview:
                    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        f"{score:.2f}",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        if self.config.preview:
            cv2.imshow("YOLO AutoLoot Preview", frame_bgr)
            cv2.waitKey(1)

        return best

    def _perform_click(self, x: int, y: int) -> None:
        screen_x = self.config.region_x1 + x + self.config.click_offset_x
        screen_y = self.config.region_y1 + y + self.config.click_offset_y

        if self.config.click_mode == "left":
            pyautogui.click(screen_x, screen_y, button="left")
        elif self.config.click_mode == "right":
            pyautogui.click(screen_x, screen_y, button="right")
        else:  # both
            pyautogui.click(screen_x, screen_y, button="left")
            time.sleep(self.config.second_click_delay)
            pyautogui.click(screen_x, screen_y, button="right")

        self._log(f"Tıklama: ({screen_x}, {screen_y}) | mod={self.config.click_mode}")

    def _run_loop(self) -> None:
        assert self.model is not None
        try:
            while not self.stop_event.is_set():
                img_rgb = self._capture_region()
                frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                best = self._find_best_box_center(frame)

                if best is not None:
                    x, y, score = best
                    self._log(f"Nesne bulundu: merkez=({x}, {y}) skor={score:.3f}")
                    self._perform_click(x, y)
                else:
                    self._log("Nesne bulunamadı.")

                time.sleep(self.config.interval_sec)
        except Exception as exc:
            self._log(f"Çalışma hatası: {exc}")
            messagebox.showerror("Çalışma Hatası", str(exc))
        finally:
            self.running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            cv2.destroyAllWindows()
            self._log("Tarama sonlandı.")

    def _on_close(self) -> None:
        self.stop()
        self.root.after(150, self.root.destroy)

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {message}\n")
        self.log_text.see("end")


def main() -> None:
    root = tk.Tk()
    app = AutoLootApp(root)
    app._log("Hazır. Ayarları kontrol edip Başlat'a basın.")
    root.mainloop()


if __name__ == "__main__":
    main()
