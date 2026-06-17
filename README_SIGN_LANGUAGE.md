# Panduan Sistem Penerjemah Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia)

Sistem ini adalah implementasi dari rancangan Tugas Akhir: **"Pengembangan Sistem Pengenalan Bahasa Isyarat Berbasis Computer Vision dan Machine Learning Menggunakan CNN dan MediaPipe"**.

Sistem ini dikhususkan secara eksklusif untuk **SIBI** karena menggunakan isyarat satu tangan (tangan kanan) untuk mengeja huruf A-Z, yang sangat ideal untuk pelacakan *real-time* satu kamera menggunakan MediaPipe Hands.

---

## 📂 Struktur Berkas
- **`requirements.txt`**: Daftar pustaka Python yang diperlukan (OpenCV, MediaPipe, pyttsx3).
- **`train.py`**: Skrip pelatihan model CNN (dapat dilatih menggunakan dataset gambar SIBI).
- **`main.py`**: Aplikasi utama penerjemah *real-time* via webcam dengan visualisasi HUD, stabilisasi buffer SIBI, dan suara TTS.
- **`rancangan_tugas_akhir.md`**: Dokumen rancangan proposal TA lengkap terfokus secara eksklusif pada SIBI.

---

## 🛠️ Instalasi & Persiapan

### 1. Instalasi Python & Library
Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi SIBI Real-Time
Jalankan program penerjemah:

```bash
python main.py
```

*Catatan: Saat pertama kali dijalankan, sistem akan mengunduh model pelacak tangan resmi MediaPipe `hand_landmarker.task` (~5.6 MB) dari server Google ke direktori proyek secara otomatis.*

---

## 🎮 Gestur Abjad SIBI yang Didukung (Mode Geometris)
Sistem berjalan menggunakan **Mode Geometris Sendi SIBI** yang sangat akurat mendeteksi abjad jari SIBI berikut sesuai dengan petunjuk visual resmi:

1. **`A`** $\rightarrow$ Mengepalkan tangan rapat (fist), ibu jari rapat di samping jari telunjuk.
2. **`B`** $\rightarrow$ Telapak tangan terbuka lebar menghadap kamera, ibu jari ditekuk ke arah telapak.
3. **`D`** $\rightarrow$ Hanya jari telunjuk yang tegak lurus ke atas.
4. **`F`** $\rightarrow$ Jari telunjuk, tengah, dan manis tegak sejajar **rapat** ke atas, kelingking ditekuk.
5. **`G`** $\rightarrow$ Jari telunjuk dan jempol direntangkan secara **horizontal (menyamping)**, lainnya ditekuk.
6. **`H`** $\rightarrow$ Jari telunjuk dan tengah direntangkan secara **horizontal (menyamping)**, lainnya ditekuk.
7. **`I`** $\rightarrow$ Hanya jari kelingking yang tegak ke atas.
8. **`K`** $\rightarrow$ Jari telunjuk dan tengah tegak membentuk huruf V dengan jempol melintang tegak di depannya.
9. **`L`** $\rightarrow$ Jempol dan telunjuk terbuka lebar membentuk sudut siku 90 derajat (L shape).
10. **`R`** $\rightarrow$ Jari telunjuk dan tengah berdiri tegak dalam posisi **saling menyilang (crossed)**.
11. **`U`** $\rightarrow$ Jari telunjuk dan tengah tegak lurus ke atas dalam posisi **rapat berdampingan**.
12. **`V`** $\rightarrow$ Jari telunjuk dan tengah tegak lurus ke atas dalam posisi **membuka renggang (V)**.
13. **`W`** $\rightarrow$ Jari telunjuk, tengah, dan manis tegak lurus dalam posisi **membuka renggang (W)**.
14. **`X`** $\rightarrow$ Jari telunjuk berdiri dalam posisi **menekuk/melengkung (hooked)** seperti kail.
15. **`Y`** $\rightarrow$ Hanya jempol dan kelingking yang terbuka lebar ke arah samping (Shaka/hang loose).

---

## 🎹 Kontrol Keyboard pada Aplikasi
* **`[SPASI] (Spacebar)`**: Mengonfirmasi kata saat ini menjadi suara (Text-to-Speech) bahasa Indonesia dan mereset buffer kata.
* **`[BACKSPACE]`**: Menghapus huruf SIBI terakhir pada kata.
* **`[C] / [c]`**: Mengosongkan kata di layar (*Clear*).
* **`[Q] / [q]`**: Keluar dari aplikasi.
