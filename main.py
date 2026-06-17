import os
import cv2
import numpy as np
import threading
import urllib.request
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. TENSORFLOW DETECTION & OPTIONAL IMPORT
# ==========================================
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    print("\n[SISTEM INFO] TensorFlow tidak terdeteksi di lingkungan Python Anda (umum terjadi pada versi Python terbaru seperti 3.14).")
    print("[SISTEM INFO] Mengaktifkan mode deteksi cerdas abjad SIBI (Sistem Isyarat Bahasa Indonesia) berbasis koordinat MediaPipe!\n")

# ==========================================
# 2. TEXT-TO-SPEECH (TTS) BACKGROUND THREAD
# ==========================================
class TextToSpeechThread(threading.Thread):
    """
    Thread khusus untuk menjalankan Text-to-Speech (TTS) agar
    tidak memblokir (freeze) frame OpenCV saat mengucapkan kata.
    """
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.daemon = True

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 140) # Laju bicara
            engine.setProperty('volume', 1.0)
            engine.say(self.text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR TTS] Gagal memproses suara: {e}")

# ==========================================
# 3. SIBI GEOMETRIC HAND GESTURE CLASSIFIER
# ==========================================
class SibiGestureModel:
    """
    Model klasifikasi berbasis aturan koordinat sendi tangan (geometric hand rules) yang invarian skala.
    Disesuaikan secara khusus berdasarkan standar abjad SIBI (Sistem Isyarat Bahasa Indonesia)
    dan dilengkapi deteksi gerakan dinamis berbasis lintasan untuk huruf J dan Z.
    """
    def __init__(self):
        self.index_path = [] # Menyimpan (x, y) ujung telunjuk untuk Z
        self.pinky_path = [] # Menyimpan (x, y) ujung kelingking untuk J
        
    def clear_history(self):
        self.index_path.clear()
        self.pinky_path.clear()

    def predict_gesture(self, hand_landmarks):
        def dist(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
        # Ujung Jari (Tips)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]
        
        # Sendi Pembanding (MCP/PIP/DIP)
        thumb_ip = hand_landmarks[3]
        index_mcp = hand_landmarks[5]
        index_pip = hand_landmarks[6]
        middle_mcp = hand_landmarks[9]
        middle_pip = hand_landmarks[10]
        ring_mcp = hand_landmarks[13]
        ring_pip = hand_landmarks[14]
        pinky_mcp = hand_landmarks[17]
        pinky_pip = hand_landmarks[18]
        
        # Lebar telapak tangan sebagai referensi skala invarian
        palm_width = dist(index_mcp, pinky_mcp)
        if palm_width < 0.01:
            palm_width = 0.01

        # Hitung rasio ekstensi jari (1.0 = lurus, <0.5 = mengepal erat)
        def finger_extension(tip, pip, mcp):
            s_len = dist(tip, pip) + dist(pip, mcp)
            a_len = dist(tip, mcp)
            return a_len / s_len if s_len > 0 else 0.0

        index_ext = finger_extension(index_tip, index_pip, index_mcp)
        middle_ext = finger_extension(middle_tip, middle_pip, middle_mcp)
        ring_ext = finger_extension(ring_tip, ring_pip, ring_mcp)
        pinky_ext = finger_extension(pinky_tip, pinky_pip, pinky_mcp)

        # Status Terbuka/Tertutup (Boolean)
        index_open = index_ext > 0.78
        middle_open = middle_ext > 0.78
        ring_open = ring_ext > 0.78
        pinky_open = pinky_ext > 0.78

        index_closed = index_ext < 0.35
        middle_closed = middle_ext < 0.35
        ring_closed = ring_ext < 0.35
        pinky_closed = pinky_ext < 0.35

        # Jarak relatif invarian skala
        dist_thumb_index = dist(thumb_tip, index_tip) / palm_width
        dist_index_middle = dist(index_tip, middle_tip) / palm_width
        dist_middle_ring = dist(middle_tip, ring_tip) / palm_width
        dist_thumb_middle = dist(thumb_tip, middle_pip) / palm_width
        dist_thumb_mcp = dist(thumb_tip, index_mcp) / palm_width

        # Arah kemiringan jari (Deteksi Horizontal untuk G dan H)
        index_horizontal = index_open and abs(index_tip.x - index_mcp.x) > abs(index_tip.y - index_mcp.y) * 1.1
        middle_horizontal = middle_open and abs(middle_tip.x - middle_mcp.x) > abs(middle_tip.y - middle_mcp.y) * 1.1

        # Cek persilangan jari telunjuk & tengah (Deteksi R secara matematika - invarian rotasi)
        # Proyeksikan vektor dari middle_tip ke index_tip pada vektor telapak (Index MCP ke Pinky MCP)
        dot_crossed = (index_tip.x - middle_tip.x) * (pinky_mcp.x - index_mcp.x) + (index_tip.y - middle_tip.y) * (pinky_mcp.y - index_mcp.y)
        is_crossed = dot_crossed > -0.002

        # ==========================================
        # DETEKSI GERAKAN DINAMIS (J & Z)
        # ==========================================
        # 1. Update lintasan
        if index_open and not middle_open and not ring_open and not pinky_open:
            self.index_path.append((index_tip.x, index_tip.y))
            if len(self.index_path) > 25:
                self.index_path.pop(0)
        else:
            self.index_path.clear()

        if pinky_open and not index_open and not middle_open and not ring_open:
            self.pinky_path.append((pinky_tip.x, pinky_tip.y))
            if len(self.pinky_path) > 25:
                self.pinky_path.pop(0)
        else:
            self.pinky_path.clear()

        # 2. Cek Lintasan Z (Gerak ujung telunjuk kanan-kiri-kanan)
        if len(self.index_path) >= 12:
            xs = [p[0] for p in self.index_path]
            ys = [p[1] for p in self.index_path]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            # Gerakan harus cukup signifikan
            if x_range > 0.08 and y_range > 0.08:
                idx_max = np.argmax(xs)
                # Cari min setelah max agar urutan kanan -> kiri benar
                idx_min = idx_max + np.argmin(xs[idx_max:])
                
                # Pola Z: Kanan (max_x) terjadi sebelum Kiri (min_x)
                if 2 < idx_max < idx_min < len(xs) - 2:
                    move1_right = xs[idx_max] - xs[0] > 0.03
                    move2_left = xs[idx_max] - xs[idx_min] > 0.04
                    move3_right = xs[-1] - xs[idx_min] > 0.03
                    move_down = ys[idx_min] - ys[idx_max] > 0.03
                    
                    if move1_right and move2_left and move3_right and move_down:
                        self.index_path.clear()
                        return "Z", 0.99

        # 3. Cek Lintasan J (Gerak ujung kelingking melengkung ke bawah lalu naik-kiri)
        if len(self.pinky_path) >= 12:
            xs = [p[0] for p in self.pinky_path]
            ys = [p[1] for p in self.pinky_path]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            if x_range > 0.06 and y_range > 0.08:
                idx_max_y = np.argmax(ys) # Titik terbawah lengkungan J
                
                if 2 < idx_max_y < len(ys) - 2:
                    downward = ys[idx_max_y] - ys[0] > 0.04
                    upward = ys[idx_max_y] - ys[-1] > 0.03
                    curve_left = xs[idx_max_y] - xs[-1] > 0.02
                    
                    if downward and upward and curve_left:
                        self.pinky_path.clear()
                        return "J", 0.99

        # ==========================================
        # PENCOCOKAN ATURAN GESTUR ALFABET SIBI
        # ==========================================
        
        # B: Semua jari tegak terbuka lebar
        if index_open and middle_open and ring_open and pinky_open:
            return "B", 0.98

        # F: Jari telunjuk ditekuk menyentuh jempol, tiga jari lainnya tegak sejajar ke atas
        elif middle_open and ring_open and pinky_open and not index_open and dist_thumb_index < 0.35:
            return "F", 0.98
            
        # W: Jari telunjuk, tengah, dan manis tegak lurus terbuka, kelingking tertekuk
        elif index_open and middle_open and ring_open and not pinky_open:
            return "W", 0.97
            
        # H: Jari Telunjuk dan Tengah direntangkan sejajar secara HORIZONTAL (menyamping)
        elif index_horizontal and middle_horizontal and not ring_open and not pinky_open:
            return "H", 0.96
            
        # G: Jari Telunjuk direntangkan secara HORIZONTAL (menyamping), jempol terbuka, lainnya ditekuk
        elif index_horizontal and not middle_open and not ring_open and not pinky_open:
            return "G", 0.96
            
        # R: Jari Telunjuk dan Tengah berdiri tegak menyilang (crossed)
        elif index_open and middle_open and not ring_open and not pinky_open and is_crossed:
            return "R", 0.97
            
        # Y: Jempol dan Kelingking membuka lebar ke samping (Shaka sign)
        elif pinky_open and not index_open and not middle_open and not ring_open and dist_thumb_mcp > 0.48:
            return "Y", 0.98
            
        # L: Jempol dan Telunjuk membuka membentuk sudut siku (L shape)
        elif index_open and not middle_open and not ring_open and not pinky_open and dist_thumb_mcp > 0.48:
            return "L", 0.99
            
        # K: Telunjuk & Tengah terbuka (V shape), Jempol tegak di depan jari tengah
        elif index_open and middle_open and not ring_open and not pinky_open and dist_thumb_middle < 0.5:
            return "K", 0.95
            
        # U & V: Deteksi jari telunjuk dan tengah tegak, lainnya tertekuk ke bawah
        elif index_open and middle_open and not ring_open and not pinky_open:
            # U: Rapat berdampingan
            if dist_index_middle < 0.35:
                return "U", 0.96
            # V: Renggang membuka (peace sign)
            else:
                return "V", 0.96

        # P: Jari telunjuk & tengah mengarah ke bawah, jempol menempel di tengah
        elif index_ext > 0.75 and middle_ext > 0.75 and index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y and not ring_open and not pinky_open and dist_thumb_middle < 0.5:
            return "P", 0.95

        # Q: Jari telunjuk dan jempol mengarah ke bawah, lainnya tertekuk rapat
        elif index_ext > 0.75 and index_tip.y > index_mcp.y and not middle_open and not ring_open and not pinky_open and dist_thumb_index < 0.4:
            return "Q", 0.95
                
        # D & X: Hanya Jari Telunjuk yang aktif (tidak mengepal), lainnya mengepal erat
        elif not index_closed and middle_closed and ring_closed and pinky_closed:
            # X: Telunjuk menekuk/melengkung (hooked)
            if index_ext <= 0.78:
                return "X", 0.95
            # D: Telunjuk tegak lurus sempurna
            else:
                return "D", 0.95
                
        # I & J: Hanya Kelingking tegak
        elif pinky_open and not index_open and not middle_open and not ring_open:
            # J: Kelingking ditekuk melengkung / miring menyamping saat membentuk pola J (jika deteksi dinamis terlewati)
            if abs(pinky_tip.x - pinky_pip.x) / palm_width > 0.2:
                return "J", 0.95
            else:
                return "I", 0.96

        # ========================================================
        # KLASIFIKASI KEPALAN & BENTUK RAPAT (C, O, A, E, M, N, S, T)
        # ========================================================
        # 1. C & O: Telapak melengkung (jari tidak terbuka penuh tapi tidak mengepal erat)
        elif (0.35 <= index_ext < 0.78) and (0.35 <= middle_ext < 0.78) and \
             (0.35 <= ring_ext < 0.78) and (0.35 <= pinky_ext < 0.78):
            # O: Membentuk lingkaran rapat (ujung telunjuk dekat dengan ujung jempol)
            if dist_thumb_index < 0.35:
                return "O", 0.96
            # C: Membentuk lengkungan terbuka (C shape)
            else:
                return "C", 0.96

        # 2. JIKA KEPALAN ERAT (A, E, M, N, S, T)
        elif index_closed and middle_closed and ring_closed and pinky_closed:
            # Proyeksi vektor X ujung ibu jari terhadap lebar telapak tangan
            dx = pinky_mcp.x - index_mcp.x
            dy = pinky_mcp.y - index_mcp.y
            len_sq = dx**2 + dy**2
            
            tx = thumb_tip.x - index_mcp.x
            ty = thumb_tip.y - index_mcp.y
            
            dot = tx * dx + ty * dy
            rel_proj = dot / len_sq if len_sq > 0 else 0.0
                
            # A: Ibu jari berada rapat di samping luar jari telunjuk
            if rel_proj < 0.1:
                return "A", 0.98
            # T: Ibu jari diselipkan di bawah jari telunjuk
            elif 0.1 <= rel_proj < 0.38:
                return "T", 0.96
            # S & N: Ibu jari berada di depan jari telunjuk & tengah
            elif 0.38 <= rel_proj < 0.65:
                # S: Ibu jari melintang di DEPAN jari-jari (posisi Y lebih tinggi)
                if thumb_tip.y < middle_pip.y:
                    return "S", 0.96
                # N: Ibu jari diselipkan di bawah 2 jari (posisi Y ditekuk masuk/rendah)
                else:
                    return "N", 0.95
            # M: Ibu jari diselipkan di bawah 3 jari (berada di bawah jari manis)
            elif 0.65 <= rel_proj < 0.90:
                return "M", 0.95
            # E: Ibu jari dilipat flat rapat melintang di telapak bawah jari-jari
            else:
                return "E", 0.97
            
        else:
            return "-", 0.0

# ==========================================
# 4. DESIGN CONSTANTS & PALETTE
# ==========================================
COLOR_PRIMARY_GREEN = (52, 199, 89)   # Hijau Modern (Sleek Apple Green)
COLOR_TEXT = (255, 255, 255)          # Putih
COLOR_SECONDARY_RED = (255, 69, 58)   # Merah Cerah
COLOR_BG_HUD = (30, 30, 30)            # Background gelap HUD

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),      # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),  # Tengah
    (9, 13), (13, 14), (14, 15), (15, 16),# Manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17)                              # Telapak/Pangkal bawah
]

def draw_custom_landmarks(frame, hand_landmarks, w, h, color):
    """
    Menggambar skeletal model tangan secara estetik menggunakan OpenCV.
    """
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
        p2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        
    for lm in hand_landmarks:
        p = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, p, 5, (44, 62, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 3, COLOR_TEXT, -1, cv2.LINE_AA)

# ==========================================
# 5. APLIKASI UTAMA
# ==========================================
def main():
    task_file = 'hand_landmarker.task'
    if not os.path.exists(task_file):
        print("[INFO] Mengunduh model pelacak tangan MediaPipe (hand_landmarker.task)...")
        try:
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, task_file)
            print("[INFO] Unduhan model berhasil selesai!")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh file model MediaPipe: {e}")
            return

    # Inisialisasi Model SIBI Geometris
    model = SibiGestureModel()
    cnn_model = None
    
    # Inisialisasi Model CNN jika TensorFlow aktif dan model ada
    model_path = 'model_cnn_asl.h5'
    if HAS_TENSORFLOW and os.path.exists(model_path):
        print(f"[INFO] Memuat model CNN dari: {model_path} ...")
        try:
            cnn_model = tf.keras.models.load_model(model_path)
            print("[INFO] Model CNN berhasil dimuat!")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model CNN: {e}. Menggunakan Rule-Based Model.")

    # Inisialisasi MediaPipe Tasks Hand Landmarker (Khusus 1 Tangan)
    print("[INFO] Memulai MediaPipe Tasks Hand Landmarker (Single-Hand SIBI Mode)...")
    base_options = python.BaseOptions(model_asset_path=task_file)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Inisialisasi Webcam
    print("\n[INFO] Mengaktifkan webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses!")
        return
        
    print("\n=== SISTEM SIAP DIJALANKAN (SIBI TRANSLATOR LENGKAP) ===")
    print("Petunjuk Penggunaan:")
    print(" - Posisikan Tangan Kanan di depan kamera.")
    print(" - Tekan [SPASI] untuk bersuara (TTS) dan mengosongkan kata.")
    print(" - Tekan [BACKSPACE] untuk menghapus huruf terakhir.")
    print(" - Tekan [C] untuk mengosongkan kalimat.")
    print(" - Tekan [Q] untuk keluar.")

    current_word = ""
    stable_letter = ""
    frame_counter = 0
    stable_threshold = 12
    retry_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Jeda toleransi driver webcam
            retry_count += 1
            if retry_count > 30:
                print("[ERROR] Gagal membaca frame dari webcam secara konsisten.")
                break
            cv2.waitKey(100)
            continue
            
        retry_count = 0
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Deteksi tangan
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        predicted_letter = "-"
        confidence = 0.0
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Gambar visualisasi tangan
            draw_custom_landmarks(frame, hand_landmarks, w, h, COLOR_PRIMARY_GREEN)
            
            # Hitung Bounding Box tangan
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x_max, cx), max(y_max, cy)
                x_min, y_min = min(x_min, cx), min(y_min, cy)
            
            margin = 25
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COLOR_PRIMARY_GREEN, 2)
            
            # Klasifikasi gestur SIBI
            if cnn_model is not None:
                # Menggunakan CNN jika ada
                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size > 0:
                    try:
                        resized = cv2.resize(hand_roi, (64, 64))
                        normalized = resized / 255.0
                        input_data = np.expand_dims(normalized, axis=0)
                        prediction = cnn_model.predict(input_data, verbose=0)
                        class_idx = np.argmax(prediction)
                        confidence = prediction[0][class_idx]
                        predicted_letter = chr(65 + class_idx)
                    except Exception:
                        predicted_letter, confidence = model.predict_gesture(hand_landmarks)
            else:
                # Menggunakan Geometris Sendi SIBI Kustom
                predicted_letter, confidence = model.predict_gesture(hand_landmarks)
                
            # Logika Debounce / Kestabilan Deteksi Huruf
            if predicted_letter != "-" and predicted_letter != "":
                if predicted_letter == stable_letter:
                    frame_counter += 1
                else:
                    stable_letter = predicted_letter
                    frame_counter = 0
                    
                if frame_counter >= stable_threshold:
                    current_word += stable_letter
                    print(f"[SIBI] Menambahkan huruf: '{stable_letter}' -> Kata: '{current_word}'")
                    frame_counter = 0
                    stable_letter = ""
        
        # ==========================================
        # 6. RENDER HUD / GUI WINDOW
        # ==========================================
        # Bar HUD bagian bawah
        cv2.rectangle(frame, (0, h - 80), (w, h), COLOR_BG_HUD, -1)
        cv2.putText(frame, f"KATA SIBI: {current_word}", (20, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_PRIMARY_GREEN, 2, cv2.LINE_AA)
        
        # Panel info di kanan atas
        if predicted_letter != "-":
            cv2.rectangle(frame, (10, 10), (240, 120), COLOR_BG_HUD, -1)
            cv2.putText(frame, "MODE: SIBI (1 Tangan)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"HURUF: {predicted_letter}", (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PRIMARY_GREEN, 2, cv2.LINE_AA)
            cv2.putText(frame, f"CONF: {confidence*100:.1f}%", (20, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            
            # Progress bar kestabilan
            progress_w = int((frame_counter / stable_threshold) * 200)
            cv2.rectangle(frame, (20, 112), (220, 117), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 112), (20 + progress_w, 117), COLOR_PRIMARY_GREEN, -1)
        else:
            cv2.rectangle(frame, (10, 10), (240, 60), COLOR_BG_HUD, -1)
            cv2.putText(frame, "Tangan Kosong", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SECONDARY_RED, 1, cv2.LINE_AA)
            frame_counter = 0
            model.clear_history()

        # Panduan pintasan tombol
        cv2.putText(frame, "Space: Speak | Backspace: Delete | C: Clear | Q: Quit", (w - 380, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Status mode deteksi
        mode_descr = "Mode Deteksi SIBI Aktif"
        cv2.putText(frame, mode_descr, (20, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Sign Language Translator SIBI - OpenCV", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            if current_word:
                print(f"[SPEAK] Mengucapkan: '{current_word}'")
                tts_thread = TextToSpeechThread(current_word)
                tts_thread.start()
                current_word = ""
        elif key == 8: # Backspace
            if len(current_word) > 0:
                current_word = current_word[:-1]
                print(f"[DELETE] Kata sekarang: '{current_word}'")
        elif key == ord('c') or key == ord('C'):
            current_word = ""
            print("[CLEAR] Buffer dikosongkan.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
