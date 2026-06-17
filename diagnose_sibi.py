import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from main import SibiGestureModel

def main():
    task_file = 'hand_landmarker.task'
    if not os.path.exists(task_file):
        print("[INFO] Mengunduh model pelacak tangan MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, task_file)

    # Inisialisasi MediaPipe Tasks Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=task_file)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Inisialisasi Model Geometris Baru
    model = SibiGestureModel()
    
    cap = cv2.VideoCapture(0)
    print("=== DIAGNOSIS GEOMETRIS SIBI INVARIAN SKALA ===")
    print("Tekan [Q] untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        # Draw background info box
        cv2.rectangle(frame, (0, 0), (480, 480), (30, 30, 30), -1)
        
        predicted_letter = "-"
        confidence = 0.0
        
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            
            # Predict SIBI letter
            predicted_letter, confidence = model.predict_gesture(landmarks)
            
            # Draw skeleton
            for connection in [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]:
                p1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
                p2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
                cv2.line(frame, p1, p2, (52, 199, 89), 2, cv2.LINE_AA)
            
            for lm in landmarks:
                p = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Hitung fitur-fitur untuk ditampilkan
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_mcp = landmarks[9]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            ring_mcp = landmarks[13]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            pinky_mcp = landmarks[17]
            
            def dist(p1, p2):
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                
            palm_width = dist(index_mcp, pinky_mcp)
            if palm_width < 0.01:
                palm_width = 0.01
                
            def finger_extension(tip, pip, mcp):
                s_len = dist(tip, pip) + dist(pip, mcp)
                a_len = dist(tip, mcp)
                return a_len / s_len if s_len > 0 else 0.0
                
            index_ext = finger_extension(index_tip, index_pip, index_mcp)
            middle_ext = finger_extension(middle_tip, middle_pip, middle_mcp)
            ring_ext = finger_extension(ring_tip, ring_pip, ring_mcp)
            pinky_ext = finger_extension(pinky_tip, pinky_pip, pinky_mcp)
            
            # Proyeksi jempol
            dx = pinky_mcp.x - index_mcp.x
            dy = pinky_mcp.y - index_mcp.y
            len_sq = dx**2 + dy**2
            tx = thumb_tip.x - index_mcp.x
            ty = thumb_tip.y - index_mcp.y
            dot = tx * dx + ty * dy
            rel_proj = dot / len_sq if len_sq > 0 else 0.0
            
            dist_thumb_index = dist(thumb_tip, index_tip) / palm_width
            dist_index_middle = dist(index_tip, middle_tip) / palm_width
            dist_thumb_middle = dist(thumb_tip, middle_pip) / palm_width
            
            # Tampilkan data di layar
            y_offset = 25
            def put_info(text, color=(255, 255, 255)):
                nonlocal y_offset
                cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y_offset += 22
                
            put_info("=== HASIL KLASIFIKASI ===", (52, 199, 89))
            put_info(f"HURUF SIBI: {predicted_letter}", (0, 255, 255))
            put_info(f"CONFIDENCE: {confidence*100:.1f}%")
            put_info("=========================")
            
            put_info("--- RASIO EKSTENSI JARI ---")
            put_info(f"Index Ext: {index_ext:.3f} | Middle Ext: {middle_ext:.3f}")
            put_info(f"Ring Ext:  {ring_ext:.3f} | Pinky Ext:  {pinky_ext:.3f}")
            
            put_info("--- PROYEKSI JEMPOL (rel_proj) ---")
            put_info(f"rel_proj (A < 0.1 < T < 0.38 < S/N < 0.65 < M < 0.9 < E):")
            put_info(f"NILAI rel_proj: {rel_proj:.3f}", (0, 255, 255))
            put_info(f"Thumb Tip Y: {thumb_tip.y:.3f} | Middle PIP Y: {middle_pip.y:.3f}")
            
            put_info("--- JARAK RELATIF KELAS INVARIAN ---")
            put_info(f"Palm Width (Scale ref): {palm_width:.3f}")
            put_info(f"Dist Thumb-Index: {dist_thumb_index:.3f}")
            put_info(f"Dist Index-Middle: {dist_index_middle:.3f}")
            put_info(f"Dist Thumb-MiddlePIP: {dist_thumb_middle:.3f}")
            
            # Tampilkan buffer lintasan
            put_info("--- BUFFER TRAJECTORY ---")
            put_info(f"Index Path Len (Z): {len(model.index_path)} / 25")
            put_info(f"Pinky Path Len (J): {len(model.pinky_path)} / 25")
            
        else:
            model.clear_history()
            cv2.putText(frame, "Tangan Tidak Terdeteksi", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("DIAGNOSIS GEOMETRIS SIBI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
