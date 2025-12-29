#!/usr/bin/env python3
"""
Integrated Face Recognition System
Combines: Recognition + Liveness + Head Pose Detection
FIXED: Head pose axes + macOS cleanup
"""

import cv2
import numpy as np
import pickle
import os
import time
import sqlite3
import csv
from datetime import datetime
import platform
import signal
import sys

# Try optional dependencies
try:
    from scipy.spatial import distance as dist
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

from insightface.app import FaceAnalysis

# Global exit flag
_should_exit = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _should_exit
    print("\n\n[INFO] ⚠️ Stopping...")
    _should_exit = True

signal.signal(signal.SIGINT, signal_handler)

# ========= CONFIG =========
MODELS_DIR = "models"
EMBEDDING_PATH = os.path.join(MODELS_DIR, "insightface_embeddings.pickle")
LOG_DIR = "logs"
CSV_PATH = os.path.join(LOG_DIR, "integrated_log.csv")
DB_PATH = os.path.join(LOG_DIR, "integrated_log.sqlite")

SIM_THRESHOLD = 0.45
FRAME_RESIZE = 0.5
PROVIDERS = ["CPUExecutionProvider"]
BOX_COLOR_KNOWN = (0, 255, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)
BOX_COLOR_FAKE = (0, 165, 255)  # Orange
TEXT_COLOR = (255, 255, 255)
LOG_COOLDOWN_SEC = 5.0

# ========= HEAD POSE DETECTION (FIXED) =========

def detect_head_pose_insightface(face):
    """
    FIXED: Correct axis mapping for head pose.
    Uses InsightFace 5-point landmarks (kps).
    """
    try:
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            kps = face.kps
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            
            # YAW (Left/Right): Horizontal offset of nose from eye center
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose[0]
            horizontal_offset = nose_x - eye_center_x
            
            # PITCH (Up/Down): Vertical offset of nose from eye center
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            nose_y = nose[1]
            vertical_offset = nose_y - eye_center_y
            
            # Normalize by eye distance
            eye_distance = np.linalg.norm(left_eye - right_eye)
            
            if eye_distance > 0:
                yaw_normalized = horizontal_offset / eye_distance
                pitch_normalized = vertical_offset / eye_distance
            else:
                yaw_normalized = 0
                pitch_normalized = 0
            
            # Classify YAW (Left/Right) - FIXED threshold
            if yaw_normalized > 0.10:
                yaw_text = "Right"
            elif yaw_normalized < -0.10:
                yaw_text = "Left"
            else:
                yaw_text = "Straight"
            
            # Classify PITCH (Up/Down)
            if pitch_normalized > 0.08:
                pitch_text = "Down"
            elif pitch_normalized < -0.08:
                pitch_text = "Up"
            else:
                pitch_text = "Center"
            
            # Combined pose
            if yaw_text == "Straight" and pitch_text == "Center":
                pose = "Straight"
                color = (0, 255, 0)  # Green
            elif yaw_text != "Straight":
                pose = yaw_text  # Prioritize yaw
                color = (0, 255, 255) if yaw_text == "Right" else (255, 0, 255)
            else:
                pose = pitch_text
                color = (255, 255, 0) if pitch_text == "Down" else (255, 0, 128)
            
            return pose, color, (yaw_normalized, pitch_normalized)
    except:
        pass
    
    return "Unknown", (128, 128, 128), (0, 0)

def detect_head_pose_basic(frame, face_bbox):
    """Basic fallback for head pose."""
    x1, y1, x2, y2 = face_bbox
    
    # Horizontal position (Yaw)
    face_center_x = (x1 + x2) // 2
    frame_center_x = frame.shape[1] // 2
    h_offset = face_center_x - frame_center_x
    h_ratio = h_offset / (frame.shape[1] / 2)
    
    # Vertical position (Pitch)
    face_center_y = (y1 + y2) // 2
    frame_center_y = frame.shape[0] // 2
    v_offset = face_center_y - frame_center_y
    v_ratio = v_offset / (frame.shape[0] / 2)
    
    # Classify
    if abs(h_ratio) > abs(v_ratio):  # Yaw dominates
        if h_ratio > 0.08:
            return "Right", (0, 255, 255), (h_ratio, v_ratio)
        elif h_ratio < -0.08:
            return "Left", (255, 0, 255), (h_ratio, v_ratio)
    else:  # Pitch dominates
        if v_ratio > 0.08:
            return "Down", (255, 255, 0), (h_ratio, v_ratio)
        elif v_ratio < -0.08:
            return "Up", (255, 0, 128), (h_ratio, v_ratio)
    
    return "Straight", (0, 255, 0), (h_ratio, v_ratio)

# ========= LIVENESS DETECTION =========

def euclidean_distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_liveness_simple(face, prev_face):
    """Simple liveness based on movement."""
    if prev_face is None:
        return False, 0, "Initializing"
    
    score = 0
    reasons = []
    
    # Movement
    curr_center = [(face.bbox[0] + face.bbox[2])/2, (face.bbox[1] + face.bbox[3])/2]
    prev_center = [(prev_face.bbox[0] + prev_face.bbox[2])/2, (prev_face.bbox[1] + prev_face.bbox[3])/2]
    movement = euclidean_distance(curr_center, prev_center)
    
    if movement > 2:
        score += 1
        reasons.append(f"Move:{movement:.1f}")
    
    # Size change
    curr_size = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    prev_size = (prev_face.bbox[2] - prev_face.bbox[0]) * (prev_face.bbox[3] - prev_face.bbox[1])
    size_change = abs(curr_size - prev_size) / prev_size if prev_size > 0 else 0
    
    if size_change > 0.05:
        score += 1
        reasons.append(f"Depth:{size_change*100:.0f}%")
    
    # Eye distance (blink)
    if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
        if hasattr(prev_face, 'kps') and prev_face.kps is not None and len(prev_face.kps) >= 2:
            curr_eye = euclidean_distance(face.kps[0], face.kps[1])
            prev_eye = euclidean_distance(prev_face.kps[0], prev_face.kps[1])
            eye_change = abs(curr_eye - prev_eye) / prev_eye if prev_eye > 0 else 0
            
            if eye_change > 0.03:
                score += 2
                reasons.append("Blink")
    
    is_live = score >= 2
    reason_str = ", ".join(reasons) if reasons else "Static"
    
    return is_live, score, reason_str

# ========= FACE RECOGNITION =========

def load_embeddings():
    """Load face embeddings."""
    if not os.path.exists(EMBEDDING_PATH):
        return None
    
    with open(EMBEDDING_PATH, "rb") as f:
        data = pickle.load(f)
    
    encodings = np.asarray(data["encodings"], dtype="float32")
    names = data["names"]
    norms = np.linalg.norm(encodings, axis=1, keepdims=True)
    encodings = encodings / np.clip(norms, 1e-8, None)
    
    return {"encodings": encodings, "names": np.array(names)}

def cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a, b)

# ========= LOGGING =========

def init_csv():
    """Initialize CSV log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "name", "similarity", "liveness", "pose", "x1", "y1", "x2", "y2"])

def log_to_csv(timestamp, name, sim, liveness, pose, bbox):
    """Log to CSV."""
    x1, y1, x2, y2 = bbox
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, f"{sim:.4f}", liveness, pose, x1, y1, x2, y2])

def init_db():
    """Initialize SQLite database."""
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            similarity REAL,
            liveness TEXT,
            pose TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
        )
    """)
    conn.commit()
    return conn

def log_to_db(conn, timestamp, name, sim, liveness, pose, bbox):
    """Log to database."""
    x1, y1, x2, y2 = bbox
    c = conn.cursor()
    c.execute("""
        INSERT INTO recognition_log (timestamp, name, similarity, liveness, pose, x1, y1, x2, y2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, name, float(sim), liveness, pose, int(x1), int(y1), int(x2), int(y2)))
    conn.commit()

# ========= CLEANUP =========

def cleanup_opencv_aggressive():
    """Aggressive cleanup for macOS."""
    print("[INFO] 🧹 Cleaning up...")
    try:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        
        if platform.system() == "Darwin":
            for i in range(25):  # Extra iterations for integrated system
                cv2.waitKey(1)
                if i % 5 == 0:
                    time.sleep(0.05)
            time.sleep(0.3)
            for _ in range(10):
                cv2.waitKey(1)
        else:
            for _ in range(5):
                cv2.waitKey(1)
        
        print("[INFO] ✅ Windows closed")
    except Exception as e:
        print(f"[WARNING] Cleanup: {e}")

def clear_input_buffer():
    """Clear stdin buffer."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass

# ========= MAIN =========

def main():
    """Main integrated system."""
    global _should_exit
    
    print("\n" + "="*60)
    print(" 🚀 INTEGRATED FACE RECOGNITION SYSTEM")
    print(" Features: Recognition + Liveness + Head Pose")
    print("="*60)
    
    # Load embeddings
    print("\n[INFO] Loading face embeddings...")
    data = load_embeddings()
    if data is None:
        print("[ERROR] No embeddings found. Run Option 4 first!")
        input("\nPress ENTER...")
        return
    
    known_encodings = data["encodings"]
    known_names = data["names"]
    print(f"[INFO] ✅ Loaded {len(known_encodings)} face embeddings")
    
    # Initialize InsightFace
    print("\n[INFO] Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] ✅ InsightFace ready")
    
    # Initialize logging
    init_csv()
    db_conn = init_db()
    print(f"[INFO] ✅ Logging: {CSV_PATH}, {DB_PATH}")
    
    # Open camera
    print("\n[INFO] Opening camera...")
    system = platform.system()
    
    if system == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif system == "Windows":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        try:
            db_conn.close()
        except:
            pass
        input("\nPress ENTER...")
        return
    
    # Set properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if system == "Darwin":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    
    print("[INFO] ✅ Camera ready")
    print("\n" + "="*60)
    print("📹 SYSTEM ACTIVE")
    print("="*60)
    print("Features:")
    print(" • Face Recognition (Green=Known, Red=Unknown)")
    print(" • Liveness Detection (Orange=Fake)")
    print(" • Head Pose (Left/Right/Straight/Up/Down)")
    print("\nPress 'q' to quit")
    print("="*60 + "\n")
    
    last_log_time_per_name = {}
    prev_face = None
    frame_count = 0
    
    try:
        while not _should_exit:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
            faces = app.get(small_frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                
                # Scale back
                x1_full = int(x1 / FRAME_RESIZE)
                y1_full = int(y1 / FRAME_RESIZE)
                x2_full = int(x2 / FRAME_RESIZE)
                y2_full = int(y2 / FRAME_RESIZE)
                bbox_full = (x1_full, y1_full, x2_full, y2_full)
                
                # Get embedding
                if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                    emb = face.normed_embedding
                else:
                    emb = face.embedding
                    emb = emb / np.linalg.norm(emb)
                emb = emb.astype("float32")
                
                # Face recognition
                sims = cosine_similarity(known_encodings, emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_name = known_names[best_idx]
                
                if best_sim >= SIM_THRESHOLD:
                    name_label = best_name
                    base_color = BOX_COLOR_KNOWN
                else:
                    name_label = "Unknown"
                    base_color = BOX_COLOR_UNKNOWN
                
                # Liveness detection
                is_live, live_score, live_reason = detect_liveness_simple(face, prev_face)
                
                if not is_live and frame_count > 10:  # Skip first 10 frames
                    liveness_label = "FAKE"
                    final_color = BOX_COLOR_FAKE
                else:
                    liveness_label = "LIVE"
                    final_color = base_color
                
                # Head pose detection (FIXED)
                pose, pose_color, (yaw, pitch) = detect_head_pose_insightface(face)
                if pose == "Unknown":
                    pose, pose_color, (yaw, pitch) = detect_head_pose_basic(frame, bbox_full)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), final_color, 2)
                
                # Display info
                y_offset = y1_full - 10
                
                # Name + Similarity
                cv2.putText(frame, f"{name_label} ({best_sim:.2f})", (x1_full, y_offset),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, final_color, 1)
                
                # Liveness
                y_offset -= 25
                cv2.putText(frame, f"Liveness: {liveness_label}", (x1_full, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, final_color, 1)
                
                # Pose
                y_offset -= 25
                cv2.putText(frame, f"Pose: {pose}", (x1_full, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
                
                # Log (with cooldown)
                if best_sim >= SIM_THRESHOLD and is_live:
                    now = time.time()
                    last_time = last_log_time_per_name.get(best_name, 0.0)
                    
                    if now - last_time >= LOG_COOLDOWN_SEC:
                        last_log_time_per_name[best_name] = now
                        ts_str = datetime.now().isoformat(timespec="seconds")
                        
                        log_to_csv(ts_str, best_name, best_sim, liveness_label, pose, bbox_full)
                        log_to_db(db_conn, ts_str, best_name, best_sim, liveness_label, pose, bbox_full)
                        
                        print(f"[LOG] {ts_str} - {best_name} ({best_sim:.3f}) - {liveness_label} - {pose}")
                
                # Save for next frame
                prev_face = face
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Integrated System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed - stopping...")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] ⚠️ Keyboard interrupt")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Critical cleanup
        print("\n[INFO] 🧹 Shutting down...")
        
        # 1. Release camera
        try:
            cap.release()
            time.sleep(0.4)  # Longer delay for integrated system
            print("[INFO] ✅ Camera released")
        except Exception as e:
            print(f"[WARNING] Camera: {e}")
        
        # 2. Close database
        try:
            db_conn.close()
            print("[INFO] ✅ Database closed")
        except Exception as e:
            print(f"[WARNING] DB: {e}")
        
        # 3. Aggressive OpenCV cleanup
        cleanup_opencv_aggressive()
        
        # 4. Platform-specific final steps
        if platform.system() == "Darwin":
            time.sleep(0.6)
            clear_input_buffer()
        
        print("\n" + "="*60)
        print(" ✅ SHUTDOWN COMPLETE")
        print("="*60)
        print(f"📊 Logs saved:")
        print(f"   CSV: {CSV_PATH}")
        print(f"   DB:  {DB_PATH}")
        print("="*60)
        
        time.sleep(0.3)
        try:
            input("\nPress ENTER to return to menu...")
        except:
            pass

if __name__ == "__main__":
    main()
