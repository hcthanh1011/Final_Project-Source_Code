#!/usr/bin/env python3
"""
Real-time Face Recognition with Logging
macOS Optimized - Fixed terminal freeze issue
"""

import os
import time
import csv
import sqlite3
from datetime import datetime
import cv2
import numpy as np
import pickle
import signal
import platform
from insightface.app import FaceAnalysis

# ========= SIGNAL HANDLING =========
_should_exit = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _should_exit
    print("\n\n[INFO] ⚠️ Stopping recognition system...")
    _should_exit = True
    try:
        cv2.destroyAllWindows()
        for _ in range(10):  # Increased from 5 to 10
            cv2.waitKey(1)
    except:
        pass

signal.signal(signal.SIGINT, signal_handler)
if platform.system() == "Windows":
    try:
        signal.signal(signal.SIGBREAK, signal_handler)
    except AttributeError:
        pass

# ========= CONFIG =========
MODELS_DIR = "models"
EMBEDDING_PATH = os.path.join(MODELS_DIR, "insightface_embeddings.pickle")
LOG_DIR = "logs"
CSV_PATH = os.path.join(LOG_DIR, "recognition_log.csv")
DB_PATH = os.path.join(LOG_DIR, "recognition_log.sqlite")

SIM_THRESHOLD = 0.45
FRAME_RESIZE = 0.5
PROVIDERS = ["CPUExecutionProvider"]
BOX_COLOR_KNOWN = (0, 255, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
LOG_COOLDOWN_SEC = 5.0

# ========= FUNCTIONS =========

def cleanup_opencv_macos():
    """macOS-specific cleanup to prevent terminal freeze."""
    print("[INFO] 🧹 Cleaning up OpenCV...")
    try:
        cv2.destroyAllWindows()
        
        if platform.system() == "Darwin":
            # Critical for macOS: multiple waitKey calls
            for i in range(15):  # Increased to 15 for extra safety
                cv2.waitKey(1)
                if i % 5 == 0:
                    time.sleep(0.05)
        else:
            cv2.waitKey(1)
        
        print("[INFO] ✅ Windows closed")
    except Exception as e:
        print(f"[WARNING] Cleanup error: {e}")

def init_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_embeddings():
    if not os.path.exists(EMBEDDING_PATH):
        print(f"[ERROR] Embedding file not found: {EMBEDDING_PATH}")
        print("[TIP] Run Option 4 (Build Embeddings) first")
        return None
    
    with open(EMBEDDING_PATH, "rb") as f:
        data = pickle.load(f)
    
    encodings = data["encodings"]
    names = data["names"]
    encodings = np.asarray(encodings, dtype="float32")
    norms = np.linalg.norm(encodings, axis=1, keepdims=True)
    encodings = encodings / np.clip(norms, 1e-8, None)
    
    print(f"[INFO] Loaded {encodings.shape[0]} embeddings")
    return {"encodings": encodings, "names": names}

def init_csv():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "name", "similarity", "x1", "y1", "x2", "y2"])
    print(f"[INFO] CSV logging: {CSV_PATH}")

def log_to_csv(timestamp, name, sim, bbox):
    x1, y1, x2, y2 = bbox
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, f"{sim:.4f}", x1, y1, x2, y2])

def init_db():
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            similarity REAL,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
        )
    """)
    conn.commit()
    print(f"[INFO] SQLite logging: {DB_PATH}")
    return conn

def log_to_db(conn, timestamp, name, sim, bbox):
    x1, y1, x2, y2 = bbox
    c = conn.cursor()
    c.execute("""
        INSERT INTO recognition_log (timestamp, name, similarity, x1, y1, x2, y2)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, name, float(sim), int(x1), int(y1), int(x2), int(y2)))
    conn.commit()

def cosine_similarity(a, b):
    """a: (N,D), b: (D,) -> (N,)"""
    return np.dot(a, b)

# ========= MAIN =========

def main():
    global _should_exit
    
    print("\n" + "="*60)
    print(" REAL-TIME FACE RECOGNITION")
    print(" macOS Optimized")
    print("="*60)
    
    # Load embeddings
    data = load_embeddings()
    if data is None:
        input("\nPress ENTER to continue...")
        return
    
    known_encodings = data["encodings"]
    known_names = np.array(data["names"])
    
    # Initialize InsightFace
    print("\n[INFO] Initializing InsightFace...")
    app = init_insightface()
    
    # Initialize logging
    init_csv()
    db_conn = init_db()
    
    # Open camera
    print("\n[INFO] Opening webcam...")
    system = platform.system()
    backend = cv2.CAP_AVFOUNDATION if system == "Darwin" else cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_ANY
    
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback
    
    if not cap.isOpened():
        print("[ERROR] Failed to open webcam")
        try:
            db_conn.close()
        except:
            pass
        input("\nPress ENTER to continue...")
        return
    
    print("[INFO] ✅ Camera ready")
    print("\n" + "="*60)
    print("📹 RECOGNITION ACTIVE")
    print("="*60)
    print("Instructions:")
    print(" - Green box = known person")
    print(" - Red box = unknown")
    print(" - Press 'q' to quit")
    print("="*60 + "\n")
    
    last_log_time_per_name = {}
    
    try:
        while not _should_exit:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Resize for speed
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
            faces = app.get(small_frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                
                # Scale back to original
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
                
                # Compare with known faces
                sims = cosine_similarity(known_encodings, emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_name = known_names[best_idx]
                
                if best_sim >= SIM_THRESHOLD:
                    name_to_show = f"{best_name} ({best_sim:.2f})"
                    color = BOX_COLOR_KNOWN
                    
                    # Log with cooldown
                    now = time.time()
                    last_time = last_log_time_per_name.get(best_name, 0.0)
                    if now - last_time >= LOG_COOLDOWN_SEC:
                        last_log_time_per_name[best_name] = now
                        ts_str = datetime.now().isoformat(timespec="seconds")
                        log_to_csv(ts_str, best_name, best_sim, bbox_full)
                        log_to_db(db_conn, ts_str, best_name, best_sim, bbox_full)
                        print(f"[LOG] {ts_str} - {best_name} ({best_sim:.3f})")
                else:
                    name_to_show = f"Unknown ({best_sim:.2f})"
                    color = BOX_COLOR_UNKNOWN
                
                # Draw box
                cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), color, 2)
                
                # Draw label
                (tw, th), _ = cv2.getTextSize(name_to_show, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1_full, y2_full), 
                             (x1_full + tw + 10, y2_full + th + 10), color, cv2.FILLED)
                cv2.putText(frame, name_to_show, (x1_full + 5, y2_full + th),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, TEXT_COLOR, 1)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 'q' pressed - stopping...")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] ⚠️ Keyboard interrupt")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup sequence (critical order)
        print("\n[INFO] 🧹 Shutting down...")
        
        # 1. Release camera first
        try:
            cap.release()
            time.sleep(0.2)  # Give OS time
            print("[INFO] ✅ Camera released")
        except Exception as e:
            print(f"[WARNING] Camera release: {e}")
        
        # 2. Close database
        try:
            db_conn.close()
            print("[INFO] ✅ Database closed")
        except Exception as e:
            print(f"[WARNING] DB close: {e}")
        
        # 3. Clean up OpenCV windows (platform-specific)
        cleanup_opencv_macos()
        
        # 4. Final flush for macOS
        if platform.system() == "Darwin":
            time.sleep(0.3)
        
        print("\n✅ Cleanup complete")
        print("="*60)
        input("\nPress ENTER to return to menu...")

if __name__ == "__main__":
    main()
