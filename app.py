#!/usr/bin/env python3
"""
Flask Web Interface - Face Recognition System
Tích hợp với integrated_system.py có sẵn
Version: 2.0 - With 8 Menu Options
"""
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis
import sqlite3
import time
import platform

app = Flask(__name__)
CORS(app)

# ========= IMPORT TỪ INTEGRATED_SYSTEM =========
import sys
sys.path.append(os.path.dirname(__file__))

# Config từ integrated_system.py
MODELS_DIR = "models"
EMBEDDING_PATH = os.path.join(MODELS_DIR, "insightface_embeddings.pickle")
LOG_DIR = "logs"
DB_PATH = os.path.join(LOG_DIR, "integrated_log.sqlite")
SIM_THRESHOLD = 0.45
FRAME_RESIZE = 0.5
PROVIDERS = ["CPUExecutionProvider"]
BOX_COLOR_KNOWN = (0, 255, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)

# ========= GLOBAL VARIABLES =========
face_app = None
known_encodings = None
known_names = None
camera_instance = None

# ========= HELPER FUNCTIONS =========
def cosine_similarity(a, b):
    """Compute cosine similarity"""
    return np.dot(a, b)

def detect_head_pose_insightface(face):
    """Head pose detection"""
    try:
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            kps = face.kps
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose[0]
            horizontal_offset = nose_x - eye_center_x
            
            eye_distance = np.linalg.norm(left_eye - right_eye)
            if eye_distance > 0:
                yaw_normalized = horizontal_offset / eye_distance
            else:
                yaw_normalized = 0
            
            if yaw_normalized > 0.10:
                return "Right", (0, 255, 255)
            elif yaw_normalized < -0.10:
                return "Left", (255, 0, 255)
            else:
                return "Straight", (0, 255, 0)
    except:
        pass
    return "Unknown", (128, 128, 128)

def euclidean_distance(p1, p2):
    """Calculate distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_liveness_simple(face, prev_face):
    """Simple liveness detection - RELAXED VERSION"""
    if prev_face is None:
        return True, 0, "Initializing"
    
    score = 0
    reasons = []
    
    # Movement - GIẢM threshold từ 2 → 1
    curr_center = [(face.bbox[0] + face.bbox[2])/2, (face.bbox[1] + face.bbox[3])/2]
    prev_center = [(prev_face.bbox[0] + prev_face.bbox[2])/2, (prev_face.bbox[1] + prev_face.bbox[3])/2]
    movement = euclidean_distance(curr_center, prev_center)
    
    if movement > 1:
        score += 1
        reasons.append(f"Move:{movement:.1f}")
    
    # Size change - GIẢM threshold từ 0.05 → 0.02
    curr_size = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    prev_size = (prev_face.bbox[2] - prev_face.bbox[0]) * (prev_face.bbox[3] - prev_face.bbox[1])
    size_change = abs(curr_size - prev_size) / prev_size if prev_size > 0 else 0
    
    if size_change > 0.02:
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
    
    # GIẢM yêu cầu score từ 2 → 1
    is_live = score >= 1
    reason_str = ", ".join(reasons) if reasons else "Static"
    return is_live, score, reason_str

# ========= MODEL INITIALIZATION =========
def initialize_model():
    """Load embeddings và InsightFace model"""
    global face_app, known_encodings, known_names
    
    print("\n[INFO] Initializing model...")
    
    if not os.path.exists(EMBEDDING_PATH):
        print(f"[ERROR] Embedding file not found: {EMBEDDING_PATH}")
        return False
    
    # Load embeddings
    with open(EMBEDDING_PATH, "rb") as f:
        data = pickle.load(f)
    
    encodings = np.asarray(data["encodings"], dtype="float32")
    names = data["names"]
    
    # Normalize
    norms = np.linalg.norm(encodings, axis=1, keepdims=True)
    known_encodings = encodings / np.clip(norms, 1e-8, None)
    known_names = np.array(names)
    
    # Initialize InsightFace
    face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    print(f"[INFO] ✅ Model loaded: {len(known_encodings)} faces")
    return True

# ========= DATABASE LOGGING =========
def log_to_db(timestamp, name, sim, liveness, pose, bbox):
    """Log recognition to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create table if not exists
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
        
        x1, y1, x2, y2 = bbox
        c.execute("""
            INSERT INTO recognition_log (timestamp, name, similarity, liveness, pose, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, name, float(sim), liveness, pose, int(x1), int(y1), int(x2), int(y2)))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARNING] DB Log failed: {e}")

# ========= VIDEO CAMERA CLASS =========
class VideoCamera:
    def __init__(self):
        """Initialize camera"""
        system = platform.system()
        if system == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif system == "Windows":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY
        
        self.video = cv2.VideoCapture(0, backend)
        if not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
        
        if not self.video.isOpened():
            raise Exception("Cannot open camera")
        
        # Set properties
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame_count = 0
        self.last_log_time = {}
        self.prev_face = None
        
        print("[INFO] Camera initialized")
    
    def __del__(self):
        """Release camera"""
        if hasattr(self, 'video'):
            self.video.release()
    
    def get_frame(self):
        """Process and return frame"""
        success, frame = self.video.read()
        if not success:
            return None
        
        self.frame_count += 1
        
        # Process frame
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
        faces = face_app.get(small_frame)
        
        for face in faces:
            # Scale bbox back to full size
            x1, y1, x2, y2 = face.bbox.astype(int)
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
            is_live, live_score, live_reason = detect_liveness_simple(face, self.prev_face)
            if not is_live and self.frame_count > 10:
                liveness_label = "FAKE"
                final_color = (0, 165, 255)
            else:
                liveness_label = "LIVE"
                final_color = base_color
            
            # Head pose
            pose, pose_color = detect_head_pose_insightface(face)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), final_color, 2)
            
            # Display info
            y_offset = y1_full - 10
            cv2.putText(frame, f"{name_label} ({best_sim:.2f})", 
                       (x1_full, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, final_color, 2)
            
            y_offset -= 25
            cv2.putText(frame, f"Liveness: {liveness_label}", 
                       (x1_full, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, final_color, 1)
            
            y_offset -= 25
            cv2.putText(frame, f"Pose: {pose}", 
                       (x1_full, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
            
            # Log with cooldown
            if best_sim >= SIM_THRESHOLD and is_live:
                now = time.time()
                last_time = self.last_log_time.get(best_name, 0.0)
                if now - last_time >= 1.0:
                    self.last_log_time[best_name] = now
                    ts_str = datetime.now().isoformat(timespec="seconds")
                    log_to_db(ts_str, best_name, best_sim, liveness_label, pose, bbox_full)
                    print(f"[LOG] {ts_str} - {best_name} ({best_sim:.3f}) - {liveness_label} - {pose}")
            
            # Save for next frame
            self.prev_face = face
        
        # Display frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()

# ========= VIDEO GENERATOR =========
def generate_frames():
    """Generator for video streaming"""
    global camera_instance
    
    if camera_instance is None:
        camera_instance = VideoCamera()
    
    while True:
        try:
            frame = camera_instance.get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"[ERROR] Frame generation: {e}")
            break

# ========= FLASK ROUTES - 8 MENU OPTIONS =========
@app.route('/')
def index():
    """Trang chủ - Home menu"""
    return render_template('index.html')

@app.route('/collect')
def collect_images():
    """Option 1: Collect Images"""
    return render_template('collect.html')

@app.route('/clean')
def clean_dataset():
    """Option 2: Clean Dataset"""
    return render_template('clean.html')

@app.route('/analyze')
def analyze_pose():
    """Option 3: Analyze Pose"""
    return render_template('analyze.html')

@app.route('/build')
def build_embeddings():
    """Option 4: Build Embeddings"""
    return render_template('build.html')

@app.route('/recognition')
def recognition():
    """Option 5: Real-time Recognition"""
    return render_template('recognition.html')

@app.route('/liveness')
def liveness():
    """Option 6: Liveness Detection"""
    return render_template('liveness.html')

@app.route('/pose')
def pose():
    """Option 7: Head Pose Detection"""
    return render_template('pose.html')

@app.route('/integrated')
def integrated():
    """Option 8: Integrated System - Main Feature"""
    return render_template('integrated.html')

# ========= API ROUTES =========
@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Statistics API"""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({
                'total_logs': 0,
                'unique_persons': 0,
                'recent_logs': []
            })
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total logs
        c.execute("SELECT COUNT(*) FROM recognition_log")
        total_logs = c.fetchone()[0]
        
        # Unique persons
        c.execute("SELECT COUNT(DISTINCT name) FROM recognition_log WHERE name != 'Unknown'")
        unique_persons = c.fetchone()[0]
        
        # Recent logs (last 10)
        c.execute("""
            SELECT timestamp, name, similarity, pose 
            FROM recognition_log 
            ORDER BY id DESC LIMIT 10
        """)
        recent_logs = c.fetchall()
        
        conn.close()
        
        return jsonify({
            'total_logs': total_logs,
            'unique_persons': unique_persons,
            'recent_logs': recent_logs
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'total_logs': 0,
            'unique_persons': 0,
            'recent_logs': []
        })

@app.route('/stats/rotations')
def stats_rotations():
    """Statistics API - ONLY Left/Right poses"""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({
                'total_rotations': 0,
                'unique_persons': 0,
                'rotation_logs': []
            })
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total Left/Right rotations
        c.execute("SELECT COUNT(*) FROM recognition_log WHERE pose IN ('Left', 'Right')")
        total_rotations = c.fetchone()[0]
        
        # Unique persons with Left/Right detections
        c.execute("SELECT COUNT(DISTINCT name) FROM recognition_log WHERE pose IN ('Left', 'Right') AND name != 'Unknown'")
        unique_persons = c.fetchone()[0]
        
        # Recent rotation logs (last 10 - ONLY Left/Right)
        c.execute("""
            SELECT timestamp, name, similarity, pose 
            FROM recognition_log 
            WHERE pose IN ('Left', 'Right')
            ORDER BY id DESC LIMIT 10
        """)
        rotation_logs = c.fetchall()
        
        conn.close()
        
        return jsonify({
            'total_rotations': total_rotations,
            'unique_persons': unique_persons,
            'rotation_logs': rotation_logs
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'total_rotations': 0,
            'unique_persons': 0,
            'rotation_logs': []
        })
    
@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'running',
        'model_loaded': face_app is not None,
        'embeddings_count': len(known_encodings) if known_encodings is not None else 0
    })

# ========= MAIN =========
if __name__ == '__main__':
    print("\n" + "="*60)
    print(" 🚀 FLASK FACE RECOGNITION WEB APP")
    print(" Version 2.0 - With 8 Menu Options")
    print("="*60)
    
    # Initialize model
    if not initialize_model():
        print("\n[ERROR] Cannot load embeddings!")
        print("[INFO] Please run: python main_system.py")
        print("[INFO] Then select Option 4 (Build Embeddings)")
        exit(1)
    
    # Initialize database
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Port 8080 (tránh conflict với AirPlay trên macOS)
    PORT = 8080
    
    print("\n[INFO] ✅ Server ready!")
    print(f"[INFO] Open browser: http://localhost:{PORT}")
    print("[INFO] Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n[ERROR] Port {PORT} is already in use!")
            print(f"[INFO] Try: lsof -i :{PORT} to find the process")
            print("[INFO] Or change PORT variable in app.py")
        else:
            raise
