#!/usr/bin/env python3
"""
Head Pose Detection - Tuned Sensitivity
Left/Right detection more responsive
"""

import cv2
import numpy as np
import platform
import time
import signal
import sys

# Try importing MediaPipe - optional
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Global exit flag
_should_exit = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _should_exit
    print("\n\n[INFO] ⚠️ Stopping...")
    _should_exit = True

signal.signal(signal.SIGINT, signal_handler)

def detect_head_pose_insightface(face):
    """
    Use InsightFace landmarks for pose estimation.
    Best method without MediaPipe.
    """
    try:
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps
            
            if len(kps) >= 5:
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                nose_x = nose[0]
                offset = nose_x - eye_center_x
                
                eye_distance = np.linalg.norm(left_eye - right_eye)
                normalized_offset = offset / eye_distance if eye_distance > 0 else 0
                
                # TUNED: 0.10 threshold for better sensitivity
                if normalized_offset > 0.10:
                    return "Right", (0, 255, 255), normalized_offset
                elif normalized_offset < -0.10:
                    return "Left", (255, 0, 255), normalized_offset
                else:
                    return "Straight", (0, 255, 0), normalized_offset
    except Exception as e:
        pass
    
    return None, None, None

def detect_head_pose_basic(frame, face_bbox):
    """
    Enhanced basic fallback with improved sensitivity.
    """
    x1, y1, x2, y2 = face_bbox
    w = x2 - x1
    h = y2 - y1
    
    # Method 1: Face position
    face_center_x = (x1 + x2) // 2
    frame_center_x = frame.shape[1] // 2
    offset = face_center_x - frame_center_x
    offset_ratio = offset / (frame.shape[1] / 2)
    
    # Method 2: Brightness symmetry
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size > 0:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        mid = w // 2
        left_half = gray_face[:, :mid]
        right_half = gray_face[:, mid:]
        
        if left_half.size > 0 and right_half.size > 0:
            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            brightness_diff = (right_brightness - left_brightness) / 255.0
            combined_score = offset_ratio * 0.6 + brightness_diff * 0.4
        else:
            combined_score = offset_ratio
    else:
        combined_score = offset_ratio
    
    # TUNED: 0.08 threshold for better sensitivity
    if combined_score > 0.08:
        return "Right", (0, 255, 255), combined_score
    elif combined_score < -0.08:
        return "Left", (255, 0, 255), combined_score
    else:
        return "Straight", (0, 255, 0), combined_score

def detect_head_pose_mediapipe(frame):
    """
    MediaPipe-based detection (highest accuracy).
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None, None, None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        nose_tip = landmarks.landmark[1]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        nose_x = int(nose_tip.x * w)
        left_eye_x = int(left_eye.x * w)
        right_eye_x = int(right_eye.x * w)
        eye_center_x = (left_eye_x + right_eye_x) // 2
        offset = nose_x - eye_center_x
        
        # TUNED: 7px threshold for better sensitivity
        if offset > 7:
            return "Right", (0, 255, 255), offset
        elif offset < -7:
            return "Left", (255, 0, 255), offset
        else:
            return "Straight", (0, 255, 0), offset

def cleanup_opencv_aggressive():
    """
    Aggressive cleanup for macOS terminal freeze fix.
    """
    print("[INFO] 🧹 Cleaning up...")
    try:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        
        if platform.system() == "Darwin":
            for i in range(20):
                cv2.waitKey(1)
                if i % 4 == 0:
                    time.sleep(0.05)
            time.sleep(0.2)
            for _ in range(5):
                cv2.waitKey(1)
        else:
            for _ in range(5):
                cv2.waitKey(1)
        
        print("[INFO] ✅ Windows closed")
    except Exception as e:
        print(f"[WARNING] Cleanup: {e}")

def clear_input_buffer():
    """Clear stdin buffer to prevent terminal issues."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass

def main():
    """Main entry point for head pose detection."""
    global _should_exit
    
    print("\n" + "="*60)
    print(" HEAD POSE DETECTION")
    if MEDIAPIPE_AVAILABLE:
        print(" Mode: MediaPipe (Highest Accuracy)")
    else:
        print(" Mode: Enhanced InsightFace Landmarks")
        print(" 💡 Install MediaPipe for best results:")
        print("    pip install mediapipe --no-cache-dir")
    print("="*60)
    
    # Initialize face detector
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        use_insightface = True
        print("[INFO] ✅ InsightFace loaded (with landmarks)")
    except Exception as e:
        use_insightface = False
        print("[INFO] Using OpenCV Haar Cascade")
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            print("[ERROR] Cannot load face detector")
            input("\nPress ENTER...")
            return
    
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
        input("\nPress ENTER...")
        return
    
    # Set camera properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if system == "Darwin":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    
    print("[INFO] ✅ Camera ready")
    print("\n" + "="*60)
    print("📹 DETECTION ACTIVE")
    print("="*60)
    print("Instructions:")
    print(" - Turn head left/right/straight")
    print(" - Green = Straight | Yellow = Right | Magenta = Left")
    print(" - Score shows detection confidence")
    print(" - Press 'q' to quit")
    print("="*60 + "\n")
    
    try:
        while not _should_exit:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            pose_text = "No face"
            color = (0, 0, 255)
            score = 0.0
            
            # Detect faces
            if use_insightface:
                try:
                    faces = app.get(frame)
                    if faces:
                        face = faces[0]
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        
                        # Priority 1: MediaPipe (if available)
                        if MEDIAPIPE_AVAILABLE:
                            result = detect_head_pose_mediapipe(frame)
                            if result[0] is not None:
                                pose_text, color, score = result
                            else:
                                result = detect_head_pose_insightface(face)
                                if result[0] is not None:
                                    pose_text, color, score = result
                                else:
                                    pose_text, color, score = detect_head_pose_basic(frame, (x1, y1, x2, y2))
                        else:
                            # Priority 2: InsightFace landmarks
                            result = detect_head_pose_insightface(face)
                            if result[0] is not None:
                                pose_text, color, score = result
                            else:
                                # Priority 3: Basic method
                                pose_text, color, score = detect_head_pose_basic(frame, (x1, y1, x2, y2))
                        
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                except Exception as e:
                    pose_text = "Error"
                    score = 0.0
            else:
                # OpenCV Haar Cascade fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    pose_text, color, score = detect_head_pose_basic(frame, (x, y, x+w, y+h))
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            # Display pose with score
            cv2.putText(display, f"Pose: {pose_text}", (20, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
            cv2.putText(display, f"Score: {score:.2f}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            if not MEDIAPIPE_AVAILABLE:
                mode_text = "InsightFace Mode" if use_insightface else "Basic Mode"
                cv2.putText(display, mode_text, (20, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            cv2.imshow("Head Pose Detection", display)
            
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
        # Critical cleanup sequence
        print("\n[INFO] 🧹 Shutting down...")
        
        try:
            if cap is not None:
                cap.release()
                time.sleep(0.3)
                print("[INFO] ✅ Camera released")
        except Exception as e:
            print(f"[WARNING] Camera release: {e}")
        
        cleanup_opencv_aggressive()
        
        if platform.system() == "Darwin":
            time.sleep(0.5)
            clear_input_buffer()
        
        print("\n" + "="*60)
        print(" ✅ CLEANUP COMPLETE")
        print("="*60)
        time.sleep(0.2)
        
        try:
            input("\nPress ENTER to return to menu...")
        except:
            pass

if __name__ == "__main__":
    main()
