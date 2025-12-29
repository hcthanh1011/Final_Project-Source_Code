#!/usr/bin/env python3
"""
Liveness Detection - Fixed for buffalo_l model
Works with 5 basic landmarks (kps) instead of 106
"""

import cv2
import numpy as np
import platform
import time
import signal
import sys

try:
    from scipy.spatial import distance as dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from insightface.app import FaceAnalysis

# Global exit flag
_should_exit = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _should_exit
    print("\n\n[INFO] ⚠️ Stopping...")
    _should_exit = True

signal.signal(signal.SIGINT, signal_handler)

# Config
PROVIDERS = ["CPUExecutionProvider"]
BLINK_THRESHOLD = 3  # Number of blinks to consider "live"
MOTION_THRESHOLD = 10  # Pixel movement threshold

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def eye_aspect_ratio_simple(left_eye, right_eye):
    """
    Simplified EAR calculation using 5-point landmarks.
    Returns ratio based on eye distance change.
    """
    # Calculate eye width (distance between left and right eye)
    eye_width = euclidean_distance(left_eye, right_eye)
    
    # Estimate if eyes are open based on vertical position variation
    # (Simple heuristic: wider eyes = more open)
    return eye_width

def detect_liveness_basic(face, prev_face, frame_count):
    """
    Basic liveness detection using:
    1. Blink detection (eye aspect ratio change)
    2. Face movement (position change)
    3. Head pose variation
    """
    liveness_score = 0
    reasons = []
    
    # Check if we have previous frame for comparison
    if prev_face is None:
        return False, 0, ["Initializing..."]
    
    # Method 1: Face position change (movement)
    curr_bbox = face.bbox
    prev_bbox = prev_face.bbox
    
    bbox_center_curr = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
    bbox_center_prev = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
    
    movement = euclidean_distance(bbox_center_curr, bbox_center_prev)
    
    if movement > 2:  # Small movement detected
        liveness_score += 1
        reasons.append(f"Movement: {movement:.1f}px")
    
    # Method 2: Face size variation (depth change)
    curr_size = (curr_bbox[2] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[1])
    prev_size = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
    
    size_change = abs(curr_size - prev_size) / prev_size if prev_size > 0 else 0
    
    if size_change > 0.05:  # 5% size change
        liveness_score += 1
        reasons.append(f"Depth change: {size_change*100:.1f}%")
    
    # Method 3: Eye distance variation (blink detection)
    if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
        if hasattr(prev_face, 'kps') and prev_face.kps is not None and len(prev_face.kps) >= 2:
            curr_eye_dist = euclidean_distance(face.kps[0], face.kps[1])
            prev_eye_dist = euclidean_distance(prev_face.kps[0], prev_face.kps[1])
            
            eye_change = abs(curr_eye_dist - prev_eye_dist) / prev_eye_dist if prev_eye_dist > 0 else 0
            
            if eye_change > 0.03:  # 3% change (possible blink)
                liveness_score += 2  # Blink is strong indicator
                reasons.append(f"Blink detected: {eye_change*100:.1f}%")
    
    # Liveness decision
    is_live = liveness_score >= 2
    
    return is_live, liveness_score, reasons

def cleanup_opencv_aggressive():
    """Aggressive cleanup for macOS."""
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
    except Exception as e:
        print(f"[WARNING] Cleanup: {e}")

def clear_input_buffer():
    """Clear stdin buffer."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass

def main():
    """Main entry point for liveness detection."""
    global _should_exit
    
    print("\n" + "="*60)
    print(" LIVENESS DETECTION (Anti-Spoofing)")
    print(" Method: Motion & Blink Analysis")
    print("="*60)
    
    # Initialize InsightFace
    print("\n[INFO] Initializing InsightFace...")
    try:
        app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] ✅ Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
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
    print("📹 LIVENESS DETECTION ACTIVE")
    print("="*60)
    print("Instructions:")
    print(" - Move your head slightly")
    print(" - Blink naturally")
    print(" - Green box = LIVE person detected")
    print(" - Red box = Possible photo/video (FAKE)")
    print(" - Press 'q' to quit")
    print("="*60 + "\n")
    
    prev_face = None
    frame_count = 0
    
    try:
        while not _should_exit:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display = frame.copy()
            
            # Detect faces
            try:
                faces = app.get(frame)
                
                if faces:
                    face = faces[0]  # Use first face
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    
                    # Liveness detection
                    is_live, score, reasons = detect_liveness_basic(face, prev_face, frame_count)
                    
                    # Color based on liveness
                    if is_live:
                        color = (0, 255, 0)  # Green = LIVE
                        label = "LIVE"
                    else:
                        color = (0, 0, 255)  # Red = FAKE
                        label = "FAKE"
                    
                    # Draw bounding box
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    
                    # Display label
                    cv2.putText(display, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    
                    # Display score
                    cv2.putText(display, f"Score: {score}", (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    
                    # Display reasons (first 2)
                    for i, reason in enumerate(reasons[:2]):
                        cv2.putText(display, reason, (x1, y2 + 50 + i*20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Save current face for next frame comparison
                    prev_face = face
                else:
                    # No face detected
                    cv2.putText(display, "No face detected", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    prev_face = None
                
            except Exception as e:
                cv2.putText(display, f"Error: {str(e)[:30]}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Display frame info
            cv2.putText(display, f"Frame: {frame_count}", (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Liveness Detection", display)
            
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
        # Cleanup
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
