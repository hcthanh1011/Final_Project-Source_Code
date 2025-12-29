#!/usr/bin/env python3
"""
Face Dataset Collector - Saves ORIGINAL images for later processing
macOS Compatible - Fixed camera opening issue
"""

import os
import sys
import cv2
import time
import signal
import platform
import numpy as np
from insightface.app import FaceAnalysis

# ========= GLOBAL FLAGS =========
_should_exit = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global _should_exit
    print("\n\n[INFO] ⚠️ Interrupt detected (Ctrl+C)")
    print("[INFO] Stopping capture...")
    _should_exit = True
    try:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
    except:
        pass

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if platform.system() == "Windows":
    try:
        signal.signal(signal.SIGBREAK, signal_handler)
    except AttributeError:
        pass

# ========= CONFIG =========
DATASET_DIR = "dataset"
NUM_IMAGES_TO_SAVE = 80
MIN_FACE_SIZE = 80
BLUR_THRESHOLD = 100.0
SAVE_COOLDOWN = 0.10
PROVIDERS = ["CPUExecutionProvider"]
DETECTION_SIZE = (640, 640)

# Save mode - choose one:
# "original": Save full frame (recommended for build_embeddings)
# "cropped": Save face crop 256x256
# "aligned": Save 112x112 aligned (skip build_embeddings step)
SAVE_MODE = "cropped"  # Changed to cropped for better results

# ========= UTILITY FUNCTIONS =========

def compute_blur_score(gray_img):
    """Check if image is blurry using Laplacian variance."""
    score = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD, score

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_face_image(frame, face, filepath, mode="cropped"):
    """
    Save face image in different modes.
    
    Args:
        frame: Original frame
        face: InsightFace face object
        filepath: Save path
        mode: "original", "cropped", or "aligned"
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if mode == "original":
            # Save full original frame
            cv2.imwrite(filepath, frame)
            return True
            
        elif mode == "cropped":
            # Save larger crop (256x256) with padding
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # Add padding around face
            h, w = frame.shape[:2]
            pad = 50
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                # Resize to 256x256 for consistency
                cropped = cv2.resize(cropped, (256, 256))
                cv2.imwrite(filepath, cropped)
                return True
            return False
            
        elif mode == "aligned":
            # Save 112x112 aligned
            if hasattr(face, "align_mat") and face.align_mat is not None:
                M = face.align_mat
                aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
                cv2.imwrite(filepath, aligned)
                return True
            
            # Fallback: simple crop + resize
            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                aligned = cv2.resize(cropped, (112, 112))
                cv2.imwrite(filepath, aligned)
                return True
            return False
            
    except Exception as e:
        print(f"[ERROR] save_face_image failed: {e}")
        return False

def cleanup_opencv_macos():
    """macOS-specific OpenCV cleanup."""
    print("[INFO] 🧹 Cleaning up OpenCV windows...")
    try:
        cv2.destroyAllWindows()
        
        if platform.system() == "Darwin":
            # macOS requires multiple waitKey() calls
            for i in range(10):
                cv2.waitKey(1)
                time.sleep(0.05)
        else:
            cv2.waitKey(1)
        
        print("[INFO] ✅ Windows closed")
    except Exception as e:
        print(f"[WARNING] Cleanup error: {e}")

# ========= MAIN FUNCTION =========

def main():
    global _should_exit
    
    print("=" * 60)
    print(" FACE DATASET COLLECTOR")
    print(f" Save Mode: {SAVE_MODE.upper()}")
    print(" macOS Optimized")
    print("=" * 60 + "\n")

    # Get person name
    person_name = input("Enter person's name (e.g., JohnDoe): ").strip()
    if not person_name:
        print("[ERROR] Name cannot be empty.")
        input("\nPress ENTER to continue...")
        return

    person_dir = os.path.join(DATASET_DIR, person_name)
    ensure_dir(person_dir)
    print(f"[INFO] Dataset folder: {person_dir}")

    # Initialize InsightFace
    print("\n[INFO] Initializing InsightFace...")
    try:
        app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        print("[INFO] ✅ InsightFace ready!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize InsightFace: {e}")
        print("[TIP] Run: pip install insightface onnxruntime")
        input("\nPress ENTER to continue...")
        return

    # Open camera with platform-specific backend
    print("\n[INFO] Opening webcam...")
    system = platform.system()
    
    if system == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
        print("[INFO] Using AVFoundation backend for macOS")
    elif system == "Windows":
        backend = cv2.CAP_DSHOW
        print("[INFO] Using DirectShow backend for Windows")
    else:
        backend = cv2.CAP_ANY
        print("[INFO] Using default backend")
    
    cap = None
    camera_opened = False
    
    # Try multiple camera opening strategies
    for attempt in range(3):
        print(f"[INFO] Camera opening attempt {attempt + 1}/3...")
        
        if attempt == 0:
            cap = cv2.VideoCapture(0, backend)
        elif attempt == 1:
            cap = cv2.VideoCapture(0)  # Without backend
        else:
            cap = cv2.VideoCapture(1, backend)  # Try camera index 1
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret:
                camera_opened = True
                print("[INFO] ✅ Camera opened successfully!")
                break
            else:
                print("[WARNING] Camera opened but cannot read frame")
                cap.release()
        else:
            print(f"[WARNING] Attempt {attempt + 1} failed")
        
        time.sleep(0.5)
    
    if not camera_opened:
        print("\n" + "="*60)
        print(" ❌ CAMERA ERROR")
        print("="*60)
        print("\n🔍 TROUBLESHOOTING:")
        print(" 1. macOS: Check camera permission in System Settings")
        print("    → Privacy & Security → Camera → Terminal/Python")
        print(" 2. Close other apps using camera (Zoom, Skype, etc.)")
        print(" 3. Try external USB webcam")
        print(" 4. Run this test:")
        print("    python3 -c 'import cv2; print(cv2.VideoCapture(0).isOpened())'")
        print(" 5. macOS: Force quit Terminal and try again")
        print("="*60)
        input("\nPress ENTER to continue...")
        return

    # Set camera properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if system == "Darwin":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"[WARNING] Could not set camera properties: {e}")

    # Warm up camera
    print("[INFO] Warming up camera...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    saved_count = 0
    last_save_time = 0.0

    print("\n" + "="*60)
    print("📸 CAPTURE MODE")
    print("="*60)
    print("Instructions:")
    print(" - Look at the camera and move slightly")
    print(" - Green box = saving image")
    print(" - Yellow box = face too small")
    print(" - Orange box = image is blurry")
    print(" - Press 'q' to quit early")
    print("="*60 + "\n")

    try:
        while saved_count < NUM_IMAGES_TO_SAVE and not _should_exit:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera")
                break

            display_frame = frame.copy()
            faces = app.get(frame)
            info_text = f"Saved: {saved_count}/{NUM_IMAGES_TO_SAVE} | Faces: {len(faces)}"

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                w, h = x2 - x1, y2 - y1

                # Check face size
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(display_frame, "Too small", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    continue

                # Check blur
                gray_face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                is_blur, blur_score = compute_blur_score(gray_face)
                if is_blur:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(display_frame, f"Blur: {blur_score:.0f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    continue

                # Save with cooldown
                current_time = time.time()
                if current_time - last_save_time >= SAVE_COOLDOWN:
                    filename = f"{person_name}_{saved_count+1:04d}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    
                    # Save image
                    if save_face_image(frame, face, filepath, mode=SAVE_MODE):
                        saved_count += 1
                        last_save_time = current_time

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(display_frame, f"SAVED {saved_count}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        progress = (saved_count / NUM_IMAGES_TO_SAVE) * 100
                        print(f"[{saved_count:3d}/{NUM_IMAGES_TO_SAVE}] {progress:5.1f}% - {filename}")
                        
                        if saved_count >= NUM_IMAGES_TO_SAVE:
                            print("\n[INFO] ✅ Target reached! Finishing...")
                            break

            # Display frame
            cv2.putText(display_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Dataset Collector", display_frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] 'q' pressed - stopping...")
                break

            # Break if target reached
            if saved_count >= NUM_IMAGES_TO_SAVE:
                break

    except KeyboardInterrupt:
        print("\n[INFO] ⚠️ Keyboard interrupt detected")
    except Exception as e:
        print(f"\n[ERROR] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n[INFO] 🧹 Cleaning up...")
        
        # Release camera first
        try:
            if cap is not None:
                cap.release()
                time.sleep(0.2)
                print("[INFO] ✅ Camera released")
        except Exception as e:
            print(f"[WARNING] Camera release error: {e}")
        
        # Cleanup OpenCV windows
        cleanup_opencv_macos()
        
        # Final flush for macOS
        if platform.system() == "Darwin":
            time.sleep(0.2)

        # Summary
        print("\n" + "="*60)
        print(" 📊 COLLECTION SUMMARY")
        print("="*60)
        print(f"👤 Person: {person_name}")
        print(f"📸 Images saved: {saved_count}/{NUM_IMAGES_TO_SAVE}")
        print(f"📁 Location: {person_dir}")
        print(f"💾 Save mode: {SAVE_MODE}")
        
        if saved_count >= NUM_IMAGES_TO_SAVE:
            print("\n✅ SUCCESS: All images collected!")
            print("⏭️  NEXT STEP: Run Option 4 (Build Embeddings)")
        elif saved_count > 0:
            print(f"\n⚠️  PARTIAL: {saved_count} images collected")
            print("💡 TIP: Run this again to collect more")
        else:
            print("\n❌ FAILED: No images saved")
            print("💡 TIPS:")
            print(" - Ensure good lighting")
            print(" - Move closer to camera")
            print(" - Check camera permissions")
        
        print("="*60)
        
        # Safe input with cleanup
        if platform.system() == "Darwin":
            time.sleep(0.3)
        
        print("\n✅ Ready to return to menu")
        input("Press ENTER to continue...")

if __name__ == "__main__":
    main()
