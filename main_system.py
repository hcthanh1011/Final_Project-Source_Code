#!/usr/bin/env python3
"""
=================================================
   INTEGRATED FACE RECOGNITION SYSTEM
   Combines: Face Recognition + Liveness + Pose
=================================================
"""

import sys
import os
import signal
import platform

# Global cleanup flag
_cleanup_done = False

def global_signal_handler(sig, frame):
    """Global signal handler for the entire system"""
    global _cleanup_done
    
    if _cleanup_done:
        return  # Prevent double cleanup
    
    print("\n\n" + "="*60)
    print("  ⚠️ INTERRUPT DETECTED - SHUTTING DOWN")
    print("="*60)
    
    try:
        import cv2
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
    except:
        pass
    
    _cleanup_done = True
    print("\n✅ Cleanup complete. Goodbye!")
    sys.exit(0)

# Register global handlers
signal.signal(signal.SIGINT, global_signal_handler)
if platform.system() == "Windows":
    try:
        signal.signal(signal.SIGBREAK, global_signal_handler)
    except AttributeError:
        pass


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\n" + "="*60)
    print("   🎭 FACE RECOGNITION SYSTEM - AI DETECTION PROJECT")
    print("="*60)

def print_menu():
    print("\n📋 MAIN MENU:")
    print("-" * 60)
    print("  [1] 📸 Collect Face Images (Step 1: Data Collection)")
    print("  [2] 🧹 Clean Dataset (Step 2: Remove Bad Quality)")
    print("  [3] 📊 Analyze Pose Distribution (Check Dataset Quality)")
    print("  [4] 🤖 Build Face Embeddings (Step 3: Train Model)")
    print("  [5] 🎯 Real-time Face Recognition (Main Feature)")
    print("  [6] 👁️  Liveness Detection (Anti-Spoofing)")
    print("  [7] 🔄 Head Pose Detection (Left/Right/Straight)")
    print("  [8] 🚀 INTEGRATED SYSTEM (All-in-One)")
    print("  [0] ❌ Exit")
    print("-" * 60)

def run_collect_images():
    print("\n🚀 Starting Image Collection Module...")
    try:
        import collect_images
        collect_images.main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (this is expected)")
        print("[INFO] Signal handler already cleaned up")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force cleanup
        import cv2
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        input("\n👉 Press ENTER to continue...")

def run_dataset_cleaner():
    print("\n🚀 Starting Dataset Cleaner Module...")
    try:
        import dataset_quality_cleaner
        dataset_quality_cleaner.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_pose_analyzer():
    print("\n🚀 Starting Pose Analyzer Module...")
    try:
        import dataset_pose_analyzer
        dataset_pose_analyzer.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_build_embeddings():
    print("\n🚀 Starting Embedding Builder Module...")
    try:
        import build_embeddings
        build_embeddings.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_face_recognition():
    print("\n🚀 Starting Face Recognition Module...")
    try:
        import realtime_recognition
        realtime_recognition.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_liveness_detection():
    print("\n🚀 Starting Liveness Detection Module...")
    try:
        import liveness_detector
        liveness_detector.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_head_pose():
    print("\n🚀 Starting Head Pose Detection Module...")
    try:
        import head_pose_detector
        head_pose_detector.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to continue...")

def run_integrated_system():
    """
    INTEGRATED SYSTEM: Combines all 3 modules
    - Face Recognition
    - Head Pose Detection (Left/Right/Straight)
    - Recognition Logging
    """
    print("\n🚀 Starting INTEGRATED SYSTEM...")
    print("📝 Features: Recognition + Pose Detection + Logging")
    print("⚡ Press 'q' to quit\n")
    
    try:
        import cv2
        import numpy as np
        import pickle
        import time
        from datetime import datetime
        from insightface.app import FaceAnalysis
        
        # Load face embeddings
        EMBEDDING_PATH = "models/insightface_embeddings.pickle"
        if not os.path.exists(EMBEDDING_PATH):
            print("❌ ERROR: Embeddings not found!")
            print("💡 Please run Option 4 (Build Embeddings) first.")
            input("Press Enter to continue...")
            return
        
        with open(EMBEDDING_PATH, "rb") as f:
            data = pickle.load(f)
        
        known_encodings = data["encodings"]
        known_names = np.array(data["names"])
        
        # Normalize encodings
        norms = np.linalg.norm(known_encodings, axis=1, keepdims=True)
        known_encodings = known_encodings / np.clip(norms, 1e-8, None)
        
        # Initialize InsightFace
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("✅ System ready! Look at the camera...")
        
        # Settings
        SIM_THRESHOLD = 0.45
        YAW_LEFT = -20.0
        YAW_RIGHT = 20.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = app.get(small_frame)
            
            for face in faces:
                # Bounding box
                x1, y1, x2, y2 = (face.bbox * 2).astype(int)  # Scale back
                
                # Face Recognition
                emb = face.normed_embedding if hasattr(face, 'normed_embedding') else face.embedding / np.linalg.norm(face.embedding)
                sims = np.dot(known_encodings, emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_name = known_names[best_idx]
                
                # Head Pose Detection
                yaw, pitch, roll = face.pose
                if yaw < YAW_LEFT:
                    pose_label = "LEFT"
                    pose_color = (0, 255, 0)
                elif yaw > YAW_RIGHT:
                    pose_label = "RIGHT"
                    pose_color = (0, 0, 255)
                else:
                    pose_label = "STRAIGHT"
                    pose_color = (255, 165, 0)
                
                # Display results
                if best_sim >= SIM_THRESHOLD:
                    name_display = f"{best_name} ({best_sim:.2f})"
                    box_color = (0, 255, 0)  # Green for known
                else:
                    name_display = f"Unknown ({best_sim:.2f})"
                    box_color = (0, 0, 255)  # Red for unknown
                
                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Name label
                cv2.rectangle(frame, (x1, y2), (x1+250, y2+30), box_color, cv2.FILLED)
                cv2.putText(frame, name_display, (x1+5, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                # Pose label
                cv2.rectangle(frame, (x1, y1-30), (x1+150, y1), pose_color, cv2.FILLED)
                cv2.putText(frame, f"Pose: {pose_label}", (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            cv2.imshow("Integrated Face Recognition System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System stopped successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")

def main():
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("\n👉 Select option (0-8): ").strip()
        
        if choice == "1":
            run_collect_images()
        elif choice == "2":
            run_dataset_cleaner()
        elif choice == "3":
            run_pose_analyzer()
        elif choice == "4":
            run_build_embeddings()
        elif choice == "5":
            run_face_recognition()
        elif choice == "6":
            run_liveness_detection()
        elif choice == "7":
            run_head_pose()
        elif choice == "8":
            import integrated_system
            integrated_system.main()
        elif choice == "0":
            print("\n👋 Thank you for using Face Recognition System!")
            print("🎓 Developed by: Group 9")
            sys.exit(0)
        else:
            print("\n❌ Invalid option. Please select 0-8.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
