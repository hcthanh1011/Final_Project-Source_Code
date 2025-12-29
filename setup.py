#!/usr/bin/env python3
"""
Cross-platform setup script for Face Recognition System
"""
import os
import sys
import platform
import subprocess

def main():
    print("="*60)
    print("  FACE RECOGNITION SYSTEM - SETUP")
    print("="*60)
    
    system = platform.system()
    print(f"\n[INFO] Detected OS: {system}")
    print(f"[INFO] Python: {sys.version}")
    
    # Create directories
    dirs = ["dataset", "models", "logs", "utils"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[INFO] ✅ Created directory: {d}")
    
    # Install dependencies
    print("\n[INFO] Installing dependencies...")
    
    if system == "Windows":
        # Windows specific
        print("[INFO] Installing for Windows...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    elif system == "Darwin":
        # macOS specific
        print("[INFO] Installing for macOS...")
        
        # Check for Apple Silicon
        if platform.processor() == "arm":
            print("[INFO] Detected Apple Silicon (M1/M2/M3)")
            print("[INFO] Using optimized packages...")
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    else:
        # Linux
        print("[INFO] Installing for Linux...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\n" + "="*60)
    print("  ✅ SETUP COMPLETE!")
    print("="*60)
    print("\n💡 Next steps:")
    print("  1. Run: python main_system.py")
    print("  2. Select Option 1 to collect face images")
    print("  3. Select Option 4 to build embeddings")
    print("  4. Select Option 5 or 8 for recognition")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
