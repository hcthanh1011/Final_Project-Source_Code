#!/usr/bin/env python3
"""
DEBUG TOOL: Test dataset images and detection
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
PROVIDERS = ["CPUExecutionProvider"]

def main():
    print("\n" + "="*60)
    print(" DATASET DIAGNOSTIC TOOL")
    print("="*60)
    
    # Check dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: {DATASET_DIR}/ not found")
        return
    
    people = [p for p in os.listdir(DATASET_DIR)
              if os.path.isdir(os.path.join(DATASET_DIR, p))]
    
    if not people:
        print(f"\n❌ ERROR: No person folders in {DATASET_DIR}/")
        return
    
    print(f"\n📁 Found {len(people)} person(s): {', '.join(people)}")
    
    # Pick first person
    person = people[0]
    person_dir = os.path.join(DATASET_DIR, person)
    images = [f for f in os.listdir(person_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"\n❌ ERROR: No images in {person_dir}/")
        return
    
    print(f"\n📸 Testing with person: {person}")
    print(f"   Total images: {len(images)}")
    
    # Test first image
    test_img_path = os.path.join(person_dir, images[0])
    print(f"\n🔍 Testing image: {images[0]}")
    
    img = cv2.imread(test_img_path)
    if img is None:
        print(f"❌ ERROR: Cannot read image!")
        return
    
    h, w, c = img.shape
    print(f"✅ Image loaded: {w}x{h}x{c}")
    print(f"   File size: {os.path.getsize(test_img_path)} bytes")
    
    # Initialize InsightFace with different det_sizes
    det_sizes = [(640, 640), (320, 320), (256, 256), (192, 192)]
    
    for det_size in det_sizes:
        print(f"\n🧪 Testing with det_size={det_size}")
        try:
            app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
            app.prepare(ctx_id=0, det_size=det_size)
            
            # Test direct detection
            faces = app.get(img)
            print(f"   Direct detection: {len(faces)} face(s) found")
            if faces:
                for i, face in enumerate(faces):
                    score = getattr(face, 'det_score', 0.0)
                    print(f"   Face {i+1}: det_score={score:.3f}")
            
            # Test with upscaling
            if max(h, w) <= 300:
                scale = 2.0
                upscaled = cv2.resize(img, (int(w*scale), int(h*scale)),
                                     interpolation=cv2.INTER_CUBIC)
                faces_up = app.get(upscaled)
                print(f"   Upscaled (2x): {len(faces_up)} face(s) found")
                if faces_up:
                    for i, face in enumerate(faces_up):
                        score = getattr(face, 'det_score', 0.0)
                        print(f"   Face {i+1}: det_score={score:.3f}")
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Show image info summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"✅ Images are readable")
    print(f"📏 Image dimensions: {w}x{h}")
    
    if w == 256 and h == 256:
        print("✅ Images are 256x256 (cropped mode)")
    elif w == 112 and h == 112:
        print("⚠️  Images are 112x112 (aligned mode)")
        print("💡 SOLUTION: Re-run Option 1 with SAVE_MODE='cropped'")
    else:
        print(f"ℹ️  Images are {w}x{h} (original/custom size)")
    
    print("\n💡 RECOMMENDATIONS:")
    if max(h, w) < 200:
        print(" - Images are very small, detection may fail")
        print(" - Use SAVE_MODE='cropped' (256x256) in collect_images.py")
    
    print("\n🔧 QUICK FIX:")
    print(" 1. Delete dataset folder: rm -rf dataset/")
    print(" 2. Edit collect_images.py: SAVE_MODE='cropped'")
    print(" 3. Re-run Option 1 to collect new images")
    print(" 4. Run this test again to verify")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
