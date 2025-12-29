#!/usr/bin/env python3
"""
Build Face Embeddings - CPU ONLY (CoreML bypass)
CRITICAL FIX: Force CPUExecutionProvider on macOS
"""

import os
import pickle
import numpy as np
import cv2
import platform
import sys
from insightface.app import FaceAnalysis

# ========= CRITICAL FIX: FORCE CPU PROVIDER =========
# CoreML causes "different ranks" error on macOS
# Do NOT use provider_helper - force CPU
PROVIDERS = ["CPUExecutionProvider"]
print("[INFO] FORCED CPU provider (CoreML disabled due to compatibility issues)")

DATASET_DIR = "dataset"
MODELS_DIR = "models"
EMBEDDING_PATH = os.path.join(MODELS_DIR, "insightface_embeddings.pickle")

def init_insightface():
    """Initialize InsightFace with CPU provider and optimal det_size."""
    try:
        print(f"[INFO] Initializing InsightFace (OS: {platform.system()})")
        print(f"[INFO] Providers: {PROVIDERS}")
        
        app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
        
        # Use det_size=(192,192) - verified working in test_dataset.py
        app.prepare(ctx_id=0, det_size=(192, 192))
        
        print("[INFO] Detection size: 192x192")
        print("[INFO] ✅ InsightFace initialized successfully")
        return app
    except Exception as e:
        raise RuntimeError(f"InsightFace init failed: {e}")

def extract_embedding_from_image(img_path, app):
    """
    Extract embedding from 256x256 cropped image.
    
    Returns:
        (embedding, status_message)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, "cannot read image"
        
        h, w = img.shape[:2]
        
        # Try direct detection first
        faces = app.get(img)
        
        if faces:
            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            
            # Get embedding
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                emb = face.normed_embedding
            elif hasattr(face, "embedding") and face.embedding is not None:
                emb = face.embedding
                emb = emb / np.linalg.norm(emb)
            else:
                return None, "no embedding attribute"
            
            emb = emb.astype("float32")
            score = getattr(face, "det_score", 0.0)
            return emb, f"OK (score={score:.3f})"
        
        # Fallback: upscale 2x
        if max(h, w) <= 300:
            upscaled = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            faces = app.get(upscaled)
            
            if faces:
                face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
                
                if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                    emb = face.normed_embedding
                elif hasattr(face, "embedding"):
                    emb = face.embedding
                    emb = emb / np.linalg.norm(emb)
                else:
                    return None, "no embedding"
                
                emb = emb.astype("float32")
                score = getattr(face, "det_score", 0.0)
                return emb, f"OK (upscaled, score={score:.3f})"
        
        return None, f"no face detected ({w}x{h})"
        
    except Exception as e:
        return None, f"error: {str(e)[:50]}"

def main():
    print("\n" + "="*60)
    print(" BUILDING FACE EMBEDDINGS")
    print(" CPU ONLY MODE (CoreML disabled)")
    print("="*60)

    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: {DATASET_DIR}/ not found")
        input("\nPress ENTER...")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n[INFO] Initializing InsightFace...")
    try:
        app = init_insightface()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress ENTER...")
        return

    people = [p for p in os.listdir(DATASET_DIR)
              if os.path.isdir(os.path.join(DATASET_DIR, p))]
    
    if not people:
        print(f"\n❌ No person folders in {DATASET_DIR}/")
        input("\nPress ENTER...")
        return

    print(f"\n[INFO] Found {len(people)} person(s): {', '.join(people)}")
    print("\n" + "="*60)

    known_encodings = []
    known_names = []
    total_ok, total_fail = 0, 0

    for idx, name in enumerate(people, 1):
        person_dir = os.path.join(DATASET_DIR, name)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\n[{idx}/{len(people)}] Person: {name}")
        print(f"Images: {len(image_files)}")
        print("-"*60)

        ok_person = 0
        shown_fails = 0
        
        for i, fname in enumerate(image_files, 1):
            path = os.path.join(person_dir, fname)
            
            # Progress indicator
            if i % 10 == 1 or i == len(image_files):
                print(f" [{i}/{len(image_files)}]", end="\r", flush=True)
            
            emb, status = extract_embedding_from_image(path, app)
            
            if emb is not None:
                known_encodings.append(emb)
                known_names.append(name)
                ok_person += 1
                total_ok += 1
            else:
                total_fail += 1
                # Show first 5 failures
                if shown_fails < 5:
                    print(f" ❌ {fname}: {status}                    ")
                    shown_fails += 1

        rate = (ok_person / len(image_files) * 100) if image_files else 0
        print(f"\n✅ Result: {ok_person}/{len(image_files)} ({rate:.1f}%)")
        
        if ok_person == 0:
            print("⚠️  WARNING: No embeddings!")
        elif rate < 50:
            print(f"⚠️  Low success rate")

    # Save
    if not known_encodings:
        print("\n" + "="*60)
        print("❌ FAILED: No embeddings created")
        print("="*60)
        print("\n🔧 DEBUG STEPS:")
        print(" 1. Run: python3 test_dataset.py")
        print(" 2. Check if PROVIDERS shows CoreML")
        print(" 3. Share full terminal output")
        print("="*60)
        input("\nPress ENTER...")
        return

    data = {
        "encodings": np.stack(known_encodings, axis=0),
        "names": known_names,
    }
    
    try:
        with open(EMBEDDING_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"\n✅ Saved: {EMBEDDING_PATH}")
    except Exception as e:
        print(f"\n❌ Save error: {e}")
        input("\nPress ENTER...")
        return

    total = total_ok + total_fail
    rate = (total_ok / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print(" ✅ SUCCESS")
    print("="*60)
    print(f"📁 {EMBEDDING_PATH}")
    print(f"👥 Persons: {len(set(known_names))}")
    print(f"🎯 Vectors: {len(known_encodings)}")
    print(f"📊 Rate: {total_ok}/{total} ({rate:.1f}%)")
    print("="*60)
    
    if rate >= 70:
        print("\n🎉 Excellent! Ready for Option 5")
    elif rate >= 40:
        print("\n✅ Good enough for Option 5")
    else:
        print("\n⚠️  Low rate, but try Option 5 anyway")
    
    print("\n" + "="*60)
    import time
    time.sleep(0.5)
    input("\nPress ENTER to return to menu...")

if __name__ == "__main__":
    main()
