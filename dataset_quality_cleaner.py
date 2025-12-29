import os
import cv2
import numpy as np
import shutil
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = "dataset"
CLEAN_DIR = "clean_dataset"

BLUR_THRESHOLD = 80.0
BRIGHTNESS_MIN = 60
BRIGHTNESS_MAX = 200
DUPLICATE_THRESHOLD = 0.92  # cosine similarity threshold

PROVIDERS = ["CPUExecutionProvider"]


def compute_blur(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def compute_brightness(img):
    return np.mean(img)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    print("[INFO] Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))

    ensure_dir(CLEAN_DIR)

    print("[INFO] Scanning dataset...")

    people = [p for p in os.listdir(DATASET_DIR)
              if os.path.isdir(os.path.join(DATASET_DIR, p))]

    for person in people:
        src_dir = os.path.join(DATASET_DIR, person)
        dst_dir = os.path.join(CLEAN_DIR, person)
        ensure_dir(dst_dir)

        print(f"\n[INFO] Processing: {person}")

        all_embeddings = []
        all_files = []

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(("jpg", "png"))]

        for file in files:
            filepath = os.path.join(src_dir, file)
            img = cv2.imread(filepath)

            if img is None:
                print(f"  - skipped (cannot read): {file}")
                continue

            faces = app.get(img)
            if len(faces) != 1:
                print(f"  - skipped (multiple/zero faces): {file}")
                continue

            face = faces[0]

            # Crop aligned
            M = face.align_mat
            aligned = cv2.warpAffine(img, M, (112,112))

            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            blur_score = compute_blur(gray)
            if blur_score < BLUR_THRESHOLD:
                print(f"  - skipped (blurry): {file}")
                continue

            brightness = compute_brightness(gray)
            if brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
                print(f"  - skipped (bad lighting): {file}")
                continue

            # Compute embedding
            emb = face.normed_embedding
            all_embeddings.append(emb)
            all_files.append((file, aligned))

        # Duplicate removal
        print("[INFO] Checking duplicates...")
        kept = []
        kept_files = []

        for i, emb in enumerate(all_embeddings):
            if not kept:
                kept.append(emb)
                kept_files.append(all_files[i])
                continue

            sims = cosine_similarity([emb], kept)[0]

            if np.max(sims) < DUPLICATE_THRESHOLD:
                kept.append(emb)
                kept_files.append(all_files[i])
            else:
                print(f"  - removed duplicate: {all_files[i][0]}")

        # Save cleaned images
        for filename, img in kept_files:
            cv2.imwrite(os.path.join(dst_dir, filename), img)

        print(f"[INFO] {len(kept_files)} clean images saved for {person}")

    print("\n[INFO] Dataset cleaning finished!")


if __name__ == "__main__":
    main()
