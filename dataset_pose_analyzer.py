import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
PROVIDERS = ["CPUExecutionProvider"]

YAW_LEFT = -25
YAW_RIGHT = 25
PITCH_UP = 15
PITCH_DOWN = -15


def classify_pose(yaw, pitch):
    if yaw < YAW_LEFT:
        return "left"
    elif yaw > YAW_RIGHT:
        return "right"
    elif pitch > PITCH_UP:
        return "up"
    elif pitch < PITCH_DOWN:
        return "down"
    else:
        return "frontal"


def main():
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("[INFO] Analyzing dataset pose distribution...")

    people = [p for p in os.listdir(DATASET_DIR)
              if os.path.isdir(os.path.join(DATASET_DIR, p))]

    for person in people:
        pose_count = {"frontal":0, "left":0, "right":0, "up":0, "down":0}

        folder = os.path.join(DATASET_DIR, person)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(("jpg","png"))]

        print(f"\n[INFO] Person: {person}")

        for img_name in imgs:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            faces = app.get(img)
            if len(faces) != 1:
                continue

            face = faces[0]
            yaw, pitch, roll = face.pose

            pose = classify_pose(yaw, pitch)
            pose_count[pose] += 1

        print("  Pose distribution:")
        for p, c in pose_count.items():
            print(f"    {p}: {c}")

        print("  Recommendation:")
        missing = [p for p,c in pose_count.items() if c < 5]
        if missing:
            print("    Missing poses → capture more:", missing)
        else:
            print("    Good pose diversity!")


if __name__ == "__main__":
    main()
