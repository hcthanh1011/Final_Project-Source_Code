import cv2
import platform
import time

def get_camera_backend():
    """Get optimal camera backend for current OS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_ANY


def open_camera(camera_id=0, width=640, height=480, fps=30, max_retries=3):
    """
    Open camera with cross-platform compatibility.
    Returns cv2.VideoCapture object or None if failed.
    """
    backend = get_camera_backend()

    for attempt in range(max_retries):
        print(f"[INFO] Opening camera (attempt {attempt+1}/{max_retries})...")
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            # Fallback without backend
            cap.release()
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            # Last fallback: camera_id = -1
            cap.release()
            cap = cv2.VideoCapture(-1, backend)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            if platform.system() == "Darwin":
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print("[INFO] Warming up camera...")
            ok = False
            for _ in range(10):
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break
                time.sleep(0.1)

            if ok:
                print("[INFO] Camera ready!")
                return cap

            cap.release()
            time.sleep(0.5)

        cap.release()
        time.sleep(0.5)

    print("[ERROR] Failed to open camera after all attempts")
    return None


def safe_release_camera(cap):
    """Safely release camera and destroy windows."""
    if cap is not None and cap.isOpened():
        cap.release()
        print("[INFO] Camera released")

    try:
        cv2.destroyAllWindows()
        if platform.system() == "Darwin":
            for _ in range(5):
                cv2.waitKey(1)
                time.sleep(0.05)
        else:
            cv2.waitKey(1)
    except Exception:
        pass
