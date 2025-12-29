import platform
from pathlib import Path

def get_insightface_model_path():
    """Get InsightFace model directory for current OS."""
    home = Path.home()
    # Same path for all OS; just documented
    model_dir = home / ".insightface" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)


def verify_insightface_models():
    """Check if InsightFace models are downloaded."""
    model_path = get_insightface_model_path()
    buffalo_path = Path(model_path) / "buffalo_l"

    if buffalo_path.exists():
        print(f"[INFO] InsightFace models found: {buffalo_path}")
        return True

    print(f"[WARNING] InsightFace models NOT found at: {buffalo_path}")
    print("[INFO] Models will be downloaded automatically on first use.")
    print("[INFO] This may take a few minutes depending on your connection.")
    return False
