#!/usr/bin/env python3
"""
Verify installation and system compatibility
"""
import sys
import platform

def check_installation():
    print("="*60)
    print("  INSTALLATION CHECKER")
    print("="*60)
    print(f"\nOS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check required packages
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'insightface': 'insightface',
        'onnxruntime': 'onnxruntime',
        'sklearn': 'scikit-learn',
    }
    
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES:")
    print("="*60)
    
    all_good = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_good = False
    
    # Check ONNX providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"\n📊 ONNX Providers: {', '.join(providers)}")
        
        if 'CUDAExecutionProvider' in providers:
            print("🚀 GPU acceleration available!")
        else:
            print("ℹ️ Running on CPU")
    except:
        pass
    
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL DEPENDENCIES INSTALLED!")
        print("\nYou can run: python main_system.py")
    else:
        print("❌ MISSING DEPENDENCIES!")
        print("\nRun: pip install -r requirements.txt")
    print("="*60)

if __name__ == "__main__":
    check_installation()
