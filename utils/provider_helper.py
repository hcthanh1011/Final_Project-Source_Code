#!/usr/bin/env python3
"""
Provider Helper - FIXED for macOS
CoreML causes issues with InsightFace - use CPU only
"""

import platform

def get_optimal_providers():
    """
    CRITICAL FIX for macOS: Do NOT use CoreML with InsightFace
    CoreML causes "different ranks" error
    """
    providers = []
    
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        
        system = platform.system()
        
        # Priority 1: GPU (CUDA) - if available
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
            print("[INFO] Using CUDA GPU acceleration")
        
        # CRITICAL: Skip CoreML on macOS due to InsightFace incompatibility
        # The error: "CoreML static output shape ... have different ranks"
        # happens when using CoreML with buffalo_l model
        
        # Priority 2: CPU (always use as fallback)
        providers.append('CPUExecutionProvider')
        print(f"[INFO] Using CPU provider")
        
        if system == "Darwin":
            print("[INFO] macOS detected - CoreML disabled (InsightFace compatibility)")
        
        print(f"[INFO] Provider list: {providers}")
        
    except ImportError:
        print("[WARNING] onnxruntime not found, defaulting to CPU")
        providers = ['CPUExecutionProvider']
    except Exception as e:
        print(f"[WARNING] Provider detection error: {e}")
        providers = ['CPUExecutionProvider']
    
    return providers

def check_gpu_support():
    """Check if CUDA GPU is available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("[INFO] ✅ CUDA GPU detected")
            return True
        else:
            print("[INFO] ℹ️  Running on CPU")
            return False
    except:
        return False
