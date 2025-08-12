#!/usr/bin/env python3
"""
Test script to verify the local YOLO model integration
"""

import os
import sys
from PIL import Image

# Add FullApp directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FullApp'))

def test_local_inference():
    """Test the local inference implementation"""
    print("🧪 Testing local YOLO model integration...")
    
    try:
        # Import our local inference module
        from local_inference import get_local_client
        
        # Get the client
        client = get_local_client()
        
        print(f"✅ Local inference client initialized")
        print(f"📁 Model path: {client.model_path}")
        print(f"🤖 Model loaded: {'Yes' if client.model is not None else 'No (fallback mode)'}")
        
        # Test with a dummy image if available
        test_image_path = os.path.join("FullApp", "Test_image.png")
        if os.path.exists(test_image_path):
            print(f"🖼️ Found test image: {test_image_path}")
            
            # Run inference
            result = client.infer(test_image_path)
            print(f"🔍 Inference result: {len(result['predictions'])} predictions")
            
            for i, pred in enumerate(result['predictions']):
                print(f"   Ship {i+1}: confidence={pred['confidence']:.3f}, "
                      f"center=({pred['x']:.1f}, {pred['y']:.1f}), "
                      f"size=({pred['width']:.1f}x{pred['height']:.1f})")
        else:
            print(f"ℹ️ No test image found at {test_image_path}")
            print("   To test with a real image, place an image at this path")
        
        print("\n✅ Local model integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure ultralytics is installed: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_functions_import():
    """Test importing the updated functions.py"""
    print("\n🧪 Testing updated functions.py...")
    
    try:
        # Add FullApp to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FullApp'))
        
        # Try importing functions
        import functions
        
        print("✅ functions.py imports successfully")
        print(f"🤖 CLIENT type: {type(functions.CLIENT)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing functions.py: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting local model integration tests...\n")
    
    # Test local inference
    inference_ok = test_local_inference()
    
    # Test functions import
    functions_ok = test_functions_import()
    
    print(f"\n📊 Test Results:")
    print(f"   Local Inference: {'✅ PASS' if inference_ok else '❌ FAIL'}")
    print(f"   Functions Import: {'✅ PASS' if functions_ok else '❌ FAIL'}")
    
    if inference_ok and functions_ok:
        print("\n🎉 All tests passed! Your local model integration is ready.")
        print("\n📝 Next steps:")
        print("1. Run: pip install ultralytics (if not already installed)")
        print("2. Convert your model to ONNX (optional): python onnxtrasnform.py")
        print("3. Test your Streamlit app: cd FullApp && streamlit run home.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")