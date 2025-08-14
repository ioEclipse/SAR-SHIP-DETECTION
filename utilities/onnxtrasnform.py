from ultralytics import YOLO
import os

# 1. Load the trained YOLOv11m model
model_path = os.path.join("..", "YOLOv11m", "best.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# 2. Export to ONNX format
#    - dynamic: True to avoid fixed dimensions
#    - simplify: True to simplify the graph
#    - opset: 12 for compatibility
print("Exporting model to ONNX format...")
onnx_path = model.export(format="onnx", dynamic=True, simplify=True, opset=12)
print(f"Model successfully exported to: {onnx_path}")

# 3. Verify the export
if os.path.exists(onnx_path):
    print("✅ ONNX export successful!")
    print(f"ONNX model saved at: {onnx_path}")
else:
    print("❌ ONNX export failed!")
