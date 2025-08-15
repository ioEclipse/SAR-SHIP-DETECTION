from ultralytics import YOLO

# 1. Charger le modèle PyTorch
model = YOLO("best1.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("best1.onnx")