from ultralytics import YOLO

# 1. Charger le modèle PyTorch
model = YOLO("yolov8n.pt")

# 2. Exporter vers ONNX
#    - dynamic: True pour éviter des dimensions fixes
#    - simplify: True pour alléger le graph
model.export(format="onnx", dynamic=True, simplify=True, opset=12)
