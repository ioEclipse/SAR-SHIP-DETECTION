import os
import sys
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import tempfile

# Add the parent directory to the path to import preprocessing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class LocalYOLOInference:
    """
    Local YOLO inference class to replace Roboflow model calls
    Uses the trained best.pt model from YOLOv11m folder
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the local YOLO model
        
        Args:
            model_path: Path to the model weights file (best.pt)
        """
        if model_path is None:
            # Default path relative to FullApp directory
            model_path = os.path.join(os.path.dirname(__file__), '..', 'YOLOv11m', 'best.pt')
        
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading local YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ Local YOLO model loaded successfully")
            
        except ImportError:
            print("❌ ultralytics not installed. Please run: pip install ultralytics")
            print("For now, using fallback detection method")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model = None
    
    def infer(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Run inference on an image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            Dictionary with predictions in Roboflow-compatible format
        """
        if self.model is None:
            # Fallback: return empty predictions
            return {"predictions": []}
        
        try:
            # Run inference
            results = self.model(image_path, conf=confidence_threshold, verbose=False)
            
            # Convert results to Roboflow-compatible format
            predictions = []
            
            if len(results) > 0:
                result = results[0]  # Take first result
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box in xyxy format
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        if confidence >= confidence_threshold:
                            # Convert to center coordinates and width/height (Roboflow format)
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                            center_x = x1 + width / 2
                            center_y = y1 + height / 2
                            
                            prediction = {
                                "x": float(center_x),
                                "y": float(center_y), 
                                "width": float(width),
                                "height": float(height),
                                "confidence": confidence,
                                "class": "ship",  # Our model detects ships
                                "class_id": 0
                            }
                            predictions.append(prediction)
            
            return {"predictions": predictions}
            
        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
            return {"predictions": []}
    
    def batch_infer(self, image_paths: List[str], confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image file paths
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.infer(image_path, confidence_threshold)
            results.append(result)
        return results

# Global instance to replace CLIENT
LOCAL_CLIENT = LocalYOLOInference()

def get_local_client():
    """Get the global local inference client"""
    return LOCAL_CLIENT

def reinitialize_client(model_path: str = None):
    """Reinitialize the global client with a new model path"""
    global LOCAL_CLIENT
    LOCAL_CLIENT = LocalYOLOInference(model_path)