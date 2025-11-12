import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import io

class LBPHRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.label_map: Dict[int, str] = {}
        self.is_trained = False

    def is_ready(self) -> bool:
        """Check if the model is trained and ready for recognition"""
        return self.is_trained and len(self.label_map) > 0

    def detect_face_roi(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face in image and return ROI"""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Use the largest face found
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Extract face ROI with some padding
        padding = int(w * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray_image.shape[1], x + w + padding)
        y2 = min(gray_image.shape[0], y + h + padding)

        face_roi = gray_image[y1:y2, x1:x2]
        return face_roi

    def train_from_faces(self, faces: List[np.ndarray], labels: List[int], label_map: Dict[int, str]):
        """Train the recognizer with face images"""
        if len(faces) == 0 or len(labels) == 0:
            raise ValueError("No training data provided")

        if len(faces) != len(labels):
            raise ValueError("Faces and labels arrays must have the same length")

        print(f"Training with {len(faces)} faces and {len(set(labels))} unique labels")

        # Convert faces to numpy array
        faces_array = np.array(faces, dtype=np.uint8)
        labels_array = np.array(labels, dtype=np.int32)

        # Train the recognizer
        self.recognizer.train(faces_array, labels_array)
        self.label_map = label_map.copy()
        self.is_trained = True

        print(f"Training completed. Label map: {self.label_map}")

    def predict(self, gray_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict face in image"""
        if not self.is_ready():
            raise ValueError("Model not trained")

        # Detect face
        face_roi = self.detect_face_roi(gray_image)
        if face_roi is None:
            return None, 0.0

        # Ensure face ROI is the right size for recognition
        face_roi = cv2.resize(face_roi, (200, 200))

        # Predict
        label, confidence = self.recognizer.predict(face_roi)

        # LBPH returns distance, lower is better. Convert to confidence score
        # Typical LBPH distances: 0-100, where <50 is good match
        confidence_score = max(0, 100 - confidence)

        # Get employee ID from label
        employee_id = self.label_map.get(label)

        return employee_id, confidence_score

    def export_model(self) -> Tuple[bytes, bytes, str]:
        """Export model files for storage"""
        if not self.is_ready():
            raise ValueError("Model not trained")

        # Export recognizer model
        yml_bytes = io.BytesIO()
        self.recognizer.write(yml_bytes)
        yml_bytes = yml_bytes.getvalue()

        # Export label map as numpy array and JSON
        labels_array = np.array(list(self.label_map.keys()), dtype=np.int32)
        labels_npy = io.BytesIO()
        np.save(labels_npy, labels_array)
        labels_npy = labels_npy.getvalue()

        labels_json = json.dumps(self.label_map)

        return yml_bytes, labels_npy, labels_json

    def load_model_bytes(self, yml_bytes: bytes, npy_bytes: bytes) -> bool:
        """Load model from bytes"""
        try:
            # Load YML model
            yml_buffer = io.BytesIO(yml_bytes)
            self.recognizer.read(yml_buffer)

            # Load label map from NPY
            npy_buffer = io.BytesIO(npy_bytes)
            labels_array = np.load(npy_buffer)

            # Reconstruct label map (this is simplified - in real implementation
            # you'd need to store the mapping separately)
            self.label_map = {i: f"emp_{i}" for i in labels_array}
            self.is_trained = True

            print(f"Model loaded with {len(self.label_map)} labels")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def load_model_files(self, yml_path: str, labels_json_path: str):
        """Load model from file paths"""
        try:
            self.recognizer.read(yml_path)

            with open(labels_json_path, 'r') as f:
                self.label_map = json.load(f)

            self.is_trained = True
            print(f"Model loaded from files with {len(self.label_map)} labels")

        except Exception as e:
            print(f"Error loading model files: {e}")
            raise