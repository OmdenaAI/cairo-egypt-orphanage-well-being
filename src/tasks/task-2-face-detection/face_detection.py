from ultralytics import YOLO
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt"):
        self.model = YOLO(model_path)

    def detect_face(self, frame, return_oneface=False, return_coordinates=False):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = []

        # Detect faces
        results = self.model.predict(frame, verbose=False, show=False, conf=0.25)[0]

        # For each face, extract the bounding box, the landmarks, and confidence
        for result in results:
            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            x1, x2, y1, y2 = x, x + w, y, y + h

            detected_face = frame[y: y + h, x: x + w].copy()
            detected_face = cv2.resize(detected_face, (240, 240))

            if return_oneface:
                if return_coordinates:
                    return [detected_face, (x1, x2, y1, y2)]

                return detected_face

            if return_coordinates:
                detections.append([detected_face, (int(x1), int(x2), int(y1), int(y2))])
            else:
                detections.append(detected_face)

        return detections
