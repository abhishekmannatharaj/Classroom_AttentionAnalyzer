import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def detect(self, frame):
        h, w = frame.shape[:2]
        results = self.detector.process(frame)

        faces = []
        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = x1 + int(box.width * w)
                y2 = y1 + int(box.height * h)
                faces.append((x1, y1, x2, y2))
        return faces
