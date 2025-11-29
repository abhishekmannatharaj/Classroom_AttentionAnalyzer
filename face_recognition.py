import cv2
import os
import numpy as np

class FaceRecognition:
    def __init__(self, data_dir="students"):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.data_dir = data_dir

        self.labels = []
        self.images = []
        self.usns = []

        self.load_students()
        if len(self.images) > 1:
            print("Training face recognizer...")
            self.recognizer.train(self.images, np.array(self.labels))

    def load_students(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        label_id = 0
        for usn in os.listdir(self.data_dir):
            folder = os.path.join(self.data_dir, usn)
            if not os.path.isdir(folder):
                continue
            self.usns.append(usn)

            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                self.images.append(img)
                self.labels.append(label_id)

            label_id += 1

    def recognize(self, face_gray):
        if len(self.images) < 1:
            return "UNKNOWN", 0.0

        label, confidence = self.recognizer.predict(face_gray)
        usn = self.usns[label]
        return usn, confidence
