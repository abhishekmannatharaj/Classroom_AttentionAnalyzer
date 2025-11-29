# engagement_algorithms.py
# Stable, classroom-ready facial engagement helper functions

import cv2
import numpy as np
import math
import time


# ------------------------------------------------------------
# 1. Calculate iris-based gaze direction using MediaPipe Face Mesh
# ------------------------------------------------------------
def calculate_gaze_with_iris(landmarks, width, height):
    """
    Returns gaze_x, gaze_y (normalized 0-1)
    Uses MediaPipe iris indices:
        Left: 468 center
        Right: 473 center
    """

    # Eye centers (left & right iris)
    left_iris = landmarks.landmark[468]
    right_iris = landmarks.landmark[473]

    left_x = int(left_iris.x * width)
    left_y = int(left_iris.y * height)
    right_x = int(right_iris.x * width)
    right_y = int(right_iris.y * height)

    # average of both iris positions
    gaze_x = (left_x + right_x) / 2
    gaze_y = (left_y + right_y) / 2

    # Normalize 0–1
    gaze_x_norm = gaze_x / width
    gaze_y_norm = gaze_y / height

    return gaze_x_norm, gaze_y_norm


# ------------------------------------------------------------
# 2. Determine if student is looking at screen
# ------------------------------------------------------------
def calculate_eye_contact(gaze_x, gaze_y):
    """
    Defines eye contact when iris is roughly centered.
    These values were derived from your reference model.
    """

    if gaze_x is None or gaze_y is None:
        return False

    # Acceptable centered region
    if 0.38 <= gaze_x <= 0.62 and 0.35 <= gaze_y <= 0.62:
        return True
    return False


# ------------------------------------------------------------
# 3. Blinking ratio using EAR-like method
# ------------------------------------------------------------
def calculate_blinking_ratio(landmarks, width, height):
    """
    EAR based blink detection:
    We use face mesh indices:
        Left eye: 33, 160, 158, 133, 153, 144
        Right eye: 362, 385, 387, 263, 373, 380
    """

    LEFT = [33, 160, 158, 133, 153, 144]
    RIGHT = [362, 385, 387, 263, 373, 380]

    def eye_aspect(lmk_set):
        pts = []
        for i in lmk_set:
            lm = landmarks.landmark[i]
            pts.append((lm.x * width, lm.y * height))
        pts = np.array(pts)

        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])

        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    left_ear = eye_aspect(LEFT)
    right_ear = eye_aspect(RIGHT)
    ear = (left_ear + right_ear) / 2.0

    return ear


# ------------------------------------------------------------
# 4. Mouth aspect ratio (yawning)
# ------------------------------------------------------------
def calculate_mouth_aspect_ratio(landmarks, width, height):
    """
    MAR using:
        Top: 13
        Bottom: 14
        Left: 78
        Right: 308
    """

    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    left = landmarks.landmark[78]
    right = landmarks.landmark[308]

    top = np.array([top.x * width, top.y * height])
    bottom = np.array([bottom.x * width, bottom.y * height])
    leftp = np.array([left.x * width, left.y * height])
    rightp = np.array([right.x * width, right.y * height])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(leftp - rightp)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


# ------------------------------------------------------------
# 5. Head pose estimation (yaw, pitch, roll)
# ------------------------------------------------------------
def calculate_head_pose(landmarks, width, height):
    """
    Uses Mediapipe 6-point pose estimation.
    Returns pitch, yaw, roll in degrees.
    """

    image_points = np.array([
        (landmarks.landmark[1].x * width, landmarks.landmark[1].y * height),     # nose tip
        (landmarks.landmark[152].x * width, landmarks.landmark[152].y * height), # chin
        (landmarks.landmark[33].x * width, landmarks.landmark[33].y * height),   # left eye outer
        (landmarks.landmark[263].x * width, landmarks.landmark[263].y * height), # right eye outer
        (landmarks.landmark[61].x * width, landmarks.landmark[61].y * height),   # left mouth corner
        (landmarks.landmark[291].x * width, landmarks.landmark[291].y * height)  # right mouth corner
    ], dtype="double")

    # 3D model reference points
    model_points = np.array([
        (0.0, 0.0, 0.0),      # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal = width
    camera_matrix = np.array(
        [[focal, 0, width / 2],
         [0, focal, height / 2],
         [0, 0, 1]],
        dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(model_points, image_points,
                                       camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

    pitch = float(euler[0])
    yaw   = float(euler[1])
    roll  = float(euler[2])

    return pitch, yaw, roll


# ------------------------------------------------------------
# 6. Eye Contact Buffer (temporal smoothing)
# ------------------------------------------------------------
class EyeContactBuffer:
    """
    Tracks duration of eye contact / no contact.
    Used for stable attention recognition.
    """

    def __init__(self):
        self.eye_contact_start = None
        self.eye_off_start = None
        self.current_state = "unknown"
        self.min_eye_contact_duration = 0.8   # sec
        self.min_eye_off_duration = 0.8       # sec

    def update_eye_contact(self, is_eye_contact):
        t = time.time()

        if is_eye_contact:
            # reset no-eye-contact timer
            self.eye_off_start = None
            if self.eye_contact_start is None:
                self.eye_contact_start = t
            elif t - self.eye_contact_start >= self.min_eye_contact_duration:
                self.current_state = "eye_contact"
        else:
            # reset eye-contact timer
            self.eye_contact_start = None
            if self.eye_off_start is None:
                self.eye_off_start = t
            elif t - self.eye_off_start >= self.min_eye_off_duration:
                self.current_state = "no_eye_contact"
