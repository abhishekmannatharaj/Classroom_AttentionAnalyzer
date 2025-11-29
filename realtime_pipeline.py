# ===============================================
#  realtime_pipeline.py  (FINAL UPDATED VERSION)
#  Multi-student realtime classroom engagement
#  Writes compact JSON for frontend every frame
# ===============================================

import time
import traceback
import json
import os
from collections import deque, defaultdict

import cv2
import numpy as np

# optional modules
USE_CUSTOM_FACE_DETECTOR = False
USE_CUSTOM_FACE_RECOGNIZER = False
try:
    from face_detector import FaceDetector
    USE_CUSTOM_FACE_DETECTOR = True
except:
    FaceDetector = None

try:
    from face_recognition import FaceRecognition
    USE_CUSTOM_FACE_RECOGNIZER = True
except:
    FaceRecognition = None

# mediapipe
try:
    import mediapipe as mp
except:
    print("Install mediapipe: pip install mediapipe")
    raise

# ---------------- CONFIG ----------------
SMOOTH_WINDOW = 8
YAW_THRESHOLD = 20.0
EYE_CONTACT_MIN = 0.6
EAR_SLEEP = 0.14
EAR_DROWSY = 0.19
MAR_YAWN = 0.60
HAND_MARGIN = 0.05

# JSON OUTPUT
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.json")
TMP_JSON = OUTPUT_JSON + ".tmp"

# ---------------- Mediapipe Initialization ----------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=8, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_landmark_pts(lm, idxs, w, h):
    return [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in idxs]

def eye_aspect_ratio(eye):
    eye = np.array(eye)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def mouth_aspect_ratio(mouth):
    top, bottom, left, right = map(np.array, mouth)
    return np.linalg.norm(top - bottom) / np.linalg.norm(left - right)

def estimate_head_pose(lm, w, h):
    try:
        image_points = np.array([
            (lm.landmark[1].x * w, lm.landmark[1].y * h),
            (lm.landmark[152].x * w, lm.landmark[152].y * h),
            (lm.landmark[33].x * w, lm.landmark[33].y * h),
            (lm.landmark[263].x * w, lm.landmark[263].y * h),
            (lm.landmark[61].x * w, lm.landmark[61].y * h),
            (lm.landmark[291].x * w, lm.landmark[291].y * h)
        ], dtype="double")

        model_points = np.array([
            (0, 0, 0),
            (0, -63, -12),
            (-43, 32, -26),
            (43, 32, -26),
            (-28, -28, -24),
            (28, -28, -24)
        ], dtype="double")

        focal = w
        cam_matrix = np.array([[focal, 0, w/2],
                               [0, focal, h/2],
                               [0, 0, 1]])
        dist = np.zeros((4,1))

        _, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_matrix, dist)
        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch, yaw, roll = euler.flatten()

        return float(pitch), float(yaw), float(roll)

    except:
        return 0,0,0

def save_json(data):
    try:
        with open(TMP_JSON, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(TMP_JSON, OUTPUT_JSON)
    except Exception as e:
        print("JSON write error:", e)

# -----------------------------------------------------------
# Tracking system
# -----------------------------------------------------------

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        if not self.tracks:
            for d in detections:
                self.tracks[self.next_id] = d
                self.next_id += 1
            return self.tracks.copy()

        new_tracks = {}
        used = set()

        for tid, prev in self.tracks.items():
            px, py = bbox_center(prev)
            best = None
            bestD = float("inf")

            for d in detections:
                if tuple(d) in used: continue
                cx, cy = bbox_center(d)
                dist = (px - cx)**2 + (py - cy)**2
                if dist < bestD:
                    bestD = dist
                    best = d

            if best is not None:
                new_tracks[tid] = best
                used.add(tuple(best))

        for d in detections:
            if tuple(d) not in used:
                new_tracks[self.next_id] = d
                self.next_id += 1

        self.tracks = new_tracks
        return new_tracks.copy()

# -----------------------------------------------------------
# Main Loop
# -----------------------------------------------------------

def run_camera(cam=0):
    print("Starting Pipeline...")
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera error")
        return

    tracker = SimpleTracker()

    bufEAR = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    bufMAR = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    bufYAW = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    bufEYE = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))

    while True:
        ok, frame = cap.read()
        if not ok: break

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection
        boxes = []
        det = mp_face_detection.process(rgb)
        if det.detections:
            for d in det.detections:
                bb = d.location_data.relative_bounding_box
                x1 = int(bb.xmin * W)
                y1 = int(bb.ymin * H)
                x2 = int((bb.xmin+bb.width)*W)
                y2 = int((bb.ymin+bb.height)*H)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1,y1,x2,y2])

        tracks = tracker.update(boxes)

        # Mesh
        mesh_map = []
        mesh = mp_face_mesh.process(rgb)
        if mesh.multi_face_landmarks:
            for lm in mesh.multi_face_landmarks:
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                mesh_map.append(((sum(xs)/len(xs))*W, (sum(ys)/len(ys))*H, lm))

        # Per-frame JSON
        json_students = {}
        total_distraction = 0
        hand_raised_flag = False

        for tid, box in tracks.items():

            x1,y1,x2,y2 = map(int, box)
            cx, cy = bbox_center(box)

            # Find matching face mesh
            best = None
            bestDist = float("inf")

            for mx,my,lm in mesh_map:
                d = (mx-cx)**2+(my-cy)**2
                if d < bestDist:
                    bestDist = d
                    best = lm

            if not best:
                continue

            # EAR / MAR
            leftEye = get_landmark_pts(best, LEFT_EYE_IDX, W, H)
            rightEye = get_landmark_pts(best, RIGHT_EYE_IDX, W, H)

            EAR = (eye_aspect_ratio(leftEye)+eye_aspect_ratio(rightEye))/2
            bufEAR[tid].append(EAR)
            EARs = sum(bufEAR[tid]) / len(bufEAR[tid])

            mouth = get_landmark_pts(best, MOUTH_IDX, W, H)
            MAR = mouth_aspect_ratio(mouth)
            bufMAR[tid].append(MAR)
            MARs = sum(bufMAR[tid]) / len(bufMAR[tid])

            # Head pose
            pitch,yaw,roll = estimate_head_pose(best, W, H)
            bufYAW[tid].append(yaw)
            sYaw = sum(bufYAW[tid]) / len(bufYAW[tid])

            # Status Logic
            if EARs < EAR_SLEEP:
                status = "SLEEPING"
                distraction = "OFF"
            elif EARs < EAR_DROWSY:
                status = "DROWSY"
                distraction = "OFF"
            elif MARs > MAR_YAWN:
                status = "YAWNING"
                distraction = "OFF"
            elif abs(sYaw) > YAW_THRESHOLD:
                status = "DISTRACTED"
                distraction = "ON"
                total_distraction += 1
            else:
                status = "ENGAGED"
                distraction = "OFF"

            # Fill JSON
            json_students[str(tid)] = {
                "status": status,
                "ear": round(EARs,3),
                "mar": round(MARs,3),
                "yaw": round(sYaw,2),
                "distraction": distraction
            }

            # Draw
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, status, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,255,255),2)

        # ---- GLOBAL DATA FOR FRONTEND ----
        global_distraction = int((total_distraction / max(len(json_students),1)) * 100)

        json_payload = {
            "timestamp": int(time.time()),
            "detected_count": len(json_students),
            "global_distraction": global_distraction,
            "global_hands_raised": hand_raised_flag,
            "students": json_students
        }

        save_json(json_payload)

        cv2.imshow("Realtime Classroom Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline Stopped.")


if __name__ == "__main__":
    try:
        run_camera(0)
    except:
        traceback.print_exc()
