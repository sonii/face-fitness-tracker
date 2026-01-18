import cv2
import numpy as np
import mediapipe as mp

# ---------------------------
# Choose Face Exercise
# ---------------------------
print("Choose Face Exercise:")
print("1. Jaw Open-Close")
print("2. Eyebrow Raise")
print("3. Cheek Puff")

choice = input("Enter number (1-3): ")
exercise_map = {"1": "Jaw Exercise", "2": "Eyebrow Exercise", "3": "Cheek Exercise"}
exercise_type = exercise_map.get(choice, "Jaw Exercise")

# ---------------------------
# Mediapipe FaceMesh
# ---------------------------
mp_face = mp.solutions.face_mesh
face_model = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Counters & Calibration
# ---------------------------
counter = 0
stage = None
calibration = {"jaw": [], "eyebrow": [], "cheek": []}
neutral_values = {"jaw": None, "eyebrow": None, "cheek": None}

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_model.process(rgb)

    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]

        # Key landmarks
        brow = face.landmark[105]
        eye = face.landmark[159]
        top_lip = face.landmark[13]
        bottom_lip = face.landmark[14]
        left_cheek = face.landmark[234]
        right_cheek = face.landmark[454]

        # ---------------------------
        # Calibration & Exercise
        # ---------------------------
        if exercise_type == "Jaw Exercise":
            mouth_open = abs(top_lip.y - bottom_lip.y) * h
            # Calibration
            if neutral_values["jaw"] is None:
                calibration["jaw"].append(mouth_open)
                if len(calibration["jaw"]) >= 60:
                    neutral_values["jaw"] = np.mean(calibration["jaw"])
                    print(f"[Jaw Calibration Done] Neutral={neutral_values['jaw']:.2f}")
                cv2.putText(frame, "Calibrating Jaw...", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            neutral = neutral_values["jaw"]
            # Exercise
            if stage is None:
                stage = "closed"
            if mouth_open > neutral * 1.5 and stage != "open":
                stage = "open"
            if mouth_open < neutral * 1.2 and stage == "open":
                counter += 1
                stage = "closed"

        elif exercise_type == "Eyebrow Exercise":
            brow_dist = abs(brow.y - eye.y) * h
            # Calibration
            if neutral_values["eyebrow"] is None:
                calibration["eyebrow"].append(brow_dist)
                if len(calibration["eyebrow"]) >= 60:
                    neutral_values["eyebrow"] = np.mean(calibration["eyebrow"])
                    print(f"[Eyebrow Calibration Done] Neutral={neutral_values['eyebrow']:.2f}")
                cv2.putText(frame, "Calibrating Eyebrow...", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            neutral = neutral_values["eyebrow"]
            # Exercise
            if stage is None:
                stage = "neutral"
            if brow_dist > neutral * 1.10 and stage != "raised":
                stage = "raised"
            if brow_dist < neutral * 1.03 and stage == "raised":
                counter += 1
                stage = "neutral"

        elif exercise_type == "Cheek Exercise":
            cheek_dist = abs(left_cheek.x - right_cheek.x) * w
            # Calibration
            if neutral_values["cheek"] is None:
                calibration["cheek"].append(cheek_dist)
                if len(calibration["cheek"]) >= 60:
                    neutral_values["cheek"] = np.mean(calibration["cheek"])
                    print(f"[Cheek Calibration Done] Neutral={neutral_values['cheek']:.2f}")
                cv2.putText(frame, "Calibrating Cheeks...", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            neutral = neutral_values["cheek"]

            # Thresholds
            raise_thresh = neutral * 1.15
            relax_thresh = neutral * 1.05

            # Initialize stage if None
            if stage is None:
                stage = "relaxed"

            # Detect puff
            if cheek_dist > raise_thresh and stage != "puffed":
                stage = "puffed"

            # Detect relax → count rep
            if cheek_dist < relax_thresh and stage == "puffed":
                counter += 1
                stage = "relaxed"

    # ---------------------------
    # UI Box (Name + Reps + Stage)
    # ---------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (380, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, exercise_type, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Reps: {counter}", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Stage: {stage}", (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Face Fitness Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

