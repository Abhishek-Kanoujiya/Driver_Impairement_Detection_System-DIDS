# Driver Impairment Detection System (DIDS)

This repository contains modular components for the Driver Impairment Detection System:
- `phase2.py` — Video frame acquisition and preprocessing (Phase 2)
- `face_landmarks.py` — Dlib 68-point facial landmark detector (Phase 3)
- `ear_mar.py` — Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) metrics (Phase 4A)
- `drowsiness.py` — Duration-based prolonged eye-closure and yawn event tracking (Phase 4A)
- `head_pose.py` — 3D head pose estimation and distraction tracking via solvePnP (Phase 4B)
- `main.py` — Unified camera pipeline connecting Phase 2 → Phase 3 → Phase 4A + Phase 4B
- `test_phase2.py`, `test_phase3_phase4.py`, `test_phase4b.py` — Offline unit tests

## Setup (Windows PowerShell)
Install dependencies in your Python environment:
```powershell
python -m pip install -r requirements.txt
```

Download the Dlib 68-point model file `shape_predictor_68_face_landmarks.dat` and place it in the `models` folder:
```text
models/shape_predictor_68_face_landmarks.dat
```

## Run Offline Unit Tests
Run the complete offline test suite (no camera or landmark weights required):
```powershell
python -m unittest -v test_phase2.py test_phase3_phase4.py test_phase4b.py
```

## Run Integrated Pipeline
Run the live pipeline using the default camera and landmark predictor:
```powershell
python main.py --camera 0 --model models/shape_predictor_68_face_landmarks.dat
```
Press `q` or `Esc` to quit.

Configurable CLI arguments:
- `--camera`: Camera index (default: `0`)
- `--model`: Path to 68-point landmark model (default: `models/shape_predictor_68_face_landmarks.dat`)
- `--nod-pitch`: Pitch threshold in degrees for head nodding (default: `15.0`)
- `--distract-yaw`: Yaw threshold in degrees for distraction (default: `20.0`)
- `--width`: Frame width (default: `640`)
- `--height`: Frame height (default: `480`)

## Module Interfaces
```python
# Phase 2: Frame Preprocessing
gray, preprocessing_ms = FramePreprocessor().process(bgr_frame)

# Phase 3: Facial Landmark Localization
faces = FaceLandmarkDetector(model_path).detect(gray)

# Phase 4A: Drowsiness & Fatigue
ear = average_ear(face.left_eye, face.right_eye)
mar = mouth_aspect_ratio(face.mouth)
drowsiness_result = drowsiness_analyzer.update(ear, mar)

# Phase 4B: 3D Head Pose & Distraction Tracking
image_2d_points = extract_image_points(face.points)
camera_matrix = get_camera_matrix(width, height)
dist_coeffs = get_distortion_coefficients()
pitch_deg, yaw_deg, roll_deg, rvec, tvec = estimate_head_pose(image_2d_points, camera_matrix, dist_coeffs)
pose_result = head_pose_tracker.update(pitch_deg, yaw_deg, roll_deg, rvec, tvec, camera_matrix, dist_coeffs)
```

## Default Thresholds
- **EAR Threshold:** `0.23` (closure duration: `1.0` s)
- **MAR Threshold:** `0.65` (yawn duration: `1.2` s)
- **Pitch Threshold (Nodding):** `15.0` deg (`15` consecutive frames / ~0.5 s)
- **Yaw Threshold (Distraction):** `20.0` deg (`20` consecutive frames / ~0.67 s)
