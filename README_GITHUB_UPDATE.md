# DIDS Phase 3 + Phase 4A integration (Saswat + Shashank)

This folder contains reusable modules for Phase 3 facial landmarks and Phase 4A EAR/MAR drowsiness analysis, integrated with the supplied Phase 2 preprocessing module.

## Files
- `phase2.py`: existing Phase 2 preprocessing.
- `face_landmarks.py`: Saswat's Dlib 68-point landmark detector. Accepts a processed grayscale frame; no camera loop.
- `ear_mar.py`: Shashank's EAR/MAR calculations; no camera loop.
- `drowsiness.py`: temporal smoothing and duration-based eye-closure/yawn events.
- `main.py`: one-camera-loop demo integrating Phase 2 → Phase 3 → Phase 4A.
- `test_phase2.py`, `test_phase3_phase4.py`: offline tests.

## Setup (PowerShell)
Use a Python version supported by your Dlib build. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```
Install a compatible Dlib package separately if it is not already installed. Download `shape_predictor_68_face_landmarks.dat` and place it at:
```text
models/shape_predictor_68_face_landmarks.dat
```
The model is not included in this repository.

## Tests
```powershell
python -m unittest -v test_phase2.py test_phase3_phase4.py
```

## Run
```powershell
python main.py --camera 0 --model models/shape_predictor_68_face_landmarks.dat
```
Press Q or Esc to quit. Only `main.py` opens the camera. All other modules receive frames or coordinates.

## Interfaces
```python
gray, preprocessing_ms = FramePreprocessor().process(bgr_frame)
faces = FaceLandmarkDetector(model_path).detect(gray)
ear = average_ear(face.left_eye, face.right_eye)
mar = mouth_aspect_ratio(face.mouth)
result = analyzer.update(ear, mar)
```
Dlib indices (Python zero-based): right eye 36:42, left eye 42:48, mouth 48:68. Coordinates are in the processed/resized frame.

## Scope and limitations
This update implements the Saswat Phase 3 and Shashank Phase 4A modules and connects them to Phase 2. It does not complete Phase 4B (head pose/reaction test) or Phase 5 (decision/alerts/database). EAR/MAR thresholds and durations are prototype values, not clinically validated. Offline tests do not prove live webcam/model accuracy or performance.
