# DIDS — Saswat + Shashank deliverables (Phase 2, Phase 3, Phase 4A)

This bundle preserves the supplied Phase 2 preprocessing module and adds:
- `face_landmarks.py` — Dlib 68-point face/landmark detector (Saswat, Phase 3)
- `ear_mar.py` — EAR and MAR calculations (Shashank, Phase 4A)
- `drowsiness.py` — smoothed, duration-based prolonged eye-closure and yawn logic
- `main.py` — one-camera-loop demonstration connecting Phase 2 → Phase 3 → Phase 4A
- `test_phase3_phase4.py` — offline tests (no webcam or landmark model required)

## Important scope
This bundle completes the assigned Phase 3 and Phase 4A modules and demonstrates their connection to Phase 2. It does **not** claim that Phases 1, 4B (head pose/reaction test), or 5 (decision/alerts/database) are complete. No accuracy or FPS result is claimed.

## Setup (Windows PowerShell)
Use a virtual environment with a Python version compatible with your Dlib package:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Install a compatible Dlib distribution for your Python/Windows version. Dlib wheel support varies by Python version, so if installation fails, use a supported Python environment rather than changing the code blindly.

## Required model
Download the Dlib 68-point model file `shape_predictor_68_face_landmarks.dat` separately. Create the `models` directory and put the file here:
```text
models/shape_predictor_68_face_landmarks.dat
```
The model is not bundled. Do not commit the large model file to Git unless your team explicitly agrees to the repository/licensing policy.

## Run offline tests
From this folder:
```powershell
python -m unittest -v test_phase2.py test_phase3_phase4.py
```
These tests exercise the Phase 2 output contract, EAR/MAR math, temporal state logic, and landmark coordinate slicing/drawing. Live Dlib face detection still needs the model and a camera.

## Run integrated demo
```powershell
python main.py --camera 0 --model models/shape_predictor_68_face_landmarks.dat
```
Press `q` or `Esc` to exit. Only `main.py` owns the camera loop; modules receive frames/coordinates and do not open cameras themselves.

## Interfaces
```python
gray, preprocessing_ms = FramePreprocessor().process(bgr_frame)
faces = FaceLandmarkDetector(model_path).detect(gray)
ear = average_ear(face.left_eye, face.right_eye)
mar = mouth_aspect_ratio(face.mouth)
result = analyzer.update(ear, mar)
```
`FaceLandmarks.points` has shape `(68, 2)`. Dlib indices are zero-based in Python: right eye `[36:42]`, left eye `[42:48]`, mouth `[48:68]`. All coordinates refer to the processed grayscale frame, so draw them on that same resized frame.

## Prototype caveats
- EAR/MAR thresholds and durations are configurable starter values, not validated medical thresholds.
- Webcam lighting, face angle, glasses, occlusion, and frame rate can affect measurements.
- This is a student prototype, not a certified driver-safety device. It should not be relied on to decide whether someone is safe to drive.
