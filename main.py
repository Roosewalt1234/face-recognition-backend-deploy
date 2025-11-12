from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, List, Dict
from supabase import create_client
import numpy as np
import cv2

from app.settings import settings
from app.supabase_io import (
    list_face_images,
    fetch_image_as_gray,
    upload_model_files,
    download_model_files,
    get_employee_map,
)
from app.recognizer import LBPHRecognizer

# ──────────────────────────────
# Initialize
# ──────────────────────────────
app = FastAPI(title="Face Recognition Attendance Service", version="1.0")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = LBPHRecognizer()
sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

# ──────────────────────────────
# Pydantic Models
# ──────────────────────────────
class RecognizeRequest(BaseModel):
    image_path: str                    # Supabase path or URL
    mode: Optional[str] = "checkin"    # checkin or checkout

class RecognizeResponse(BaseModel):
    matched: bool
    employee_id: Optional[str]
    employee_name: Optional[str]
    confidence: float
    mode: str
    status: str

class TrainRequest(BaseModel):
    employee_ids: Optional[List[str]] = None

class TrainResponse(BaseModel):
    trained_people: int
    samples: int
    labels: int

# ──────────────────────────────
# Load model on startup
# ──────────────────────────────
@app.on_event("startup")
def _load_model():
    yml_bytes, npy_bytes = download_model_files()
    if yml_bytes and npy_bytes:
        recognizer.load_model_bytes(yml_bytes, npy_bytes)
        print("[Startup] Model loaded successfully.")
    else:
        print("[Startup] No model found in storage. Train first via /train.")

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": recognizer.is_ready()}

# ──────────────────────────────
# Recognize Face + Record Attendance
# ──────────────────────────────
@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest):
    if not recognizer.is_ready():
        yml_bytes, npy_bytes = download_model_files()
        if not (yml_bytes and npy_bytes and recognizer.load_model_bytes(yml_bytes, npy_bytes)):
            raise HTTPException(status_code=400, detail="Model not loaded. Train first.")

    # Fetch image from Supabase storage
    gray = fetch_image_as_gray(req.image_path)
    if gray is None:
        raise HTTPException(status_code=400, detail="Could not load image.")

    emp_id, conf = recognizer.predict(gray)
    matched = bool(emp_id) and conf >= settings.CONFIDENCE_PASS
    emp_name = None
    if emp_id:
        emp_map = get_employee_map([emp_id])
        emp_name = emp_map.get(emp_id)

    # Determine status
    status = "recognized" if matched else "unrecognized"

    # Record attendance in Supabase
    record = {
        "employee_id": emp_id if matched else None,
        "action": req.mode,
        "confidence": round(conf, 2),
        "image_path": req.image_path,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        sb.table("attendance_sheet").insert(record).execute()
    except Exception as e:
        print("Supabase insert error:", e)
        raise HTTPException(status_code=500, detail="Failed to insert attendance record.")

    return RecognizeResponse(
        matched=matched,
        employee_id=emp_id if matched else None,
        employee_name=emp_name,
        confidence=round(conf, 2),
        mode=req.mode,
        status=status,
    )

# ──────────────────────────────
# Train Model
# ──────────────────────────────
@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    rows = list_face_images()
    if req.employee_ids:
        rows = [r for r in rows if r.get("employee_id") in req.employee_ids]

    if not rows:
        raise HTTPException(400, "No training data found.")

    by_emp: Dict[str, List[str]] = {}
    for r in rows:
        emp_id = r.get("employee_id")
        img_path = r.get("image_path")
        if not emp_id or not img_path:
            continue
        by_emp.setdefault(emp_id, []).append(img_path)

    faces, labels, label_map = [], [], {}
    label_id = 0

    for emp_id, paths in by_emp.items():
        for path in paths:
            gray = fetch_image_as_gray(path)
            if gray is None:
                continue
            roi = recognizer.detect_face_roi(gray) or gray
            roi = cv2.resize(roi, (200, 200))
            faces.append(roi)
            labels.append(label_id)
        label_map[label_id] = emp_id
        label_id += 1

    if not faces:
        raise HTTPException(400, "No valid faces found in dataset.")

    recognizer.train_from_faces(faces, labels, label_map)
    yml_bytes, label_npy, label_json = recognizer.export_model()
    upload_model_files(yml_bytes, label_npy, label_json)

    return TrainResponse(
        trained_people=len(label_map),
        samples=len(faces),
        labels=len(set(labels))
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)