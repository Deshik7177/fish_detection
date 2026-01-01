from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO
import cv2
import uuid
import os

from normalize import normalize_species

# -------------------------------
# App setup
# -------------------------------
app = FastAPI()

# Static + template mounting
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

templates = Jinja2Templates(directory="templates")

# -------------------------------
# Load YOLO model ONCE
# -------------------------------
model = YOLO("runs/detect/train/weights/best.pt")

# -------------------------------
# Directories
# -------------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# All supported species (UI only)
# -------------------------------
ALL_SPECIES = [
    "angel",
    "damsel",
    "grouper",
    "jack",
    "parrot",
    "shark",
    "snapper",
    "spade",
    "surgeon",
    "trigger",
    "tuna",
    "wrasse",
    "moorish idol"
]

# -------------------------------
# Home page
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original": None,
            "detected": None,
            "species": [],
            "all_species": ALL_SPECIES
        }
    )

# -------------------------------
# Prediction route
# -------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{uid}.jpg"
    output_path = f"{OUTPUT_DIR}/{uid}.jpg"

    # Save uploaded image
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run YOLO inference
    results = model(input_path)
    img = cv2.imread(input_path)

    detected_species = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_label = model.names[cls_id]

            # Normalize to common name
            species = normalize_species(raw_label)
            detected_species.add(species)

            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{species} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Save output image
    cv2.imwrite(output_path, img)

    # Sort species for clean UI
    detected_species = sorted(detected_species)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original": f"/uploads/{uid}.jpg",
            "detected": f"/outputs/{uid}.jpg",
            "species": detected_species,
            "all_species": ALL_SPECIES
        }
    )
