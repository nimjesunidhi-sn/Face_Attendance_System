from fastapi import FastAPI, UploadFile, File
from recognition import recognize_face

import cv2
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Face Attendance API Running"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"error": "Image decoding failed"}

    name = recognize_face(image)

    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_file = Path(f"{date_str}_attendance.csv")

    file_exists = csv_file.exists()
    already_marked = False

    if file_exists:
        with open(csv_file, "r", encoding="utf-8") as r:
            for row in r.readlines():
                if name in row:
                    already_marked = True
                    break

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Time"])

        if name != "Unknown" and not already_marked:
            writer.writerow([
                name,
                datetime.now().strftime("%H:%M:%S")
            ])

    return {
        "name": name,
        "attendance": "Marked" if name != "Unknown" else "Not Marked"
    }