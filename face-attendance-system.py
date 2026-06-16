# %%
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

# Folder that contains your known-face photos
KNOWN_DIR = Path("D:/face_attendance_system/photos")

# (filename, display name)
PEOPLE = [
    ("francisco_lachowski.jpg", "Francisco Lachowski"),
    ("kendall_jenner.jpg", "Kendall Jenner"),
    ("tom_holland.jpg", "Tom Holland"),
    ("emma_waston.jpg", "Emma Waston"),
    ("gigi_hadid.jpg", "Gigi Hadid")
]


# %%
def load_face(path: Path, name: str):
    """Return (encoding, name) or (None, None) if file missing or face not found."""
    if not path.is_file():
        print(f"[WARN] File not found: {path}")
        return None, None
    image = face_recognition.load_image_file(str(path))
    encs = face_recognition.face_encodings(image)
    if encs:
        return encs[0], name
    print(f"[WARN] No face detected in: {path}")
    return None, None

known_face_encodings = []
known_face_names = []

for filename, name in PEOPLE:
    enc, nm = load_face(KNOWN_DIR / filename, name)
    if enc is not None:
        known_face_encodings.append(enc)
        known_face_names.append(nm)

if not known_face_encodings:
    raise SystemExit("[ERROR] No known faces loaded. Check your image paths/photos.")




# %%
date_str = datetime.now().strftime("%Y-%m-%d")
csv_path = Path(f"{date_str}_attendance.csv")
f = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(f)
writer.writerow(["Name", "Time"])  # CSV header

# track who is still unmarked to avoid duplicates
students = set(known_face_names)


# %%
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    f.close()
    raise SystemExit("[ERROR] Cannot access camera.")

print("[INFO] Starting... Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] Frame grab failed, stopping.")
        break

    # speed-up: process a smaller frame
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # find faces & encodings in current frame
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"

        if known_face_encodings:  # safety guard
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_idx = int(np.argmin(distances)) if len(distances) else -1
            if best_idx >= 0 and matches[best_idx]:
                name = known_face_names[best_idx]

        # scale back up to original frame size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 8),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        # mark attendance once
        if name in students:
            students.remove(name)
            time_str = datetime.now().strftime("%H:%M:%S")
            writer.writerow([name, time_str])
            print(f"[MARKED] {name} at {time_str}")

    # small on-screen status
    cv2.putText(frame,
                f"Marked: {len(known_face_names)-len(students)}/{len(known_face_names)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame,
                f"Marked: {len(known_face_names)-len(students)}/{len(known_face_names)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
f.close()
print(f"[INFO] Saved attendance to {csv_path.resolve()}")


# %%



