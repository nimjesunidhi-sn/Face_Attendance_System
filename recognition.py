from pathlib import Path
import face_recognition
import cv2
import numpy as np

# Photos folder
KNOWN_DIR = Path("photos")

# Known people
PEOPLE = [
    ("francisco_lachowski.jpg", "Francisco Lachowski"),
    ("kendall_jenner.jpg", "Kendall Jenner"),
    ("tom_holland.jpg", "Tom Holland"),
    ("emma_waston.jpg", "Emma Waston"),
    ("gigi_hadid.jpg", "Gigi Hadid")
]

known_face_encodings = []
known_face_names = []

# Load all known faces
for filename, name in PEOPLE:

    image_path = KNOWN_DIR / filename

    if image_path.exists():

        image = face_recognition.load_image_file(str(image_path))

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:

            known_face_encodings.append(encodings[0])
            known_face_names.append(name)

print("Known faces loaded:", known_face_names)


def recognize_face(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)

    face_encodings = face_recognition.face_encodings(
        rgb,
        face_locations
    )

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(distances) > 0:

            best_match_index = np.argmin(distances)

            if matches[best_match_index]:

                return known_face_names[best_match_index]

    return "Unknown"