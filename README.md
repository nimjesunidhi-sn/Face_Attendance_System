Face Attendance System

Overview

This is an AI-based Face Recognition Attendance System built using Python, FastAPI, OpenCV, and the face_recognition library. The system detects and recognizes faces through a webcam and automatically marks attendance in a CSV file.

Features

- Face Recognition using AI
- Real-time Webcam Capture
- Automatic Attendance Marking
- Duplicate Attendance Prevention
- FastAPI Backend
- Web-based User Interface
- CSV Attendance Report Generation

Technologies Used

- Python
- FastAPI
- OpenCV
- face_recognition
- NumPy
- HTML
- JavaScript

Project Structure

face_attendance_system/

├── app.py

├── recognition.py

├── index.html

├── requirements.txt

├── photos/

│ ├── person1.jpg

│ ├── person2.jpg

│ └── ...

└── attendance.csv

Installation

1. Clone the repository

git clone <repository-url>
cd Face_Attendance_System

2. Install dependencies

pip install -r requirements.txt

3. Start FastAPI server

python -m uvicorn app:app --reload

4. Open the frontend

Run index.html using Live Server in VS Code or a local web server.

Usage

1. Start the FastAPI server.
2. Open the web application.
3. Allow webcam access.
4. Click "Mark Attendance".
5. The system recognizes the face and stores attendance in a CSV file.

Attendance Output

Attendance is stored in:

YYYY-MM-DD_attendance.csv

Example:

Name| Time
Francisco Lachowski| 09:15:23
Tom Holland| 09:17:40

Future Improvements

- Database Integration
- Admin Dashboard
- Student Registration Module
- Cloud Deployment
- Attendance Analytics
- Multi-Face Recognition

Author

Sunidhi Nimje

License

This project is for educational and learning purposes.
