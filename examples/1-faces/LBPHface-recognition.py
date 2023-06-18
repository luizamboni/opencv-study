import cv2
from typing import Tuple
import os
cap = cv2.VideoCapture(0)

class FaceDetector:
    def __init__(self):
        self.detector = None
        self.recognizer = None

    def add_face_detector(self, xml_training_path: str, color: Tuple[int, int, int] ):
        self.detector = (cv2.CascadeClassifier(xml_training_path), color)

    def add_recognizer(self, yml_training_path: str, color: Tuple[int, int, int] ):
        eigenface = cv2.face.LBPHFaceRecognizer_create()
        eigenface.read(yml_training_path)
        self.recognizer = (eigenface, color)
    
    def detect_in_image(self, img) -> None:
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect objects
        detector, color = self.detector
        recognizer, color = self.recognizer
        
        detected_objects = detector.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in detected_objects:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cropped = cv2.resize(gray[y+1: y+h-1, x+1:x+w-1], (220, 220))

            id, confidence = recognizer.predict(cropped)
            cv2.putText(img, f"confidence: {confidence}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
            cv2.putText(img, f"id: {id}", (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)

        # Display the output
        cv2.imshow('img', img)


blue  = (255, 0, 0)
red   = (0, 0, 255)
green = (0, 255, 0)

detector = FaceDetector()
detector.add_face_detector(os.getenv("PRETRAIN") + "/lbpcascade_frontalface_improved.xml", red)
# detector.add_classifier('haarcascade_frontalface_default.xml', blue)
detector.add_recognizer(os.getenv("MODELS") + "/lbph-classifier.yml", green)

while True:
    _, img = cap.read()
    detector.detect_in_image(img)
    
    pressed_key = cv2.waitKey(30) & 0xff
    if pressed_key == 27:
        break

cap.release()
