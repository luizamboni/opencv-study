import cv2
from typing import Tuple
import os
cap = cv2.VideoCapture(0)

class ObjectsDetector:
    def __init__(self):
        self.detectors = []

    def add_classifier(self, xml_training_path: str, color: Tuple[int, int, int] ):
        self.detectors.append(
            (cv2.CascadeClassifier(xml_training_path), color)
        )

    def detect_in_image(self, img) -> None:
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect objects
        for detector, color in self.detectors: 
            detected_objects = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in detected_objects:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Display the output
        cv2.imshow('img', img)


blue  = (255, 0, 0)
red   = (0, 0, 255)
green = (0, 255, 0)

detector = ObjectsDetector()
detector.add_classifier(os.getenv("PRETRAIN") + "/lbpcascade_frontalface_improved.xml", red)
# detector.add_classifier('haarcascade_frontalface_default.xml', blue)
detector.add_classifier(os.getenv("PRETRAIN") + "/haarcascade_eye.xml", green)

while True:
    _, img = cap.read()
    detector.detect_in_image(img)
    
    pressed_key = cv2.waitKey(30) & 0xff
    if pressed_key == 27:
        break

cap.release()
