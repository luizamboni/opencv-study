import cv2
from typing import Tuple
import os
import numpy as np
cap = cv2.VideoCapture(0)

class ObjectsDetector:
    def __init__(self):
        self.detectors = []

    def add_classifier(self, xml_training_path: str, color: Tuple[int, int, int], border: int):
        self.detectors.append(
            (cv2.CascadeClassifier(xml_training_path), color, border)
        )

    def detect_in_image(self, img) -> None:
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect objects
        objects = []
        for detector, color, border in self.detectors: 
            detected_objects = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in detected_objects:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, border)
                cv2.putText(img, f"({x}, {y})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, border)
                cv2.putText(img, f"({x+w},{y+h})", (x+h, y+w), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, border)
                cropped = img[y+border: y+h-border, x+border:x+w-border]
                objects.append(cropped)
                cv2.imshow("cropped", cropped)

        # Display the output
        cv2.imshow('img', img)
        return objects



blue  = (255, 0, 0)
red   = (0, 0, 255)
green = (0, 255, 0)

detector = ObjectsDetector()
detector.add_classifier(os.getenv("PRETRAIN") + "/lbpcascade_frontalface_improved.xml", green, 1)

img_count = 0
id = input("Digite o id para estas images: ")

while True:
    if img_count >= 25:
        break

    _, img = cap.read()
    objects = detector.detect_in_image(img)

    pressed_key = cv2.waitKey(30) & 0xff
    if pressed_key == 13:
        print("Enter pressed")
        for object in objects:
            luminosity = np.average(object)
            print(f"luminosity: {luminosity}")

            if luminosity > 110:
                img_count += 1
                print(f"saving image {img_count}")
                object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)
                object = cv2.resize(object, (220, 220))
                cv2.imwrite(os.getenv("TRAIN_IMAGES") + f"/{id}-{img_count}-obj.png", object)
            else:
                print(f"not enough luminosity: {luminosity}")

    if pressed_key == 27:
        break

cap.release()
