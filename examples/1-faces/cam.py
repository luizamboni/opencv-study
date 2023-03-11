import cv2

cap = cv2.VideoCapture(0)

class ObjectsDetector:
    def __init__(self):
        self.detectors = []

    def add_classifier(self, xml_trein_path: str):
        self.detectors.append(
            cv2.CascadeClassifier(xml_trein_path)
        )

    def detect_in_image(self, img) -> None:
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect objects
        for detector in self.detectors: 
            detected_objects = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in detected_objects:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('img', img)

detector = ObjectsDetector()
detector.add_classifier('haarcascade_frontalface_default.xml')
detector.add_classifier('haarcascade_eye.xml')

while True:
    _, img = cap.read()
    detector.detect_in_image(img)
    
    pressed_key = cv2.waitKey(30) & 0xff
    if pressed_key == 27:
        break

cap.release()
