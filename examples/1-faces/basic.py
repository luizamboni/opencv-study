import cv2



class DetectObject:
    def __init__(self, xml_train_path: str):
        self.detector = cv2.CascadeClassifier(xml_train_path)

    def detect_in_image(self, image_path: str) -> None:
        # Read the input image
        img = cv2.imread(image_path)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect objects
        detected_objects = self.detector.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in detected_objects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        cv2.imshow('img', img)

detector = DetectObject('haarcascade_frontalface_default.xml')

# detect_faces('./images/20230118_103814.jpg') 
detector.detect_in_image('./images/DSC05420.jpg') 

cv2.waitKey()
cv2.destroyAllWindows()
