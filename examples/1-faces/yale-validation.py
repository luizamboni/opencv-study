
import cv2
import os
import numpy as np
from PIL import Image

models_path = os.getenv("MODELS")
eigenRecognizer = cv2.face.EigenFaceRecognizer_create()
eigenRecognizer.read(models_path + "/classificadorEigenYale.yml")


fisherRecognizer = cv2.face.FisherFaceRecognizer_create()
fisherRecognizer.read(models_path + "/classificadorFisherYale.yml")

LBPHRecognizer = cv2.face.LBPHFaceRecognizer_create()
LBPHRecognizer.read(models_path + "/classificadorLBPHYale.yml")

recognizers = [ 
    (eigenRecognizer, "eigen"), 
    (fisherRecognizer, "fisher"), 
    (LBPHRecognizer, "LBPH"),
]

class DetectObject:
    def __init__(self, xml_training_path: str):
        self.detector = cv2.CascadeClassifier(xml_training_path)

    def detect_in_image(self, img_np) -> None:
        return self.detector.detectMultiScale(img_np)


detector = DetectObject(os.getenv("PRETRAIN") + "/haarcascade_frontalface_default.xml")


def read_images():
    base_dir = os.getenv("TRAIN_IMAGES") + "/yalefaces/teste/"
    ids = []
    faces = []

    for img_path in os.listdir(base_dir):
        full_img_path = f"{base_dir}/{img_path}"

        id = img_path.split(".")[0].replace("subject","")
        img_data = Image.open(full_img_path).convert("L")
        img_np = np.array(img_data, "uint8")
        ids.append(int(id))
        faces.append(img_np)
    
    return np.array(ids), faces


ids, photos = read_images()
for recognizer, recognizer_name in recognizers:
    print(f"{recognizer_name} -----------------------")
    total_positives = 0
    total_confidance = 0

    for id, photo in zip(ids, photos):
        predicted_id, confidence = recognizer.predict(photo)

        faces = detector.detect_in_image(photo)
        for x,y, w, h in faces:
            predicted_id, confidence = recognizer.predict(photo)
            if predicted_id == id:
                # print(f"{recognizer_name} {id} ok", confidence)
                total_positives +=1
                total_confidance += confidence
            else:
                pass
                # print(f"{recognizer_name} predicted was {predicted_id} but should be {id} fail", confidence)

    print(f"total_positives: {total_positives}")
    print(f"total_confidance: {total_confidance}")