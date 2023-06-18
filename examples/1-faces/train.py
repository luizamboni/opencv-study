import cv2
import numpy as np
import os

eigenface = cv2.face.EigenFaceRecognizer_create(
    num_components=50, 
    # threshold=1.5,
)
fisherface = cv2.face.FisherFaceRecognizer_create(
    num_components=50, 
    # threshold=1.5,
)
lbph = cv2.face.LBPHFaceRecognizer_create()

def read_images():
    base_dir = os.getenv("TRAIN_IMAGES")
    ids = []
    faces = []

    for img_path in os.listdir(base_dir):
        full_img_path = f"{base_dir}/{img_path}"
        id, _ , _  = img_path.split("-")

        img_data = cv2.cvtColor(cv2.imread(full_img_path), cv2.COLOR_BGR2GRAY)

        ids.append(int(id))
        faces.append(img_data)
        # cv2.imshow("face", img_data)
        # cv2.waitKey(10)
    
    return np.array(ids), faces


ids, faces = read_images()

print("Training")
eigenface.train(faces, ids)
eigenface.write(os.getenv("MODELS") + "/eigenface-classifier.yml")

fisherface.train(faces, ids)
fisherface.write(os.getenv("MODELS") + "/fisherface-classifier.yml")

lbph.train(faces, ids)
lbph.write(os.getenv("MODELS") + "/lbph-classifier.yml")