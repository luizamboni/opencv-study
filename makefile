PRETRAIN=$(shell pwd)/pretrain
MODELS=$(shell pwd)/models
IMAGES=$(shell pwd)/images
EXAMPLES=$(shell pwd)/examples
TRAIN_IMAGES=$(shell pwd)/train_images

faces-basic:
	PRETRAIN=${PRETRAIN} \
	IMAGES=${IMAGES} \
	python3 ./examples/1-faces/basic.py

faces-cam-basic:
	PRETRAIN=${PRETRAIN} \
	IMAGES=${IMAGES} \
	python3 ./examples/1-faces/cam.py

faces-cam-train:
	PRETRAIN=${PRETRAIN} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/record-faces.py

faces-train:
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/train.py	

eigen-faces:
	PRETRAIN=${PRETRAIN} \
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/eigenface-recognition.py

fisher-faces:
	PRETRAIN=${PRETRAIN} \
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/fisherface-recognition.py

lbph-faces:
	PRETRAIN=${PRETRAIN} \
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/LBPHface-recognition.py

yale-train:
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/yale-train.py


yale-validation:
	MODELS=${MODELS} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	PRETRAIN=${PRETRAIN} \
	python3 ./examples/1-faces/yale-validation.py