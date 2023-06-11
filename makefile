PRETRAIN=$(shell pwd)/pretrain
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
	IMAGES=${IMAGES} \
	TRAIN_IMAGES=${TRAIN_IMAGES} \
	python3 ./examples/1-faces/record-faces.py
