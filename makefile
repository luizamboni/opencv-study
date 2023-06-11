PRETRAIN=$(shell pwd)/pretrain
IMAGES=$(shell pwd)/images
EXAMPLES=$(shell pwd)/examples

faces-basic:
	PRETRAIN=${PRETRAIN} IMAGES=${IMAGES} python3 ./examples/1-faces/basic.py

faces-cam-basic:
	PRETRAIN=${PRETRAIN} IMAGES=${IMAGES} python3 ./examples/1-faces/cam.py
