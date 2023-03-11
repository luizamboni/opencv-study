build:
	docker build -t opencv-python .

run: 
	docker run -v $(shell pwd)/examples/:/examples/ \
	opencv-python \
	python3 /examples/1-basic.py
