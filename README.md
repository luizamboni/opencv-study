OpenCV study
===

A simple project for aid me archive better knowledge about computer vision possibilities.
Initialy it has tried use Docker, but i encontered much problems with docker using camera.

External dependencies
```
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
```
# 1 - Faces
these examples uses a pre-treined models to track objects
Some of these models can be found here
https://github.com/opencv/opencv/tree/master/data/ 

## LBP models
LBPs instead compute a local representation of texture. This local representation is constructed by comparing each pixel with its surrounding neighborhood of pixels.
More in: https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/