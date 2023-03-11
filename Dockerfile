FROM --platform=arm64 python:3.8

COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install python3-opencv -y
# RUN conda install -c conda-forge opencv
RUN pip install -r requirements.txt
