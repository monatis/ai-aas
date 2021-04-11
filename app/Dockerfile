FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

RUN apt update && apt install -y python3 python3-pip libsm6 libxext6 libxrender-dev libgl1-mesa-glx
#copy graph.pbtxt /
#COPY detector.pb /
#COPY labels_boxable.csv /
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt
COPY ./ /app
ENV LC_ALL C.UTF-8
ENV  LANG C.UTF-8