FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-alpine3.10

RUN apk update && apk --no-cache add python3 py3-pip && pip3 install --upgrade pip
#copy graph.pbtxt /
#COPY detector.pb /
#COPY labels_boxable.csv /
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt
COPY ./ /app
ENV LC_ALL C.UTF-8
ENV  LANG C.UTF-8