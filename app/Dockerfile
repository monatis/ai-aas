FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY ./ /app
ENV LC_ALL C.UTF-8
ENV  LANG C.UTF-8