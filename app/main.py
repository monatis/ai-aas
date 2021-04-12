import time

from fastapi import Depends, FastAPI, HTTPException
from starlette.requests import Request
import redis
import uuid
import json
from schemas import (ImageSchema, DetectedObject, ImageMetaData,
                     ObjectDetectionAPIDescription, ObjectDetectionResponse, ZSLTextInput)

import logging
queue = redis.StrictRedis(host="redis")

app = FastAPI(
    title="Intelligence as a Service",
    description="Fast, easy-to-use, consistent AI APIs APIs for developers. Use one of our SDKs or simply make an http request to our endpoints from anywhere you need it.",
    version="0.0.1-alpha"
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
def get_status_version():
    return {"status": "ok", "version": "0.0.1-alpha", "author": "M. Yusuf Sarıgöz", "license": "Apache 2.0"}


@app.get("/detect/objects", response_model=ObjectDetectionAPIDescription)
def describe_object_detection_api():
    return ObjectDetectionAPIDescription()


@app.post("/detect/objects", response_model=ObjectDetectionResponse)
def detect_objects(img: ImageSchema):
    """
    Post an image URL or base64 encoding of an image,
    get a list of detected objects.
    """
    return []


@app.post('/classify/text')
async def classify_text(zsl_input: ZSLTextInput):
    """
    Post a list of texts with a list of possible labels
    for zero shot text classification. Get the recognized label and score.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    zsl_input = zsl_input.dict()
    zsl_input["id"] = k
    queue.rpush("zsl", json.dumps(zsl_input))
    num_tries, max_tries = 0, 20
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.1)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp