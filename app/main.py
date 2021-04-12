import time

#import cv2
#import imutils
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from starlette.requests import Request
import redis
import uuid
import json
from schemas import (DetectedObject, ImageMetaData,
                     ObjectDetectionAPIDescription, ObjectDetectionResponse, ZSLTextInput)
from utils import get_image, load_model_and_labels
import logging
queue = redis.StrictRedis(host="redis")
object_detection_model, labels = load_model_and_labels()

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

@app.get("/detect/objects", response_model=ObjectDetectionAPIDescription)
def describe_object_detection_api():
    return ObjectDetectionAPIDescription()


@app.post("/detect/objects", response_model=ObjectDetectionResponse)
def detect_objects(img: np.ndarray = Depends(get_image)):
    """
    Post an image URL or base64 encoding of an image,
    get a list of detected objects.
    """
    h, w = img.shape[:2] # Save original height and width
    metadata = ImageMetaData(width=w, height=h)
    img_small = imutils.resize(img, width=500) # Resize image to reduce compute time
    blob = cv2.dnn.blobFromImage(img_small, size=(300, 300), swapRB=True, crop=False) # Our object detection model expects an image of size 300x300
    object_detection_model.setInput(blob) # Set image to our model
    preds = object_detection_model.forward() # Get predictions from our model

    detections = [] # Initialize an empty list to append individual predictions to
    for detection in preds[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3: # this confidence threshold will be configured with  API request
            label = labels[int(detection[1])-1] # Get te he human-readable label of the detected object
            left = max(int(detection[3] * w), 0) # Get the bounding boxes of the detected object
            top = max(int(detection[4] * h), 0)
            right = min(int(detection[5] * w), w)
            bottom = min(int(detection[6] * h), h)
            detection = DetectedObject(label=label, score=score, top=top, right=right, bottom=bottom, left=left)
            detections.append(detection)
    return ObjectDetectionResponse(metadata=metadata, detections=detections)



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