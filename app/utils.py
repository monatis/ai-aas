
#import cv2
import numpy as np
from fastapi import Depends
import base64
from schemas import ImageSchema
from urllib import request

def load_model_and_labels():
    """
    Load neural network model and labels from appropriate files.
    """
    labels = []
    with open('./labels_boxable.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.split(',')[1].strip()
            labels.append(label)
    model = cv2.dnn.readNetFromTensorflow('./detector.pb', './graph.pbtxt')
    return model, labels


def get_image(image_payload: ImageSchema):
    """
    Get the image type and data from ImageSchema,
    and call the appropriate parser based on its type.
    """
    if image_payload.type == 'url':
        return parse_image_from_url(image_payload.data)
    else:
        return parse_image_from_base64_string(image_payload.data)


def parse_image_from_url(url):
    """
    Try to parse image from url,
    raise ValueError otherwise, wich will be presented by the API to the client.
    """
    try:
        return cv2.imdecode(np.array(bytearray(request.urlopen(url).read()), dtype='uint8'), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ValueError("Image cannot be read from URL")


def parse_image_from_base64_string(b64_str):
    """
    Parse base64-encoded string as an image in np.ndarray format,
    raise ValueError otherwise, which will be presented by the API to the client.
    """
    try:
        return cv2.imdecode(np.array(bytearray(base64.b64decode(b64_str)), dtype='uint8'), cv2.IMREAD_UNCHANGED)
    except Exception:
        raise ValueError("image data cannot be read as a base64-encoded image string")

