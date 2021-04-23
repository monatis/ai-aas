from pydantic import BaseModel, ValidationError, validator, Field, Extra, AnyUrl
import re
from typing import List


class ImageSchema(BaseModel):
    """
    Image to be understood. You can simply pass an image URL or a base64 encoded byte string from an image.
    """
    type: str = Field(..., title="Image type", description="must be 'url' for remotely hosted image or 'b64' for base64-encoded image.")
    data: str = Field(..., title="Image data", description="must be a publically accessible URL to an image if type is 'url', or base64 encoded byte string of an image if type is 'base64Image'.")

    @validator('type')
    def is_valid_image_type(cls, v):
        if v not in ["url", "b64"]:
            raise ValueError("must be one of 'url' or 'b64'")
        return v

    @validator('data')
    def data_must_be_url_or_base64(cls, v, values):
        if values['type'] == 'url':
            return AnyUrl.validate(v)
        elif values['type'] == 'b64':
            match = re.match("^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", v)
            if match is not None:
                return match.group(0)
        else:
            raise ValueError()


    class Config:
        extra = Extra.forbid



class ImageMetaData(BaseModel):
    width: int = Field(..., title="Image width", description="Image width in pixels")
    height: int = Field(..., title="Image height", description="Image height in pixels")

class DetectedObject(BaseModel):
    label: str = Field(..., title="Object label", description="Human-readable label of the detected object")
    score: float = Field(..., title="Confidence score", description="Value that represents how sure neural network is for this particular detection. 0 = lowest, 1 = highest")
    top: int = Field(..., title="Top position", description="Upper-most point of the bounding box containing this object")
    right: int = Field(..., title="Right position", description="Right-most point of the bounding box containing this object")
    bottom: int = Field(..., title="Bottom position", description="Lower-most point of the bounding box containing this object")
    left: int = Field(..., title="Left position", description="Left-most point of the bounding box containing this object")



class ObjectDetectionResponse(BaseModel):
    metadata: ImageMetaData = Field(..., title="Image metadata", description="Metadata information about the image to be understood")
    detections: List[DetectedObject] = Field(..., title="List of detections", description="List of detections with object labels and bounding boxes")



class ObjectDetectionAPIDescription(BaseModel):
    """
    Describe object detection API for documentation purposes.
    """
    version: str = "0.99-alpha"
    title: str = "Object Detection API"
    description: str = "Detect objects in images from remote URLs or base64 encoding of images."



class ZSLTextInput(BaseModel):
    """
    Text to be labelled with a zero-shot technique.
    """
    texts: List[str] = Field(..., title="List of texts", description="List of texts to be zero-shot labelled")
    labels: List[str] = Field(..., title="Possible labels", description="List of labels to be possibly applied to texts")

    class Config:
        extra = Extra.forbid

    
class SingleTextInput(BaseModel):
    """
    Input with a single field called text for general purpose NLP.
    """
    text: str = Field(..., title="Text", description="Text to be processed")
    
    class Config:
        extra = Extra.forbid

        
class QAInput(SingleTextInput):
    """
    Input for question answering model with a context and a question.
    """
    question: str = Field(..., title="Question", description="Question to be answered")
    