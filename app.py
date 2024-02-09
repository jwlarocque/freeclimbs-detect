from typing import Annotated
from io import BytesIO
from configparser import ConfigParser

from litestar import Litestar, post, status_codes
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from ultralytics import YOLO
from PIL import Image


model = None

class Prediction:
    boxes: list
    confidences: list


@post("/", status_code=status_codes.HTTP_200_OK)
async def index(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)]
) -> Prediction:
    content = await data.read()
    predictions = None
    with Image.open(BytesIO(content)) as img:
        predictions = model(img)
        return {
            "boxes": predictions[0].boxes.xyxy.tolist(),
            "confidences": predictions[0].boxes.conf.tolist()
        }


config = ConfigParser()
config.read("config.ini")
model_path = config.get(
    "main",
    "model_path",
    fallback="../yolov8n-freeclimbs-detect-2/yolov8n-freeclimbs-detect-2.pt")
model = YOLO(model_path)
app = Litestar([index])