from pydantic import BaseModel
from typing import List


# /
class ImageInfo(BaseModel):
    width: int = 224
    height: int = 224


class ModelInfo(BaseModel):
    available: bool
    image: ImageInfo
    model: str = "static"


class SimSwapInfo(BaseModel):
    model: ModelInfo
    version: str
    name: str
    author: str


# /names
class People(BaseModel):
    name: str
    images: int


class AvailableFaces(BaseModel):
    people: List[People]
