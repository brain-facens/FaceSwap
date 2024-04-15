"""
Main module to run the SimSwap API.
"""
import os
import sys

# absolute path to this folder
abs_filepath = os.path.dirname(p=os.path.abspath(path=__file__))
abs_filepath = abs_filepath.split(sep=os.sep)[:-1]
abs_filepath = os.path.join(os.sep, *abs_filepath)
sys.path.append(abs_filepath)

# === API ===
from fastapi import BackgroundTasks, FastAPI, UploadFile, status, Depends, HTTPException
from fastapi.responses import FileResponse

# =====================================
# -------------------------------------
# dependencies
# -------------------------------------
from app.faceswap import FaceSwap
from app.logger import logger
from app.schemas import SimSwapInfo, AvailableFaces

try:
    from app.faceswap import FaceSwap
    model = FaceSwap()
except:
    FaceSwap.available = False
else:
    FaceSwap.available = True
# =====================================

# initializing api
logger.debug(msg="instanciating `FastAPI` object.")
version = "1.0.0"
app = FastAPI(title="FaceSwap", version=version)


def get_model():
    if not FaceSwap.available:
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail="model could not loaded")


@app.get(
        path="/",
        description="Returns the current status of API.",
        status_code=status.HTTP_200_OK,
        response_model=SimSwapInfo,
        tags=["FaceSwap"]
)
async def root() -> SimSwapInfo:
    logger.info(msg="calling `/` endpoint.")
    return {
        "model": {
            "available": FaceSwap.available,
            "image": {
                "width": 224,
                "height": 224
            },
            "mode": "static"
        },
        "version": version,
        "name": "Face Swap",
        "author": "BRAIN"
    }

@app.get(
        path="/names",
        description="Return all available people for swap.",
        status_code=status.HTTP_200_OK,
        response_model=AvailableFaces,
        tags=["FaceSwap"]
)
async def swap_list() -> AvailableFaces:
    logger.info(msg="calling `/names` endpoint.")
    return await FaceSwap.get_available_swaps()

@app.get(
        path="/names/{name}/{image_id}",
        description="Return all images related to a target.",
        status_code=status.HTTP_200_OK,
        response_class=FileResponse,
        tags=["FaceSwap"]
)
async def get_target_image(name: str, image_id: int) -> FileResponse:
    image_path = await FaceSwap.get_image(name=name, image_id=image_id)
    return FileResponse(path=image_path)

@app.post(
        path="/inference/{name}/{image_id}",
        description="Run the model inference and return it results.",
        status_code=status.HTTP_201_CREATED,
        response_class=FileResponse,
        dependencies=[Depends(dependency=get_model)],
        tags=["FaceSwap"]
)
async def inference(
    image: UploadFile, name: str, image_id: int, background_tasks: BackgroundTasks
    ) -> FileResponse:
    logger.info(msg="calling `/inference` endpoint.")
    predict_filepath = await model.swap(frame=image, name=name, image_id=image_id)
    background_tasks.add_task(func=model.delete_tmp, tmp_filepath=predict_filepath)
    return FileResponse(path=predict_filepath, background=background_tasks)
