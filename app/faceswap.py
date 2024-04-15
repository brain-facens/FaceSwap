# SimSwap imports used to run the inference over
# two target images.
import io
import os
import time
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from PIL import Image

from app.logger import abs_filepath, critical_error, logger

try:
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torchvision import transforms

    from lib.models.fs_model import fsModel
except ImportError as error:
    logger.critical(msg=f"could not import some dependecy : {error}")
except Exception as error:
    logger.critical(msg=f"unexpected error : {error}.")
    print(critical_error)
logger.debug(msg="all dependencies was loaded.")


class FaceSwap:

    available = False
    image_folder = Path(os.path.join(abs_filepath, "images"))
    available_names = {"people": []}
    for name in os.listdir(path=image_folder):
        data = {"name": name, "images": len(list(image_folder.joinpath(name).glob(pattern="*.jpg")))}
        available_names.get("people").append(data)

    @classmethod
    def is_available(cls) -> bool:
        """Returns if the model is available."""
        return cls.available

    @classmethod
    async def get_available_swaps(cls) -> dict:
        """Return all available swaps."""
        logger.debug(msg="calling `FaceSwap.get_available_swaps` classmethod.")
        return cls.available_names

    @classmethod
    async def get_image(cls, name: str, image_id: int) -> Path:
        """Return one especific image of one person."""
        logger.debug(msg="calling `FaceSwap.get_image` classmethod.")
        tgt_image = cls.image_folder.joinpath(name, str(image_id) + ".jpg")
        if tgt_image.exists():
            return tgt_image
        logger.error(msg=f"ID {image_id} is out of range for `{name}`.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid ID or name.")

    def __init__(self) -> None:
        # ================================
        # --------------------------------
        # mandatory configs
        # --------------------------------
        logger.debug(msg="calling `FaceSwap.__init__` method.")
        logger.debug(msg="defining mandatory options for `TestOptions`.")
        self.opt = Namespace(
            no_simswaplogo = True,
            use_mask = True,
            output_path = os.path.join(abs_filepath, "results"),
            gpu_ids = "0",
            checkpoints_dir = "./lib/checkpoints",
            name = "people",
            resize_or_crop = "scale_width",
            Arc_path = "./lib/arcface_model/arcface_checkpoint.tar",
            which_epoch = "latest",
            verbose = False,
            fp16 = False,
        )
        # ================================
        # instanciating model.
        try:
            torch.nn.Module.dump_patches = True
            logger.debug(msg="creating model.")
            self.model = fsModel()
            self.model.initialize(opt=self.opt)
            logger.debug(msg="activating evaluating mode for model.")
            self.model.eval()
        except Exception as error:
            logger.critical(msg=f"unexpected error : {error}.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="unexpected error has been occoured."
            )
        logger.info(msg="main model was instanciated.")

        # preprocessing layers
        logger.debug(msg="defining preprocessing layers to handle input images.")
        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transformer = transforms.Compose([
            transforms.ToTensor()
        ])

    @torch.no_grad
    async def swap(self, frame: UploadFile, name: str, image_id: int) -> str:
        """Method used to run inference."""
        logger.debug(msg="calling `FaceSwap.swap` method.")

        # user data
        try:
            logger.debug(msg="reading user image.")
            raw_user_image = await frame.read()
            logger.debug(msg=f"converting raw user image into a `{Image.Image}` object.")
            pil_user_image = Image.open(fp=io.BytesIO(initial_bytes=raw_user_image), mode="r").convert('RGB')
        except Exception as error:
            logger.critical(msg=f"unexpected error : {error}.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="unexpected error has been occoured."
            )
        finally:
            logger.debug(msg="closing user image.")
            await frame.close()

        # target image
        try:
            target_image_path = self.image_folder.joinpath(name, str(image_id) + ".jpg")
            assert target_image_path.exists(), f"invalid target image, {name}/{image_id}.jpg does not exist."
            pil_tgt_image = Image.open(fp=target_image_path, mode="r").convert('RGB')
        except AssertionError as error:
            logger.critical(msg=f"unexpected error : {error}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="image not found."
            )
        except Exception as error:
            logger.critical(msg=f"unexpected error : {error}.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="unexpected error has been occoured."
            )

        # preprocessing
        logger.debug(msg="appying transformetions into user and target images.")

        tgt_img = self.transformer_Arcface(pil_tgt_image)
        img_id = tgt_img.view(-1, tgt_img.shape[0], tgt_img.shape[1], tgt_img.shape[2])
        logger.debug(msg="target tensor image is ready.")

        user_img = self.transformer(pil_user_image)
        img_att = user_img.view(-1, user_img.shape[0], user_img.shape[1], user_img.shape[2])
        logger.debug(msg="user tensor image is ready.")

        # create latent id
        logger.debug(msg="creating latent ID of user image.")
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to("cpu")
        latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)

        # feed forward
        logger.info(msg="starting feed-forward.")
        infer_time = time.time()
        img_fake = self.model(img_att, latend_id)

        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        # result
        logger.debug(msg="preparing final tensor output.")
        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to("cpu")
        output = np.array(output)
        output = output[..., ::-1]
        output = output*255
        logger.debug(msg=f"inference time took {time.time() - infer_time} to run.")

        # saving result
        logger.debug(msg="saving temp file of final tensor output.")
        tmp_file_id = str(uuid4())
        tmp_filepath = os.path.join(abs_filepath, "results", tmp_file_id + ".jpg")
        cv2.imwrite(tmp_filepath, output)

        return tmp_filepath

    async def delete_tmp(self, tmp_filepath: str) -> None:
        """Deletes the temp file."""
        logger.debug(msg="calling `FaceSwap.delete_tmp` method.")
        os.remove(path=tmp_filepath)
