import math
import os
from argparse import Namespace

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from PIL import Image, ImageDraw
from torchvision.transforms import transforms

from lib.models.fs_model import fsModel


def create_model(abs_filepath: str) -> fsModel:
    opt = Namespace(
        no_simswaplogo = True,
        use_mask = True,
        output_path = os.path.join(abs_filepath, "results"),
        gpu_ids = "0",
        checkpoints_dir = os.path.join(abs_filepath, "lib", "checkpoints"),
        name = "people",
        resize_or_crop = "scale_width",
        Arc_path = os.path.join(abs_filepath, "lib", "arcface_model", "arcface_checkpoint.tar"),
        which_epoch = "latest",
        verbose = False,
        fp16 = False,
    )
    model = fsModel()
    model.initialize(opt=opt)
    model.eval()
    return model

class SquarePad:

    def __call__(self, image: torch.Tensor, pad: int = None):
        _, h, w = image.size()
        if pad:
            max_wh = pad
        else:
            max_wh = np.max([h, w])
        hp = (max_wh - h) / 2
        wp = (max_wh - w) / 2
        hpf, hpc = math.floor(hp), math.ceil(hp)
        wpf, wpc = math.floor(wp), math.ceil(wp)
        padding = (wpc, wpf, hpc, hpf)
        return F.pad(input=image, pad=padding, value=0, mode="constant")


class FaceRecognitionModel:

    def __init__(self) -> None:
        model_path = os.path.abspath(path=os.path.dirname(p=__file__))
        model_path = os.path.join(model_path, "face_landmarks", "face_landmarker.task")

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

    def __get_facebox(self, landmarks: FaceLandmarkerResult, width: int, height: int) -> tuple:
        
        faces = landmarks.face_landmarks
        if not faces:
            return (0,0,0,0)

        # unique face
        tgt_face = faces[0]

        indices = (10, 152, 234, 454) # top, bottom, right, left
        top_pt = tgt_face[indices[0]]
        bottom_pt = tgt_face[indices[1]]
        right_pt = tgt_face[indices[2]]
        left_pt = tgt_face[indices[3]]

        # box
        x1 = int(right_pt.x * width)
        y1 = int(top_pt.y * height)
        x2 = int(left_pt.x * width)
        y2 = int(bottom_pt.y * height)

        return (x1, y1, x2, y2)

    def __call__(self, image: Image.Image) -> dict:
        cv_image = np.asarray(a=image)
        cv_image = cv2.flip(src=cv_image, flipCode=1)

        with self.FaceLandmarker.create_from_options(self.options) as model:
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
            landmarks = model.detect_for_video(mp_frame, 30)

        # face box
        face_box = self.__get_facebox(landmarks=landmarks, width=cv_image.shape[0], height=cv_image.shape[1])

        return {"landmarks": landmarks, "box": face_box}


class FaceSwapModel:

    def __init__(self, abs_filepath: str) -> None:
        self.opt = Namespace(
            no_simswaplogo = True,
            use_mask = True,
            output_path = os.path.join(abs_filepath, "results"),
            gpu_ids = "0",
            checkpoints_dir = os.path.join(abs_filepath, "lib", "checkpoints"),
            name = "people",
            resize_or_crop = "scale_width",
            Arc_path = os.path.join(abs_filepath, "lib", "arcface_model", "arcface_checkpoint.tar"),
            which_epoch = "latest",
            verbose = False,
            fp16 = False,
        )
        self.model = fsModel()
        self.model.initialize(opt=self.opt)
        self.model.eval()

        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.pad = SquarePad()

    @torch.no_grad
    def __call__(self, src: Image.Image, tgt: Image.Image, src_box: tuple, tgt_box: tuple) -> Image.Image:
        # crop face
        face_mask = Image.new(mode="RGB", size=src.size, color="#000000")
        draw = ImageDraw.Draw(im=face_mask)
        draw.rectangle(xy=[(src_box[0], src_box[1]), (src_box[2], src_box[3])], fill="#FFFFFF")
        crop_src = Image.composite(src, face_mask, face_mask.convert(mode="L"))

        face_mask = Image.new(mode="RGB", size=tgt.size, color="#000000")
        draw = ImageDraw.Draw(im=face_mask)
        draw.rectangle(xy=[(tgt_box[0], tgt_box[1]), (tgt_box[2], tgt_box[3])], fill="#FFFFFF")
        crop_tgt = Image.composite(tgt, face_mask, face_mask.convert(mode="L"))

        tgt_img = self.transformer_Arcface(crop_tgt)
        img_id = tgt_img.view(-1, tgt_img.shape[0], tgt_img.shape[1], tgt_img.shape[2])

        user_img = self.transformer(crop_src)
        img_att = user_img.view(-1, user_img.shape[0], user_img.shape[1], user_img.shape[2])

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = self.model.netArc(img_id_downsample)
        latend_id = latend_id.detach().cpu()
        latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)

        # feed forward
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
        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.cpu()
        output = np.array(output)
        output = output[..., ::-1]
        output = output*255
        output = cv2.cvtColor(src=output, code=cv2.COLOR_BGR2RGB)
        output = output.astype(dtype=np.uint8)

        # remove padding
        no_pad = Image.fromarray(obj=output, mode="RGB")
        no_pad = no_pad.resize(size=src.size)
        no_pad = no_pad.crop(box=src_box)

        # pasting result
        results = src.copy()
        results.paste(im=no_pad, box=src_box)
        return results
