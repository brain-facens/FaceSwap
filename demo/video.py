import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class VideoCapture:

    model_path = os.path.abspath(path=os.path.dirname(p=__file__))
    model_path = os.path.join(model_path, "models", "face_landmarker.task")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )

    @classmethod
    def draw_landmarks_on_image(cls, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape
        face_points = []

        for face_landmarks in face_landmarks_list:

            # Target points
            target_points = [10, 152]
            for idx in target_points:
                ldmk = face_landmarks[idx]
                face_points.append((int(ldmk.x * width), int(ldmk.y * height)))

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())

        return annotated_image, face_points

    @classmethod
    def is_aligned(cls, face_points, box1, box2) -> bool:
        if not len(face_points):
            return False

        top_point = face_points[0]
        bottom_point = face_points[1]

        top_check = (top_point[0] > box1[0][0] and top_point[1] > box1[0][1] and box1[1][0] > top_point[0] and box1[1][1] > top_point[1]) and \
                    not (top_point[0] > box2[0][0] and top_point[1] > box2[0][1] and box2[1][0] > top_point[0] and box2[1][1] > top_point[1])
        bottom_check = (bottom_point[0] > box1[0][0] and bottom_point[1] > box1[0][1] and box1[1][0] > bottom_point[0] and box1[1][1] > bottom_point[1]) and \
                    not (bottom_point[0] < box2[0][0] and bottom_point[1] < box2[0][1] and box2[1][0] > bottom_point[0] and box2[1][1] > bottom_point[1])

        if top_check and bottom_check:
            return True
        return False

    @classmethod
    def recv(cls, frame: np.ndarray) -> np.ndarray:

        frame = cv2.flip(src=frame, flipCode=1)
        frame = cv2.resize(src=frame, dsize=(960,720))

        with cls.FaceLandmarker.create_from_options(cls.options) as model:
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker = model.detect_for_video(mp_frame, 30)
            np_frame, face_points = cls.draw_landmarks_on_image(rgb_image=frame, detection_result=landmarker)

        height, width, _ = np_frame.shape
        cx, cy = width // 2, height // 2
        x1, y1 = cx - 200, cy - 250
        x2, y2 = cx + 200, cy + 250

        limit = 80
        x3, y3 = x1 + limit, y1 + limit
        x4, y4 = x2 - limit, y2 - limit

        check = cls.is_aligned(face_points=face_points, box1=((x1, y1), (x2, y2)), box2=((x3, y3), (x4, y4)))

        if check:
            color = (0,255,0)
        else:
            color = (0,0,255)

        np_frame = cv2.rectangle(img=np_frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)
        np_frame = cv2.rectangle(img=np_frame, pt1=(x3, y3), pt2=(x4, y4), color=color, thickness=2)

        return np_frame, check, frame[y1:y2, x1:x2]
