import asyncio
import io
import json
import os
import sys
import time
from uuid import uuid4

# absolute path to this folder
abs_filepath = os.path.abspath(path=os.path.dirname(p=__file__))
abs_filepath = abs_filepath.split(sep=os.sep)[:-1]
abs_filepath = os.path.join(os.sep, *abs_filepath)
sys.path.append(abs_filepath)
sys.path.append(os.path.join(abs_filepath, "demo"))

import cv2
import httpx
import streamlit as st
from PIL import Image

from demo.video import VideoCapture

SERVER_OPTIONS = {
    "local",
    "container"
}

async def get_request(url: str, is_image: bool = False):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url=url)
        except httpx.ConnectError:
            return None
        if response.status_code == 200:
            if is_image:
                return response.read()
            return response.json()
    return None


async def post_request(url: str, image: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=url, files={"image": open(file=image, mode="rb")})
        except httpx.ConnectError:
            return None
        if response.status_code == 200:
            return response.read()
    return None


with open(file=os.path.join(abs_filepath, "demo", "config.json"), mode="r") as json_buffer:
    server = os.getenv(key="FACE_SWAP_SERVER") if os.getenv(key="FACE_SWAP_SERVER") in SERVER_OPTIONS else "local"
    config = json.load(fp=json_buffer).get(server)
    URL = config.get("protocol") + "://" + config.get("IP")
    if config.get("port"):
        URL = URL + ":" + config.get("port")

if __name__=="__main__":

    # configs
    brain_col, liga_col = st.columns(spec=2)
    with brain_col:
        with Image.open(fp=os.path.join(abs_filepath, "demo", "assets", "brain_logo.png"), mode="r") as pil_img:
            st.image(image=pil_img, width=250)
    with liga_col:
        with Image.open(fp=os.path.join(abs_filepath, "demo", "assets", "liga_logo.png"), mode="r") as pil_img:
            st.image(image=pil_img, width=450)

    # API config
    cols = st.columns(spec=3)
    with cols[1]:
        with st.spinner(text="conectando.."):
            response = asyncio.run(main=get_request(url=URL))
        
        if response is None:
            st.error(body="Não foi possível manter conexão com a API.")
            st.stop()

        TITLE = response.get("name")
        AVAILABLE = response.get("model").get("available")

        st.title(body=TITLE)

        if not AVAILABLE:
            st.error(body="Modelo não está disponivél para uso.")
            st.stop()
    st.divider()

    # options
    with st.spinner(text="requesting faces.."):
        response = asyncio.run(main=get_request(url=f"{URL}/names"))
        if response is None:
            st.error(body="Não foi possível coletar as pessoas cadastradas.")
            st.stop()

    face = st.selectbox(
        label="Rosto alvo",
        help="Selecione um rosto para trocá-lo pelo seu!",
        options=response.get("people"),
        format_func=lambda face: face.get("name").replace("_", " ").upper()
    )
    image_id = face.get("images")

    # target image
    image_id = st.selectbox(label="Imagem alvo.", options=range(image_id))
    with st.spinner(text="requesting image.."):
        response = asyncio.run(main=get_request(url=f"{URL}/names/{face.get('name')}/{image_id}", is_image=True))
        if response is None:
            st.error(body="Não foi possível coletar as imagens cadastradas.")
            st.stop()

    # preview
    cols = st.columns(spec=3)
    with cols[1]:
        st.header(body="Alvo", divider=True)
        with Image.open(io.BytesIO(initial_bytes=response)) as pil_img:
            st.image(image=pil_img)

    st.header(body="Camera", divider=True)
    cam_index = st.number_input(label="Porta USB da camera", min_value=0, max_value=5, step=1)
    st.text(body="Enquadre seu rosto na região marcada na imagem até ela ficar verde.")
    cap = cv2.VideoCapture(cam_index)
    frame_placeholder = st.empty()
    counter = 3
    timer = 0
    crop = None
    while cap.isOpened():
        check, raw_frame = cap.read()
        if not check:
            st.error(body="Nenhum vídeo capturado.")
            st.stop()
        else:
            frame, ready, crop = VideoCapture.recv(frame=raw_frame)
            if ready:
                if not timer:
                    timer = time.time()
                else:
                    if time.time() - timer >= 1:
                        counter -= 1
                        timer = time.time()
                height, width, _ = frame.shape
                cx, cy = width // 2, height // 2
                frame = cv2.putText(img=frame, text=f"{counter}", org=(cx, cy), color=(0,255,0), fontFace=1, fontScale=3, thickness=2)
                if counter == 0:
                    break
            else:
                counter = 3
                timer = 0
            frame_placeholder.image(image=frame, channels="BGR")

    if crop is not None:
        img_id = str(uuid4())
        img_path = os.path.join(abs_filepath, "demo", img_id + ".jpg")
        pil_img = Image.fromarray(obj=cv2.cvtColor(src=crop, code=cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize(size=(224,224))
        pil_img.save(fp=img_path, format="JPEG")
        del img_id, pil_img

        with st.spinner(text="Gerando imagem.."):
            try:
                response = asyncio.run(main=post_request(url=f"{URL}/inference/{face.get('name')}/{image_id}", image=img_path))
            except Exception as error:
                st.error(body=str(error))
                st.stop()
            finally:
                os.remove(path=img_path)
        if response is None:
            st.error(body="Não foi possível gerar a imagem.")
            st.stop()
        with Image.open(fp=io.BytesIO(initial_bytes=response), mode="r") as pil_img:
            frame_placeholder.image(image=pil_img, caption=f"Rosto trocado com {face.get('name').replace("_", " ").upper()}")
    else:
        st.error(body="Nenhuma camera foi detectada.")

    if st.button(label="Reinicar"):
        st.rerun()

    cap.release()
    cv2.destroyAllWindows()
