import asyncio
import io
import json
import os

abs_filepath = os.path.dirname(p=os.path.abspath(path=__file__))
abs_filepath = abs_filepath.split(sep=os.sep)[:-1]
abs_filepath = os.path.join(os.sep, *abs_filepath)

import httpx
import streamlit as st
from PIL import Image


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


with open(file=os.path.join(abs_filepath, "demo", "config.json"), mode="r") as json_buffer:
    config = json.load(fp=json_buffer)
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
    camera = st.camera_input(label="Camera", label_visibility="hidden")
    if camera:
        pass
